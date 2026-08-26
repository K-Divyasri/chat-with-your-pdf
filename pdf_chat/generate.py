"""Step 6 of RAG: turn retrieved passages into an answer -- WITH citations, or "I don't know".

This file is where "retrieval" meets "generation". It does three jobs:

  1. The guardrail. If the best retrieved passage is only weakly similar to the
     question, the honest move is to refuse: "I don't know based on these
     documents." A RAG system that makes something up when it has nothing to go on
     is worse than useless. We gate on the top similarity score.

  2. The prompt. This is the single most important habit in all of RAG: we build a
     prompt that says, in effect, "Answer using ONLY the context below, cite the
     page, and if it is not in the context, say you don't know." Then we paste the
     retrieved chunks in. The model is not recalling; it is reading.

  3. The two backends, one interface:
       OFFLINE (default)  No key, no network. We build a short answer by quoting
                          the most relevant retrieved sentences and attaching
                          their page citations. It is extractive, not fluent -- but
                          it proves the pipeline end to end and every citation is
                          real. It also never hallucinates, by construction.
       REAL               Sends the same prompt to a real LLM through LiteLLM.
                          Fluent prose, still grounded in (and citing) the context.

Whichever backend runs, the return type is the same `Answer`, so callers -- the
CLI, the Streamlit app, the tests -- never care which one produced it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from . import DEFAULT_MIN_SCORE
from .retrieval import Retrieved

REFUSAL = "I don't know based on the provided documents."


@dataclass
class Answer:
    """The final output: the text, the sources it leaned on, and the confidence."""

    text: str
    sources: list[Retrieved] = field(default_factory=list)
    top_score: float = 0.0
    refused: bool = False

    def citations(self) -> list[str]:
        """De-duplicated list of '(file, p.N)' tags across the sources used."""
        seen: list[str] = []
        for item in self.sources:
            tag = item.chunk.citation()
            if tag not in seen:
                seen.append(tag)
        return seen


# --------------------------------------------------------------------------- #
#  The prompt -- shared by both backends                                      #
# --------------------------------------------------------------------------- #
def build_prompt(question: str, retrieved: list[Retrieved]) -> list[dict[str, str]]:
    """Build the chat messages that ground the model in the retrieved context.

    Each context block is numbered and labelled with its citation so the model can
    refer to sources by page. The system message is where the grounding rules live
    -- copy this pattern; it is 80% of what makes RAG trustworthy.
    """
    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        tag = item.chunk.citation()
        context_blocks.append(f"[Source {i}] {tag}\n{item.chunk.text}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no context found)"

    system = (
        "You answer questions using ONLY the context passages provided. "
        "Every context passage is labelled with its source file and page. "
        "Rules:\n"
        "1. If the answer is in the context, answer concisely and cite the "
        "source like (file.pdf, p.N).\n"
        "2. If the context does not contain the answer, reply exactly: "
        f'"{REFUSAL}"\n'
        "3. Never use outside knowledge and never invent a citation."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
#  Offline extractive backend                                                 #
# --------------------------------------------------------------------------- #
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _best_sentences(question: str, retrieved: list[Retrieved], max_sentences: int = 2) -> list[tuple[str, str]]:
    """Pick the sentences from the retrieved chunks that overlap the question most.

    Returns (sentence, citation) pairs. This is a tiny extractive summariser: it
    scores each candidate sentence by how many question words it contains, so the
    offline answer actually responds to what was asked instead of dumping a chunk.
    """
    from .embedding import tokenize  # local import to avoid a cycle at module load

    q_words = set(tokenize(question))
    scored: list[tuple[float, int, str, str]] = []
    for rank, item in enumerate(retrieved):
        tag = item.chunk.citation()
        for sentence in _SENTENCE_RE.split(item.chunk.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = set(tokenize(sentence))
            overlap = len(q_words & words)
            # Break ties by retrieval rank (earlier chunk = more relevant) and length.
            scored.append((overlap, -rank, sentence, tag))
    # Keep only sentences that share at least one meaningful word with the question.
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    picked: list[tuple[str, str]] = []
    for _, _neg_rank, sentence, tag in scored:
        if sentence not in [p[0] for p in picked]:
            picked.append((sentence, tag))
        if len(picked) >= max_sentences:
            break
    return picked


def offline_answer(question: str, retrieved: list[Retrieved]) -> str:
    """Build an extractive answer by quoting the most on-point retrieved sentences."""
    picks = _best_sentences(question, retrieved)
    if not picks:
        # Retrieval returned something, but nothing in it addresses the question.
        return REFUSAL
    lines = [f"{sentence} {tag}" for sentence, tag in picks]
    return "Based on the documents: " + " ".join(lines)


# --------------------------------------------------------------------------- #
#  Real LLM backend (LiteLLM)                                                 #
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = os.environ.get("PDFCHAT_MODEL", "gemini/gemini-1.5-flash")


def real_answer(question: str, retrieved: list[Retrieved], model: str) -> str:
    """Send the grounded prompt to a real model via LiteLLM. Imported lazily."""
    from litellm import completion  # noqa: PLC0415

    messages = build_prompt(question, retrieved)
    response = completion(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
#  The one function callers use                                               #
# --------------------------------------------------------------------------- #
def generate_answer(
    question: str,
    retrieved: list[Retrieved],
    *,
    offline: bool = True,
    min_score: float = DEFAULT_MIN_SCORE,
    model: str | None = None,
) -> Answer:
    """Produce an `Answer` from retrieved chunks, honouring the confidence guardrail.

    The guardrail runs first and identically for both backends: if the top
    retrieved score is below `min_score`, we refuse rather than guess. Only if we
    clear the bar do we actually generate.
    """
    top_score = retrieved[0].score if retrieved else 0.0
    if not retrieved or top_score < min_score:
        return Answer(text=REFUSAL, sources=[], top_score=top_score, refused=True)

    if offline:
        text = offline_answer(question, retrieved)
    else:
        text = real_answer(question, retrieved, model or DEFAULT_MODEL)

    refused = text.strip().lower().startswith("i don't know")
    if refused:
        sources: list[Retrieved] = []
    elif offline:
        # Offline mode quotes specific sentences and tags them inline, so we can
        # narrow "Sources" to exactly the chunks the answer actually drew from
        # (fall back to all retrieved if, somehow, none matched).
        used = [r for r in retrieved if r.chunk.citation() in text]
        sources = used or retrieved
    else:
        # A real LLM reads all the context we gave it; show the full retrieved set.
        sources = retrieved
    return Answer(text=text, sources=sources, top_score=top_score, refused=refused)
