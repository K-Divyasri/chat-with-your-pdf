"""Step 2 of RAG: cut each page into small, overlapping passages ("chunks").

Why chunk at all? Two reasons.

  1. Retrieval gets sharper. If you embed a whole 500-word page as one vector,
     that vector is a blurry average of everything on the page. A question about
     one sentence on it may not match well. Smaller chunks = more focused
     vectors = more precise retrieval.

  2. Prompts have a budget. You cannot paste an entire book into the model. You
     retrieve a handful of small chunks and paste only those.

Why *overlap* them? Because a fixed cut is blind to meaning -- it will happily
slice a sentence, or split a question from its answer, right at a boundary. By
repeating the last little bit of one chunk at the start of the next, an idea that
straddles the seam still lives, whole, inside at least one chunk. The usual rule
of thumb is a chunk of a few hundred characters with roughly 10-20% overlap; the
defaults here (500 / 100) follow that. There is a whole notebook (02) and a lab
(07) devoted to feeling why the numbers matter.

We chunk each page *separately* so a chunk never spans two pages -- that keeps its
page citation honest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loading import Page


@dataclass
class Chunk:
    """A small passage of text plus everything needed to cite it."""

    text: str
    source: str
    page_number: int
    chunk_index: int  # position of this chunk within the whole corpus (assigned later)

    def citation(self) -> str:
        """A short human-readable source tag, e.g. '(solar_system.pdf, p.3)'."""
        return f"({self.source}, p.{self.page_number})"


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces/newlines into single spaces and trim the ends.

    PDF text extraction is messy -- stray newlines mid-sentence, double spaces.
    Cleaning it up first makes chunk boundaries predictable and embeddings a
    little cleaner. We keep it deliberately simple.
    """
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """Split one string into overlapping windows of about `chunk_size` characters.

    We slide a window of `chunk_size` characters across the text, then step
    forward by `chunk_size - overlap` so each window shares `overlap` characters
    with the one before it. To avoid chopping words in half we nudge each cut to
    the nearest space when there is one nearby.

    Returns a list of chunk strings. An empty/blank input returns [].
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    text = _normalise_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer to end on a space so we don't split a word, but only if there is
        # one in the last quarter of the window (else just cut where we are).
        if end < len(text):
            space = text.rfind(" ", start + int(chunk_size * 0.75), end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap  # step back so the next window overlaps this one
    return chunks


def chunk_pages(
    pages: list[Page],
    *,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """Chunk a list of `Page`s into a flat list of `Chunk`s, numbered corpus-wide.

    Each chunk carries its source file and page number, so its citation survives
    all the way to the final answer. `chunk_index` is assigned here in reading
    order and is what the vector store uses to point back at the right chunk.
    """
    chunks: list[Chunk] = []
    running_index = 0
    for page in pages:
        for piece in chunk_text(page.text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                Chunk(
                    text=piece,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=running_index,
                )
            )
            running_index += 1
    return chunks
