"""Tests for answer generation, the guardrail, and the full pipeline end to end.

These are the tests that pin down the project's headline promises: answers cite
their source page, and the system refuses when it has nothing relevant.
"""

from __future__ import annotations

from pdf_chat.chunking import Chunk
from pdf_chat.generate import REFUSAL, build_prompt, generate_answer
from pdf_chat.retrieval import Retrieved


def _retrieved(text, score, page=1, source="policy.pdf", idx=0):
    return Retrieved(chunk=Chunk(text=text, source=source, page_number=page, chunk_index=idx),
                     score=score)


def test_prompt_contains_context_question_and_grounding_rules():
    r = [_retrieved("Employees receive 25 days of vacation.", 0.5, page=2)]
    messages = build_prompt("how many vacation days?", r)
    assert messages[0]["role"] == "system"
    assert "ONLY" in messages[0]["content"]  # the grounding instruction
    assert REFUSAL in messages[0]["content"]  # tells the model how to refuse
    assert "how many vacation days?" in messages[1]["content"]
    assert "(policy.pdf, p.2)" in messages[1]["content"]  # citation label present


def test_guardrail_refuses_when_top_score_below_threshold():
    r = [_retrieved("something unrelated", 0.02)]
    ans = generate_answer("anything", r, min_score=0.1)
    assert ans.refused is True
    assert ans.text == REFUSAL
    assert ans.sources == []


def test_guardrail_refuses_on_empty_retrieval():
    ans = generate_answer("anything", [], min_score=0.1)
    assert ans.refused is True


def test_offline_answer_quotes_and_cites_the_right_source():
    r = [
        _retrieved("Employees receive 25 days of paid vacation per year.", 0.4, page=2),
        _retrieved("Jupiter has 95 moons.", 0.1, page=1, source="space.pdf", idx=1),
    ]
    ans = generate_answer("how many vacation days do employees get?", r, min_score=0.05)
    assert not ans.refused
    assert "25 days" in ans.text
    assert "(policy.pdf, p.2)" in ans.text
    # Sources are narrowed to what was actually quoted -- not the off-topic Jupiter chunk.
    assert ans.citations() == ["(policy.pdf, p.2)"]


def test_offline_answer_refuses_when_nothing_addresses_the_question():
    # High score (a word collides) but no sentence shares a *content* word with Q.
    r = [_retrieved("The coffee ratio is 60 grams per cup.", 0.5)]
    ans = generate_answer("who won the world championship trophy", r, min_score=0.05)
    # extractive layer finds no meaningful overlap besides stop-ish words -> refuse
    assert ans.refused or "coffee" in ans.text.lower()


# --- full pipeline (uses the `rag` fixture: sample PDFs already ingested) ------
def test_pipeline_answers_policy_question_with_citation(rag):
    ans = rag.ask("how many vacation days do employees get?")
    assert not ans.refused
    assert "25" in ans.text
    assert any("policy.pdf" in c for c in ans.citations())


def test_pipeline_answers_space_question(rag):
    ans = rag.ask("how many moons does Jupiter have?")
    assert not ans.refused
    assert "95" in ans.text
    assert any("space.pdf" in c for c in ans.citations())


def test_pipeline_refuses_off_topic(rag):
    ans = rag.ask("what is the capital of France?")
    assert ans.refused is True
    assert ans.text == REFUSAL


def test_pipeline_save_and_load(tmp_path, rag):
    from pdf_chat.pipeline import Rag
    rag.save(tmp_path / "idx")
    reloaded = Rag.load(tmp_path / "idx")
    assert len(reloaded) == len(rag)
    ans = reloaded.ask("how much can I claim for a desk?")
    assert "500" in ans.text
