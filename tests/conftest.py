"""Shared pytest fixtures. These build a tiny corpus once so tests run fast and offline."""

from __future__ import annotations

import pytest

from pdf_chat.loading import Page
from pdf_chat.pipeline import Rag
from pdf_chat.simple_pdf import write_text_pdf

# A minimal three-topic corpus, small enough to reason about by hand.
SAMPLE_PAGES = {
    "policy.pdf": [
        ["COMPANY POLICY", "Full-time employees receive 25 days of paid vacation per year.",
         "Sick leave is capped at 10 paid days per year."],
        ["EQUIPMENT", "The company provides a laptop and a monitor to every employee.",
         "You may claim up to 500 dollars for a desk and chair."],
    ],
    "space.pdf": [
        ["PLANETS", "Jupiter is the largest planet and has 95 confirmed moons.",
         "Mars is the red planet because its soil is rich in iron oxide."],
    ],
}


@pytest.fixture
def sample_pdfs(tmp_path):
    """Write the sample PDFs into a temp dir and return their paths."""
    paths = []
    for name, pages in SAMPLE_PAGES.items():
        paths.append(write_text_pdf(tmp_path / name, pages))
    return paths


@pytest.fixture
def rag(sample_pdfs):
    """A Rag engine with the sample PDFs already ingested (fully offline)."""
    engine = Rag()
    engine.ingest(sample_pdfs)
    return engine


@pytest.fixture
def sample_page():
    """A single hand-made Page for chunking tests."""
    return Page(source="demo.txt", page_number=1, text="word " * 400)
