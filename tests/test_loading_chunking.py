"""Tests for the load and chunk steps -- the parts that must preserve page numbers."""

from __future__ import annotations

from pdf_chat.chunking import chunk_pages, chunk_text
from pdf_chat.loading import load_pdf, load_txt


def test_pdf_round_trips_with_page_numbers(sample_pdfs):
    pages = load_pdf(sample_pdfs[0])  # policy.pdf has 2 pages
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert pages[0].source == "policy.pdf"
    assert "vacation" in pages[0].text.lower()
    assert "desk" in pages[1].text.lower()


def test_load_txt_is_single_page(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world", encoding="utf-8")
    pages = load_txt(f)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == "hello world"


def test_chunk_text_short_input_is_one_chunk():
    assert chunk_text("just a short sentence") == ["just a short sentence"]


def test_chunk_text_empty_is_no_chunks():
    assert chunk_text("   ") == []


def test_chunks_overlap():
    # Distinct words so we can see the overlap: the end of one chunk should
    # reappear at the start of the next.
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    # The tail of chunk 0 and the head of chunk 1 must share some words.
    tail = set(chunks[0].split()[-5:])
    head = set(chunks[1].split()[:15])
    assert tail & head, "adjacent chunks should overlap"


def test_chunk_size_controls_count(sample_page):
    few = chunk_pages([sample_page], chunk_size=1000, overlap=100)
    many = chunk_pages([sample_page], chunk_size=200, overlap=40)
    assert len(many) > len(few)


def test_chunk_pages_assigns_running_index_and_keeps_pages(sample_pdfs):
    pages = load_pdf(sample_pdfs[0])
    chunks = chunk_pages(pages, chunk_size=120, overlap=20)
    # chunk_index is 0,1,2,... across the whole corpus, in order.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Every chunk still knows its page, and pages are non-decreasing.
    page_nums = [c.page_number for c in chunks]
    assert page_nums == sorted(page_nums)
    assert set(page_nums) <= {1, 2}


def test_citation_format(sample_pdfs):
    chunks = chunk_pages(load_pdf(sample_pdfs[0]), chunk_size=120, overlap=20)
    assert chunks[0].citation() == "(policy.pdf, p.1)"
