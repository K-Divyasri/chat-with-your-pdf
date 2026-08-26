"""Step 1 of RAG: turn a PDF (or a .txt file) into text, keeping page numbers.

Page numbers matter more than anything else in this file. The whole selling
point of "chat with your PDF" over a plain chatbot is that every answer can
point back to *where* it came from -- "see page 3." We can only do that if we
remember, for every scrap of text, which page it was on. So the unit we pass
around the rest of the program is not a big string; it is a `Page`: its text and
its 1-based page number, tagged with which document it came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    """One page of one document."""

    source: str      # the file name, e.g. "solar_system.pdf" -- shown in citations
    page_number: int  # 1-based, the way a human counts pages
    text: str         # the extracted text of just this page


def load_pdf(path: str | Path) -> list[Page]:
    """Read a PDF and return one `Page` per page, in order.

    We import pypdf lazily (inside the function) so that code which only touches
    .txt files does not need pypdf installed. `extract_text()` can return None
    for an image-only page, so we guard against that and never crash.
    """
    from pypdf import PdfReader  # noqa: PLC0415 (lazy import on purpose)

    path = Path(path)
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(Page(source=path.name, page_number=index + 1, text=text))
    return pages


def load_txt(path: str | Path) -> list[Page]:
    """Read a plain-text file as a single-page document.

    Handy for notebooks and tests where a full PDF is overkill. A .txt has no
    real page breaks, so the whole file is "page 1".
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return [Page(source=path.name, page_number=1, text=text)]


def load_any(path: str | Path) -> list[Page]:
    """Load a .pdf or .txt by looking at the file extension. Raises on anything else."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".txt":
        return load_txt(path)
    raise ValueError(f"Don't know how to read {path.name!r}. Use a .pdf or .txt file.")


def load_folder(folder: str | Path) -> list[Page]:
    """Load every .pdf and .txt in a folder (sorted by name) into one flat list of pages."""
    folder = Path(folder)
    pages: list[Page] = []
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in {".pdf", ".txt"}:
            pages.extend(load_any(file))
    return pages
