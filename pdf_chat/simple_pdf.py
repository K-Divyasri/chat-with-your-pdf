"""Write a plain-text PDF with the standard library only -- no reportlab, no fpdf.

Why does this file exist? Because to *learn* "chat with your PDF" you need PDFs
to chat with, and we did not want to make you install a PDF-writing library just
to get some. A PDF that only contains text is simpler than it looks: it is a few
"objects" (a catalog, a page tree, one object per page, one text stream per page,
one font) plus a small table of byte-offsets at the end. We hand-write exactly
that. pypdf reads it straight back as clean text with the page breaks intact.

You will almost never write a PDF by hand in real life. Treat this as a curiosity
you can peek at, not something to memorise. The interesting file is loading.py,
which READS PDFs -- that part you will do for real.
"""

from __future__ import annotations

import io
from pathlib import Path


def _escape(text: str) -> str:
    r"""Backslash-escape the three characters that are special inside a PDF string:
    the backslash itself, and the two parentheses that delimit a string."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_text_pdf(path: str | Path, pages: list[list[str]]) -> Path:
    """Write `pages` (a list of pages, each a list of text lines) to a PDF at `path`.

    Every page uses 12pt Helvetica starting near the top-left, with the lines
    stacked downward. Returns the Path it wrote to.
    """
    path = Path(path)
    n_pages = len(pages)
    if n_pages == 0:
        raise ValueError("A PDF needs at least one page.")

    # Object numbering: 1 = catalog, 2 = page tree, then for each page a Page
    # object and a Contents (text stream) object, then finally the shared font.
    page_obj_ids = [3 + 2 * i for i in range(n_pages)]
    font_obj_id = 3 + 2 * n_pages
    objects: list[tuple[int, str]] = []

    objects.append((1, "<< /Type /Catalog /Pages 2 0 R >>"))
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects.append((2, f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>"))

    for i, lines in enumerate(pages):
        page_id = page_obj_ids[i]
        content_id = page_id + 1
        # A text object: begin (BT), pick font (Tf), set start point (Td), set
        # line height (TL), then one "show text" (Tj) + "next line" (T*) per line.
        parts = ["BT", "/F1 12 Tf", "72 720 Td", "16 TL"]
        for line in lines:
            parts.append(f"({_escape(line)}) Tj")
            parts.append("T*")
        parts.append("ET")
        stream = "\n".join(parts)
        objects.append((
            page_id,
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>",
        ))
        objects.append((
            content_id,
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        ))

    objects.append((font_obj_id, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    objects.sort()

    # Serialise, remembering where each object starts so we can build the xref table.
    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number, body in objects:
        offsets[number] = buffer.tell()
        buffer.write(f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1"))

    xref_pos = buffer.tell()
    highest = max(offsets) + 1
    buffer.write(f"xref\n0 {highest}\n".encode("latin-1"))
    buffer.write(b"0000000000 65535 f \n")  # the mandatory free entry for object 0
    for number in range(1, highest):
        buffer.write(f"{offsets[number]:010d} 00000 n \n".encode("latin-1"))
    buffer.write(
        f"trailer\n<< /Size {highest} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF".encode("latin-1")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return path
