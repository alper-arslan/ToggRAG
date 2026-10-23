from pathlib import Path

import pymupdf

from toggrag.models import Heading, Page


def read_pages(pdf_path: Path) -> list[Page]:
    """Return the text of every PDF page.

    Page numbers are 1-based, as shown by a PDF viewer. Image-only pages
    (such as covers) come back with empty text.
    """
    with pymupdf.open(pdf_path) as pdf:
        return [
            Page(page=number, text=page.get_text())
            for number, page in enumerate(pdf, start=1)
        ]


def read_headings(pdf_path: Path) -> list[Heading]:
    """Return the PDF's bookmark outline, one Heading per entry.

    The outline carries the chapter > section > subsection hierarchy and the
    page each heading starts on, so headings do not have to be guessed from
    font sizes.
    """
    with pymupdf.open(pdf_path) as pdf:
        return [
            Heading(level=level, title=title, page=page)
            for level, title, page in pdf.get_toc()
        ]
