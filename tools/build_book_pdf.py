"""Build the complete HTML book as a single PDF using Microsoft Edge."""

from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = ROOT / "docs" / "book"
INDEX = BOOK_DIR / "index.html"
COMBINED = BOOK_DIR / "_book_combined.html"
OUTPUT = BOOK_DIR / "Building_AI_Voice_Studio.pdf"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def extract_main_content(document: str) -> str:
    start_marker = '<div class="main-content">'
    start = document.find(start_marker)
    body_end = document.rfind("</body>")
    container_end = document.rfind("</div>", 0, body_end)
    main_end = document.rfind("</div>", 0, container_end)
    if start == -1 or main_end <= start:
        raise ValueError("Could not find the document main-content container")
    return document[start + len(start_marker):main_end].strip()


def linked_chapters(index: str) -> list[Path]:
    links = re.findall(r'href="chapters/([^"#]+\.html)"', index)
    unique_links = list(dict.fromkeys(links))
    paths = [BOOK_DIR / "chapters" / link for link in unique_links]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing linked book files: {names}")
    return paths


def build_combined_html() -> tuple[int, int]:
    index = INDEX.read_text(encoding="utf-8")
    chapters = linked_chapters(index)
    sections = [extract_main_content(index)]
    for chapter in chapters:
        title = html.escape(chapter.stem.replace("_", " ").title())
        content = extract_main_content(chapter.read_text(encoding="utf-8"))
        sections.append(f'<section class="book-section" data-source="{chapter.name}">'
                        f'<div class="chapter-source">{title}</div>{content}</section>')

    style = (BOOK_DIR / "css" / "style.css").read_text(encoding="utf-8")
    style += """
.book-section { page-break-before: always; }
.chapter-source { display: none; }
@page { size: Letter; margin: 0.65in; }
@media print {
  .book-section { page-break-before: always; }
  h1, h2, h3, h4 { page-break-after: avoid; }
  pre, table, figure, .note, .tip, .warning, .info, .exercise, .project,
  .line-explanation { page-break-inside: avoid; }
}
"""
    document = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<title>Building AI Voice Studio</title><style>'
        + style
        + "</style></head><body><div class=\"book-container\"><main class=\"main-content\">"
        + "\n".join(sections)
        + "</main></div></body></html>"
    )
    COMBINED.write_text(document, encoding="utf-8")
    return len(chapters), len(sections)


def main() -> None:
    if not EDGE.is_file():
        raise FileNotFoundError(f"Microsoft Edge was not found at {EDGE}")
    chapter_count, section_count = build_combined_html()
    subprocess.run(
        [
            str(EDGE),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={OUTPUT}",
            COMBINED.resolve().as_uri(),
        ],
        check=True,
    )
    print(f"Created {OUTPUT}")
    print(f"Included {chapter_count} linked chapters/appendices in {section_count} sections")
    print(f"PDF size: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
