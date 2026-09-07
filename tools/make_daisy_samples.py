"""Generate sample documents for testing the DAISY book feature.

Produces the same small book ("A DAISY Test Book") in every format the
project's parsers support, so the DAISY pipeline can be tested end to end:

* ``daisy_test.md``    -- Markdown with # .. ###### headings
* ``daisy_test.txt``   -- plain text (paragraphs, no heading markup)
* ``daisy_test.html``  -- HTML with <h1> .. <h6>
* ``daisy_test.docx``  -- Word document with Heading 1-3 styles
* ``daisy_test.pdf``   -- two-page PDF with title-like heading lines
* ``daisy_test.epub``  -- minimal EPUB 2 with semantic headings

Run:  python tools/make_daisy_samples.py [output_dir]
"""

from __future__ import annotations

import os
import sys
import zipfile

# ---------------------------------------------------------------------------
# Shared book content
# ---------------------------------------------------------------------------
TITLE = "A DAISY Test Book"

CHAPTERS = [
    ("Introduction", 1, [
        "This is the introduction of the DAISY test book. It has a few "
        "paragraphs of text so that audio file creation can be exercised.",
        "The introduction is followed by several chapters with headings of "
        "different levels, to test both 'Heading style 1 only' and 'Break on "
        "every heading' chapter splitting.",
    ]),
    ("Getting Started", 1, [
        "Getting started explains the very basics. Everything in this book "
        "is intentionally short and simple.",
        "A chapter can contain sub sections with deeper headings.",
    ]),
    ("First Steps", 2, [
        "This is a level two section inside Getting Started. It belongs to "
        "its parent chapter but is still a heading of its own.",
    ]),
    ("Advanced Topics", 1, [
        "Advanced topics go into more detail about the format and the "
        "tools used to create talking books.",
    ]),
    ("Troubleshooting", 2, [
        "Troubleshooting covers common problems and their solutions. "
        "For example, what to do when a reader cannot open the book.",
    ]),
    ("Conclusion", 1, [
        "This is the conclusion. The book ends here with a final word and "
        "a summary of everything that was covered.",
    ]),
]


def _paragraphs_text() -> str:
    parts = [TITLE]
    for title, level, paras in CHAPTERS:
        parts.append("")
        parts.append("#" * level + " " + title)
        parts.append("")
        parts.extend(paras)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def write_markdown(path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_paragraphs_text())


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------
def write_text(path: str) -> None:
    lines = [TITLE, ""]
    for _title, _level, paras in CHAPTERS:
        lines.extend(paras)
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def write_html(path: str) -> None:
    body = [f"<h1>{TITLE}</h1>"]
    for title, level, paras in CHAPTERS:
        body.append(f"<h{level}>{title}</h{level}>")
        body.extend(f"<p>{p}</p>" for p in paras)
    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        f"<meta charset=\"utf-8\">\n<title>{TITLE}</title>\n</head>\n<body>\n"
        + "\n".join(body) +
        "\n</body>\n</html>\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ---------------------------------------------------------------------------
# DOCX (python-docx)
# ---------------------------------------------------------------------------
def write_docx(path: str) -> None:
    from docx import Document  # type: ignore

    doc = Document()
    doc.add_heading(TITLE, level=1)
    for title, level, paras in CHAPTERS:
        doc.add_heading(title, level=level)
        for para in paras:
            doc.add_paragraph(para)
    doc.save(path)


# ---------------------------------------------------------------------------
# PDF (hand-crafted, one page per chapter-group, valid xref table)
# ---------------------------------------------------------------------------
def write_pdf(path: str) -> None:
    """Write a small two-page PDF with heading lines (Helvetica, ASCII text)."""
    pages = []
    for group in (CHAPTERS[:3], CHAPTERS[3:]):
        lines = [("A DAISY Test Book", 18)]
        for title, _level, paras in group:
            lines.append((title, 14))
            lines.extend((p, 10) for p in paras)
        pages.append(lines)

    # Build the content streams for every page.
    contents = []
    for page_lines in pages:
        parts = []
        y = 720
        for text, size in page_lines:
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            parts.append(f"BT /F1 {size} Tf 72 {y} Td ({escaped}) Tj ET")
            y -= 22
        stream = "\n".join(parts).encode("latin-1")
        contents.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    # Object numbering: 1 catalog, 2 pages tree, 3..N pages, N+1 font,
    # then the content streams.
    n_pages = len(contents)
    font_no = 2 + n_pages + 1
    objs = [None]
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = b" ".join(b"%d 0 R" % (3 + i) for i in range(n_pages))
    objs.append(b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % n_pages)
    for i in range(n_pages):
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (font_no, font_no + 1 + i)
        )
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objs.extend(contents)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs[1:], start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % len(objs)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objs), xref_pos)
    )
    with open(path, "wb") as fh:
        fh.write(out)


# ---------------------------------------------------------------------------
# EPUB (minimal EPUB 2 with semantic headings)
# ---------------------------------------------------------------------------
def write_epub(path: str) -> None:
    chapter_html = []
    spine = []
    manifest = []
    for i, (title, level, paras) in enumerate(CHAPTERS, start=1):
        fname = f"chapter{i}.xhtml"
        body = f"<h{level}>{title}</h{level}>"
        body += "".join(f"<p>{p}</p>" for p in paras)
        chapter_html.append(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.1//EN\" "
            "\"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd\">\n"
            f"<html xmlns=\"http://www.w3.org/1999/xhtml\">\n<head>\n"
            f"<title>{title}</title>\n</head>\n<body>\n{body}\n</body>\n</html>"
        )
        spine.append(f'<itemref idref="c{i}"/>')
        manifest.append(
            f'<item id="c{i}" href="{fname}" media-type="application/xhtml+xml"/>'
        )

    opf = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<package xmlns=\"http://www.idpf.org/2007/opf\" unique-identifier=\"uid\" version=\"2.0\">\n"
        "  <metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n"
        f"    <dc:title>{TITLE}</dc:title>\n"
        "    <dc:identifier id=\"uid\">urn:uuid:test-daisy-book</dc:identifier>\n"
        "    <dc:language>en</dc:language>\n"
        "  </metadata>\n"
        "  <manifest>\n"
        "    <item id=\"ncx\" href=\"toc.ncx\" media-type=\"application/x-dtbncx+xml\"/>\n"
        + "\n".join(manifest) +
        "\n  </manifest>\n  <spine toc=\"ncx\">\n"
        + "\n".join(spine) +
        "\n  </spine>\n</package>\n"
    )
    ncx = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<ncx xmlns=\"http://www.daisy.org/z3986/2005/ncx/\" version=\"2005-1\">\n"
        "  <head><meta name=\"dtb:uid\" content=\"urn:uuid:test-daisy-book\"/></head>\n"
        "  <docTitle><text>A DAISY Test Book</text></docTitle>\n"
        "  <navMap>\n"
        + "\n".join(
            f'    <navPoint id="n{i}" playOrder="{i}"><navLabel><text>{t}</text></navLabel>'
            f'<content src="chapter{i}.xhtml"/></navPoint>'
            for i, (t, _l, _p) in enumerate(CHAPTERS, start=1)
        ) +
        "\n  </navMap>\n</ncx>\n"
    )
    container = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<container version=\"1.0\" "
        "xmlns=\"urn:oasis:names:tc:opendocument:xmlns:container\">\n"
        "  <rootfiles><rootfile full-path=\"OEBPS/content.opf\" "
        "media-type=\"application/oebps-package+xml\"/></rootfiles>\n"
        "</container>\n"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container)
        for i, (title, level, paras) in enumerate(CHAPTERS, start=1):
            zf.writestr(f"OEBPS/chapter{i}.xhtml", chapter_html[i - 1])
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/toc.ncx", ncx)


# ---------------------------------------------------------------------------
def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("samples", "daisy")
    os.makedirs(out_dir, exist_ok=True)
    write_markdown(os.path.join(out_dir, "daisy_test.md"))
    write_text(os.path.join(out_dir, "daisy_test.txt"))
    write_html(os.path.join(out_dir, "daisy_test.html"))
    write_docx(os.path.join(out_dir, "daisy_test.docx"))
    write_pdf(os.path.join(out_dir, "daisy_test.pdf"))
    write_epub(os.path.join(out_dir, "daisy_test.epub"))
    print("Samples written to", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()