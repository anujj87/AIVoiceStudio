"""Document parsers.

Every parser returns a :class:`Document`:

* ``blocks`` -- ordered list of :class:`Block` items:
  ``Block(kind="text"|"heading", level, text, page)``
* ``pages``  -- the number of pages (real pages for PDF, pseudo-pages for
  free-flowing formats).

Headings are extracted per format:
* DOCX -> paragraphs whose style starts with "Heading" (level 1-6)
* Markdown -> ATX headings (# .. ######)
* HTML -> <h1>..<h6>
* PDF -> best-effort heuristic on short title-like lines
* TXT / clipboard -> paragraph blocks (no heading markup)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable, List, Optional

from ..constants import TEXT_PAGE_CHARS

log = logging.getLogger(__name__)


@dataclass
class Block:
    kind: str  # "text" | "heading"
    text: str
    page: int = 0
    level: int = 0  # heading level 1-6 (0 for plain text)


@dataclass
class Document:
    blocks: List[Block] = field(default_factory=list)
    source: str = ""
    format: str = ""

    @property
    def pages(self) -> int:
        if not self.blocks:
            return 0
        return max(b.page for b in self.blocks) + 1

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


class ParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _page_from_offset(offset: int) -> int:
    return offset // TEXT_PAGE_CHARS


def _assign_pages(blocks: List[Block], page_size: int = TEXT_PAGE_CHARS) -> List[Block]:
    """Give each block a pseudo page index based on cumulative character count."""
    offset = 0
    for block in blocks:
        block.page = _page_from_offset(offset)
        offset += max(len(block.text), 1)
    return blocks


def _split_long_block(block: Block, page_size: int = TEXT_PAGE_CHARS) -> List[Block]:
    """Split a very long text block into page-sized chunks at paragraph breaks."""
    if len(block.text) <= page_size:
        return [block]
    parts = []
    paragraphs = re.split(r"\n\s*\n", block.text)
    current, current_len = [], 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > page_size and current:
            parts.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para) + 2
    if current:
        parts.append("\n\n".join(current))
    return [
        Block(kind=block.kind, level=block.level, text=p, page=block.page) for p in parts
    ]


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ParseError("Could not decode the file as text")


# ---------------------------------------------------------------------------
# Individual parsers
# ---------------------------------------------------------------------------
def parse_txt(path: str, encoding: Optional[str] = None) -> Document:
    with open(path, "rb") as fh:
        data = fh.read()
    text = data.decode(encoding) if encoding else _decode_text(data)
    blocks: List[Block] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            blocks.append(Block(kind="text", text=para))
    _assign_pages(blocks)
    return Document(blocks=blocks, source=path, format="txt")


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def parse_markdown(path: str) -> Document:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    blocks: List[Block] = []
    para_lines: List[str] = []

    def flush():
        if para_lines:
            text_ = " ".join(line.strip() for line in para_lines if line.strip())
            if text_:
                blocks.append(Block(kind="text", text=text_))
            para_lines.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("```"):
            flush()
            continue
        match = _HEADING_RE.match(stripped)
        if match:
            flush()
            blocks.append(Block(kind="heading", level=len(match.group(1)), text=match.group(2).strip()))
            continue
        # Inline markdown: remove links, emphasis, code ticks.
        cleaned = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", stripped)
        cleaned = re.sub(r"[*_`~]{1,3}", "", cleaned)
        cleaned = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", cleaned)
        cleaned = re.sub(r"^\s*\|.*\|\s*$", "", cleaned)  # table rows dropped
        if cleaned:
            para_lines.append(cleaned)
    flush()
    _assign_pages(blocks)
    return Document(blocks=blocks, source=path, format="markdown")


class _HtmlTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: List[Block] = []
        self._current: List[str] = []
        self._heading_stack: List[int] = []
        self._in_heading: Optional[int] = None
        self._skip_depth = 0

    def _flush(self):
        if self._current:
            text = " ".join(" ".join(self._current).split())
            if text:
                self.blocks.append(Block(kind="text", text=text))
            self._current = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head", "noscript"):
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._in_heading = int(tag[1])
        elif tag in ("p", "div", "br", "li", "tr", "section", "article"):
            self._flush()

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head", "noscript"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if self._in_heading and re.fullmatch(r"h[1-6]", tag):
            self._flush()
            if self.blocks and self.blocks[-1].kind == "text":
                # The extractor has flushed heading text as plain text; fix kind.
                last = self.blocks[-1]
                last.kind = "heading"
                last.level = self._in_heading
            self._in_heading = None
        elif tag in ("p", "div", "li", "tr"):
            self._flush()

    def handle_data(self, data):
        if self._skip_depth:
            return
        self._current.append(data)


def parse_html(path: str) -> Document:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    parser = _HtmlTextExtractor()
    parser.feed(text)
    parser.close()
    blocks = parser.blocks
    _assign_pages(blocks)
    return Document(blocks=blocks, source=path, format="html")


def parse_pdf(
    path: str,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> Document:
    try:
        from pypdf import PdfReader  # lazy: heavy dependency
    except ImportError as exc:
        raise ParseError("PDF support requires the 'pypdf' package.") from exc
    try:
        reader = PdfReader(path)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Could not open PDF: {exc}") from exc
    blocks: List[Block] = []
    total_pages = len(reader.pages)
    for page_idx, page in enumerate(reader.pages):
        if on_progress is not None and total_pages > 1:
            on_progress(
                f"Reading the document... page {page_idx + 1} of {total_pages}",
                0.05 + 0.8 * (page_idx / total_pages),
            )
        try:
            page_text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            page_text = ""
        if not page_text.strip():
            continue
        # Best-effort heading detection: short title-like lines.
        for line in re.split(r"\n", page_text):
            line = line.strip()
            if not line:
                continue
            if _looks_like_heading(line):
                blocks.append(Block(kind="heading", level=1, text=line, page=page_idx))
            else:
                blocks.append(Block(kind="text", text=line, page=page_idx))
    return Document(blocks=blocks, source=path, format="pdf")


_HEADING_HINT = re.compile(
    r"^(chapter\s+\d+|(unit|part|section|lesson|module)\s+\d+|"
    r"\d+(\.\d+)*\.?\s+[A-Z])",
    re.IGNORECASE,
)


def _looks_like_heading(line: str) -> bool:
    if len(line) > 70:
        return False
    if _HEADING_HINT.search(line):
        return True
    # Short title-case line followed by a shorter-than-average length.
    words = line.split()
    if 2 <= len(words) <= 8 and line[0].isupper():
        # Avoid common sentence openers.
        if re.match(r"^(the|a|an|this|that|these|those|there|it|he|she|they|we|you)\b", line, re.I):
            return False
        return True
    return False


def parse_docx(path: str) -> Document:
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise ParseError("DOCX support requires the 'python-docx' package.") from exc
    try:
        document = docx.Document(path)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Could not open DOCX: {exc}") from exc
    blocks: List[Block] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "") if para.style else ""
        match = re.match(r"heading\s*([1-6])", style, re.IGNORECASE)
        if match:
            blocks.append(Block(kind="heading", level=int(match.group(1)), text=text))
        else:
            blocks.append(Block(kind="text", text=text))
    # Tables: join cells into text blocks.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(Block(kind="text", text=" | ".join(cells)))
    _assign_pages(blocks)
    return Document(blocks=blocks, source=path, format="docx")


def parse_doc(path: str) -> Document:
    """Legacy .doc via Word COM automation (Windows + MS Word required)."""
    try:
        import win32com.client  # pywin32, optional
    except ImportError as exc:
        raise ParseError(
            "Legacy .doc files need Microsoft Word with the 'pywin32' package "
            "installed. Save the file as .docx or plain text instead."
        ) from exc
    try:
        import pythoncom  # noqa: PLC0415

        pythoncom.CoInitialize()
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(path), ReadOnly=True)
        text = doc.Content.Text
        doc.Close(False)
        word.Quit()
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Could not read .doc file (is Word installed?): {exc}") from exc
    blocks = [Block(kind="text", text=p.strip()) for p in text.splitlines() if p.strip()]
    _assign_pages(blocks)
    return Document(blocks=blocks, source=path, format="doc")


def parse_epub(
    path: str,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> Document:
    """Parse an EPUB e-book file.

    EPUB is a ZIP archive containing XHTML content. We extract the XHTML files,
    parse them with the HTML parser, and return the combined document.
    """
    import zipfile  # noqa: PLC0415
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    try:
        with zipfile.ZipFile(path, "r") as zf:
            # Read the OPF manifest to find content files in reading order
            opf_path = _find_opf(zf)
            if opf_path:
                content_files, base_dir = _parse_opf(zf, opf_path)
            else:
                # Fallback: use all XHTML files sorted by name
                content_files = sorted(
                    n for n in zf.namelist()
                    if n.lower().endswith((".xhtml", ".html", ".htm"))
                    and not n.startswith(("__MACOSX", "."))
                )
                base_dir = ""

            blocks: List[Block] = []
            total_files = len(content_files)
            for i, filename in enumerate(content_files):
                if on_progress is not None and total_files > 1:
                    on_progress(
                        f"Reading the document... chapter {i + 1} of {total_files}",
                        0.05 + 0.8 * (i / total_files),
                    )
                try:
                    raw = zf.read(filename)
                except KeyError:
                    continue
                html_text = raw.decode("utf-8", errors="replace")
                parser = _HtmlTextExtractor()
                parser.feed(html_text)
                parser.close()
                for block in parser.blocks:
                    block.page = i  # use chapter index as page
                    blocks.append(block)
    except zipfile.BadZipFile as exc:
        raise ParseError(f"Could not read EPUB file: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Could not parse EPUB: {exc}") from exc

    if not blocks:
        raise ParseError("The EPUB file contains no readable text.")
    _assign_pages(blocks)
    return Document(blocks=blocks, source=path, format="epub")


def _find_opf(zf: zipfile.ZipFile) -> str | None:
    """Find the OPF file path from the EPUB container.xml."""
    try:
        container = zf.read("META-INF/container.xml").decode("utf-8", errors="replace")
        root = ET.fromstring(container)
        ns = {"mc": "urn:oasis:names:tc:opendocument:xmlns:container"}
        for rootfile in root.findall(".//mc:rootfile", ns):
            media_type = rootfile.get("media-type", "")
            if "opendocument" in media_type or rootfile.get("full-path", ""):
                return rootfile.get("full-path")
    except (KeyError, ET.ParseError):
        pass
    # Fallback: look for any .opf file
    for name in zf.namelist():
        if name.endswith(".opf"):
            return name
    return None


def _parse_opf(
    zf: zipfile.ZipFile, opf_path: str
) -> tuple[List[str], str]:
    """Parse the OPF manifest and spine to get content files in reading order.

    Returns (list_of_xhtml_paths, base_directory).
    """
    try:
        raw = zf.read(opf_path).decode("utf-8", errors="replace")
    except KeyError:
        return [], ""

    root = ET.fromstring(raw)
    base_dir = os.path.dirname(opf_path)

    # Build manifest: id -> href
    manifest = {}
    for item in root.findall(".//{http://www.idpf.org/2007/opf}item"):
        item_id = item.get("id", "")
        href = item.get("href", "")
        manifest[item_id] = href

    # Spine: ordered list of itemref idrefs
    spine_ids = []
    for itemref in root.findall(".//{http://www.idpf.org/2007/opf}spine/{http://www.idpf.org/2007/opf}itemref"):
        idref = itemref.get("idref", "")
        if idref in manifest:
            spine_ids.append(idref)

    content_files = []
    for item_id in spine_ids:
        href = manifest[item_id]
        full_path = os.path.join(base_dir, href) if base_dir else href
        # Normalize path separators
        full_path = full_path.replace("\\", "/")
        if full_path.lower().endswith((".xhtml", ".html", ".htm", ".xml")):
            content_files.append(full_path)

    return content_files, base_dir


def parse_clipboard(text: str) -> Document:
    blocks: List[Block] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            blocks.append(Block(kind="text", text=para))
    _assign_pages(blocks)
    return Document(blocks=blocks, source="<clipboard>", format="clipboard")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
_EXT_PARSERS = {
    ".txt": parse_txt,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".html": parse_html,
    ".htm": parse_html,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".doc": parse_doc,
    ".epub": parse_epub,
}


def parse_document(
    path: str,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> Document:
    """Parse a file into a Document.

    ``on_progress(message, fraction)`` is an optional callback invoked while
    parsing so the GUI can drive a progress dialog (PDF and EPUB pages are
    reported one by one; other formats report start/end phases).
    """
    ext = os.path.splitext(path)[1].lower()
    parser = _EXT_PARSERS.get(ext)
    if parser is None:
        raise ParseError(
            f"Unsupported file type '{ext or '(none)'}'. Supported: "
            + ", ".join(sorted(_EXT_PARSERS))
        )
    if on_progress is not None:
        on_progress("Reading the document...", 0.02)
    if ext == ".pdf":
        doc = parse_pdf(path, on_progress=on_progress)
    elif ext == ".epub":
        doc = parse_epub(path, on_progress=on_progress)
    else:
        doc = parser(path)
    if on_progress is not None:
        on_progress("Analyzing the document...", 0.9)
    # Split oversized blocks so pages stay uniform.
    split_blocks: List[Block] = []
    for block in doc.blocks:
        split_blocks.extend(_split_long_block(block))
    _assign_pages(split_blocks)
    doc.blocks = split_blocks
    return doc
