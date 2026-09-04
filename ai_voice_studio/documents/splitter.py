"""Split a parsed Document into synthesis segments (SPEC 3.4).

Each segment has a ``title`` (used for the file name) and ``text`` (sent to the
TTS engine). Naming rules:

* page segments:      ``01 page 1``, ``02 page 2``, ...
* grouped pages:      ``01 pages 1 to 3``, ``02 pages 4 to 6``, ...
* heading segments:   ``01 <heading text>``, ``02 <heading text>``, ...
  (heading text is sanitized for use as a file name)

Modes:
* ``page_with_h1``  -- page by page; when a Heading-1 appears inside a page the
  text before it is emitted first, then the heading plus the rest of the page.
* ``page_only``     -- one segment per page (or N pages per file when
  ``pages_per_file`` > 1, used by the New Project wizard).
* ``h1_only``       -- each Heading-1 together with the text up to the next
  Heading-1; text before the first Heading-1 becomes a leading "page 1" file.
* ``all_headings``  -- every heading (levels 1-6) as its own short segment.
* ``one_file``      -- the whole document as a single segment (no resume in
  the middle of the one audio file).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from ..constants import (
    MODE_ALL_HEADINGS,
    MODE_H1_ONLY,
    MODE_ONE_FILE,
    MODE_PAGE_ONLY,
    MODE_PAGE_WITH_H1,
)
from ..util import sanitize_filename
from .parsers import Block, Document

log = logging.getLogger(__name__)


@dataclass
class Segment:
    index: int
    title: str  # file stem, e.g. "01 page 1"
    text: str
    page: int = 0

    @property
    def filename(self) -> str:
        return sanitize_filename(self.title, 80)


def _pad(index: int) -> str:
    return f"{index:02d}"


def _page_title(index: int, page_no: int) -> str:
    return f"{_pad(index)} page {page_no}"


def _pages_title(index: int, first_page_no: int, last_page_no: int) -> str:
    """Title for a group of consecutive pages, e.g. ``01 pages 1 to 3``."""
    if first_page_no == last_page_no:
        return _page_title(index, first_page_no)
    return f"{_pad(index)} pages {first_page_no} to {last_page_no}"


def _heading_title(index: int, heading_text: str) -> str:
    return f"{_pad(index)} {sanitize_filename(heading_text, 70)}"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def split_page_with_h1(blocks: List[Block]) -> List[Segment]:
    """Page by page; Heading-1 splits the page (SPEC 3.4 mode 1)."""
    segments: List[Segment] = []
    counter = 0
    pages: dict[int, List[Block]] = {}
    for block in blocks:
        pages.setdefault(block.page, []).append(block)

    for page_no in sorted(pages):
        page_blocks = pages[page_no]
        h1_idx = next(
            (i for i, b in enumerate(page_blocks) if b.kind == "heading" and b.level == 1),
            None,
        )
        if h1_idx is None:
            counter += 1
            text = "\n\n".join(b.text for b in page_blocks)
            segments.append(Segment(index=counter, title=_page_title(counter, page_no + 1), text=text, page=page_no))
            continue

        pre = page_blocks[:h1_idx]
        if pre and any(b.text.strip() for b in pre):
            counter += 1
            segments.append(
                Segment(
                    index=counter,
                    title=_page_title(counter, page_no + 1),
                    text="\n\n".join(b.text for b in pre),
                    page=page_no,
                )
            )
        heading = page_blocks[h1_idx]
        rest = page_blocks[h1_idx:]
        counter += 1
        segments.append(
            Segment(
                index=counter,
                title=_heading_title(counter, heading.text),
                text="\n\n".join(b.text for b in rest),
                page=page_no,
            )
        )
    return segments


def split_page_only(blocks: List[Block], pages_per_file: int = 1) -> List[Segment]:
    """One segment per page (SPEC 3.4 mode 2).

    When ``pages_per_file`` > 1, consecutive pages are grouped together and
    each group becomes one segment (one audio file), so a group of 3 pages
    is recorded as a single file named ``01 pages 1 to 3``.  Resume works per
    group: every finished group is saved immediately.
    """
    pages_per_file = max(1, int(pages_per_file or 1))
    pages: dict[int, List[Block]] = {}
    for block in blocks:
        pages.setdefault(block.page, []).append(block)
    ordered = sorted(pages)
    segments: List[Segment] = []
    counter = 0
    for start in range(0, len(ordered), pages_per_file):
        group_pages = ordered[start:start + pages_per_file]
        counter += 1
        parts: List[str] = []
        for page_no in group_pages:
            parts.append("\n\n".join(b.text for b in pages[page_no]))
        segments.append(
            Segment(
                index=counter,
                title=_pages_title(counter, group_pages[0] + 1, group_pages[-1] + 1),
                text="\n\n".join(p for p in parts if p),
                page=group_pages[0],
            )
        )
    return segments


def split_one_file(blocks: List[Block]) -> List[Segment]:
    """Whole document as one single segment (no file separation).

    Everything is joined into one audio file.  Recording can only be resumed
    from the very beginning, never from the middle of the file - the wizard
    and the Recording window warn about that.
    """
    text = "\n\n".join(b.text for b in blocks if b.text and b.text.strip())
    if not text.strip():
        return []
    page = blocks[0].page if blocks else 0
    return [
        Segment(index=1, title=f"{_pad(1)} full recording", text=text, page=page)
    ]


def split_h1_only(blocks: List[Block]) -> List[Segment]:
    """One segment per Heading-1 (with its content) (SPEC 3.4 mode 3).

    Text before the first Heading-1 becomes a leading "01 page 1" segment so no
    content is lost.
    """
    sections: List[tuple[str | None, List[Block]]] = []  # (heading_text, blocks)
    leading: List[Block] = []
    current_heading: str | None = None
    current: List[Block] = []

    for block in blocks:
        if block.kind == "heading" and block.level == 1:
            if current_heading is not None or current:
                sections.append((current_heading, current))
            current_heading = block.text
            current = [block]
        else:
            if current_heading is None:
                leading.append(block)
            else:
                current.append(block)
    if current_heading is not None or current:
        sections.append((current_heading, current))

    segments: List[Segment] = []
    counter = 0

    if leading and any(b.text.strip() for b in leading):
        counter += 1
        segments.append(
            Segment(
                index=counter,
                title=_page_title(counter, 1),
                text="\n\n".join(b.text for b in leading),
                page=leading[0].page,
            )
        )

    for heading_text, blocks_ in sections:
        if not any(b.text.strip() for b in blocks_):
            continue
        counter += 1
        segments.append(
            Segment(
                index=counter,
                title=_heading_title(counter, heading_text or "untitled section"),
                text="\n\n".join(b.text for b in blocks_),
                page=blocks_[0].page,
            )
        )
    return segments


def split_all_headings(blocks: List[Block]) -> List[Segment]:
    """Every heading (1-6) as its own segment (SPEC 3.4 mode 4)."""
    segments: List[Segment] = []
    counter = 0
    for block in blocks:
        if block.kind == "heading" and 1 <= block.level <= 6 and block.text.strip():
            counter += 1
            segments.append(
                Segment(
                    index=counter,
                    title=_heading_title(counter, block.text),
                    text=block.text,
                    page=block.page,
                )
            )
    return segments


# ---------------------------------------------------------------------------
# DAISY splitting modes
# ---------------------------------------------------------------------------
#: Maximum characters per DAISY chapter segment. DAISY readers work best
#: with chapters that are 30 seconds to 5 minutes of audio; ~1200 chars
#: is roughly 2 minutes at normal speech rate.
DAISY_MAX_CHARS = 1200


def split_daisy_chapters(blocks: List[Block]) -> List[Segment]:
    """Split into DAISY-sized chapters using headings when available.

    Strategy:
    1. Group text by Heading-1 (or Heading-2 if no Heading-1 exists).
    2. If a heading group exceeds DAISY_MAX_CHARS, split at paragraph
       boundaries into sub-segments.
    3. If there are no headings at all, split by pseudo-pages.
    """
    # Detect which heading level to use
    h1_blocks = [b for b in blocks if b.kind == "heading" and b.level == 1]
    h2_blocks = [b for b in blocks if b.kind == "heading" and b.level == 2]
    target_level = 1 if h1_blocks else (2 if h2_blocks else 0)

    if target_level == 0:
        # No headings: fall back to page-based splitting
        return _split_by_pages(blocks)

    # Group by heading level
    sections: List[tuple[str, List[Block]]] = []
    leading: List[Block] = []
    current_heading: str | None = None
    current: List[Block] = []

    for block in blocks:
        if block.kind == "heading" and block.level == target_level:
            if current_heading is not None or current:
                sections.append((current_heading, current))
            current_heading = block.text
            current = [block]
        else:
            if current_heading is None:
                leading.append(block)
            else:
                current.append(block)
    if current_heading is not None or current:
        sections.append((current_heading, current))

    # Now split sections that are too long
    segments: List[Segment] = []
    counter = 0

    if leading and any(b.text.strip() for b in leading):
        sub = _split_text_blocks(leading)
        for text in sub:
            counter += 1
            segments.append(Segment(
                index=counter,
                title=_page_title(counter, 1),
                text=text,
                page=leading[0].page,
            ))

    for heading_text, blocks_ in sections:
        if not any(b.text.strip() for b in blocks_):
            continue
        sub = _split_text_blocks(blocks_)
        if len(sub) <= 1:
            counter += 1
            segments.append(Segment(
                index=counter,
                title=_heading_title(counter, heading_text or "untitled section"),
                text="\n\n".join(b.text for b in blocks_),
                page=blocks_[0].page,
            ))
        else:
            for i, text in enumerate(sub):
                counter += 1
                suffix = f" part {i + 1}" if len(sub) > 1 else ""
                segments.append(Segment(
                    index=counter,
                    title=_heading_title(counter, (heading_text or "untitled") + suffix),
                    text=text,
                    page=blocks_[0].page,
                ))

    return segments


def _split_by_pages(blocks: List[Block]) -> List[Segment]:
    """Fallback: split by pseudo-pages when no headings exist."""
    pages: dict[int, List[Block]] = {}
    for block in blocks:
        pages.setdefault(block.page, []).append(block)
    segments: List[Segment] = []
    counter = 0
    for page_no in sorted(pages):
        counter += 1
        text = "\n\n".join(b.text for b in pages[page_no])
        segments.append(Segment(
            index=counter,
            title=_page_title(counter, page_no + 1),
            text=text,
            page=page_no,
        ))
    return segments


def _split_text_blocks(blocks: List[Block]) -> List[str]:
    """Split a list of blocks into chunks of DAISY_MAX_CHARS."""
    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if current_len + len(text) > DAISY_MAX_CHARS and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts, current_len = [], 0
        current_parts.append(text)
        current_len += len(text)
    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return chunks if chunks else ["\n\n".join(b.text for b in blocks if b.text.strip())]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
SPLITTERS = {
    MODE_PAGE_WITH_H1: split_page_with_h1,
    MODE_PAGE_ONLY: split_page_only,
    MODE_H1_ONLY: split_h1_only,
    MODE_ALL_HEADINGS: split_all_headings,
    MODE_ONE_FILE: split_one_file,
}


def split_document(
    doc: Document,
    mode: str,
    pages_per_file: int = 1,
) -> List[Segment]:
    """Split ``doc`` into segments for ``mode``.

    ``pages_per_file`` only applies to :data:`MODE_PAGE_ONLY` (how many pages
    go into one audio file).  It is ignored for every other mode.
    """
    if mode == MODE_ONE_FILE:
        segments = split_one_file(doc.blocks)
    elif mode == MODE_PAGE_ONLY:
        segments = split_page_only(doc.blocks, pages_per_file=pages_per_file)
    else:
        splitter = SPLITTERS.get(mode)
        if splitter is None:
            raise ValueError(f"Unknown audio mode: {mode}")
        segments = splitter(doc.blocks)
    # Renumber sequentially and enforce unique titles.
    seen: set[str] = set()
    for idx, seg in enumerate(segments, start=1):
        seg.index = idx
        seg.title = _renumber_title(seg.title, idx)
        if seg.title in seen:
            seg.title = f"{seg.title} ({idx})"
        seen.add(seg.title)
    return segments


def _renumber_title(title: str, index: int) -> str:
    """Replace the leading number of a title (e.g. '07 page 3') with the
    final sequential number."""
    number = f"{index:02d}"
    head, sep, tail = title.partition(" ")
    if sep and head.isdigit():
        return f"{number} {tail}"
    return f"{number} {title}" if not title[0].isdigit() else title
