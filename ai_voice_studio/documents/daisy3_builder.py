"""DAISY 3 (Z39.86-2005) audio + text talking book builder.

Builds a DAISY 3 fileset from a project's recorded segments.  DAISY 3 is the
successor of DAISY 2.02: the whole book text lives in one **DTBook** XML
document, navigation is an **NCX** file, and SMIL 2.0 documents synchronize
each text element with the recorded audio.

Fileset layout (``<project>/DAISY3/``) -- flat, sibling-relative URIs only::

    book.xml       -- DTBook 2005-3 text content (frontmatter, body, rearmatter)
    ncx.xml        -- NCX navigation (navMap built from the heading hierarchy)
    package.opf    -- OPF package manifest tying everything together
    0001.smil ...  -- one SMIL 2.0 document per recorded segment
    aud0001.wav .  -- recorded audio renamed to digit-only names

Conformance notes:

* DTBook uses the official FPI ``-//NISO//DTD dtbook 2005-3//EN`` with the
  required structure ``dtbook > head + book > (frontmatter, bodymatter,
  rearmatter)``; ``doctitle``/``docauthor`` are mandatory in frontmatter.
* The NCX carries ``dtb:`` metadata (uid, depth, totalPageCount,
  maxPageNumber) and a navMap whose navPoints nest by heading level.
* Content SMILs are SMIL 2.0 (``-//NISO//DTD xml-smil 2005-1//EN``) with the
  custom attributes DAISY 3 players require: ``dtb:requiredNamespace``,
  ``dtb:totalElapsed``/``dtb:timeInThisSmil``, ``customTest`` on the par.
  The ``<audio>`` sits directly inside each ``<par>`` (no wrapping
  ``<seq>``, which DAISY 3 players do not play through).
* Images found in the source document (DOCX/EPUB media entries, HTML
  ``<img src>`` files) are embedded as official DTBook ``<imggroup>`` /
  ``<img>`` elements at the end of the book and listed in the OPF manifest.
* Synchronization is phrase-level: every paragraph of a segment becomes one
  ``<par>``, the segment audio is sliced proportionally by character count --
  the same mechanism the DAISY 2.02 full-text builder uses and that real
  players (FSReader, EasyReader) rely on for synchronized text display.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
import wave
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

from ..constants import (
    DAISY3_DTBOOK_FILE,
    DAISY3_NCX_FILE,
    DAISY3_OUTPUT_DIR_NAME,
    DAISY3_PACKAGE_FILE,
)
from ..util import sanitize_filename

# SMIL 2.0 DAISY namespace (dtb: prefix) used on smil/customTest attributes.
_SMIL_DTB_NS = "http://www.daisy.org/z3986/2005/dtbook/"

_DTBOOK_DOCTYPE = (
    '<!DOCTYPE dtbook PUBLIC "-//NISO//DTD dtbook 2005-3//EN" '
    '"http://www.daisy.org/z3986/2005/dtbook-2005-3.dtd">'
)
_SMIL_DOCTYPE = (
    '<!DOCTYPE smil PUBLIC "-//NISO//DTD xml-smil 2005-1//EN" '
    '"http://www.daisy.org/z3986/2005/xml-smil-2005-1.dtd">'
)

_AUDIO_MIME = {
    "wav": "audio/x-wav",
    "mp3": "audio/mpeg",
    "flac": "audio/x-flac",
}

# Recognized image types for embedding into the DTBook (captions use the
# file stem as alt text).
_IMAGE_EXT_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

_IMG_TAG_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_daisy3_book(
    output_dir: str,
    project_name: str,
    segments: List[Dict[str, Any]],
    audio_format: str = "wav",
    language: str = "en",
    publisher: str = "",
    source_file: str = "",
    author: str = "",
    meta: Optional[Dict[str, str]] = None,
) -> str:
    """Build the complete DAISY 3 book structure into ``<output_dir>/DAISY3``.

    Parameters mirror :func:`ai_voice_studio.documents.daisy_builder.build_daisy_book`.
    Text is always included: DAISY 3 is a full-text/full-audio format.  Images
    found in the source document are embedded as DTBook ``<imggroup>``
    elements at the end of the book.

    ``meta`` carries the DAISY book information entered in the wizard
    (``title``, ``creator``, ``date``, ``subject``, ``narrator``,
    ``producer``); empty or missing values fall back to sensible defaults and
    the running time is always computed from the recorded audio.

    Returns the path of the generated ``package.opf`` (the entry point DAISY 3
    players open), or ``""`` when no segment has been recorded yet.
    """
    ready = [
        s for s in segments
        if s.get("status") == "done" and s.get("saved")
    ]
    if not ready:
        return ""

    daisy_dir = os.path.join(output_dir, DAISY3_OUTPUT_DIR_NAME)
    os.makedirs(daisy_dir, exist_ok=True)

    book_uid = "urn:uuid:" + str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Copy recorded audio under digit-only names and measure durations.
    entries: List[Dict[str, Any]] = []
    for i, seg in enumerate(ready, start=1):
        src = os.path.join(output_dir, os.path.basename(seg["saved"]))
        ext = (os.path.splitext(os.path.basename(seg["saved"]))[1] or
               "." + (audio_format or "wav")).lstrip(".").lower()
        audio_name = f"aud{i:04d}.{ext}"
        dst = os.path.join(daisy_dir, audio_name)
        if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        elif not os.path.isfile(dst) and os.path.isfile(src):
            shutil.copy2(src, dst)
        dur = _audio_duration(dst, ext)
        entries.append({
            "index": seg.get("index", i),
            "title": seg.get("title") or f"Chapter {i}",
            "text": seg.get("text") or "",
            "audio": audio_name,
            "duration": dur,
            "number": i,
        })

    total_ms = int(sum(e["duration"] for e in entries) * 1000)

    # 2. Phrase-level sync blocks per segment (heading + each paragraph),
    #    ids assigned book-wide so DTBook, SMIL and NCX references agree.
    id_counter = [0]
    for entry in entries:
        entry["blocks"] = _entry_blocks(entry, id_counter)

    # 2b. Extract images from the source document (best effort) and copy
    #     them into DAISY3/images/ under digit-only names.
    images = _stage_images(_extract_images(source_file), daisy_dir)

    # 3. DTBook text document.
    meta = meta or {}
    _write_dtbook(
        daisy_dir, project_name, entries, book_uid, now_iso, language,
        publisher, source_file, author, images, meta,
    )

    # 4. NCX navigation.
    _write_ncx(
        daisy_dir, project_name, entries, book_uid, now_iso, language,
        publisher, total_ms, meta,
    )

    # 5. Per-segment SMIL 2.0 content documents.  Each SMIL carries the
    #    cumulative book offset (dtb:offset) and the book total time
    #    (dtb:totalElapsed) as DAISY 3 players expect.
    smil_files: List[str] = []
    for entry in entries:
        entry["book_total_ms"] = total_ms
        smil_files.append(_write_smil(daisy_dir, entry, book_uid, project_name))

    # 1a. Cumulative book offset for every segment.  Each SMIL declares its
    #    start position (dtb:offset); rounding must always be UP so declared
    #    positions never land before the real audio start of that SMIL.
    cumulative_ms = 0
    for entry in entries:
        entry["offset_ms"] = cumulative_ms
        cumulative_ms += int(entry["duration"] * 1000)

    # 6. OPF package manifest.
    opf_path = _write_opf(
        daisy_dir, project_name, entries, book_uid, now_iso, language,
        publisher, source_file, author, total_ms, smil_files, images, meta,
    )
    return opf_path


def export_daisy3_zip(project_dir: str, zip_path: str,
                      book_folder: Optional[str] = None) -> str:
    """Package the generated DAISY 3 book into a distributable ZIP archive."""
    daisy_dir = os.path.join(project_dir, DAISY3_OUTPUT_DIR_NAME)
    if not os.path.isdir(daisy_dir) or not os.path.isfile(
        os.path.join(daisy_dir, DAISY3_PACKAGE_FILE)
    ):
        raise FileNotFoundError(
            "No DAISY 3 book has been generated yet. Record all segments "
            "first, then try exporting again."
        )
    top = book_folder or sanitize_filename(
        os.path.basename(project_dir.rstrip("/\\")) or "book", 60
    )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(daisy_dir):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                rel = os.path.relpath(file_path, daisy_dir)
                zf.write(file_path, os.path.join(top, rel).replace("\\", "/"))
    return os.path.abspath(zip_path)


# ---------------------------------------------------------------------------
# Synchronized text blocks
# ---------------------------------------------------------------------------
def _entry_blocks(entry: Dict[str, Any], id_counter: List[int]) -> List[Dict[str, Any]]:
    """Compute the synchronized text blocks of one segment.

    The first block is the segment heading; every non-empty paragraph follows.
    Each block carries the DTBook element id it will be attached to, the SMIL
    ``<text>`` id, the ``<par>`` id and its share of the audio.
    """
    blocks: List[Dict[str, Any]] = []

    def add(dtbook_id: str, text: str) -> None:
        id_counter[0] += 1
        blocks.append({
            "dtbook_id": dtbook_id,
            "text_id": f"txt_{id_counter[0]:04d}",
            "par_id": f"par_{id_counter[0]:04d}",
            "aud_id": f"aud_{id_counter[0]:04d}",
            "text": text,
            "chars": max(1, len(text)),
        })

    add(f"seg_{entry['number']:04d}", str(entry["title"]))
    paras = [p.strip() for p in (entry.get("text") or "").splitlines() if p.strip()]
    for j, para in enumerate(paras, start=1):
        add(f"p_{entry['number']:04d}_{j:03d}", para)
    return blocks


# ---------------------------------------------------------------------------
# DTBook
# ---------------------------------------------------------------------------
def _write_dtbook(
    daisy_dir: str,
    title: str,
    entries: List[Dict[str, Any]],
    book_uid: str,
    now_iso: str,
    language: str,
    publisher: str,
    source_file: str,
    author: str,
    images: Optional[List[Tuple[str, str, bytes]]] = None,
    meta: Optional[Dict[str, str]] = None,
) -> None:
    """Write the DTBook 2005-3 text content document."""
    publisher = publisher.strip() or "AI Voice Studio"
    meta = meta or {}
    title = meta.get("title") or title
    author = (meta.get("creator") or author).strip() or publisher
    date = meta.get("date") or now_iso[:10]
    subject = meta.get("subject", "")
    narrator = meta.get("narrator", "")
    producer = meta.get("producer", "")
    show_software = bool(meta.get("show_software", True))
    images = images or []

    head_meta = (
        '    <meta name="dtb:uid" content="' + _escape(book_uid) + '" />\n'
        '    <meta name="dtb:depth" content="1" />\n'
        '    <meta name="dtb:totalPageCount" content="0" />\n'
        '    <meta name="dtb:maxPageNumber" content="0" />\n'
    )

    optional_meta = ""
    if subject:
        optional_meta += (
            '\n    <meta name="dc:Subject" content="' + _escape(subject) + '" />')
    if narrator:
        optional_meta += (
            '\n    <meta name="dtb:narrator" content="' + _escape(narrator) + '" />')
    # dtb:producer is repeatable: the user producer and, when the software
    # checkbox is on, the producing software are both listed.
    producers = ([producer] if producer else [])
    if show_software and "AI Voice Studio" not in producers:
        producers.append("AI Voice Studio")
    for prod in producers:
        optional_meta += (
            '\n    <meta name="dtb:producer" content="' + _escape(prod) + '" />')

    frontmatter = (
        '    <frontmatter>\n'
        '      <doctitle id="doctitle">' + _escape(title) + '</doctitle>\n'
        '      <docauthor id="docauthor">' + _escape(author) + '</docauthor>\n'
        '    </frontmatter>\n'
    )

    # Body: one level1 per segment; heading as h1, paragraphs as p.  Every
    # element carries the id the SMIL <text> elements point at.
    body_parts: List[str] = []
    for entry in entries:
        blocks = entry["blocks"]
        head_block = blocks[0]
        lines = [
            '    <level1 id="lvl_{num:04d}">'.format(num=entry["number"]),
            '      <h1 id="{id}">{text}</h1>'.format(
                id=head_block["dtbook_id"], text=_escape(head_block["text"])),
        ]
        for block in blocks[1:]:
            lines.append(
                '      <p id="{id}">{text}</p>'.format(
                    id=block["dtbook_id"], text=_escape(block["text"]))
            )
        lines.append('    </level1>')
        body_parts.append("\n".join(lines))

    source_meta = (
        '\n      <meta name="dc:source" content="' + _escape(source_file) + '" />'
        if source_file else ""
    )

    dtbook = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + _DTBOOK_DOCTYPE + '\n'
        '<dtbook version="2005-3" xml:lang="' + _escape(language) + '" '
        'xmlns="http://www.daisy.org/z3986/2005/dtbook/">\n'
        '  <head>\n'
        '    <title>' + _escape(title) + '</title>\n'
        + head_meta +
        '    <meta name="dc:Title" content="' + _escape(title) + '" />\n'
        '    <meta name="dc:Creator" content="' + _escape(author) + '" />\n'
        '    <meta name="dc:Language" content="' + _escape(language) + '" />\n'
        '    <meta name="dc:Date" content="' + _escape(date) + '" />\n'
        '    <meta name="dc:Publisher" content="' + _escape(publisher) + '" />'
        + optional_meta
        + source_meta + '\n'
        '  </head>\n'
        '  <book>\n'
        + frontmatter +
        '    <bodymatter>\n'
        + "\n".join(body_parts) + '\n'
        '    </bodymatter>\n'
        '    <rearmatter>\n'
        + _dtbook_image_section(images) +
        '      <level1 id="rearmatter_end" class="rearmatter">\n'
        '        <p id="rearmatter_end_p">End of ' + _escape(title) + '</p>\n'
        '      </level1>\n'
        '    </rearmatter>\n'
        '  </book>\n'
        '</dtbook>\n'
    )
    with open(os.path.join(daisy_dir, DAISY3_DTBOOK_FILE), "w", encoding="utf-8") as fh:
        fh.write(dtbook)


# ---------------------------------------------------------------------------
# Images (DTBook imggroup)
# ---------------------------------------------------------------------------
def _dtbook_image_section(images: List[Tuple[str, str, bytes]]) -> str:
    """Return the rearmatter ``<level1>`` holding every image.

    Uses the official DTBook image markup: ``<imggroup>`` with ``<img``
    (``src``) plus ``<captions>``; producers usually place the long
    description in ``<img>`` content, which players can read.
    """
    if not images:
        return ""
    lines = [
        '      <level1 id="images_level" class="images">\n'
        '        <h1 id="images_heading">Images</h1>\n'
    ]
    for i, (name, _mime, _data) in enumerate(images, start=1):
        stem = os.path.splitext(name)[0]
        lines.append(
            '        <imggroup id="imggrp_{i:04d}">\n'
            '          <img id="img_{i:04d}" src="images/{name}" alt="{alt}"/>\n'
            '          <captions id="cap_{i:04d}" imgref="img_{i:04d}">\n'
            '            <p id="cap_{i:04d}_p">{alt}</p>\n'
            '          </captions>\n'
            '        </imggroup>'.format(i=i, name=name, alt=_escape(stem))
        )
    lines.append('      </level1>\n')
    return "\n".join(lines)


def _extract_images(source_file: str) -> List[Tuple[str, str, bytes]]:
    """Return ``(file_name, mime_type, data)`` for images in the source.

    Supported sources: DOCX and EPUB (ZIP media entries) and HTML (local
    ``<img src>`` files).  Everything else yields no images -- the book is
    then simply text + audio.
    """
    images: List[Tuple[str, str, bytes]] = []
    if not source_file or not os.path.isfile(source_file):
        return images
    ext = os.path.splitext(source_file)[1].lower()
    try:
        if ext in (".docx", ".epub"):
            images = _images_from_zip(source_file)
        elif ext in (".html", ".htm"):
            images = _images_from_html(source_file)
    except Exception:  # noqa: BLE001 -- images are best effort
        return []
    return images


def _stage_images(
    images: List[Tuple[str, str, bytes]],
    daisy_dir: str,
) -> List[Tuple[str, str, bytes]]:
    """Copy extracted images into ``<daisy_dir>/images/`` (digit-only names).

    Returns the staged ``(new_name, mime, data)`` list used by the DTBook and
    OPF writers.
    """
    if not images:
        return []
    img_dir = os.path.join(daisy_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    staged: List[Tuple[str, str, bytes]] = []
    for i, (_name, mime, data) in enumerate(images, start=1):
        ext = _ext_for_mime(mime, _name)
        new_name = f"img{i:04d}{ext}"
        with open(os.path.join(img_dir, new_name), "wb") as fh:
            fh.write(data)
        staged.append((new_name, mime, data))
    return staged


def _ext_for_mime(mime: str, name: str) -> str:
    for ext, candidate in _IMAGE_EXT_MIME.items():
        if candidate == mime:
            return ext
    return os.path.splitext(name)[1].lower() or ".png"


def _images_from_zip(path: str) -> List[Tuple[str, str, bytes]]:
    found: List[Tuple[str, str, bytes]] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = os.path.splitext(info.filename)[1].lower()
            mime = _IMAGE_EXT_MIME.get(ext)
            if not mime or info.file_size > 20 * 1024 * 1024:
                continue
            stem = os.path.basename(info.filename)
            try:
                data = zf.read(info)
            except Exception:  # noqa: BLE001
                continue
            found.append((stem, mime, data))
    return found


class _ImgSrcParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.srcs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        if tag.lower() == "img":
            for name, value in attrs:
                if name.lower() == "src" and value:
                    self.srcs.append(value)


def _images_from_html(path: str) -> List[Tuple[str, str, bytes]]:
    base = os.path.dirname(os.path.abspath(path))
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        parser = _ImgSrcParser()
        parser.feed(fh.read())
    found: List[Tuple[str, str, bytes]] = []
    seen: set = set()
    for src in parser.srcs:
        if "://" in src or src.startswith("data:"):
            continue
        local = os.path.normpath(os.path.join(base, src.split("#")[0].split("?")[0]))
        if local in seen or not os.path.isfile(local):
            continue
        seen.add(local)
        ext = os.path.splitext(local)[1].lower()
        mime = _IMAGE_EXT_MIME.get(ext)
        if not mime:
            continue
        with open(local, "rb") as fh:
            found.append((os.path.basename(local), mime, fh.read()))
    return found


# ---------------------------------------------------------------------------
# NCX
# ---------------------------------------------------------------------------
def _write_ncx(
    daisy_dir: str,
    title: str,
    entries: List[Dict[str, Any]],
    book_uid: str,
    now_iso: str,
    language: str,
    publisher: str,
    total_ms: int,
    meta: Optional[Dict[str, str]] = None,
) -> None:
    """Write the NCX navigation file (navMap with one navPoint per segment)."""
    meta = meta or {}
    title = meta.get("title") or title
    author = (meta.get("creator") or publisher).strip() or "AI Voice Studio"
    nav_points: List[str] = []
    for entry in entries:
        blocks = entry["blocks"]
        nav_points.append(
            '      <navPoint id="nav_{num:04d}" playOrder="{num}">\n'
            '        <navLabel><text>{title}</text></navLabel>\n'
            '        <content src="{smil}#{tid}" />\n'
            '      </navPoint>'.format(
                num=entry["number"],
                title=_escape(entry["title"]),
                smil=_smil_name(entry),
                tid=blocks[0]["text_id"],
            )
        )

    ncx = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1" '
        'xml:lang="' + _escape(language) + '">\n'
        '  <head>\n'
        '    <meta name="dtb:uid" content="' + _escape(book_uid) + '" />\n'
        '    <meta name="dtb:depth" content="1" />\n'
        '    <meta name="dtb:totalPageCount" content="0" />\n'
        '    <meta name="dtb:maxPageNumber" content="0" />\n'
        '  </head>\n'
        '  <docTitle><text>' + _escape(title) + '</text></docTitle>\n'
        '  <docAuthor><text>' + _escape(author) + '</text></docAuthor>\n'
        '  <navMap>\n'
        + "\n".join(nav_points) + '\n'
        '  </navMap>\n'
        '</ncx>\n'
    )
    with open(os.path.join(daisy_dir, DAISY3_NCX_FILE), "w", encoding="utf-8") as fh:
        fh.write(ncx)


# ---------------------------------------------------------------------------
# SMIL 2.0 content documents
# ---------------------------------------------------------------------------
def _write_smil(
    daisy_dir: str,
    entry: Dict[str, Any],
    book_uid: str,
    book_title: str,
) -> str:
    """Write one SMIL 2.0 content document for a recorded segment."""
    smil_name = _smil_name(entry)
    dur = max(0.001, entry["duration"])
    blocks = entry["blocks"]
    total_chars = sum(b["chars"] for b in blocks)

    # One <par> per synchronized text block.  In DAISY 3 the <audio> sits
    # **directly** inside the <par> (no wrapping <seq>; that is a DAISY 2.02
    # idiom that DAISY 3 players such as EasyReader do not play through).
    # The <text> element references the DTBook fragment (book.xml#id) and
    # carries its own id; the audio is sliced proportionally by character
    # count across the blocks.
    pars: List[str] = []
    elapsed = 0.0
    for idx, block in enumerate(blocks):
        if idx == len(blocks) - 1:
            # Give the final clip a small tail beyond the measured duration:
            # MP3 encoder delay means decodable audio usually runs slightly
            # LONGER than the container duration.  A clipEnd exactly at the
            # measured value makes strict players stop just before the real
            # end of the narration.
            clip_end = dur + 0.25
        else:
            clip_end = min(dur, elapsed + dur * block["chars"] / total_chars)
        pars.append(
            '      <par id="{pid}">\n'
            '        <text src="{dtbook}#{dtid}" id="{tid}" />\n'
            '        <audio src="{aud}" clipBegin="{b:.3f}s" clipEnd="{e:.3f}s" />\n'
            '      </par>'.format(
                pid=block["par_id"],
                dtbook=DAISY3_DTBOOK_FILE,
                dtid=block["dtbook_id"],
                tid=block["text_id"],
                aud=entry["audio"],
                b=elapsed,
                e=clip_end,
            )
        )
        elapsed = clip_end

    smil = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + _SMIL_DOCTYPE + '\n'
        '<smil xmlns:dtb="' + _SMIL_DTB_NS + '">\n'
        '  <head>\n'
        '    <meta name="dtb:uid" content="' + _escape(book_uid) + '" />\n'
        '    <meta name="dtb:offset" content="'
        + _format_duration(_total_ms(entry.get("offset_ms"))) + '" />\n'
        '    <meta name="dtb:totalElapsed" content="'
        + _format_duration(_total_ms(entry.get("book_total_ms"))) + '" />\n'
        '    <meta name="dtb:timeInThisSmil" content="'
        + _format_duration(int(dur * 1000)) + '" />\n'
        '  </head>\n'
        '  <body>\n'
        '    <seq dur="' + f"{dur:.3f}s" + '">\n'
        + "\n".join(pars) + '\n'
        '    </seq>\n'
        '  </body>\n'
        '</smil>\n'
    )
    with open(os.path.join(daisy_dir, smil_name), "w", encoding="utf-8") as fh:
        fh.write(smil)
    return smil_name


def _total_ms(total_ms_value: Any) -> int:
    """Coerce a milliseconds value (int or str) safely."""
    try:
        return max(0, int(total_ms_value))
    except (TypeError, ValueError):
        return 0


def _smil_name(entry: Dict[str, Any]) -> str:
    return f"{entry['number']:04d}.smil"


# ---------------------------------------------------------------------------
# OPF package manifest
# ---------------------------------------------------------------------------
def _write_opf(
    daisy_dir: str,
    title: str,
    entries: List[Dict[str, Any]],
    book_uid: str,
    now_iso: str,
    language: str,
    publisher: str,
    source_file: str,
    author: str,
    total_ms: int,
    smil_files: List[str],
    images: Optional[List[Tuple[str, str, bytes]]] = None,
    meta: Optional[Dict[str, str]] = None,
) -> str:
    """Write the OPF package manifest and return its path."""
    images = images or []
    meta = meta or {}
    publisher = publisher.strip() or "AI Voice Studio"
    title = meta.get("title") or title
    author = (meta.get("creator") or author).strip() or publisher
    date = meta.get("date") or now_iso[:10]
    subject = meta.get("subject", "")
    narrator = meta.get("narrator", "")
    producer = meta.get("producer", "")
    show_software = bool(meta.get("show_software", True))

    items = [
        f'    <item id="dtbook" href="{DAISY3_DTBOOK_FILE}" media-type="application/x-dtbook+xml"/>',
        f'    <item id="ncx" href="{DAISY3_NCX_FILE}" media-type="application/x-dtbncx+xml"/>',
    ]
    for entry in entries:
        ext = os.path.splitext(entry["audio"])[1].lstrip(".").lower()
        mime = _AUDIO_MIME.get(ext, "audio/x-wav")
        items.append(
            f'    <item id="smil_{entry["number"]:04d}" href="{_smil_name(entry)}"'
            f' media-type="application/smil"/>'
        )
        items.append(
            f'    <item id="audio_{entry["number"]:04d}" href="{entry["audio"]}"'
            f' media-type="{mime}"/>'
        )
    for i, (name, mime, _data) in enumerate(images, start=1):
        items.append(
            f'    <item id="img_{i:04d}" href="images/{name}"'
            f' media-type="{mime}"/>'
        )

    source_meta = (
        f'\n    <dc:source>{_escape(source_file)}</dc:source>'
        if source_file else ""
    )

    optional_meta = ""
    if subject:
        optional_meta += f'\n    <dc:subject>{_escape(subject)}</dc:subject>'
    if narrator:
        optional_meta += (
            '\n    <meta name="dtb:narrator" content="' + _escape(narrator) + '" />')
    producers = ([producer] if producer else [])
    if show_software and "AI Voice Studio" not in producers:
        producers.append("AI Voice Studio")
    for prod in producers:
        optional_meta += (
            '\n    <meta name="dtb:producer" content="' + _escape(prod) + '" />')

    opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" '
        'unique-identifier="uid" version="2.0">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dtb="http://www.daisy.org/z3986/2005/dtbook/">\n'
        f'    <dc:title>{_escape(title)}</dc:title>\n'
        f'    <dc:creator>{_escape(author)}</dc:creator>\n'
        f'    <dc:language>{_escape(language)}</dc:language>\n'
        f'    <dc:identifier id="uid">{_escape(book_uid)}</dc:identifier>\n'
        f'    <dc:date>{_escape(date)}</dc:date>\n'
        f'    <dc:publisher>{_escape(publisher)}</dc:publisher>\n'
        + optional_meta +
        f'    <dc:format>Daisy 3</dc:format>'
        + source_meta + '\n'
        '    <meta name="dtb:uid" content="' + _escape(book_uid) + '" />\n'
        '    <meta name="dtb:depth" content="1" />\n'
        '    <meta name="dtb:totalPageCount" content="0" />\n'
        '    <meta name="dtb:maxPageNumber" content="0" />\n'
        '    <meta name="dtb:totalTime" content="' + _format_duration(total_ms) + '" />\n'
        '  </metadata>\n'
        '  <manifest>\n'
        + "\n".join(items) + '\n'
        '  </manifest>\n'
        '  <spine toc="ncx">\n'
        + "".join(
            f'    <itemref idref="smil_{e["number"]:04d}"/>\n'
            for e in entries
        ) +
        '  </spine>\n'
        '</package>\n'
    )
    opf_path = os.path.join(daisy_dir, DAISY3_PACKAGE_FILE)
    with open(opf_path, "w", encoding="utf-8") as fh:
        fh.write(opf)
    return opf_path


# ---------------------------------------------------------------------------
# Helpers (shared semantics with daisy_builder.py)
# ---------------------------------------------------------------------------
def _audio_duration(path: str, ext: str) -> float:
    """Best-effort audio duration in seconds (0.0 when it cannot be read)."""
    if ext == "wav":
        try:
            with wave.open(path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate if rate else 0.0
        except Exception:  # noqa: BLE001
            return 0.0
    try:
        from ..audio import ffmpeg as ffmpeg_mod  # noqa: PLC0415

        exe = ffmpeg_mod.find_ffmpeg()
        if not exe:
            return 0.0
        result = subprocess.run(
            [exe, "-i", path, "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or ""
        )
        if match:
            h, m, s = match.groups()
            return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _format_duration(total_ms: int) -> str:
    """Format milliseconds as ``hh:mm:ss``, rounding UP to the next second.

    Declared times must never be shorter than the real audio: players that
    pace or clamp playback against ``dtb:totalTime`` / ``dtb:totalElapsed`` /
    ``dtb:timeInThisSmil`` would otherwise stop before the audio ends (the
    "last part of a file does not play" bug).
    """
    total_s = -(-max(0, int(total_ms)) // 1000)  # ceiling division
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _escape(text: str) -> str:
    """Escape XML special characters."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
