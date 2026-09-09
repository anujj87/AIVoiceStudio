"""DAISY 2.02 (Digital Talking Book) audio book builder.

Builds a standards-compliant DAISY 2.02 (Z39.86) fileset from a project's
recorded segments, modelled on genuine DAISY Consortium sample books
("Climbing the Highest Mountain", "WIPO Treaty") so that mainstream DAISY
players (AMIS, EasyReader, Thorin, ...) open it, show the text and play it.

Fileset layout (``<project>/DAISY/``) -- **flat**, exactly like real DAISY
2.02 books: every file lives in the same folder so that sibling-relative
URIs always resolve::

    ncc.html      -- Navigation Control Center (XHTML 1.0 + DAISY metadata).
                     The file players open first.
    master.smil   -- optional Master SMIL chaining every content SMIL.
    0001.smil ... -- one SMIL 1.0 content document per recorded segment.
    0001.html ... -- XHTML text content document per segment (always
                     generated; carries the full text when ``include_text``).
    aud0001.mp3 . -- recorded audio renamed to digit-only names so that
                     alphabetical order equals playback order.

Conformance notes (DAISY 2.02 specification, Feb 2001):

* Every NCC ``<body>`` child used for navigation carries an ``id`` and one
  ``<a href="smil-file#fragment">`` whose fragment resolves to a SMIL
  ``<text>`` element inside that content SMIL -- never to a ``<body>``
  element.
* Content SMIL documents are SMIL 1.0 (``<!DOCTYPE smil PUBLIC ...
  SMIL10.dtd>``), carry a ``<layout><region id="txtView"/></layout>`` in the
  head, contain one main ``<seq dur=...>``, and each ``<par>`` holds one
  ``<text>`` plus one ``<audio>`` wrapped in a nested ``<seq>`` (the
  structure used by the official samples).  All media objects carry ``id``.
* All references between files are sibling-relative (``src="0001.smil#..."``,
  ``src="aud0001.mp3"``) -- no subfolders, no ``../`` escapes, which several
  players cannot resolve.
* ``ncc.html`` carries the mandatory ``dc:``/``ncc:`` metadata set including
  ``ncc:charset``, ``ncc:setInfo``, ``ncc:pageFront/pageNormal/pageSpecial``,
  ``ncc:tocItems`` and ``ncc:totalTime``.
* ``ncc:multimediaType`` is ``audioNcc`` for audio-only books and
  ``audioFullText`` when synchronized text documents are included.
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
from typing import Any, Dict, List, Optional

from ..constants import (
    DAISY_NCC_FILE,
    DAISY_OUTPUT_DIR_NAME,
    DAISY_PACKAGE_FILE,
    DAISY_MASTER_SMIL_FILE,
)
from ..util import sanitize_filename

_AUDIO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
}

_XHTML_DOCTYPE = (
    '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"\n'
    '  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
)
_STRICT_DOCTYPE = (
    '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n'
    '  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">'
)
_SMIL_DOCTYPE = (
    '<!DOCTYPE smil PUBLIC "-//W3C//DTD SMIL 1.0//EN"\n'
    '  "http://www.w3.org/TR/REC-smil/SMIL10.dtd">'
)

_SMIL_LAYOUT = "    <layout>\n      <region id=\"txtView\" />\n    </layout>\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_daisy_book(
    output_dir: str,
    project_name: str,
    segments: List[Dict[str, Any]],
    audio_format: str = "wav",
    include_text: bool = False,
    language: str = "en",
    publisher: str = "",
    source_file: str = "",
) -> str:
    """Build the complete DAISY 2.02 book structure.

    Parameters
    ----------
    output_dir:
        Project folder. Recorded audio files live here and are copied into
        ``DAISY/`` under digit-only names. The book is written into
        ``DAISY/`` as a flat fileset.
    project_name:
        Human-readable book title.
    segments:
        List of segment dicts with keys ``index``, ``title``, ``text``,
        ``saved`` (audio filename) and ``status`` (``"done"`` when recorded).
    audio_format:
        ``"wav"``, ``"mp3"`` or ``"flac"`` (used as fallback extension and for
        the OPF media-type when a segment's own extension cannot be read).
    include_text:
        If True, the per-segment XHTML documents carry the full segment text
        (audio+text book, ``ncc:multimediaType=audioFullText``). If False the
        text documents still exist (they hold the segment heading that SMIL
        ``<text>`` elements synchronize with) but the book is declared
        ``audioNcc``.
    language:
        ISO 639 language code for ``dc:language``.
    publisher:
        Publisher name for ``dc:publisher``.
    source_file:
        Original document path, stored as ``dc:source`` metadata.

    Returns
    -------
    str
        Path to the generated ``ncc.html``, or ``""`` when no segment has been
        recorded yet.
    """
    ready = [
        s for s in segments
        if s.get("status") == "done" and s.get("saved")
    ]
    if not ready:
        return ""

    daisy_dir = os.path.join(output_dir, DAISY_OUTPUT_DIR_NAME)
    os.makedirs(daisy_dir, exist_ok=True)

    book_uid = "urn:uuid:" + str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    multimedia = "audioFullText" if include_text else "audioNcc"

    # 1. Copy recorded audio into DAISY/ under digit-only names and measure
    #    durations.  Digit-only names are required so players can order audio
    #    correctly (and because names with spaces break SMIL/NCC URI
    #    resolution in several players).
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

    # 1a. Cumulative book offset for every segment.  Each content SMIL
    #     declares its start position (ncc:totalElapsedTime); genuine DAISY
    #     2.02 books set this per SMIL, and players use it for book-level
    #     positioning.
    cumulative_ms = 0
    for entry in entries:
        entry["offset_ms"] = cumulative_ms
        cumulative_ms += int(entry["duration"] * 1000)

    # 1b. Split each segment into synchronized text blocks.  Full-text books
    #     get one <par> per text block (heading + each paragraph) with the
    #     segment audio sliced proportionally by character count -- the
    #     structure genuine DAISY 2.02 full-text books use.  Without this,
    #     players only ever see the heading and treat the book as audio-only.
    id_counter = [0]
    for entry in entries:
        entry["blocks"] = _entry_blocks(entry, include_text, id_counter)

    # 2. Per-segment XHTML text content documents (siblings of the SMILs).
    for entry in entries:
        _write_text_document(daisy_dir, entry, language, include_text)

    # 3. Per-segment content SMIL documents (SMIL 1.0).
    smil_files: List[str] = []
    for entry in entries:
        smil_files.append(
            _write_smil(daisy_dir, entry, book_uid, project_name)
        )

    # 4. Master SMIL chaining content SMIL files for linear playback.
    _write_master_smil(daisy_dir, entries, book_uid, project_name)

    # 5. NCC -- the file players actually open.
    ncc_path = _write_ncc(
        daisy_dir, project_name, entries, book_uid, now_iso, language,
        publisher, source_file, multimedia, total_ms,
    )

    # 6. Optional OPF package manifest (informational; AMIS reads ncc.html).
    _write_opf(
        daisy_dir, project_name, entries, book_uid, now_iso, language,
        publisher, multimedia,
    )

    return ncc_path


# ---------------------------------------------------------------------------
# Synchronized text blocks
# ---------------------------------------------------------------------------
def _entry_blocks(
    entry: Dict[str, Any],
    include_text: bool,
    id_counter: List[int],
) -> List[Dict[str, Any]]:
    """Compute the synchronized text blocks of one segment.

    The first block is always the segment heading; when ``include_text`` is
    True every paragraph follows.  Ids (``txt_``/``par_``/``aud_``) are
    assigned from a book-wide counter so SMIL, text and NCC references agree.
    """
    blocks: List[Dict[str, Any]] = []

    def add(html_id: str, text: str) -> None:
        id_counter[0] += 1
        blocks.append({
            "html_id": html_id,
            "text_id": f"txt_{id_counter[0]:04d}",
            "par_id": f"par_{id_counter[0]:04d}",
            "aud_id": f"aud_{id_counter[0]:04d}",
            "text": text,
            "chars": max(1, len(text)),
        })

    add(f"seg_{entry['number']:04d}", str(entry["title"]))
    if include_text:
        paras = [p.strip() for p in (entry.get("text") or "").splitlines() if p.strip()]
        for j, para in enumerate(paras, start=1):
            add(f"p_{entry['number']:04d}_{j:03d}", para)
    return blocks


# ---------------------------------------------------------------------------
# Public API (continued)
# ---------------------------------------------------------------------------
def export_daisy_zip(project_dir: str, zip_path: str, book_folder: Optional[str] = None) -> str:
    """Package the generated DAISY book into a distributable ZIP archive.

    The ZIP contains a single top-level folder (``book_folder``, defaulting to
    a sanitised version of the project folder name) holding ``ncc.html``,
    ``master.smil``, the content SMIL/text documents and the audio files --
    the flat layout DAISY readers expect.

    Parameters
    ----------
    project_dir:
        Project folder containing the ``DAISY/`` book.
    zip_path:
        Destination path for the ZIP file.
    book_folder:
        Name of the top-level folder inside the ZIP.

    Returns
    -------
    str
        Absolute path to the created ZIP file.

    Raises
    ------
    FileNotFoundError
        When no DAISY book has been generated yet.
    """
    daisy_dir = os.path.join(project_dir, DAISY_OUTPUT_DIR_NAME)
    if not os.path.isdir(daisy_dir) or not os.path.isfile(
        os.path.join(daisy_dir, DAISY_NCC_FILE)
    ):
        raise FileNotFoundError(
            "No DAISY book has been generated yet. Record all segments first, "
            "then try exporting again."
        )
    top = book_folder or sanitize_filename(os.path.basename(project_dir.rstrip("/\\")) or "book", 60)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(daisy_dir):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                rel = os.path.relpath(file_path, daisy_dir)
                zf.write(file_path, os.path.join(top, rel).replace("\\", "/"))
    return os.path.abspath(zip_path)


# ---------------------------------------------------------------------------
# NCC
# ---------------------------------------------------------------------------
def _write_ncc(
    daisy_dir: str,
    title: str,
    entries: List[Dict[str, Any]],
    book_uid: str,
    now_iso: str,
    language: str,
    publisher: str,
    source_file: str,
    multimedia: str,
    total_ms: int,
) -> str:
    """Write the Navigation Control Center (XHTML 1.0 + DAISY metadata)."""
    publisher = publisher.strip() or "AI Voice Studio"

    # First body entry must be an <h1 class="title"> pointing into the first
    # content SMIL (spec 2.1.6.1). Then one heading per recorded segment.
    # Anchors resolve to SMIL <text> ids (sibling-relative, no subfolders).
    body = [
        f'    <h1 class="title" id="ncc_0000">'
        f'<a href="{_smil_name(entries[0])}#'
        f'{_text_id(entries[0])}">{_escape_html(title)}</a></h1>'
    ]
    for entry in entries:
        body.append(
            f'    <h1 id="ncc_{entry["number"]:04d}">'
            f'<a href="{_smil_name(entry)}#'
            f'{_text_id(entry)}">{_escape_html(entry["title"])}</a></h1>'
        )

    source_meta = (
        f'\n    <meta name="dc:source" content="{_escape_html(source_file)}" />'
        if source_file else ""
    )

    ncc = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f"{_XHTML_DOCTYPE}\n"
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        '  <head>\n'
        '    <title>' + _escape_html(title) + '</title>\n'
        '    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n'
        '    <meta name="dc:title" content="' + _escape_html(title) + '" />\n'
        '    <meta name="dc:identifier" content="' + _escape_html(book_uid) + '" />\n'
        '    <meta name="dc:date" content="' + now_iso[:10] + '" scheme="yyyy-mm-dd" />\n'
        '    <meta name="dc:format" content="Daisy 2.02" />\n'
        '    <meta name="dc:language" content="' + _escape_html(language) + '" scheme="ISO 639" />\n'
        '    <meta name="dc:publisher" content="' + _escape_html(publisher) + '" />'
        + source_meta + '\n'
        '    <meta name="ncc:pageFront" content="0" />\n'
        '    <meta name="ncc:pageNormal" content="0" />\n'
        '    <meta name="ncc:pageSpecial" content="0" />\n'
        '    <meta name="ncc:maxPageNormal" content="0" />\n'
        '    <meta name="ncc:setInfo" content="1 of 1" />\n'
        '    <meta name="ncc:depth" content="1" />\n'
        '    <meta name="ncc:multimediaType" content="' + multimedia + '" />\n'
        '    <meta name="ncc:generator" content="AI Voice Studio" />\n'
        '    <meta name="ncc:totalTime" content="' + _format_duration(total_ms) + '" scheme="hh:mm:ss" />\n'
        '    <meta name="ncc:charset" content="utf-8" />\n'
        '    <meta name="ncc:tocItems" content="' + str(len(entries) + 1) + '" />\n'
        '  </head>\n'
        '  <body>\n'
        + "\n".join(body) + "\n"
        '  </body>\n'
        '</html>\n'
    )
    ncc_path = os.path.join(daisy_dir, DAISY_NCC_FILE)
    with open(ncc_path, "w", encoding="utf-8") as fh:
        fh.write(ncc)
    return ncc_path


# ---------------------------------------------------------------------------
# Content SMIL documents
# ---------------------------------------------------------------------------
def _write_smil(
    daisy_dir: str,
    entry: Dict[str, Any],
    book_uid: str,
    book_title: str,
) -> str:
    """Write one SMIL 1.0 content document for a recorded segment.

    Mirrors the structure used by the official DAISY 2.02 samples::

        <smil>
          <head>
            ... metadata ...
            <layout><region id="txtView" /></layout>
          </head>
          <body>
            <seq dur="123.456s">
              <par endsync="last" id="par_0001">
                <text src="0001.html#seg_0001" id="txt_0001" />
                <seq>
                  <audio src="aud0001.mp3" clip-begin="npt=0.000s"
                         clip-end="npt=12.345s" id="aud_0001" />
                </seq>
              </par>
              ... one <par> per text block, audio sliced proportionally ...
            </seq>
          </body>
        </smil>

    Full-text books carry one ``<par>`` per synchronized text block (heading
    plus each paragraph); the single segment audio is divided between them
    proportionally by character count, exactly the mechanism the official
    samples use to slice one audio file across several ``<par>``s.
    """
    smil_name = _smil_name(entry)
    dur = max(0.001, entry["duration"])
    blocks = entry["blocks"]
    total_chars = sum(b["chars"] for b in blocks)

    pars: List[str] = []
    elapsed = 0.0
    for idx, block in enumerate(blocks):
        if idx == len(blocks) - 1:
            # Give the final clip a small tail beyond the measured duration:
            # MP3 encoder delay means decodable audio usually runs slightly
            # LONGER than the container duration.  A clip-end exactly at the
            # measured value makes strict players stop just before the real
            # end of the narration.
            clip_end = dur + 0.25
        else:
            clip_end = min(dur, elapsed + dur * block["chars"] / total_chars)
        pars.append(
            f'      <par endsync="last" id="{block["par_id"]}">\n'
            f'        <text src="{entry["number"]:04d}.html#{block["html_id"]}" '
            f'id="{block["text_id"]}" />\n'
            '        <seq>\n'
            f'          <audio src="{entry["audio"]}" '
            f'clip-begin="npt={elapsed:.3f}s" clip-end="npt={clip_end:.3f}s" '
            f'id="{block["aud_id"]}" />\n'
            '        </seq>\n'
            '      </par>'
        )
        elapsed = clip_end

    smil = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f"{_SMIL_DOCTYPE}\n"
        '<smil>\n'
        '  <head>\n'
        '    <meta name="dc:format" content="Daisy 2.02" />\n'
        '    <meta name="dc:identifier" content="' + _escape_html(book_uid) + '" />\n'
        '    <meta name="dc:title" content="' + _escape_html(book_title) + '" />\n'
        '    <meta name="title" content="' + _escape_html(entry["title"]) + '" />\n'
        '    <meta name="ncc:generator" content="AI Voice Studio" />\n'
        '    <meta name="ncc:totalElapsedTime" content="'
        + _format_duration(_total_ms(entry.get("offset_ms"))) + '" />\n'
        '    <meta name="ncc:timeInThisSmil" content="'
        + _format_duration(int(dur * 1000)) + '" />\n'
        + _SMIL_LAYOUT +
        '  </head>\n'
        '  <body>\n'
        f'    <seq dur="{dur + 0.25:.3f}s">\n'
        + "\n".join(pars) + "\n"
        '    </seq>\n'
        '  </body>\n'
        '</smil>\n'
    )
    with open(os.path.join(daisy_dir, smil_name), "w", encoding="utf-8") as fh:
        fh.write(smil)
    return smil_name


def _write_master_smil(
    daisy_dir: str,
    entries: List[Dict[str, Any]],
    book_uid: str,
    book_title: str,
) -> None:
    """Write the optional Master SMIL chaining every content SMIL document."""
    refs = []
    for entry in entries:
        refs.append(
            f'    <ref title="{_escape_html(entry["title"])}" '
            f'src="{_smil_name(entry)}#'
            f'{_text_id(entry)}" id="smil_{entry["number"]:04d}"/>'
        )
    total_ms = int(sum(e["duration"] for e in entries) * 1000)
    smil = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f"{_SMIL_DOCTYPE}\n"
        '<smil>\n'
        '  <head>\n'
        '    <meta name="ncc:generator" content="AI Voice Studio" />\n'
        '    <meta name="dc:format" content="Daisy 2.02" />\n'
        '    <meta name="dc:title" content="' + _escape_html(book_title) + '" />\n'
        '    <meta name="dc:identifier" content="' + _escape_html(book_uid) + '" />\n'
        '    <meta name="ncc:timeInThisSmil" content="'
        + _format_duration(total_ms) + '" />\n'
        + _SMIL_LAYOUT +
        '  </head>\n'
        '  <body>\n'
        + "\n".join(refs) + "\n"
        '  </body>\n'
        '</smil>\n'
    )
    with open(os.path.join(daisy_dir, DAISY_MASTER_SMIL_FILE), "w", encoding="utf-8") as fh:
        fh.write(smil)


# ---------------------------------------------------------------------------
# Text content documents
# ---------------------------------------------------------------------------
def _write_text_document(
    daisy_dir: str,
    entry: Dict[str, Any],
    language: str,
    include_text: bool,
) -> str:
    """Write one XHTML text content document and return its file name.

    The document always contains a heading element with the segment's sync id
    (``seg_0001``) that the content SMIL ``<text>`` points at.  When
    ``include_text`` is True the full segment text follows as paragraphs and
    every text block carries a back-link to its SMIL ``<text>`` element (the
    bidirectional linking genuine DAISY 2.02 full-text books use, which
    players rely on to render and highlight the text view)::

        <h1 id="seg_0001"><a href="0001.smil#txt_0001">Chapter</a></h1>
        <p id="p_0001_001"><a href="0001.smil#txt_0002">Paragraph.</a></p>
    """
    text_name = f"{entry['number']:04d}.html"
    blocks = entry["blocks"]
    smil_name = _smil_name(entry)
    if include_text:
        doctype = _STRICT_DOCTYPE
        content_type = "application/xhtml+xml; charset=utf-8"
    else:
        doctype = _XHTML_DOCTYPE
        content_type = "text/html; charset=utf-8"

    body_lines: List[str] = []
    for block in blocks:
        escaped = _escape_html(block["text"])
        if include_text:
            back = f'<a href="{smil_name}#{block["text_id"]}">{escaped}</a>'
        else:
            back = escaped
        tag = "h1" if block is blocks[0] else "p"
        body_lines.append(f'    <{tag} id="{block["html_id"]}">{back}</{tag}>')

    html = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f"{doctype}\n"
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        f'xml:lang="{_escape_html(language)}" lang="{_escape_html(language)}">\n'
        '  <head>\n'
        f'    <title>{_escape_html(entry["title"])}</title>\n'
        f'    <meta http-equiv="Content-Type" content="{content_type}" />\n'
        '  </head>\n'
        '  <body>\n'
        + "\n".join(body_lines) + "\n"
        '  </body>\n'
        '</html>\n'
    )
    with open(os.path.join(daisy_dir, text_name), "w", encoding="utf-8") as fh:
        fh.write(html)
    return text_name


# ---------------------------------------------------------------------------
# OPF package manifest (informational)
# ---------------------------------------------------------------------------
def _write_opf(
    daisy_dir: str,
    title: str,
    entries: List[Dict[str, Any]],
    book_uid: str,
    now_iso: str,
    language: str,
    publisher: str,
    multimedia: str,
) -> None:
    """Write an OPF package file listing every component of the book."""
    publisher = publisher.strip() or "AI Voice Studio"

    items = [
        f'    <item id="ncc" href="{DAISY_NCC_FILE}" media-type="application/xhtml+xml"/>',
        f'    <item id="master" href="{DAISY_MASTER_SMIL_FILE}" media-type="application/smil"/>',
    ]
    for entry in entries:
        ext = os.path.splitext(entry["audio"])[1].lstrip(".").lower()
        mime = _AUDIO_MIME.get(ext, "audio/wav")
        items.append(
            f'    <item id="smil_{entry["number"]:04d}" href="{_smil_name(entry)}"'
            f' media-type="application/smil"/>'
        )
        items.append(
            f'    <item id="text_{entry["number"]:04d}" href="{entry["number"]:04d}.html"'
            f' media-type="application/xhtml+xml"/>'
        )
        items.append(
            f'    <item id="audio_{entry["number"]:04d}" href="{entry["audio"]}"'
            f' media-type="{mime}"/>'
        )

    opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid" version="2.0">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        f'    <dc:title>{_escape_html(title)}</dc:title>\n'
        f'    <dc:identifier id="uid">{_escape_html(book_uid)}</dc:identifier>\n'
        f'    <dc:language>{_escape_html(language)}</dc:language>\n'
        f'    <dc:date>{now_iso[:10]}</dc:date>\n'
        f'    <dc:publisher>{_escape_html(publisher)}</dc:publisher>\n'
        '    <dc:format>Daisy 2.02</dc:format>\n'
        f'    <meta name="ncc:multimediaType" content="{multimedia}" />\n'
        '  </metadata>\n'
        '  <manifest>\n'
        + "\n".join(items) + "\n"
        '  </manifest>\n'
        '  <spine>\n'
        '    <itemref idref="master"/>\n'
        '  </spine>\n'
        '</package>\n'
    )
    with open(os.path.join(daisy_dir, DAISY_PACKAGE_FILE), "w", encoding="utf-8") as fh:
        fh.write(opf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _smil_name(entry: Dict[str, Any]) -> str:
    return f"{entry['number']:04d}.smil"


def _text_id(entry: Dict[str, Any]) -> str:
    """The SMIL ``<text>`` id of the segment's first (heading) block."""
    blocks = entry.get("blocks")
    if blocks:
        return blocks[0]["text_id"]
    return f"txt_{entry['number']:04d}"


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
    # MP3/FLAC: probe with FFmpeg if available.
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
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or "")
        if match:
            h, m, s = match.groups()
            return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _total_ms(total_ms_value: Any) -> int:
    """Coerce a milliseconds value (int or str) safely."""
    try:
        return max(0, int(total_ms_value))
    except (TypeError, ValueError):
        return 0


def _format_duration(total_ms: int) -> str:
    """Format milliseconds as ``hh:mm:ss``, rounding UP to the next second.

    Declared times must never be shorter than the real audio: players that
    pace or clamp playback against ``ncc:totalTime`` /
    ``ncc:timeInThisSmil`` would otherwise stop before the audio ends
    (the "last part of a file does not play" bug).
    """
    total_s = -(-max(0, int(total_ms)) // 1000)  # ceiling division
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _escape_html(text: str) -> str:
    """Escape HTML/XML special characters."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
