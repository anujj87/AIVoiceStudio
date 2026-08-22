"""DAISY 2.02 book builder.

Generates the DAISY 2.02 book structure after all segments have been recorded:

* ``ncc.html``          -- Navigation Center Combined (master TOC)
* ``package.opf``       -- OPF content manifest
* ``audio/<segment>``   -- audio files (MP3/WAV)
* ``smil/<segment>.smil`` -- SMIL files (audio-text sync or audio-only nav)

The builder works in two modes:

* **Audio only** -- SMIL files reference audio for navigation; NCC provides
  chapter-level links.  No text content is stored.
* **Audio + text** -- SMIL files contain both ``<audio>`` and ``<text>``
  references, enabling synchronized playback in DAISY-capable readers.

All paths and identifiers follow the DAISY 2.02 / Z39.86 specification.
"""

from __future__ import annotations

import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..constants import (
    DAISY_AUDIO_DIR_NAME,
    DAISY_NCC_FILE,
    DAISY_PACKAGE_FILE,
)
from ..util import sanitize_filename


def build_daisy_book(
    output_dir: str,
    project_name: str,
    segments: List[Dict[str, Any]],
    audio_format: str = "mp3",
    include_text: bool = False,
    language: str = "en",
    publisher: str = "",
) -> str:
    """Build the complete DAISY 2.02 book structure.

    Parameters
    ----------
    output_dir:
        Project folder where audio files already exist.
    project_name:
        Human-readable book title.
    segments:
        List of segment dicts with keys: ``index``, ``title``, ``text``,
        ``saved`` (audio filename), ``status``.
    audio_format:
        ``"mp3"`` or ``"wav"``.
    include_text:
        If True, generate audio+text SMIL files (DAISY full-text+audio).
    language:
        ISO 639 language code for the ``dc:language`` metadata.
    publisher:
        Publisher name for metadata.

    Returns
    -------
    str
        Path to the ``ncc.html`` file.
    """
    # Create output subdirectories
    audio_dir = os.path.join(output_dir, DAISY_AUDIO_DIR_NAME)
    os.makedirs(audio_dir, exist_ok=True)

    # Collect ready segments
    ready = [
        s for s in segments
        if s.get("status") == "done" and s.get("saved")
    ]
    if not ready:
        return ""

    # Build the unique id base for this book
    book_uid = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Generate NCC (ncc.html)
    ncc_path = _write_ncc(
        output_dir, project_name, ready, book_uid, now_iso,
        language, publisher, audio_format, include_text,
    )

    # 2. Generate OPF (package.opf)
    _write_opf(
        output_dir, project_name, ready, book_uid, now_iso,
        language, publisher, audio_format, include_text,
    )

    # 3. Generate SMIL files for each chapter
    _write_smil_files(
        output_dir, ready, audio_format, include_text,
    )

    return ncc_path


def _sanitize_id(text: str) -> str:
    """Create a DAISY-safe identifier from text."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", text.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:60] or "item"


def _write_ncc(
    output_dir: str,
    title: str,
    segments: List[Dict[str, Any]],
    book_uid: str,
    now_iso: str,
    language: str,
    publisher: str,
    audio_format: str,
    include_text: bool,
) -> str:
    """Write the NCC (Navigation Center Combined) HTML file."""
    ext = audio_format.lower()
    entries = []
    for seg in segments:
        safe_title = sanitize_filename(seg["title"], 60)
        smil_file = f"{safe_title}.smil"
        audio_file = seg["saved"]
        entries.append(
            f'  <h2><a href="{smil_file}">{_escape_html(seg["title"])}</a></h2>'
        )

    publisher_line = f'\n  <meta name="dc:publisher" content="{_escape_html(publisher)}" />' if publisher else ""
    text_class = "fulltext" if include_text else "audioonly"
    doctype = "-//W3C//DTD XHTML 1.0 Transitional//EN"
    dtd_url = "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"

    ncc = f"""<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "{dtd_url}">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{language}" lang="{language}">
<head>
  <title>{_escape_html(title)}</title>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <meta name="dc:title" content="{_escape_html(title)}" />
  <meta name="dc:identifier" content="{book_uid}" />
  <meta name="dc:language" content="{language}" />
  <meta name="dc:date" content="{now_iso[:10]}" />{publisher_line}
  <meta name="ncc:generator" content="AI Voice Studio" />
  <meta name="ncc:multimediaType" content="{"DTBText" if include_text else "DTBAudio"}" />
  <meta name="ncc:totalTime" content="00:00:00" />
  <meta name="ncc:pageFront" content="1" />
  <style type="text/css">
    body {{ font-family: sans-serif; margin: 2em; }}
    h1 {{ font-size: 1.5em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
    h2 {{ font-size: 1.1em; margin: 0.8em 0 0.2em; }}
    a {{ text-decoration: none; color: #0066cc; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body class="{text_class}">
  <h1>{_escape_html(title)}</h1>
{chr(10).join(entries)}
</body>
</html>
"""
    ncc_path = os.path.join(output_dir, DAISY_NCC_FILE)
    with open(ncc_path, "w", encoding="utf-8") as fh:
        fh.write(ncc)
    return ncc_path


def _write_opf(
    output_dir: str,
    title: str,
    segments: List[Dict[str, Any]],
    book_uid: str,
    now_iso: str,
    language: str,
    publisher: str,
    audio_format: str,
    include_text: bool,
) -> None:
    """Write the OPF (Open Packaging Format) content manifest."""
    ext = audio_format.lower()
    publisher_el = f"\n    <dc:publisher>{_escape_xml(publisher)}</dc:publisher>" if publisher else ""

    manifest_items = []
    spine_refs = []
    for seg in segments:
        safe_title = sanitize_filename(seg["title"], 60)
        smil_id = _sanitize_id(safe_title)
        manifest_items.append(
            f'    <item id="{smil_id}" media-type="application/smil" '
            f'href="{safe_title}.smil"/>'
        )
        spine_refs.append(f'    <itemref idref="{smil_id}"/>')

    opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE package PUBLIC "-//NISO//DTD dtbook 2005-2//EN"
  "http://www.daisy.org/z3986/2005/dtbook-2005-2.dtd">
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"
            xmlns:oebps="http://openebook.org/namespaces/oebps/1.0/">
    <dc:title>{_escape_xml(title)}</dc:title>
    <dc:identifier id="uid">{_escape_xml(book_uid)}</dc:identifier>
    <dc:language>{_escape_xml(language)}</dc:language>
    <dc:date>{now_iso[:10]}</dc:date>{publisher_el}
    <dc:format>application/DTBook+XML</dc:format>
    <oebps:compression>gzip</oebps:compression>
  </metadata>
  <manifest>
{chr(10).join(manifest_items)}
  </manifest>
  <spine>
{chr(10).join(spine_refs)}
  </spine>
</package>
"""
    opf_path = os.path.join(output_dir, DAISY_PACKAGE_FILE)
    with open(opf_path, "w", encoding="utf-8") as fh:
        fh.write(opf)


def _write_smil_files(
    output_dir: str,
    segments: List[Dict[str, Any]],
    audio_format: str,
    include_text: bool,
) -> None:
    """Write SMIL files for each chapter segment."""
    ext = audio_format.lower()
    for seg in segments:
        safe_title = sanitize_filename(seg["title"], 60)
        smil_path = os.path.join(output_dir, f"{safe_title}.smil")
        audio_file = seg["saved"]
        text_file = f"{safe_title}.txt" if include_text else None

        # Build SMIL content
        if include_text and text_file:
            smil = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE smil PUBLIC "-//W3C//DTD SMIL 2.0//EN"
  "http://www.w3.org/TR/2005/REC-SMIL2-20050107/DTD/smil20.dtd">
<smil xmlns="http://www.w3.org/2001/SMIL20/Language">
  <head>
    <meta name="dc:title" content="{_escape_xml(seg['title'])}" />
  </head>
  <body>
    <seq>
      <par>
        <text src="{text_file}" />
        <audio src="{DAISY_AUDIO_DIR_NAME}/{audio_file}" />
      </par>
    </seq>
  </body>
</smil>
"""
        else:
            smil = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE smil PUBLIC "-//W3C//DTD SMIL 2.0//EN"
  "http://www.w3.org/TR/2005/REC-SMIL-20050107/DTD/smil20.dtd">
<smil xmlns="http://www.w3.org/2001/SMIL20/Language">
  <head>
    <meta name="dc:title" content="{_escape_xml(seg['title'])}" />
  </head>
  <body>
    <seq>
      <par>
        <audio src="{DAISY_AUDIO_DIR_NAME}/{audio_file}" />
      </par>
    </seq>
  </body>
</smil>
"""
        with open(smil_path, "w", encoding="utf-8") as fh:
            fh.write(smil)

        # For audio+text, also write the text file next to the SMIL
        if include_text:
            txt_path = os.path.join(output_dir, f"{safe_title}.txt")
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(seg.get("text", ""))


def export_daisy_zip(project_dir: str, zip_path: str) -> str:
    """Package the DAISY book into a distributable ZIP archive.

    Parameters
    ----------
    project_dir:
        Path to the project folder containing DAISY files.
    zip_path:
        Destination path for the ZIP file.

    Returns
    -------
    str
        Absolute path to the created ZIP file.
    """
    # Files and directories to include in the DAISY book
    daisy_files = [
        DAISY_NCC_FILE,
        DAISY_PACKAGE_FILE,
    ]
    daisy_dirs = [
        DAISY_AUDIO_DIR_NAME,
        "smil",
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add individual files
        for fname in daisy_files:
            fpath = os.path.join(project_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)

        # Add directories recursively
        for dirname in daisy_dirs:
            dirpath = os.path.join(project_dir, dirname)
            if not os.path.isdir(dirpath):
                continue
            for root, _dirs, files in os.walk(dirpath):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, project_dir)
                    zf.write(file_path, arcname)

        # Also include any chapter text files (for audio+text books)
        for fname in os.listdir(project_dir):
            if fname.endswith(".txt") and fname not in ("segments.txt",):
                fpath = os.path.join(project_dir, fname)
                if os.path.isfile(fpath):
                    zf.write(fpath, fname)

    return os.path.abspath(zip_path)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _escape_xml(text: str) -> str:
    """Escape XML special characters."""
    return _escape_html(text)
