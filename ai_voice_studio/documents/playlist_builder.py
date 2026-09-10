"""Playlist file generation for the ``audio_playlist`` project type.

After recording completes, three standard playlist files are written next to
the audio files in the project folder, so the narration opens in modern
media players (VLC, PotPlayer, GOM, Windows Media Player, Winamp, ...):

* ``<project>.m3u8`` -- extended M3U (UTF-8); VLC, PotPlayer, GOM, mpv ...
* ``<project>.pls``  -- Winamp-style playlist; VLC, GOM, PotPlayer, Winamp
* ``<project>.wpl``  -- Windows Media Player playlist (Zune/XP style);
  Windows Media Player, PotPlayer, GOM

All three reference the audio by file name only (sibling-relative paths), so
the playlists keep working when the project folder is copied or zipped and
opened elsewhere.  Entries are ordered by segment index -- the same order
the audio was recorded in.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from xml.sax.saxutils import escape as _xml_escape


def _quoted(value: str) -> str:
    """XML attribute value, always wrapped in double quotes."""
    return '"%s"' % _xml_escape(value).replace('"', "&quot;")


def _ready_entries(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recorded segments (``status == "done"`` with a saved file), in order."""
    entries = [
        s for s in segments
        if s.get("status") == "done" and s.get("saved")
    ]
    entries.sort(key=lambda s: s.get("index", 0))
    return entries


def _titles(entries: List[Dict[str, Any]]) -> List[str]:
    return [str(s.get("title") or os.path.basename(s["saved"])) for s in entries]


def _files(entries: List[Dict[str, Any]]) -> List[str]:
    return [os.path.basename(str(s["saved"])) for s in entries]


def _write_m3u8(path: str, title: str, names: List[str], titles: List[str]) -> None:
    lines = ["#EXTM3U", f"#PLAYLIST:{title}"]
    for name, entry_title in zip(names, titles):
        lines.append(f"#EXTINF:-1,{entry_title}")
        lines.append(name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_pls(path: str, title: str, names: List[str], titles: List[str]) -> None:
    lines = ["[playlist]"]
    for i, (name, entry_title) in enumerate(zip(names, titles), start=1):
        lines.append(f"File{i}={name}")
        lines.append(f"Title{i}={entry_title}")
        lines.append(f"Length{i}=-1")
    lines.append(f"NumberOfEntries={len(names)}")
    lines.append("Version=2")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_wpl(path: str, title: str, names: List[str]) -> None:
    # Windows Media Player playlist (SMIL).  Duration is unknown per entry;
    # WMP tolerates a duration of 0 and reads the real length from the file.
    body = "".join(
        f'      <media src={_quoted(name)}/>\n' for name in names
    )
    wpl = (
        '<?wpl version="1.0"?>\n'
        '<smil>\n'
        '  <head>\n'
        '    <meta name="Generator" content="AI Voice Studio"/>\n'
        f'    <title>{_quoted(title)}</title>\n'
        '  </head>\n'
        '  <body>\n'
        '    <seq>\n'
        + body +
        '    </seq>\n'
        '  </body>\n'
        '</smil>\n'
    )
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(wpl)


def build_playlists(
    project_dir: str,
    project_name: str,
    segments: List[Dict[str, Any]],
) -> List[str]:
    """Write the M3U8/PLS/WPL playlists into ``project_dir``.

    Returns the list of written file paths; empty when no segment has been
    recorded yet.  Existing playlist files are overwritten so they stay in
    sync with the project's segments (e.g. after re-recording).
    """
    entries = _ready_entries(segments)
    if not entries:
        return []
    safe = "".join(
        c if c not in '\\/:*?"<>|' else "_" for c in project_name
    ).strip() or "playlist"
    names = _files(entries)
    titles = _titles(entries)

    written: List[str] = []
    m3u8_path = os.path.join(project_dir, safe + ".m3u8")
    _write_m3u8(m3u8_path, project_name, names, titles)
    written.append(m3u8_path)

    pls_path = os.path.join(project_dir, safe + ".pls")
    _write_pls(pls_path, project_name, names, titles)
    written.append(pls_path)

    wpl_path = os.path.join(project_dir, safe + ".wpl")
    _write_wpl(wpl_path, project_name, names)
    written.append(wpl_path)
    return written
