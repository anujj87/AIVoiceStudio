"""One-off tool: rebuild the DAISY 2.02 book of an existing project.

Uses the same settings the recording dialog uses (project.json daisy/tts
sections) so the regenerated book is identical to what the app would build,
but with the fixed SMIL clips.  Run from the repository root::

    .venv/Scripts/python.exe tools/rebuild_daisy2_book.py <project-dir>

Audio is NOT re-recorded: only the DAISY/ book structure is regenerated
from the already recorded segment files.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import project  # noqa: E402
from ai_voice_studio.documents.daisy_builder import build_daisy_book  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python tools/rebuild_daisy2_book.py <project-dir>")
        return 2
    project_dir = sys.argv[1]
    fresh = project.load_project(project_dir)
    pd = fresh.get("daisy") or {}
    meta = {
        "title": pd.get("title") or fresh.get("name", "Untitled"),
        "creator": pd.get("creator", ""),
        "date": pd.get("date"),
        "subject": pd.get("subject", ""),
        "narrator": pd.get("narrator", ""),
        "producer": pd.get("producer", ""),
        "show_software": pd.get("show_software", True),
    }
    result = build_daisy_book(
        output_dir=project_dir,
        project_name=fresh.get("name", "Untitled"),
        segments=fresh.get("segments", []),
        audio_format=fresh.get("tts", {}).get("output_format", "wav"),
        include_text=False,  # daisy_audio project type
        language=pd.get("language", "en"),
        publisher=pd.get("publisher", ""),
        source_file=fresh.get("source_file", ""),
        meta=meta,
    )
    print("ncc:", result or "(no completed segments)")
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
