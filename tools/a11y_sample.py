"""A throwaway project for the accessibility audit tools.

The recording window is the largest dialog in the application, but it can
only be built from a real project folder, so the audits used to skip it
whenever ``build/e2e_omnivoice/design`` was absent.  This module creates a
small, self-contained project under ``build/a11y_sample`` on demand, so
``tools/a11y_audit.py`` and ``tools/a11y_adjacency.py`` always cover it.

Nothing here is imported by the application.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ai_voice_studio import project  # noqa: E402
from ai_voice_studio.constants import MODE_PAGE_WITH_H1  # noqa: E402

SAMPLE_DIR = os.path.join("build", "a11y_sample")


def ensure_sample_project(audio_mode: str = MODE_PAGE_WITH_H1) -> str:
    """Create (or reuse) a small project and return its folder."""
    if not os.path.isfile(os.path.join(SAMPLE_DIR, "project.json")):
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        project.create_project(
            SAMPLE_DIR,
            "A11y sample",
            "a11y_sample.txt",
            audio_mode,
            [
                {
                    "index": index + 1,
                    "title": f"{index + 1:02d} chapter {index + 1}",
                    "text": f"This is the text of chapter {index + 1}.",
                }
                for index in range(6)
            ],
        )
    return SAMPLE_DIR


if __name__ == "__main__":
    print(ensure_sample_project())
