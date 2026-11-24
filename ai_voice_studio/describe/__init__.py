"""Audio Description: describe the picture of a video, offline.

This package is what the ``Audio description`` project base in the New Project
wizard, the describe dialog and the *Manage the video describer* Settings
category are built on.  Like :mod:`ai_voice_studio.stt` it shares nothing with
the TTS side except the managed Python environments, so a describing model can
never break a voice or a transcription model - and vice versa.

The parts, in the order a run uses them:

* :mod:`~ai_voice_studio.describe.media` - how long is the video (FFmpeg probe).
* :mod:`~ai_voice_studio.describe.pipeline` - pure planning: chunking, the time
  estimate and fitting a description into a silent gap.
* :mod:`~ai_voice_studio.describe.prompts` - the professional audio-description
  guidelines every request carries.
* :mod:`~ai_voice_studio.describe.catalog` - which models, with their licences,
  download sizes, video-memory needs and hallucination notes.
* :mod:`~ai_voice_studio.describe.environments` - one virtualenv per model.
* :mod:`~ai_voice_studio.describe.engine` - running the worker subprocess.
* :mod:`~ai_voice_studio.describe.narration` - speaking the descriptions with
  an installed offline voice, so the project also exports a spoken track.
* :mod:`~ai_voice_studio.describe.extended` - **extended** audio description:
  stopping the film for each description, which is the only way to describe a
  video whose dialogue never leaves a pause, and rendering the described film.
* :mod:`~ai_voice_studio.describe.job` - the background run and its documents.
"""

from __future__ import annotations

from . import (
    catalog,
    engine,
    environments,
    extended,
    job,
    media,
    narration,
    output,
    pipeline,
    prompts,
)

__all__ = [
    "catalog", "engine", "environments", "extended", "job", "media",
    "narration", "output", "pipeline", "prompts",
]
