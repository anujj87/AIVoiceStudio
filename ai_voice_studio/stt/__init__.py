"""Speech-to-Text: model catalog, output documents and the transcribe job.

This package is what the ``Speech to text`` project base in the New Project
wizard, the transcribe dialog and the *Manage the STT* Settings category are
built on.  It shares nothing with the TTS side except the managed Python
environments, so an STT model can never break a TTS engine's dependencies.
"""

from __future__ import annotations

from . import catalog, engine, environments, job, output

__all__ = ["catalog", "engine", "environments", "job", "output"]
