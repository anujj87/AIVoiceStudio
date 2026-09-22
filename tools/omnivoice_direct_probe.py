"""Dev probe: the direct OmniVoice engine with the drone recovery in place.

Runs the real worker (``omnivoice-triton`` in the managed environment) through
``OmniVoiceEngine.synthesize`` and reports, for each text, what the detector
thinks of the delivered take.  Nothing is written into a project.

    .venv\\Scripts\\python tools/omnivoice_direct_probe.py

Exit code is non-zero if a delivered take still looks like a drone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.path.join(REPO_ROOT, "build", "tmp", "direct_probe")

#: The managed environment that owns omnivoice-triton + torch.  In development
#: the worker is spawned with ``sys.executable``, so point it there (the app
#: venv has neither).
MANAGED_PYTHON = os.path.join(
    os.environ.get("APPDATA", ""), "AIVoiceStudio", "tts_envs", "omnivoice",
    "Scripts", "python.exe",
)

TEXTS = [
    "Hello. This is the direct OmniVoice engine with drone recovery enabled.",
    ("रैंड ने देखा कि सफ़ेद झंडा दिखाने वाले दो आदमी सड़क के उस पार खड़े थे और वे "
     "उसकी ओर देख रहे थे। उनमें से एक की आँख काली थी और उसका जबड़ा सूजा हुआ था, "
     "जिससे वह और भी डरावना लग रहा था। रैंड ने धीरे से सिर हिलाया और अपनी तलवार "
     "की मूठ पर हाथ रख दिया।"),
]


PROJECTS = os.path.join(os.path.expandvars("%APPDATA%"), "AIVoiceStudio", "projects")


def _paragraphs(project: str, segment: int, limit: int) -> list:
    """Real paragraphs of a project, the units a recording sends one by one."""
    path = os.path.join(PROJECTS, project, "project.json")
    if not os.path.isfile(path):
        raise SystemExit(f"no project file at {path}")
    data = json.load(open(path, encoding="utf-8"))
    segments = data.get("segments") or []
    if not segments:
        raise SystemExit("project has no segments")
    seg = segments[max(0, min(segment, len(segments) - 1))]
    text = seg.get("text") or ""
    out: list = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if len(paragraph) < 20:
            continue
        out.append(paragraph)
        if len(out) >= limit:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-project", default="",
                        help="measure real paragraphs of this project instead")
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--paragraphs", type=int, default=8)
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    from ai_voice_studio import omnivoice_quality as quality
    from ai_voice_studio.omnivoice import OmniVoiceEngine

    if not os.path.isfile(MANAGED_PYTHON):
        print(f"managed python not found: {MANAGED_PYTHON}")
        return 2
    sys.executable = MANAGED_PYTHON
    os.makedirs(OUT_DIR, exist_ok=True)

    texts = TEXTS
    if args.from_project:
        texts = _paragraphs(args.from_project, args.segment, args.paragraphs)
        print(f"measuring {len(texts)} paragraph(s) of {args.from_project}")

    engine = OmniVoiceEngine({
        "engine": "omnivoice",
        "tts": "omnivoice",
        "voice": "alloy",
        "variant": "triton",
    })
    engine.language = args.language
    bad = 0
    repaired = 0
    try:
        for index, text in enumerate(texts, start=1):
            started = time.time()
            samples = engine.synthesize(text)
            took = time.time() - started
            verdict = quality.judge(samples)
            path = os.path.join(OUT_DIR, f"take{index}.wav")
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(samples.tobytes())
            print(
                f"[{index}] {len(text)} chars, {len(samples) / 24000:.1f}s audio, "
                f"{took:.1f}s, speech ratio {verdict.speech_ratio:.2f}, "
                f"drone {verdict.drone_s:.1f}s -> {'BAD' if verdict.bad else 'ok'}"
            )
            if engine.last_repairs:
                record = engine.last_repairs[0]
                repaired += 1
                print(
                    f"    repaired: {record['attempts']} attempts, "
                    f"{record['pieces']} piece(s), {record['draws']} draws, "
                    f"{record['unrepaired']} unresolved"
                )
            for warning in engine.last_warnings:
                print(f"    warning: {warning}")
            bad += 1 if verdict.bad else 0
    finally:
        engine.close()
    print(
        f"delivered takes that still look like a drone: {bad}/{len(texts)}; "
        f"takes the recovery had to repair: {repaired}/{len(texts)}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
