"""Measure OmniVoice's drone failure against the real server.

Run it from the project root with the app's own Python::

    # How often does one plain draw of a real chapter paragraph drone?
    .venv/Scripts/python.exe tools/omnivoice_drone_probe.py --mode raw --paragraphs 8

    # Does the client's repair remove them? (what a recording actually gets)
    .venv/Scripts/python.exe tools/omnivoice_drone_probe.py --mode client --paragraphs 20

``raw`` answers the first question and whether re-drawing fixes it; ``client``
goes through the shipping code path (``OmniVoiceServerManager.synthesize``) and
reports how many chunks needed a second draw, how long that cost, and whether
any drone survived into the delivered audio.

Both write a UTF-8 JSON report next to the app log; the console only prints
ASCII progress, so it survives a cp1252 terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_voice_studio.omnivoice import spec  # noqa: E402
from ai_voice_studio.omnivoice_server import (  # noqa: E402
    DRONE_REPAIR_CHARS,
    OmniVoiceServerManager,
    quality,
    split_text_for_server,
)
from ai_voice_studio.paths import logs_dir  # noqa: E402

PROJECTS = os.path.join(os.path.expandvars("%APPDATA%"), "AIVoiceStudio", "projects")


def _load_paragraphs(project: str, segment: int, limit: int) -> list:
    path = os.path.join(PROJECTS, project, "project.json")
    if not os.path.isfile(path):
        raise SystemExit(f"no project file at {path}")
    data = json.load(open(path, encoding="utf-8"))
    segments = data.get("segments") or []
    if not segments:
        raise SystemExit("project has no segments")
    seg = segments[max(0, min(segment, len(segments) - 1))]
    return [c for c in split_text_for_server(seg.get("text") or "") if c][:limit]


def _raw_draw(mgr, text, seed, voice, instruct, language):
    """One request with no repair at all: what the server actually returned."""
    wav, headers = mgr._post_speech(
        text,
        url=f"{mgr.base_url}/v1/audio/speech",
        voice=voice,
        instructions=instruct,
        speed=1.0,
        stream=False,
        response_format="wav",
        optional={"guidance_scale": 3.0, "denoise": True, "language": language},
        seed=seed,
        request_timeout_s=None,
    )
    samples = mgr._wav_bytes_to_samples(wav)
    return samples, quality.judge(
        samples, server_flagged=headers.get("x-no-speech-detected") == "true"
    )


def _raw_report(mgr, paragraphs, args, instruct) -> dict:
    report = {"mode": "raw", "paragraphs": []}
    for i, text in enumerate(paragraphs, 1):
        print(f"\n[{i}/{len(paragraphs)}] {len(text)} chars", flush=True)
        row = {"index": i, "chars": len(text), "head": text[:100], "draws": []}
        verdict = None
        for attempt in range(1, 2 + args.retries):
            samples, verdict = _raw_draw(
                mgr, text, 1000 + i + (attempt - 1) * 7919, args.voice, instruct,
                args.language,
            )
            row["draws"].append(verdict._asdict())
            print(f"  draw {attempt}: {quality.describe(samples)} "
                  f"bad={verdict.bad}", flush=True)
            if not verdict.bad:
                break
        row["recovered_by_redraw"] = (
            not verdict.bad and any(d["bad"] for d in row["draws"])
        )
        report["paragraphs"].append(row)

    first = [r for r in report["paragraphs"] if r["draws"][0]["bad"]]
    all_bad = [r for r in report["paragraphs"] if r["draws"][-1]["bad"]]
    report["summary"] = {
        "paragraphs": len(report["paragraphs"]),
        "first_draw_drones": len(first),
        "first_draw_drone_rate": round(
            len(first) / max(1, len(report["paragraphs"])), 3
        ),
        "recovered_by_redraw": len(first) - len(all_bad),
        "still_bad_after_every_draw": len(all_bad),
    }
    return report


def _client_report(mgr, paragraphs, args, instruct) -> dict:
    """The shipping path: what a recording is actually written from."""
    counts = {"requests": 0}
    original = mgr._post_speech

    def counting(*call_args, **call_kwargs):
        counts["requests"] += 1
        return original(*call_args, **call_kwargs)

    mgr._post_speech = counting
    report = {"mode": "client", "paragraphs": []}
    try:
        for i, text in enumerate(paragraphs, 1):
            before = counts["requests"]
            t0 = time.time()
            samples = mgr.synthesize(
                text, voice=args.voice, instructions=instruct,
                language=args.language, seed=1000 + i,
            )
            wall = time.time() - t0
            verdict = quality.judge(samples)
            repairs = list(mgr.last_repairs)
            row = {
                "index": i,
                "chars": len(text),
                "head": text[:100],
                "draws": counts["requests"] - before,
                "wall_s": round(wall, 1),
                "verdict": verdict._asdict(),
                "repairs": repairs,
            }
            report["paragraphs"].append(row)
            print(f"[{i}/{len(paragraphs)}] {len(text)}c draws={row['draws']} "
                  f"{wall:.0f}s ratio={verdict.speech_ratio} "
                  f"drone={verdict.drone_s}s bad={verdict.bad} "
                  f"repaired={len(repairs)}", flush=True)
    finally:
        mgr._post_speech = original

    delivered = [r for r in report["paragraphs"] if r["verdict"]["bad"]]
    retried = [r for r in report["paragraphs"] if r["draws"] > 1]
    report["summary"] = {
        "paragraphs": len(report["paragraphs"]),
        "requests": counts["requests"],
        "requests_per_chunk": round(
            counts["requests"] / max(1, len(report["paragraphs"])), 2
        ),
        "chunks_needing_a_redraw": len(retried),
        "chunks_sentence_repaired": sum(
            1 for r in report["paragraphs"] if r["repairs"]
        ),
        "drones_delivered": len(delivered),
        "seconds": round(sum(r["wall_s"] for r in report["paragraphs"]), 1),
    }
    report["config"] = {
        "voice": args.voice,
        "language": args.language,
        "drone_attempts": mgr.drone_attempts,
        "repair_attempts": mgr.repair_attempts,
        "repair_chars": DRONE_REPAIR_CHARS,
        "num_chunks_in_text": len(paragraphs),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("raw", "client"), default="raw")
    ap.add_argument("--project", default="TheEyeOfTheWorld")
    ap.add_argument("--segment", type=int, default=18)
    ap.add_argument("--paragraphs", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3, help="raw mode only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--voice", default="alloy")
    ap.add_argument("--language", default="hi")
    args = ap.parse_args()

    paragraphs = _load_paragraphs(args.project, args.segment, args.paragraphs)
    instruct = spec.resolve_instruct("omnivoice_server", args.voice)
    print(f"mode={args.mode} paragraphs={len(paragraphs)} "
          f"voice={args.voice!r} language={args.language}", flush=True)

    mgr = OmniVoiceServerManager(device=args.device, num_steps=args.num_steps)
    t0 = time.time()
    print("starting the OmniVoice server (loads the model if needed)...", flush=True)
    mgr.start()
    print(f"server ready in {time.time() - t0:.0f}s", flush=True)

    if args.mode == "raw":
        report = _raw_report(mgr, paragraphs, args, instruct)
    else:
        report = _client_report(mgr, paragraphs, args, instruct)
    report["project"] = args.project
    report["segment"] = args.segment
    report["instructions"] = instruct

    out = os.path.join(logs_dir(), f"omnivoice_drone_probe_{args.mode}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("\n=== summary ===")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
