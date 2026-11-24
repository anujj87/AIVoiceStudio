"""Speak one sentence with an installed Voice Lab engine, for real.

Drives the same path a recording uses - the voice entry from the catalog, the
app-facing ``VoicelabEngine``, the engine's own virtualenv, its worker
subprocess, the model download and the audio it returns - and reports what
came back: the sample rate, the length of the audio, how long the run took, the
real-time factor, the RMS of the samples, and the application's own
speech-versus-drone verdict.

The verdict matters as much as the RMS: a take with *no speech in it* is loud
(OmniVoice's known drone failure is a steady boom at full level), so every
level test passes while the engine is broken.  The detector used here is the
one the application itself uses to decide that a take must not be written into
a recording (``ai_voice_studio.omnivoice_quality``), and a bad verdict fails
this script the same way silence does.

**Detector and levels are still not enough: neither can tell speech from
babble.**  A take that is speech-shaped noise passes both.  The words are
therefore checked too, by transcribing the take with the application's own
speech-to-text engine - Whisper large-v3, in ``stt_envs\\whisper_large_v3``,
started through ``ai_voice_studio.stt.engine`` exactly as a recording is - and
comparing what came back with the text the engine was given (word error rate,
punctuation and case ignored).  ``--wer`` moves the limit (default 0.35, which
is what separates the engines that speak, 0.00-0.14, from the ones that only
make noise, 0.52-1.00); ``--no-words`` skips the check; ``--asr-accuracy``
chooses the beam size.  The check is a *certification*: when it cannot run -
no Whisper environment, a failed transcription - this script says so and fails
the run rather than reporting a take nobody has listened to.

Exit codes: 0 the engine spoke the words, 1 it did not, or the take could not
be checked (the reason is printed), 2 the engine is not installed / the
arguments are wrong.

    .venv\\Scripts\\python tools/e2e_voice_lab_engine.py f5tts
    .venv\\Scripts\\python tools/e2e_voice_lab_engine.py gptsovits \\
        --reference voice.wav --transcript "the words in voice.wav"
    .venv\\Scripts\\python tools/e2e_voice_lab_engine.py indic_parler --device gpu
    .venv\\Scripts\\python tools/e2e_voice_lab_engine.py vibevoice --language en \\
        --text "a sentence to speak" --wer 0.2

The engine must already be installed (Settings > Voice Clone > Install engine)
and, for the engines whose weights live in a gated repository, a Hugging Face
token with that licence accepted must be in place (Settings > Hugging Face
account).  The first run of an engine downloads its model, so it is slow; the
report says which download happened.

"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import unicodedata
import wave

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_TEXT = {
    "hi": "नमस्ते। यह एक छोटा परीक्षण है, और यह वाक्य धीरे बोला गया है।",
    "en": "This is a short end to end test of a voice engine.",
}

#: How wrong the words may come back before the take is called babble.  The
#: value is measured, not chosen.  Engines that speak the text sit at 0.00-0.23
#: through this check's own default decoding: a cloned take moves a little
#: between runs and the speech model hears one word or another of it
#: differently (0.00-0.18), and Indic Parler-TTS 0.23, with the quickest setting
#: reading the same takes at 0.14 and 0.10.  Engines that only render
#: speech-shaped noise sit at 0.52-1.00.  A limit of 0.35 sits between the two,
#: with room for a take whose accent or clone the speech model hears differently,
#: and is deliberately tight enough to have caught a half-loaded checkpoint.
WORDS_LIMIT = 0.35

#: The speech model the words are checked with: the application's own Whisper
#: large-v3, driven through the application's own STT engine, so the check needs
#: no environment the user does not already have for punctuation and audio
#: description.
WORDS_MODEL = "whisper_large_v3"
WORDS_VARIANT = "large-v3"

#: Unicode categories kept inside a word: letters, numbers and the combining
#: marks that carry the vowels of Devanagari and the other Indic scripts.  A
#: tokenizer that dropped the marks would shred every Hindi word into pieces.
_WORD_CATEGORIES = ("L", "N", "M")

#: What the words check came back with.  ``WORDS_UNCHECKED`` is a failure, not a
#: shrug: a take nobody has listened to must not be certified.  ``WORDS_SKIPPED``
#: is the caller saying so on purpose (``--no-words``), and the run says which
#: of the two it was rather than claiming the words came back either way.
WORDS_OK = "words-ok"
WORDS_FAILED = "words-failed"
WORDS_UNCHECKED = "words-unchecked"
WORDS_SKIPPED = "words-skipped"


def _entry_for(engine_id: str, args) -> dict:
    """The voice entry the panels would hand to the engine.

    A built-in voice when the engine has one; for a clone-only engine (GPT-
    SoVITS) a cloned entry built from the reference recording the caller
    passed, because that is the only way such an engine speaks.
    """
    from ai_voice_studio import voicelab
    from ai_voice_studio.voicelab import engines as engine_registry

    if not args.reference:
        entries = voicelab.builtin_voice_entries(engine_id)
        if entries:
            # The engine's own voice for the language being spoken: the first
            # entry of a multilingual engine is often English, and the engine
            # is *told* the language (OmniVoice's prompt names it), so picking
            # the first entry would speak Hindi through an English prompt.
            wanted = (args.language or "").lower()[:2]
            for entry in entries:
                if str(entry.get("language") or "").lower()[:2] == wanted:
                    return dict(entry)
            return dict(entries[0])

    if not args.reference:
        raise SystemExit(
            f"{engine_id} has no voices of its own: pass --reference and "
            "--transcript (see --help)"
        )
    if not os.path.isfile(args.reference):
        raise SystemExit(f"no such reference recording: {args.reference}")

    variant = engine_registry.clone_variant(engine_id)
    entry = {
        "engine": engine_id,
        "kind": "voicelab",
        "provider": "voicelab",
        "language": args.language,
        "variant": variant,
        "voice": "e2e probe",
        "voice_name": "e2e probe",
        # The worker's own request keys (``voicelab._VOICE_KEYS``): a clone's
        # recording and its transcript travel as ``ref_audio``/``ref_text``.
        "ref_audio": os.path.abspath(args.reference),
        "ref_text": args.transcript or "",
        "device": args.device,
    }
    if variant:
        entry["model"] = variant
    return entry


def _wav_stats(path: str) -> dict:
    with wave.open(path, "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return {
            "frames": frames,
            "rate": rate,
            "channels": handle.getnchannels(),
            "seconds": round(frames / float(rate or 1), 2),
        }


def _words(text: str) -> list:
    """A sentence as comparable words, or an empty list.

    Case, punctuation and spacing are ignored, and nothing else is.  The
    combining marks that carry the vowels of Devanagari and the other Indic
    scripts stay *inside* their word, because a comparison that dropped them
    would shred every Hindi word into pieces; a punctuation mark ends a word,
    so "hello,world" counts as two.
    """
    words = []
    current = []
    for character in unicodedata.normalize("NFC", text):
        if character.isspace() or \
                unicodedata.category(character)[0] not in _WORD_CATEGORIES:
            if current:
                words.append("".join(current))
                current = []
            continue
        current.append(character.lower())
    if current:
        words.append("".join(current))
    return words


def _word_error_rate(expected: str, heard: str) -> float:
    """The share of the expected words that did not come back.

    0.0 is every word, 1.0 is none of them; more than 1.0 means the take said
    extra words as well.  This is the usual edit distance over word lists.
    """
    reference, hypothesis = _words(expected), _words(heard)
    if not reference:
        return float("nan")
    previous = list(range(len(hypothesis) + 1))
    for index, word in enumerate(reference, start=1):
        current = [index]
        for position, heard_word in enumerate(hypothesis, start=1):
            current.append(min(
                previous[position] + 1,                       # a word was lost
                current[position - 1] + 1,                    # a word appeared
                previous[position - 1] + (word != heard_word),  # a word changed
            ))
        previous = current
    return previous[-1] / len(reference)


def _transcribe(path: str, args):
    """What the take says, as heard by the application's own speech-to-text.

    Returns ``(words, None)`` on success and ``("", reason)`` when the check
    cannot be run: no Whisper environment, or a transcription that failed.
    """
    try:
        from ai_voice_studio.stt import engine as stt_engine
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if not stt_engine.is_installed(WORDS_MODEL):
        return "", f"the {WORDS_MODEL} environment is not installed"

    def progress(message, fraction=None):
        if fraction is None:
            print(f"    {message}", flush=True)
        else:
            print(f"    {message} {fraction * 100:.0f}%", flush=True)

    try:
        segments = stt_engine.transcribe_audio(
            path,
            model_id=WORDS_MODEL,
            variant=WORDS_VARIANT,
            language=(args.language or "auto"),
            accuracy=args.asr_accuracy,
            compute="auto",
            on_progress=progress,
        )
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    heard = " ".join(str(segment.get("text") or "") for segment in segments)
    return heard.strip(), None


def _words_report(path: str, text: str, args) -> str:
    """Print what the take says and whether it is the text it was given.

    Returns one of :data:`WORDS_OK`, :data:`WORDS_FAILED`,
    :data:`WORDS_UNCHECKED` or :data:`WORDS_SKIPPED`.
    """
    if args.no_words:
        print("words      : not checked (--no-words)")
        return WORDS_SKIPPED
    if not _words(text):
        print("words      : not checked (the text carries no words)")
        return WORDS_SKIPPED

    heard, failure = _transcribe(path, args)
    if failure is not None:
        print(f"words      : NOT CHECKED ({failure})")
        print("             install Whisper large-v3 in Settings > Manage the "
              "STT, or pass --no-words")
        return WORDS_UNCHECKED
    rate = _word_error_rate(text, heard)
    print(f"heard      : {heard}")
    print(f"words      : {rate:.2f} of them wrong (limit {args.wer:.2f}, "
          f"Whisper large-v3, {args.asr_accuracy})")
    return WORDS_OK if rate <= args.wer else WORDS_FAILED


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("engine", help="Voice Lab engine id, e.g. chatterbox")
    parser.add_argument("--text", default="", help="what to say")
    parser.add_argument("--language", default="hi", help="language code (default hi)")
    parser.add_argument("--device", default="cpu", help="cpu or gpu (default cpu)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--reference", default="",
                        help="reference recording, for a clone-only engine")
    parser.add_argument("--transcript", default="",
                        help="what the reference recording says")
    parser.add_argument("--out", default="",
                        help="where to write the WAV (default build/e2e_engines/)")
    parser.add_argument("--tuning", default="",
                        help="engine tuning as key=value,key=value")
    parser.add_argument("--wer", type=float, default=WORDS_LIMIT,
                        help="word error rate above which the take has not "
                             f"spoken the text (default {WORDS_LIMIT})")
    parser.add_argument("--no-words", action="store_true",
                        help="skip the transcription check")
    parser.add_argument("--asr-accuracy", default="balanced",
                        choices=("fast", "balanced", "best"),
                        help="how carefully the take is transcribed "
                             "(default balanced)")
    args = parser.parse_args(argv)

    utf8_console()
    logging_ready()
    from ai_voice_studio import voicelab

    if not voicelab.engines.is_engine(args.engine):
        print(f"'{args.engine}' is not a Voice Lab engine "
              f"(try one of: {', '.join(voicelab.engines.engine_ids())})")
        return 2

    if not voicelab.is_installed(args.engine):
        print(f"{args.engine} is not installed. Install it in "
              f"Settings > Voice Clone, or with:\n"
              f"  {sys.executable} -c \"from ai_voice_studio import voicelab; "
              f"print(voicelab.packages_for('{args.engine}'))\"")
        return 2

    text = args.text or DEFAULT_TEXT.get(args.language, DEFAULT_TEXT["en"])
    entry = _entry_for(args.engine, args)
    if args.tuning:
        options = {}
        for part in args.tuning.split(","):
            key, _, value = part.partition("=")
            if key.strip():
                options[key.strip()] = _coerce(value.strip())
        entry["options"] = options
    entry["device"] = args.device

    out_dir = args.out or os.path.join(REPO_ROOT, "build", "e2e_engines")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.engine}_{args.device}.wav")

    print(f"engine    : {args.engine} ({voicelab.engines.engine_name(args.engine)})")
    print(f"installed : {voicelab.installed_version(args.engine)}")
    print(f"device    : {args.device}")
    print(f"text      : {text}")
    print(f"output    : {out_path}")
    print("synthesizing ...", flush=True)

    from ai_voice_studio.voicelab import VoicelabEngine

    started = time.monotonic()
    try:
        engine = VoicelabEngine(entry, device=args.device)
        samples = engine.synthesize(text, speed=args.speed)
        sample_rate = engine.sample_rate
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    elapsed = time.monotonic() - started

    import numpy as np

    audio = np.asarray(samples).reshape(-1)
    seconds = len(audio) / float(sample_rate or 1)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0
    with wave.open(out_path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(audio.astype("<i2").tobytes())

    print()
    print("=" * 70)
    print(f"samples    : {len(audio)} at {sample_rate} Hz")
    print(f"audio      : {seconds:.2f} s")
    print(f"elapsed    : {elapsed:.2f} s")
    print(f"real-time  : {seconds / elapsed:.2f}x" if elapsed else "")
    print(f"peak / rms : {int(np.max(np.abs(audio))) if len(audio) else 0} / {rms:.0f}")
    print(f"wav        : {out_path}  {_wav_stats(out_path)}")
    if seconds < 0.2:
        print("!! the run produced less than a fifth of a second of audio")
        return 1
    if rms < 30:
        print("!! the audio is (near) silence")
        return 1
    verdict = _judge(audio)
    if verdict is not None:
        print(f"speech     : {verdict.speech_ratio:.2f} of the audible windows "
              f"sound like speech, {verdict.drone_s:.1f} s drone")
        if verdict.bad:
            print(f"!! the take is not speech: {verdict.reason}")
            return 1

    # Loud, speech-shaped and still not the words: the detector above cannot
    # tell speech from babble, so the take is transcribed and compared.
    if not args.no_words:
        print("transcribing ...", flush=True)
    outcome = _words_report(out_path, text, args)
    if outcome == WORDS_UNCHECKED:
        print("!! the words could not be checked, so this take is not certified")
        return 1
    if outcome == WORDS_FAILED:
        print(f"!! the engine did not speak the text it was given (over "
              f"{args.wer:.2f} of its words wrong)")
        return 1
    if outcome == WORDS_SKIPPED:
        print("OK: the engine spoke (the words were not checked).")
        return 0
    print("OK: the engine spoke the words.")
    return 0


def _judge(audio):
    """The application's own speech-versus-drone verdict, or ``None``.

    The detector needs nothing but NumPy, and a take it refuses is a take the
    application would refuse to write into a recording - so it is reported here
    rather than found out afterwards.  A detector that cannot run (an import
    this environment does not have) must not fail an engine that just spoke.
    """
    try:
        from ai_voice_studio import omnivoice_quality as quality

        return quality.judge(audio)
    except Exception as exc:
        print(f"speech     : not judged ({type(exc).__name__}: {exc})")
        return None


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def logging_ready() -> None:
    """Show the workers' warnings and notes on the console."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def utf8_console() -> None:
    """Let the report print Devanagari.

    A Windows console defaults to cp1252, where a Hindi sentence raises
    ``UnicodeEncodeError`` - the script must be able to echo back the text it
    was asked to speak.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    sys.exit(main())
