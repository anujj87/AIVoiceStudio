"""The audio-description worker: one process, one model, one video.

This script is deliberately **standalone**.  It is started with the interpreter
of a describing model's own virtualenv (see
:mod:`ai_voice_studio.describe.environments`) and it imports only third-party
packages - ``transformers``, ``torch``, ``PIL`` - never the application.  That is
what keeps a model's dependencies out of the GUI process, exactly as the
speech-to-text worker does.

Usage::

    python worker.py request.json

The request JSON carries the video, the chunk plan, the frame rate, the device,
the Hugging Face repository and the guideline prompt.  The worker:

1. samples frames from each chunk with FFmpeg (one invocation per chunk);
2. throws away the chunks whose picture has barely changed since the last
   description, so a held shot or a slide that is simply still on screen costs
   neither a description nor - in extended mode - a pause;
3. asks the model for a short, objective description of the chunks that are
   left;
4. writes each finished chunk to stdout as ``AIVS-PROGRESS|<percent>%|<message>``
   so the dialog can show real progress on a long video;
5. writes ``{"ok": true, "segments": [...]}`` to the ``out`` path.

A request may instead carry ``"mode": "download"``, which makes the worker
fetch one variant's checkpoint files and describe nothing at all; see
:func:`_download_weights`.

Only the visual frames are passed to the model.  The models here that also take
audio (Qwen2.5-Omni, Gemma 4 E2B/E4B/12B) will still be driven from frames,
which keeps one code path for every model in the catalog; the soundtrack is used
by the optional gap-fitting pass instead, which is where timing actually needs
it.

One detail does differ between the families: some processors want the frames
embedded in the chat message, and some (Gemma 3's template, for one) want the
prompt to carry a placeholder per frame with the frames handed over beside it.
:func:`_describe_chunk` tries the first and falls back to the second, so adding
a model does not mean adding a second way of asking.

A second detail differs between the *checkpoints*: which ``transformers`` class
loads them, and how they are asked for a description at all.  A checkpoint that
ships its own modelling code registers the class it provides in its
``config.json``, and the MiniCPM-o/V family registers ``AutoModel`` while
registering nothing for the task-specific classes - so a request carries the
class the model's own catalogue entry names (:func:`_auto_class_candidates`), and
the family's documented way in, ``model.chat``, is what ``"driver": "chat"``
selects (:func:`_describe_with_chat`).
"""

from __future__ import annotations

import inspect
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile

PROGRESS_PREFIX = "AIVS-PROGRESS|"

#: One finished description, written as ``AIVS-SEGMENT|<json>`` the moment it
#: exists.  The engine turns these into incremental saves, so closing a long run
#: never throws away the chunks that were already described.
SEGMENT_PREFIX = "AIVS-SEGMENT|"

#: Never sample more than this many frames from one chunk, whatever the request
#: says: a mis-set frame rate must not turn into a memory error.
MAX_FRAMES_PER_CHUNK = 120

#: How many vision patches one prompt may hold.  The frames of a window go to
#: the model in **one** prompt and its vision encoder attends over every patch of
#: every frame at once, so what fits is a memory question, not a matter of taste.
#: Measured on a 12 GB card with a 4B model resident: one frame of a 1024-pixel
#: video (2304 patches) peaks at 9.2 GB, four frames at 11.3 GB, and six at
#: 13.6 GB - past the card, where the driver silently spills to system memory and
#: a twelve-token answer stops taking five seconds and starts taking three
#: minutes.  A window with more frames than this is thinned to fit and keeps its
#: spread over the window.
#:
#: The number is **three** frames, not the four it was, because a window's prompt
#: carries more text than a chunk's did: the instruction asks for what has
#: changed, and the previous description is quoted into it.  Measured against the
#: same three-minute lecture at half a frame per second, four frames plus that
#: text took **71 seconds a window** (the prompt no longer fitted the card and the
#: driver paged), while three frames take the ten seconds a window the vision
#: encoder actually costs.  A prompt that spills is not a slower answer, it is a
#: different order of magnitude, so the budget errs on the side of fitting.
MAX_PROMPT_PATCHES = 6912

#: How many tokens one description may take when the run does not say.  Every
#: driver asks for its own budget rather than the checkpoint's: a description is
#: one or two sentences, and on a CPU the difference between a 160-token answer
#: and Moondream 2's own 768-token default is minutes per window.
DEFAULT_MAX_NEW_TOKENS = 160


def _clock(seconds: float) -> str:
    """``H:MM:SS`` / ``M:SS`` for a progress message (from ``pipeline``'s idea)."""
    total = int(max(0.0, float(seconds or 0.0)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _emit(percent: float, message: str) -> None:
    try:
        sys.stdout.write(f"{PROGRESS_PREFIX}{percent:.0f}%|{message}\n")
        sys.stdout.flush()
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _emit: {exc}\n")


def _emit_segment(segment: dict) -> None:
    try:
        sys.stdout.write(f"{SEGMENT_PREFIX}{json.dumps(segment, ensure_ascii=False)}\n")
        sys.stdout.flush()
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _emit_segment: {exc}\n")


def _load_request(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _ffmpeg_exe(request: dict) -> str:
    """The FFmpeg to use: the one the application manages, else the PATH."""
    configured = str(request.get("ffmpeg") or "").strip()
    if configured and os.path.isfile(configured):
        return configured
    return shutil.which("ffmpeg") or "ffmpeg"


# ---------------------------------------------------------------------------
# The weights cache on a Windows machine without Developer Mode
# ---------------------------------------------------------------------------
def _hugging_face_cache_dir() -> str:
    """Where ``huggingface_hub`` will keep this model's files.

    The same answer as :func:`ai_voice_studio.describe.weights.cache_dir` - this
    script is standalone and an environment interpreter cannot import the
    application, so the few lines are repeated here rather than shared.
    """
    for name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = str(os.environ.get(name) or "").strip()
        if value:
            return os.path.abspath(os.path.expanduser(value))
    home = str(os.environ.get("HF_HOME") or "").strip()
    if home:
        return os.path.join(os.path.abspath(os.path.expanduser(home)), "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                        "hub")


def _symlinks_available(folder: str) -> bool:
    """True when a symlink can actually be created in ``folder``."""
    try:
        os.makedirs(folder, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder, prefix="aivs_symlink_") as tmp:
            source = os.path.join(tmp, "source")
            with open(source, "w", encoding="utf-8") as fh:
                fh.write("x")
            try:
                os.symlink(source, os.path.join(tmp, "target"))
            except (OSError, NotImplementedError):
                return False
            return True
    except OSError:
        return False


def prepare_weights_cache() -> bool:
    """Stop Hugging Face storing weights as symlinks when they cannot work.

    The cache keeps one copy of a file in ``blobs`` and points each snapshot at
    it with a **symlink**, which on Windows needs Developer Mode or an elevated
    process.  Without either, the library's own advice is a wall of text and the
    fetch fails part-way - measured on this machine, a describing model that had
    never been downloaded produced no description at all and the run's error was
    that advice rather than a cause.  The documented fallback is to copy the file
    into the snapshot instead, which costs disk in the cache and always works.

    Both settings are read when ``huggingface_hub`` is imported, so this has to
    run before the first import of it - which is why it is called at the top of
    :func:`run`, before any model work.  Returns True when symlinks were turned
    off by this call; False when they work here or somebody already decided.
    A machine that *can* create them keeps them: the copy costs a second copy
    of every checkpoint, which for a 37 GB model is not a small thing.
    """
    if os.environ.get("HF_HUB_DISABLE_SYMLINKS"):
        return False
    if _symlinks_available(_hugging_face_cache_dir()):
        return False
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    # The warning says the same thing this function just handled, and the run
    # would otherwise report it instead of what it was asked to do.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return True


def _extract_frames(ffmpeg: str, video: str, start: float, end: float,
                    fps: float, out_dir: str, cap: int) -> list:
    """Extract frames from ``[start, end)`` and return their file paths.

    One FFmpeg invocation per chunk using its ``fps`` filter, which is far
    cheaper than seeking once per frame.  A failure returns an empty list: the
    chunk is then described from whatever the model can say about no frames,
    which is worse but never fatal.
    """
    os.makedirs(out_dir, exist_ok=True)
    rate = float(fps) if fps and float(fps) > 0 else 0.5
    pattern = os.path.join(out_dir, "frame_%04d.jpg")
    args = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, float(start)):.3f}",
        "-to", f"{max(0.0, float(end)):.3f}",
        "-i", video,
        "-vf", f"fps={rate:g},scale='min(1024,iw)':-2",
        "-frames:v", str(int(cap)),
        "-q:v", "3",
        pattern,
    ]
    try:
        subprocess.run(
            args, capture_output=True, text=True, timeout=600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except Exception:
        return []
    names = sorted(n for n in os.listdir(out_dir) if n.endswith(".jpg"))
    return [os.path.join(out_dir, name) for name in names[:cap]]


#: Frames are compared as tiny grey pictures rather than pixel for pixel.  The
#: only question a comparison has to answer is "does the picture move at all",
#: and a small summary answers it for almost nothing while ignoring the JPEG
#: and H.264 noise that is in every decoded frame.
SIGNATURE_SIZE = (64, 36)

#: How much the whole picture must move (grey levels, averaged over the frame)
#: for a chunk to be worth describing again.  Measured, not guessed, on a real
#: three-minute lecture and on synthesised stills, in these units:
#:
#: | What the chunk shows | Mean change | Verdict |
#: | --- | --- | --- |
#: | a picture genuinely held | 0.00 - 0.07 | nothing to say |
#: | a speaker shifting, no slide change | 1.15 - 1.47 | still the same scene |
#: | a slide advanced or a cut | 3.12 - 7.53 | describe it |
#:
#: So the two bands are those of "the scene is the same and someone moved" and
#: "the picture is a different one", and the line between them is here.  What a
#: chunk repeated costs is the description and, in extended audio description, a
#: pause and the words to fill it; what a chunk wrongly skipped costs is the only
#: account the listener was going to get of something that did change.  A caller
#: may set its own ``min_picture_change`` in the request; 0 describes every chunk.
MIN_PICTURE_CHANGE = 2.0


def frame_signature(path: str):
    """A tiny grey summary of one frame, or ``None`` when it cannot be read."""
    try:
        from PIL import Image

        with Image.open(path) as picture:
            return picture.convert("L").resize(SIGNATURE_SIZE).tobytes()
    except Exception:
        return None


def picture_change(first, second) -> float:
    """How much the picture moves between two frame signatures, in grey levels.

    The mean absolute difference over the frame: 0.0 is a picture that is held
    exactly, a couple of grey levels is a speaker shifting in their seat, and a
    new slide or a cut is in the tens.  Averaging over the whole frame is what
    makes the number mean "how much of the *scene* changed" rather than "did any
    pixel move", which is the question a listener cares about.

    A signature that cannot be read answers 255.0 - "everything changed" -
    because an unreadable frame is never evidence that nothing happened.
    """
    if first is None or second is None or len(first) != len(second):
        return 255.0
    return sum(abs(a - b) for a, b in zip(first, second, strict=False)) / len(first)


def frame_signatures(paths: list) -> list:
    """The signatures of a chunk's frames, in the order they were sampled."""
    return [frame_signature(path) for path in paths]


def picture_changed(signatures: list, reference=None,
                    threshold: float = MIN_PICTURE_CHANGE) -> bool:
    """Whether a chunk's picture has changed enough to be worth describing.

    ``signatures`` are the chunk's own frames in order and ``reference`` is the
    last frame of the last chunk that **was described** - the picture the
    listener has already been told about - or ``None`` when nothing has been
    described yet.  The largest change of any neighbouring pair decides: the
    pairs inside the chunk, and the pair between ``reference`` and the chunk's
    first frame.  So a chunk that cuts to a new still picture is described even
    though its own picture never moves, and a still picture held over a chunk
    boundary is recognised as the picture the listener has already heard about
    rather than as a new scene.

    A chunk is described whenever it cannot be judged at all: a single frame with
    no reference, no frames at all, a frame that cannot be read.  The one thing
    this must never do is throw a description away on a guess.
    """
    sequence = list(signatures)
    if reference is not None:
        sequence = [reference, *sequence]
    changes = [picture_change(first, second)
               for first, second in itertools.pairwise(sequence)]
    if not changes:
        return True
    return max(changes) >= max(0.0, float(threshold))


def _min_picture_change(request: dict) -> float:
    """How much a chunk's picture must change for it to be described."""
    value = request.get("min_picture_change")
    if value is None:
        return MIN_PICTURE_CHANGE
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return MIN_PICTURE_CHANGE


#: The ``transformers`` auto classes a describing checkpoint is loaded through,
#: most specific first.  These are the *fallback*, for the checkpoints that do
#: not name a class of their own; one that ships modelling code registers the
#: implementation it provides in its ``config.json`` and the name it registers
#: under is part of the checkpoint, which is why the catalogue answers with it
#: rather than this list being guessed from.
_DEFAULT_AUTO_CLASSES = ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                         "AutoModelForCausalLM")

#: What ``transformers`` says when the class it was asked for does not recognise
#: the checkpoint's configuration.  It is a question about the checkpoint, not a
#: failure to load it: another class can still work, which is exactly how
#: MiniCPM-o 2.6 answers ``AutoModelForImageTextToText`` (the error a user saw).
#: Only these refusals move on to the next class - anything else (a corrupt
#: download, an out-of-memory load) would fail the same way again, and is raised
#: where it happened instead of being retried three times.
_UNKNOWN_ARCHITECTURE = ("unrecognized configuration class", "unrecognized model",
                         "model type should be one of")


def _auto_class_candidates(request: dict) -> list:
    """The auto classes to try, the checkpoint's own choice first."""
    names = []
    configured = str(request.get("auto_class") or "").strip()
    if configured:
        names.append(configured)
    names.extend(_DEFAULT_AUTO_CLASSES)
    return list(dict.fromkeys(names))


def _unknown_architecture(exc: BaseException) -> bool:
    """True when ``exc`` says the auto class did not recognise the checkpoint."""
    text = str(exc).lower()
    return any(marker in text for marker in _UNKNOWN_ARCHITECTURE)


def _load_model(request: dict):
    """Load the processor and the model; returns ``(processor, model, torch)``.

    The processor is ``None`` for a model asked one picture at a time, because
    such a checkpoint has none: Moondream 2's repository carries a tokenizer and
    its own modelling code, and the only reason the generic path needs a
    processor is the multimodal message it builds.  Asking ``AutoProcessor`` for
    one anyway would put a component this driver never touches between the run
    and the weights - and it would answer only through its tokenizer fallback,
    which is not something a description should depend on.

    The class the checkpoint is asked for is the one its own ``auto_map``
    registers, which the catalogue carries as ``request['auto_class']``, and
    MiniCPM-o 2.6 is why: it maps ``AutoModel`` to its own ``MiniCPMO`` and maps
    nothing to ``AutoModelForImageTextToText``, so asking for the task-specific
    class was refused with "Unrecognized configuration class <class
    '...MiniCPMOConfig'> for this kind of AutoModel: AutoModelForImageTextToText"
    *inside a correct environment*.  The environment was not the problem - one
    model per environment is what made the question wrong instead of the install -
    and answering the checkpoint's own name is the fix.  The task-specific
    classes are still tried after it, but only when the refusal is that one.
    """
    import torch

    repo = request["hf_model"]
    trust = bool(request.get("trust_remote_code"))
    if str(request.get("driver") or "") == QUERY_DRIVER:
        processor = None
    else:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(repo,
                                                 trust_remote_code=trust)
    device = str(request.get("device") or "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    import transformers

    load_kwargs = request.get("load_kwargs") or {}
    failures = []
    for name in _auto_class_candidates(request):
        model_cls = getattr(transformers, name, None)
        if model_cls is None:
            continue
        try:
            if device == "cuda":
                model = _load_on_gpu(model_cls, repo, trust, torch, request)
            else:
                model = _from_pretrained(
                    model_cls, repo, _load_dtype(request, torch.float32),
                    trust, None, load_kwargs=load_kwargs,
                )
                try:
                    model.to("cpu")
                except Exception as exc:
                    sys.stderr.write(f"Ignoring a failure in _load_model: {exc}\n")
        except Exception as exc:
            failures.append(exc)
            _free_cuda(torch)
            if not _unknown_architecture(exc):
                raise
            continue
        model.eval()
        return processor, model, torch
    if failures:
        raise failures[-1]
    raise RuntimeError(  # pragma: no cover - any transformers has one
        "This transformers version has no multimodal model class."
    )


#: Video memory held back from a full-precision load: the vision encoder's
#: activations, the KV cache and the tokens being generated all need room on the
#: card as well as the weights.  A checkpoint whose own size is larger than the
#: card minus this is asked for in 4 bits instead of being allowed to spill.
_GPU_HEADROOM_GIB = 2.0


def _gpu_total_gib(torch_module) -> float:
    """The card's total video memory in gigabytes (0 when it cannot be read)."""
    try:
        total = torch_module.cuda.get_device_properties(0).total_memory
        return float(total) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _quantized_config(torch_module):
    """A 4-bit load configuration; ``(None, why)`` when it cannot be built.

    4-bit is what lets a checkpoint larger than the card still run here: Gemma 3
    12B is about 24 GB of weights and a 12 GB card holds its 4-bit form (roughly
    a quarter of that) with room to generate.  The quantiser is
    ``bitsandbytes``, installed into a model's environment only on a machine
    with an NVIDIA card; an environment built before that was part of the GPU
    install simply does not have it, which is said rather than guessed at.
    """
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        return None, str(exc)
    try:
        import bitsandbytes  # noqa: F401 - imported to prove the backend loads
    except Exception as exc:
        return None, str(exc)
    try:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_module.bfloat16,
        )
    except Exception as exc:
        return None, str(exc)
    return config, ""


def _from_pretrained(model_cls, repo: str, dtype, trust: bool, device_map,
                     quantization_config=None, load_kwargs=None):
    """``from_pretrained`` with both spellings of the dtype argument.

    Older ``transformers`` releases spell it ``torch_dtype``; a call refused for
    an unknown keyword is retried the other way.  ``load_kwargs`` are the extra
    keywords a checkpoint's catalogue entry asks for - MiniCPM-o is loaded
    vision-only through them (see ``catalog.load_kwargs``).
    """
    extra = dict(load_kwargs or {})
    if quantization_config is not None:
        extra["quantization_config"] = quantization_config
    try:
        return model_cls.from_pretrained(
            repo, dtype=dtype, trust_remote_code=trust,
            device_map=device_map, **extra,
        )
    except TypeError:
        return model_cls.from_pretrained(
            repo, torch_dtype=dtype, trust_remote_code=trust,
            device_map=device_map, **extra,
        )


def _load_dtype(request: dict, default):
    """The dtype to load in: the checkpoint's own when its entry asks for it.

    Empty (every model but Moondream 2) keeps the run's own choice - float32 on
    a CPU, bfloat16 on a card - which is what the sizes in the catalogue are
    quoted in and what every other checkpoint is happy with.  ``"auto"`` is
    ``transformers``' spelling for "whatever this checkpoint is", taken from its
    ``config.json`` or from the weights themselves.  The field exists because
    the wrong dtype is not a slow run but a failed one for a checkpoint whose
    own forward pass casts to a dtype of its choosing: Moondream 2 casts every
    image crop to bfloat16 while the layers taking it are built in the model's
    own bfloat16, so a float32 load dies in the first linear layer.
    """
    configured = request.get("dtype")
    return configured if configured else default


def _free_cuda(torch_module) -> None:
    """Hand the freed blocks of a failed load back to the driver."""
    try:
        torch_module.cuda.empty_cache()
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _free_cuda: {exc}\n")


def _load_on_gpu(model_cls, repo: str, trust: bool, torch_module, request: dict):
    """Load on the card, never letting a model be spilled to the *disk*.

    ``device_map="auto"`` is what put part of a model on the disk on this
    machine, and a model on the disk is a set of *meta* tensors: the forward pass
    reads them, Windows dies with an access violation (``0xC0000005``), and the
    window reported whichever warning happened to sit above the crash.  The
    placement here is explicit - everything on the one card - with 4-bit as the
    way a checkpoint larger than the card still runs.  Only a model that can be
    run neither way is refused, and it is refused with its own reason instead of
    a crash.
    """
    card = _gpu_total_gib(torch_module)
    size = float(request.get("model_size_gb") or 0.0)
    budget = max(1.0, card - _GPU_HEADROOM_GIB)
    too_big = bool(size) and size > budget
    quantized, why = _quantized_config(torch_module)

    attempts = []
    if too_big and quantized is not None:
        attempts.append((quantized, (
            f"This model is about {size:.1f} GB of weights and this card holds "
            f"{card:.1f} GB, so it was loaded in 4-bit."
        )))
    else:
        attempts.append((None, ""))
        if quantized is not None:
            attempts.append((quantized, (
                "The model did not fit this card at full precision, so it was "
                "loaded in 4-bit."
            )))

    last_error = None
    load_kwargs = request.get("load_kwargs") or {}
    for quantization, note in attempts:
        try:
            model = _from_pretrained(
                model_cls, repo, _load_dtype(request, torch_module.bfloat16),
                trust, {"": 0}, quantization_config=quantization,
                load_kwargs=load_kwargs,
            )
        except Exception as exc:
            last_error = exc
            _free_cuda(torch_module)
            continue
        if note:
            _emit(2, note)
        return model

    lines = [f"Could not load {repo} on this graphics card."]
    if size and card:
        lines.append(
            f"Its weights are about {size:.1f} GB and the card holds "
            f"{card:.1f} GB."
        )
    if quantized is None:
        lines.append(
            "A model larger than the card is loaded in 4-bit, which needs the "
            "quantiser bitsandbytes in this model's environment"
            + (f" ({why})" if why else "")
            + ". Open Settings > Manage the video describer and press Repair "
            "model environment."
        )
    if last_error is not None:
        lines.append(f"The last error was: {last_error}")
    raise RuntimeError("  ".join(lines))


def _load_pictures(paths: list) -> list:
    """The sampled frames, as the RGB images a processor expects."""
    from PIL import Image

    return [Image.open(path).convert("RGB") for path in paths]


def patches_per_frame(path: str) -> int:
    """What one frame costs the vision encoder, in 16-pixel patches.

    Read from the frame itself, so a 4K video and a 480-pixel one are budgeted
    by what they actually send rather than by an assumption.
    """
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return 0
    return max(1, (int(width) // 16) * (int(height) // 16))


def thin_frames(paths: list, patches_each: int, budget: int) -> list:
    """The frames that fit ``budget`` patches, spread over the whole chunk.

    Taking the first *n* frames would describe the beginning of a chunk and
    ignore the rest, so the ones that are kept are picked at even intervals from
    the first frame to the last: a chunk is still described from end to end.

    ``patches_each`` is what one frame costs.  A frame whose size could not be
    read is not free, so without a measurement the budget is read as a plain
    frame count - which is what a retry after an out-of-memory failure passes.
    """
    paths = list(paths)
    if len(paths) < 2:
        return paths
    each = int(patches_each)
    allowed = max(1, int(budget) // each) if each > 0 else max(1, int(budget))
    if allowed >= len(paths):
        return paths
    if allowed <= 1:
        return [paths[0]]
    step = (len(paths) - 1) / (allowed - 1)
    kept = [paths[round(index * step)] for index in range(allowed)]
    return list(dict.fromkeys(kept))


def _out_of_memory(torch_module, exc: BaseException) -> bool:
    """True for the CUDA out-of-memory error, whatever it is spelled as."""
    kinds = []
    for holder in (torch_module, getattr(torch_module, "cuda", None)):
        kind = getattr(holder, "OutOfMemoryError", None)
        if isinstance(kind, type):
            kinds.append(kind)
    if kinds and isinstance(exc, tuple(kinds)):
        return True
    return "out of memory" in str(exc).lower()


def _messages(guideline: str, instruction: str, pictures: list,
              with_images: bool) -> list:
    """The chat messages one chunk is described with.

    ``with_images`` embeds the frames in the message itself.  Without it the
    message carries one image *placeholder* per frame instead, which is the form
    a processor wants when the frames are handed over beside the prompt.  The
    two forms exist because the model families in the catalog disagree about it
    (Gemma 3's template takes the second), and the picture, not the plumbing, is
    what the description is about.
    """
    content: list = []
    for picture in pictures:
        content.append({"type": "image", "image": picture} if with_images
                       else {"type": "image"})
    content.append({"type": "text", "text": instruction})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": guideline}],
        },
        {"role": "user", "content": content},
    ]


def _window_instruction(request: dict, previous: str = "") -> str:
    """The prompt for one window, told what the last description said.

    Two things go wrong when a piece of video is described with no memory of the
    one before it: the model repeats the scene it has already been told about
    ("a man sits in a chair") instead of what has just happened, and a change
    that is small in pixels but large in meaning - a line of on-screen text
    appearing, a slide advancing, a hand raised - is not reported at all, because
    nothing in the prompt asks for a change.  Measured on a real lecture, a
    two-sentence summary per two minutes said the man was sitting next to a
    whiteboard and never mentioned the code that had appeared on it.  So the
    previous description is handed back to the model and the change is asked for
    by name.
    """
    base = str(request.get("chunk_instruction")
               or "Describe this piece of video in one or two short sentences.")
    previous = str(previous or "").strip()
    if not previous:
        return base
    return (
        base
        + "\n\nThe previous description was: " + previous
        + "\nThese frames come straight after it. Say what is happening now and "
        "what has changed since that description - text, code or numbers that "
        "have appeared or changed, what the person does, anyone or anything new "
        "on screen. Do not repeat what was already described."
    )


#: ``"driver"`` value: the model is asked through its own ``chat`` method,
#: instead of the processor-plus-``generate`` path every other model takes.  It
#: is the MiniCPM-o/V family's only way in - see :func:`_describe_with_chat`.
CHAT_DRIVER = "chat"

#: ``"driver"`` value: the model is asked through its own single-image question
#: method, because it has neither a processor nor a chat template for the
#: generic path to use.  Moondream 2 is the model that needs it - see
#: :func:`_describe_with_query`.
QUERY_DRIVER = "query"


def _accepts_keyword(function, name: str) -> bool:
    """Whether ``function`` can be given a keyword called ``name``."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):  # pragma: no cover - a built-in or a wrapper
        return True
    if name in parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD
               for parameter in parameters.values())


def _describe_with_chat(processor, model, request: dict, images: list,
                        instruction: str) -> str:
    """Ask a checkpoint that is driven through its own ``chat`` method.

    The MiniCPM-o/V family reaches a description only this way.  Its processors
    ship no chat template that carries the standard multimodal message - the
    template is a plain concatenation of ``message['content']``, so a list of
    content parts is a ``TypeError`` rather than a prompt - and its own
    ``generate`` wants ``inputs_embeds``/``image_bound`` beside the tokenizer, so
    the generic path cannot drive it either.  What its own code documents is
    ``model.chat(msgs=...)`` with the frames *inside* the user turn - the shape
    the model card writes - and that is what this asks.

    The optional arguments are passed only when the method has them, because the
    family's versions differ while the call is otherwise the same: ``processor``
    (which saves loading it a second time), ``tokenizer`` (which ``chat`` decodes
    with) and ``max_new_tokens``.  Determinism is asked for under both spellings
    the family uses - ``sampling=False`` (MiniCPM-o 2.6 and MiniCPM-V 4) and
    ``do_sample=False`` (MiniCPM-o 4.5, whose other name is ignored by the
    earlier ones) - together with ``num_beams=1``, which turns the 3-beam
    beam search those earlier versions use when sampling is off into the greedy
    decode a description wants: three beams over the same frames would triple
    what a run costs for no gain on one short sentence, and a sampled answer
    would differ between runs of the same video.  The guideline prompt goes into
    the user turn rather than a system message, which every version of the
    method accepts.
    """
    pictures = _load_pictures(images)
    guideline = str(request.get("guideline_prompt") or "").strip()
    text = "\n\n".join(part for part in (guideline, instruction) if part)
    messages = [{"role": "user", "content": [*pictures, text]}]
    call = model.chat
    kwargs = {"msgs": messages}
    if _accepts_keyword(call, "tokenizer"):
        kwargs["tokenizer"] = getattr(processor, "tokenizer", None)
    if _accepts_keyword(call, "processor"):
        kwargs["processor"] = processor
    if _accepts_keyword(call, "max_new_tokens"):
        kwargs["max_new_tokens"] = int(request.get("max_new_tokens")
                                       or DEFAULT_MAX_NEW_TOKENS)
    # Both spellings of "do not sample": 2.6 and V 4 take ``sampling=``, 4.5
    # takes ``do_sample=``, and each ignores the other's name.
    if _accepts_keyword(call, "sampling"):
        kwargs["sampling"] = False
    if _accepts_keyword(call, "do_sample"):
        kwargs["do_sample"] = False
    if _accepts_keyword(call, "num_beams"):
        kwargs["num_beams"] = 1
    answer = call(**kwargs)
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else ""
    return str(answer or "").strip()


def _skill_text(result, key: str) -> str:
    """The sentence out of one of Moondream's own answers.

    Its skills answer with a dictionary - ``{"answer": ...}`` from ``query``,
    ``{"caption": ...}`` from ``caption`` - and a caller that wants the sentence
    should not have to know which skill produced it.  A bare string (what a
    revision answering with text alone gives) passes through, and so does the
    first item of a list, which is the shape its streaming answers decode to.
    """
    if isinstance(result, dict):
        answer = result.get(key)
        if answer is None:
            answer = next((value for value in result.values()
                           if isinstance(value, str)), "")
    elif isinstance(result, (list, tuple)):
        answer = result[0] if result else ""
    else:
        answer = result
    return str(answer or "").strip()


def _describe_with_query(model, request: dict, images: list,
                         instruction: str) -> str:
    """Ask a checkpoint whose own API answers one question about one picture.

    Moondream 2 is the model this exists for, and it is the sharpest case in the
    catalogue of a checkpoint ``transformers`` reaches only through its own
    code: the repository carries a tokenizer and its own modelling, and no
    processor and no chat template at all, so a description cannot be reached
    through a multimodal message however that message is spelled - there is no
    template to apply, and its ``generate`` wants ``inputs_embeds`` and an
    encoder cache beside the tokenizer rather than the ``input_ids`` a
    processor's output carries.  What its own code publishes instead is a set of
    skills, and a description *is* a question, so this asks
    ``model.query(image, question)`` and reads the ``"answer"`` it returns.

    Two properties of the checkpoint decide the rest of the call:

    * **One picture.**  ``query`` accepts a single image and its cache holds
      one, so a window is described from its **last** frame: the newest picture
      in the piece of video being described, and the same frame the run treats
      as "the picture the listener has already been told about" when it decides
      a window has not changed.  Asking once per frame instead would spend a
      forward pass on each and leave the run to merge several sentences itself,
      which is not what a 1.9B captioner is for.
    * **Greedy, and short.**  Its own defaults *sample* (``temperature=0.5``,
      ``top_p=0.3``), so the same video would be described differently on two
      runs - the thing these drivers exist to prevent - and its own cap is 768
      tokens, which on a CPU is minutes a window.  The run asks for
      ``temperature=0`` (its code then takes the ``argmax`` path) and for the
      run's own token budget.

    The documented skill is asked first and the two older spellings of it are
    fallbacks: the model card tells production users to pin a revision, and the
    revisions it names answer through ``answer_question`` and ``caption`` rather
    than ``query``.  A ``caption`` cannot follow the guideline prompt - it is a
    caption, not an answer to a question - so it is the last resort, reached
    only by a revision with no question skill at all.
    """
    pictures = _load_pictures(images)
    if not pictures:
        return ""
    picture = pictures[-1]
    guideline = str(request.get("guideline_prompt") or "").strip()
    question = "\n\n".join(part for part in (guideline, instruction) if part)
    settings = {
        "temperature": 0,
        "max_tokens": int(request.get("max_new_tokens")
                          or DEFAULT_MAX_NEW_TOKENS),
    }
    ask = getattr(model, "query", None)
    if callable(ask):
        kwargs = {}
        if _accepts_keyword(ask, "settings"):
            kwargs["settings"] = settings
        return _skill_text(ask(picture, question, **kwargs), "answer")
    ask = getattr(model, "answer_question", None)
    if callable(ask):
        return _skill_text(ask(picture, question), "answer")
    ask = getattr(model, "caption", None)
    if callable(ask):
        kwargs = {}
        if _accepts_keyword(ask, "length"):
            kwargs["length"] = "long"
        if _accepts_keyword(ask, "settings"):
            kwargs["settings"] = settings
        return _skill_text(ask(picture, **kwargs), "caption")
    raise RuntimeError(
        "This checkpoint answers neither query() nor caption(), so a "
        "description cannot be asked of it."
    )


def _describe_once(processor, model, torch, request: dict, images: list,
                   instruction: str = "") -> str:
    """One attempt: ask the model about exactly these frames."""
    instruction = instruction or _window_instruction(request)
    driver = str(request.get("driver") or "")
    if driver == CHAT_DRIVER:
        return _describe_with_chat(processor, model, request, images, instruction)
    if driver == QUERY_DRIVER:
        return _describe_with_query(model, request, images, instruction)
    guideline = str(request.get("guideline_prompt") or "")
    pictures = _load_pictures(images)
    try:
        inputs = processor.apply_chat_template(
            _messages(guideline, instruction, pictures, True),
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
    except (TypeError, ValueError, KeyError):
        # This processor wants the frames beside the prompt, not inside it: the
        # prompt keeps one placeholder per frame and receives the frames itself.
        prompt = processor.apply_chat_template(
            _messages(guideline, instruction, pictures, False),
            add_generation_prompt=True, tokenize=False,
        )
        if pictures:
            inputs = processor(text=[prompt], images=pictures,
                               return_tensors="pt")
        else:
            inputs = processor(text=[prompt], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: (value.to(device) if hasattr(value, "to") else value)
              for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=int(request.get("max_new_tokens")
                                         or DEFAULT_MAX_NEW_TOKENS),
            do_sample=False,
        )
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)
    return (text[0] if text else "").strip()


def _describe_chunk(processor, model, torch, request: dict, images: list,
                    on_note=None, instruction: str = "") -> str:
    """Ask the model for one short description of ``images``.

    A prompt whose frames do not fit the graphics card is retried with half of
    them: the driver would otherwise spill to system memory, where a description
    that should take seconds takes minutes.  A smaller prompt is therefore both
    faster and more reliable.  Only a prompt that no longer fits with a single
    frame raises, and the caller reports that one chunk rather than the run.
    """
    frames = list(images)
    while True:
        try:
            return _describe_once(processor, model, torch, request, frames,
                                  instruction)
        except Exception as exc:
            if len(frames) <= 1 or not _out_of_memory(torch, exc):
                raise
            fewer = thin_frames(frames, 0, max(1, len(frames) // 2))
            if len(fewer) >= len(frames):
                raise
            frames = fewer
            if on_note is not None:
                on_note(f"Too much for the graphics card in one go: retrying "
                        f"with {len(frames)} frame(s).")
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                sys.stderr.write(f"Ignoring a failure in _describe_chunk: {exc}\n")


#: The sentence the prompt asks for when a chunk has no visual change; treated as
#: "nothing to say" rather than as a description.
NO_CHANGE = "no significant visual change"

#: ``"mode"`` value that fetches a variant's checkpoint instead of describing
#: anything.  A variant's weights normally arrive during the first run, which is
#: why Settings > Manage the video describer can also fetch one on demand - and
#: that fetch has to happen inside this environment, because it is the one that
#: carries ``huggingface_hub``.
DOWNLOAD_MODE = "download"


#: What a gated/unauthorised download says; matched case-insensitively.
_GATED_MARKERS = ("gated", "must be authenticated", "cannot access",
                  "unauthorized", "you must have access", "please log in",
                  "access to model", "401", "403")


def _gated_hint(exc, repo: str = "") -> str:
    """The two steps a gated repository needs, or ``""`` for any other failure.

    This worker is a standalone script in the model's own environment (it
    imports no part of the application), so the wording is repeated here rather
    than imported: the engine's ``ai_voice_studio.hf_access`` says the same
    thing, and this is what the user reads when the failure happens *inside* a
    download rather than before it.
    """
    text = f"{exc}".lower()
    if not any(marker in text for marker in _GATED_MARKERS):
        return ""
    page = f"https://huggingface.co/{repo}" if repo else "its page on Hugging Face"
    return (
        f"\n\n{repo or 'This model'} downloads from a gated Hugging Face "
        "repository, which needs your own account:\n"
        f"1. Open {page} and accept the model's licence (its terms of use).\n"
        "2. Create a read token at https://huggingface.co/settings/tokens and "
        "paste it into Settings > Hugging Face account in the application.\n"
        "Nothing else is fetched until that access is in place."
    )


def _download_weights(request: dict) -> dict:
    """Fetch one variant's repository, and say where it landed.

    ``transformers`` resolves its own weights out of this same cache, so fetching
    them here means the first description of this variant starts describing
    instead of downloading - which, for a 12 GB checkpoint, is the difference
    between a run and an afternoon.
    """
    repo = str(request.get("hf_model") or "").strip()
    if not repo:
        return {"ok": False, "error": "This variant has no repository to "
                                       "download from."}
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        return {"ok": False, "error": (
            f"huggingface_hub is needed to download {repo} and is not "
            f"installed in this environment ({exc}). Install the model again "
            "from Settings > Manage the video describer.")}
    _emit(2, f"Downloading the model files of {repo}...")
    try:
        path = snapshot_download(repo_id=repo, resume_download=True)
    except TypeError:
        # huggingface_hub dropped resume_download in newer releases; the call
        # without it is the same fetch.
        try:
            path = snapshot_download(repo_id=repo)
        except Exception as exc:
            return {"ok": False, "error": f"Could not download {repo}: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": (
            f"Could not download {repo}: {exc}{_gated_hint(exc, repo)}")}
    _emit(98, f"Downloaded {repo}.")
    return {"ok": True, "repo": repo, "path": str(path),
            "note": f"{repo} is downloaded and ready to describe with."}


def run(request: dict, out_path: str) -> dict:
    # Before anything imports transformers or huggingface_hub: a machine without
    # Developer Mode cannot create the cache's symlinks, and the fallback has to
    # be switched on while the library still reads its environment.
    if prepare_weights_cache():
        _emit(1, "Storing the downloaded model files as copies: this Windows "
                  "account cannot create symlinks without Developer Mode.")
    if str(request.get("mode") or "") == DOWNLOAD_MODE:
        return _download_weights(request)
    video = str(request.get("video") or "")
    if not os.path.isfile(video):
        return {"ok": False, "error": f"Video file not found: {video}"}
    chunks = list(request.get("chunks") or [])
    if not chunks:
        return {"ok": True, "segments": [], "note": "Nothing to describe."}

    _emit(1, "Loading the describer...")
    try:
        processor, model, torch = _load_model(request)
    except Exception as exc:
        # The exact failure the user reported: a gated checkpoint answers 401
        # here, while the model is being loaded, and the message said only
        # "Please log in".
        return {"ok": False, "error": (
            f"Could not load the model: {exc}"
            f"{_gated_hint(exc, str(request.get('hf_model') or ''))}")}

    ffmpeg = _ffmpeg_exe(request)
    fps = float(request.get("fps") or 0.5)
    budget = int(request.get("prompt_patch_budget") or MAX_PROMPT_PATCHES)
    min_change = _min_picture_change(request)
    tmp_root = tempfile.mkdtemp(prefix="aivs_ad_frames_")
    segments = []
    #: The last frame of the last window that was described - the picture the
    #: listener has already been told about.  A window that looks the same as that
    #: picture is a picture the listener has already been given.
    described_reference = None
    #: What the last description said, handed to the model with the next window so
    #: it reports what has changed instead of describing the same scene again.
    previous_text = ""
    try:
        # A chunk bounds memory, progress and resume; one *window* is one
        # description.  The engine splits every chunk into its windows, and a
        # request that carries none (a recipe written before windows existed) is
        # described chunk by chunk, exactly as it used to be.
        total = sum(len(chunk.get("windows") or [chunk]) for chunk in chunks)
        described = 0
        for chunk in chunks:
            for window in (list(chunk.get("windows") or []) or [chunk]):
                described += 1
                done = described / total
                base = 5.0 + 90.0 * ((described - 1) / total)
                start = float(window.get("start") or 0.0)
                end = float(window.get("end") or 0.0)
                label = (f"{described} of {total} "
                         f"({_clock(start)} - {_clock(end)})")
                _emit(base, f"Sampling frames for piece {label}...")
                frames = _extract_frames(
                    ffmpeg, video, start, end, fps,
                    os.path.join(tmp_root, f"piece_{described:05d}"),
                    MAX_FRAMES_PER_CHUNK,
                )
                # A window whose picture has barely changed repeats the last
                # description the listener heard, and asking the model about it
                # costs that description - and, in extended audio description, a
                # pause in the film and the spoken words to fill it.  Judged on
                # every frame the window was sampled from, not only the ones that
                # fit one prompt, so a brief change inside it is not missed.
                signatures = frame_signatures(frames)
                # Nothing has been described yet, so there is nothing for the
                # opening window to repeat: it is always described, because the
                # opening is what tells the listener what they are looking at.
                repeat = described > 1 and not picture_changed(
                    signatures, described_reference, min_change)
                if repeat:
                    _emit(base + 10 * (1.0 / total),
                          f"Piece {label}: the picture has not changed since the "
                          "last description, so it is skipped.")
                    _emit(5.0 + 90.0 * done,
                          f"Skipped a piece: the picture had not changed "
                          f"({described} of {total}).")
                    continue
                described_reference = (signatures[-1] if signatures
                                       else described_reference)
                # A model that answers one picture at a time is not budgeted in
                # patches at all - its prompt holds a single image - and the one
                # it needs is the *newest*: the picture the listener has not been
                # told about yet, and the frame the run has just compared against
                # the last description.  ``thin_frames`` spreads its choice over
                # the whole window, which is right for a prompt holding several
                # frames and would otherwise hand this driver the window's first.
                if str(request.get("driver") or "") == QUERY_DRIVER:
                    frames = frames[-1:]
                # A window at a fine frame rate can sample more frames than one
                # prompt can hold on any consumer card; the ones kept are spread
                # over the whole window, so it is still described end to end.
                elif len(frames) > 1:
                    kept = thin_frames(frames, patches_per_frame(frames[0]),
                                       budget)
                    if len(kept) < len(frames):
                        _emit(base, f"Piece {label}: {len(frames)} frames is "
                                    "more than fits the graphics card at "
                                    f"once; describing {len(kept)} spread "
                                    "across it.")
                        frames = kept
                _emit(base + 20 * (1.0 / total),
                      f"Describing piece {label}...")
                try:
                    text = _describe_chunk(
                        processor, model, torch, request, frames,
                        on_note=lambda message, _b=base: _emit(
                            _b + 20 * (1.0 / total), message),
                        instruction=_window_instruction(request, previous_text),
                    )
                except Exception as exc:
                    text = ""
                    _emit(base + 40 * (1.0 / total),
                          f"Piece {label} failed: {exc}")
                if text and text.strip().lower().rstrip(".") != NO_CHANGE:
                    previous_text = text.strip()
                    segment = {
                        "start": window.get("start"),
                        "end": window.get("end"),
                        "text": text.strip(),
                        "chunk": window.get("chunk", chunk.get("index")),
                        "window": window.get("index", described - 1),
                    }
                    segments.append(segment)
                    _emit_segment(segment)
                _emit(5.0 + 90.0 * done,
                      f"Described {described} of {total} piece(s).")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"ok": True, "segments": segments}, fh, ensure_ascii=False)
    return {"ok": True, "segments": segments}


def main(argv: list) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: worker.py request.json\n")
        return 2
    request = _load_request(argv[1])
    out_path = str(request.get("out") or "")
    result = run(request, out_path)
    if not result.get("ok"):
        sys.stderr.write(str(result.get("error") or "Audio description failed."))
        # A response file with ok=false is how the engine reads the real cause;
        # write one so a failure is never just "no result".
        if out_path:
            try:
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump({"ok": False, "error": result.get("error")}, fh)
            except OSError as exc:
                sys.stderr.write(f"Ignoring a failure in main: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
