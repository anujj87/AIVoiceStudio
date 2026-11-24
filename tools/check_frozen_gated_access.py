"""Check that a frozen build really carries the gated-model access work.

A frozen executable is what ships, and the source in the repository is not
evidence about what is inside it: this reads the ``PYZ`` out of the built
``AI-Voice-Studio.exe`` and looks for the names and strings the feature consists
of - the gated-repository registry, the smallest-file access check, the Settings
panel's model picker and its two buttons, and the Voice Lab's translation of a
gated download failure.  It exists because a build that silently packaged an
older source tree looks exactly like a build that did not.

The shipped pieces that are *data* files rather than modules are read too: the
Voice Lab worker script (copied into the app so a bare interpreter in the
engine's own environment can run it) and the documents the Help menu opens.

Usage (from the repository root, with the build venv)::

    .venv\\Scripts\\python.exe tools\\check_frozen_gated_access.py
    .venv\\Scripts\\python.exe tools\\check_frozen_gated_access.py path\\to\\AI-Voice-Studio.exe
"""

from __future__ import annotations

import marshal
import os
import sys
import tempfile

#: module -> names and strings that must appear in its code objects.
WANTED = {
    "ai_voice_studio.hf_access": (
        "GATED_REPOS", "gated_repos", "gated_entry", "gated_label",
        "check_access", "_repo_info", "FILE_URL", "MODEL_API",
        "ACCESS_OK", "ACCESS_NO_TOKEN", "ACCESS_REFUSED", "ACCESS_OPEN",
        "smallest", "accept the terms", "logged in locally",
        "One token serves every gated model",
    ),
    "ai_voice_studio.voicelab.engines": ("gated_repo", "gated_repo"),
    "ai_voice_studio.voicelab": ("_error_text", "gated_repo", "access_message"),
    "ai_voice_studio.gui.settings_dialog": (
        "repo_choice", "open_btn", "access_btn", "access_status",
        "_selected_repo", "_on_open_model_page", "_on_check_access",
        "_access_checked", "_default_gated_index", "_restart_engine_workers",
        "Gated models (accept the licence first)",
        "Open the model page", "Check &access",
        "There is no per-model token",
        "one read token serves every gated model",
    ),
}

#: The Voice Lab worker ships as a *data* file: it is run by a bare interpreter
#: inside the engine's own environment and imports no part of the application.
WORKER_SCRIPT = os.path.join("_internal", "ai_voice_studio", "voicelab", "worker.py")
WORKER_NEEDLES = (
    "_POCKET_NO_CLONING_HELP", "_POCKET_NO_CLONING_NOTE",
    "_POCKET_VOICE_CLONING_REPO", "kyutai/pocket-tts", "has_voice_cloning",
    "accept the terms",
)

#: The documents the Help menu opens, and what they have to say about the gate.
SHIPPED_DOCS = {
    "UserGuide.html": ("kyutai/pocket-tts", "Check access"),
    "README.html": ("kyutai/pocket-tts", "Check access"),
    "WHATS-NEW.html": ("kyutai/pocket-tts", "Check access"),
}

_DEFAULT = os.path.join("dist", "AIVS-64", "AI-Voice-Studio", "AI-Voice-Studio.exe")


def _walk(code):
    """Every nested code object (a function's body is a constant of its own)."""
    yield code
    for const in code.co_consts:
        if hasattr(const, "co_consts"):
            yield from _walk(const)


def _vocabulary(code) -> str:
    """Everything the code object mentions: names, attributes and constants.

    Tuples are walked as well as code objects: a literal tuple of strings (the
    marker lists a gated failure is recognised by) is *one* constant, and a
    check that only looks at strings would report it missing.
    """
    words: list[str] = []
    stack = [code]
    while stack:
        node = stack.pop()
        words.extend(node.co_names)
        for const in node.co_consts:
            if hasattr(const, "co_consts"):
                stack.append(const)
            elif isinstance(const, str):
                words.append(const)
            elif isinstance(const, (tuple, frozenset, set, list)):
                for item in const:
                    if hasattr(item, "co_consts"):
                        stack.append(item)
                    elif isinstance(item, str):
                        words.append(item)
    return "\n".join(words)


def _module_report(pyz, failures: list[str]) -> None:
    for module, needles in WANTED.items():
        if module not in pyz.toc:
            failures.append(module)
            print(f"  MISSING MODULE {module}")
            continue
        code = pyz.extract(module)
        if isinstance(code, (bytes, bytearray)):
            code = marshal.loads(code)
        vocabulary = _vocabulary(code)
        missing = [name for name in needles if name not in vocabulary]
        print(f"  {module}: {'ok' if not missing else 'MISSING ' + str(missing)}")
        if missing:
            failures.append(module)


def _text_report(path: str, needles, label: str, failures: list[str]) -> None:
    if not os.path.isfile(path):
        failures.append(path)
        print(f"  MISSING {label}: {path}")
        return
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    missing = [name for name in needles if name not in text]
    print(f"  {label}: {'ok' if not missing else 'MISSING ' + str(missing)}")
    if missing:
        failures.append(path)


def main(argv: list[str]) -> int:
    from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader

    exe = argv[1] if len(argv) > 1 else _DEFAULT
    if not os.path.isfile(exe):
        print(f"No frozen executable at {exe} - build it first "
              f"(packaging\\build.ps1 -Arch x64).")
        return 2
    root = os.path.dirname(os.path.abspath(exe))
    failures: list[str] = []

    archive = CArchiveReader(exe)
    pyz_name = next((name for name in archive.toc
                     if str(name).lower().endswith(".pyz")), "")
    if not pyz_name:
        print(f"{exe} has no PYZ archive; is it a one-folder build?")
        return 2
    with tempfile.NamedTemporaryFile(suffix=".pyz", delete=False) as handle:
        handle.write(archive.extract(pyz_name))
        pyz_path = handle.name
    try:
        pyz = ZlibArchiveReader(pyz_path)
        print(f"{os.path.basename(exe)}: PYZ carries {len(pyz.toc)} modules")
        _module_report(pyz, failures)
    finally:
        os.unlink(pyz_path)

    print("\nshipped files:")
    _text_report(os.path.join(root, WORKER_SCRIPT), WORKER_NEEDLES,
                 "the Voice Lab worker script", failures)
    docs = os.path.join(root, "_internal", "docs")
    for name, needles in SHIPPED_DOCS.items():
        _text_report(os.path.join(docs, name), needles,
                     f"docs/{name}", failures)

    print()
    if failures:
        print("RESULT: this build does not carry the gated-access work:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("RESULT: the build carries the gated-access work")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
