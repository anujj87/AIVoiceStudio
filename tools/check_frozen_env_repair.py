"""Check that a frozen build really carries the environment-repair code.

A frozen executable is what ships, and the source in the repository is not
evidence about what is inside it: this reads the ``PYZ`` out of the built
``AI-Voice-Studio.exe`` and looks for the names and strings the fix consists of
- the ``pyvenv.cfg`` readers, the staging of the bundled runtime, the in-place
``venv --upgrade`` repair and the ``usable_python_exe`` every worker launch now
goes through.  It exists because a build that silently packaged an older source
tree looks exactly like a build that did not.

Usage (from the repository root, with the build venv)::

    .venv\\Scripts\\python.exe tools\\check_frozen_env_repair.py
    .venv\\Scripts\\python.exe tools\\check_frozen_env_repair.py path\\to\\AI-Voice-Studio.exe
"""

from __future__ import annotations

import marshal
import os
import sys
import tempfile

#: module -> names that must appear in its code objects (constants and
#: co_names), i.e. what the fix is made of.
WANTED = {
    "ai_voice_studio.python_runtime": ("environment_base_python", "missing_base_python",
                                       "PYVENV_CFG_NAME", "_stage_python_runtime",
                                       "repair_env", "usable_python_exe",
                                       "EnvironmentUnusable", "--without-pip",
                                       "--upgrade"),
    "ai_voice_studio.paths": ("shipped_python_runtime", "staged_python_runtime",
                              "PYTHON_EXE_NAME"),
    "ai_voice_studio.describe.engine": ("usable_python_exe",),
    "ai_voice_studio.stt.engine": ("usable_python_exe",),
    "ai_voice_studio.voicelab": ("usable_python_exe",),
    "ai_voice_studio.omnivoice": ("usable_python_exe",),
    "ai_voice_studio.omnivoice_server": ("usable_python_exe", "EnvironmentUnusable"),
}

_DEFAULT = os.path.join("dist", "AIVS-64", "AI-Voice-Studio", "AI-Voice-Studio.exe")


def _walk(code):
    yield code
    for const in code.co_consts:
        if hasattr(const, "co_consts"):
            yield from _walk(const)


def _vocabulary(code) -> str:
    """Everything the code object mentions: names, attributes and constants."""
    words: list[str] = []
    for node in _walk(code):
        words.extend(node.co_names)
        for const in node.co_consts:
            if isinstance(const, str):
                words.append(const)
            elif isinstance(const, int) and not isinstance(const, bool):
                words.append(str(const))
    return "\n".join(words)


def main(argv: list[str]) -> int:
    from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader

    exe = argv[1] if len(argv) > 1 else _DEFAULT
    if not os.path.isfile(exe):
        print(f"No frozen executable at {exe} - build it first "
              f"(packaging\\build.ps1 -Arch x64).")
        return 2

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
        missing_modules: list[str] = []
        for module, needles in WANTED.items():
            if module not in pyz.toc:
                missing_modules.append(module)
                print(f"  MISSING MODULE {module}")
                continue
            code = pyz.extract(module)
            if isinstance(code, (bytes, bytearray)):
                code = marshal.loads(code)
            vocabulary = _vocabulary(code)
            missing = [name for name in needles if name not in vocabulary]
            print(f"  {module}: {'ok' if not missing else 'MISSING ' + str(missing)}")
            if missing:
                missing_modules.append(module)
    finally:
        os.unlink(pyz_path)
    if missing_modules:
        print("RESULT: this build does not carry the environment-repair code")
        return 1
    print("RESULT: the build carries the environment-repair code")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
