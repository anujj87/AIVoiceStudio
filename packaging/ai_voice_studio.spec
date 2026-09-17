# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AI Voice Studio (onedir, Windows).

Build (from the repository root, using a 64-bit venv for the 64-bit build and
a 32-bit Python 3.13 venv for the 32-bit build)::

    python -m PyInstaller packaging/ai_voice_studio.spec --noconfirm \
        --distpath dist/AIVS-64 --workpath build/pyinstaller-64

The resulting folder (dist/AIVS-64/AI-Voice-Studio/) is consumed by the Inno
Setup scripts in this directory. ONNX Runtime and sherpa-onnx are bundled;
TTS models are NOT (they are downloaded in-app to the user folder).
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# Safety check: wx must be importable so collect_submodules can find wx.adv etc.
# If this fails, you're running PyInstaller with the wrong Python (not the .venv).
try:
    import wx
except ImportError:
    raise SystemExit(
        "FATAL: wx is not installed in this Python environment.\n"
        "Run PyInstaller with the project venv: .venv/Scripts/python.exe -m PyInstaller ..."
    )

# sherpa-onnx ships its native libraries inside the wheel; keep them.
hiddenimports = [
    "sherpa_onnx",
    "wx",
    "wx.adv",
    "pypdf",
    "docx",
    "requests",
    "numpy",
    # Subpackages not directly imported from main.py
    "ai_voice_studio.addons",
    "ai_voice_studio.python_runtime",
    "ai_voice_studio.heavy_logging",
    # Voice Lab: the clone-engine registry, its tuning vocabulary and its GUI
    # category and options dialog.
    "ai_voice_studio.voicelab",
    "ai_voice_studio.voicelab.engines",
    "ai_voice_studio.voicelab.options",
    "ai_voice_studio.gui.voicelab_options_dialog",
    # Per-category Compute combo shown next to every Preview button.
    "ai_voice_studio.gui.compute_choice",
]

datas = [
    # embedded model catalog (TTS -> language -> variant -> voice -> URLs)
    # paths are relative to this spec file (packaging/), so go up to the root
    ("../ai_voice_studio/tts/models_catalog.json", "ai_voice_studio/tts"),
    # Voice Lab worker: a standalone script run by the *managed virtualenv*
    # interpreter as a subprocess, which cannot import from the PYZ archive.
    ("../ai_voice_studio/voicelab/worker.py", "ai_voice_studio/voicelab"),
    ("../README.md", "."),
    # HTML documentation opened from the Help menu.
    ("../docs/README.html", "docs"),
    ("../docs/UserGuide.html", "docs"),
    ("../docs/AddonDevelopmentGuide.html", "docs"),
    ("../docs/AccessibilityGuide.html", "docs"),
    # Python & wxPython book (HTML, opened from Help menu via F1).
    ("../docs/book/index.html", "docs/book"),
    ("../docs/book/css/style.css", "docs/book/css"),
    ("../docs/book/chapters", "docs/book/chapters"),
]



binaries = []

# Include sherpa-onnx native libs if not auto-detected by hooks.
try:
    import sherpa_onnx  # noqa: F401

    sherpa_root = os.path.dirname(sherpa_onnx.__file__)
    lib_dir = os.path.join(sherpa_root, "lib")
    if os.path.isdir(lib_dir):
        binaries += [(os.path.join(lib_dir, f), "sherpa_onnx/lib")
                     for f in os.listdir(lib_dir) if f.endswith(".dll")]
except Exception:  # noqa: BLE001
    pass

a = Analysis(
    ["../main.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + collect_submodules("wx"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The standalone onnxruntime package must NEVER be bundled: sherpa-onnx
    # ships its own onnxruntime.dll, and two runtimes with the same module
    # name in one process corrupt the graph loader.
    excludes=["tkinter", "matplotlib", "pytest", "unittest", "onnxruntime"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI-Voice-Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AI-Voice-Studio",
)
