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

# The executable's own name and version.  Windows - and every screen reader
# answering "which application is this, and which version?" - reads them from
# the file's VERSIONINFO resource, never from its name: NVDA's "report the
# application version" gesture asks for the product name and product version,
# and a build with no resource is reported as "Application unknown, version
# not detected".  The resource is built from the same constants the About box,
# the installer and the update checker carry, so it cannot drift.
sys.path.insert(0, os.path.abspath(os.path.join(SPECPATH, "..")))
from tools.make_version_info import version_info  # noqa: E402

# The executable's own manifest (per-monitor-v2 DPI awareness, long paths and
# the "run as the current user" execution level).  Without the DPI entry the
# process is DPI-unaware: Windows renders it at 96 DPI and then stretches the
# bitmap at 150% and 200%.  That is blurry, and a screen reader reports the
# whole window as one image instead of the controls inside it.
dpi_manifest = os.path.join(SPECPATH, "dpi_aware.manifest")

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
    # Speech-to-Text: catalog, per-model environments and the transcribe GUI.
    "ai_voice_studio.stt",
    "ai_voice_studio.stt.catalog",
    "ai_voice_studio.stt.environments",
    "ai_voice_studio.gui.stt_dialog",
    # Audio description: catalog, per-model environments, the pipeline and the
    # describe window.  The worker script itself is data (see ``datas``), because
    # a describing model's own interpreter runs it as a subprocess.
    "ai_voice_studio.describe",
    "ai_voice_studio.describe.catalog",
    "ai_voice_studio.describe.engine",
    "ai_voice_studio.describe.environments",
    "ai_voice_studio.describe.extended",
    "ai_voice_studio.describe.media",
    "ai_voice_studio.describe.narration",
    "ai_voice_studio.describe.output",
    "ai_voice_studio.describe.pipeline",
    "ai_voice_studio.describe.prompts",
    "ai_voice_studio.describe.weights",
    "ai_voice_studio.gui.describe_dialog",
    # User-configurable keyboard gestures and the theme editor opened from
    # Settings (that one is imported inside a handler, not at module level).
    "ai_voice_studio.gestures",
    "ai_voice_studio.gui.theme_studio",
    # Per-category Compute combo shown next to every Preview button.
    "ai_voice_studio.gui.compute_choice",
    # First-launch terms acceptance dialog (imported by main.py).
    "ai_voice_studio.gui.accept_dialog",
]

datas = [
    # embedded model catalog (TTS -> language -> variant -> voice -> URLs)
    # paths are relative to this spec file (packaging/), so go up to the root
    ("../ai_voice_studio/tts/models_catalog.json", "ai_voice_studio/tts"),
    # Voice Lab worker: a standalone script run by the *managed virtualenv*
    # interpreter as a subprocess, which cannot import from the PYZ archive.
    ("../ai_voice_studio/voicelab/worker.py", "ai_voice_studio/voicelab"),
    # Speech-to-Text worker: the same idea - run by one STT model's own
    # environment interpreter, so it must exist as a real file on disk.
    ("../ai_voice_studio/stt/worker.py", "ai_voice_studio/stt"),
    # Audio-description worker: the same rule again.  It is deliberately
    # standalone (it imports only transformers/torch/PIL), so nothing else of
    # the describe package has to be readable outside the archive.
    ("../ai_voice_studio/describe/worker.py", "ai_voice_studio/describe"),
    ("../README.md", "."),
    # HTML documentation opened from the Help menu.
    ("../docs/README.html", "docs"),
    ("../docs/UserGuide.html", "docs"),
    ("../docs/AddonDevelopmentGuide.html", "docs"),
    ("../docs/AccessibilityGuide.html", "docs"),
    ("../docs/THIRD-PARTY-LICENSES.html", "docs"),
    ("../docs/WHATS-NEW.html", "docs"),
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
    # Name ("AI Voice Studio") and version (constants.APP_VERSION) in the
    # executable itself, for Windows and for screen readers.
    version=version_info(),
    # Per-monitor-v2 DPI awareness (packaging/dpi_aware.manifest).
    manifest=dpi_manifest,
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
