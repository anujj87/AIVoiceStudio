"""The Windows version resource for the packaged application.

Windows - and every screen reader that answers "which application is this, and
which version?" - reads an executable's own VERSIONINFO resource, never the
file name and never the application's About box.  NVDA's *report the
application version* gesture asks Windows for the product name and the product
version: a PyInstaller build without a resource answers nothing, and NVDA
reports "Application unknown, version not detected (<architecture>)".

``packaging/ai_voice_studio.spec`` hands :func:`version_info` to PyInstaller's
``EXE`` target, which writes it into ``AI-Voice-Studio.exe``.  Every value
comes from ``constants`` and ``ai_voice_studio.__author__`` - the same strings
the About box, the update checker and the installer carry - so the resource
cannot drift from the version the application reports.

Inspect the resource of a built executable (and exit non-zero if it has none)
with::

    .venv/Scripts/python.exe tools/make_version_info.py \\
        dist/AIVS-64/AI-Voice-Studio/AI-Voice-Studio.exe
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from ai_voice_studio import __author__
from ai_voice_studio.constants import APP_NAME, APP_VERSION

#: The executable's internal name and its file name, as Windows and the
#: installer see them (the PyInstaller spec builds ``AI-Voice-Studio.exe``).
INTERNAL_NAME = "AI-Voice-Studio"
ORIGINAL_FILENAME = "AI-Voice-Studio.exe"

#: English (United States) / Unicode, the usual choice for a Windows
#: executable: ``040904B0`` is language 0x0409, code page 0x04B0.
_LANGUAGE_CODEPAGE = "040904B0"
_TRANSLATION = [0x0409, 0x04B0]


def version_tuple(version: str = APP_VERSION) -> tuple[int, int, int, int]:
    """``"2027.1.0"`` becomes ``(2027, 1, 0, 0)``.

    A version resource carries four numbers.  A version with fewer parts, or
    with a suffix (``"2027.1.0-rc1"``), is padded with zeros and its
    non-numeric tail dropped, so a pre-release still builds.
    """
    numbers: list[int] = []
    for part in str(version).split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        numbers.append(int(digits))
        if len(numbers) == 4:
            break
    numbers += [0] * (4 - len(numbers))
    return tuple(numbers[:4])  # type: ignore[return-value]


def fields(version: str = APP_VERSION) -> dict[str, str]:
    """The version strings Windows keeps for the file.

    ``ProductName`` and ``ProductVersion`` are the two NVDA reads out; the
    rest are what Explorer's Details tab, the file's Properties dialog and the
    uninstall entry display.
    """
    return {
        "CompanyName": __author__,
        "FileDescription": APP_NAME,
        "FileVersion": version,
        "InternalName": INTERNAL_NAME,
        "LegalCopyright": f"{__author__} - GNU GPL v3.0 or later",
        "OriginalFilename": ORIGINAL_FILENAME,
        "ProductName": APP_NAME,
        "ProductVersion": version,
    }


def version_info(version: str = APP_VERSION):
    """The ``VSVersionInfo`` PyInstaller writes into the executable.

    Returned as the object PyInstaller's ``EXE`` target accepts directly (its
    ``version=`` argument), so the build never writes a throwaway file.
    """
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo,
        StringFileInfo,
        StringStruct,
        StringTable,
        VarFileInfo,
        VarStruct,
        VSVersionInfo,
    )

    numbers = version_tuple(version)
    table = StringTable(
        _LANGUAGE_CODEPAGE,
        [StringStruct(name, value) for name, value in fields(version).items()],
    )
    return VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=numbers, prodvers=numbers, mask=0x3F, flags=0x0,
            OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0),
        ),
        kids=[
            StringFileInfo([table]),
            VarFileInfo([VarStruct("Translation", _TRANSLATION)]),
        ],
    )


def read_version_info(path: str) -> dict[str, str]:
    """Read a built executable's version strings back.

    An empty mapping is the screen reader's own view of a file that carries no
    resource at all - the state NVDA reports as "version not detected".
    """
    if sys.platform != "win32":
        return {}
    from PyInstaller.utils.win32.versioninfo import (
        read_version_info_from_executable,
    )

    try:
        info = read_version_info_from_executable(path)
    except Exception:
        # Windows reports a missing resource the way it always does - "the
        # specified resource type cannot be found in the image file" - rather
        # than handing back an empty one, so that is the empty answer here.
        return {}
    if info is None:
        return {}
    strings: dict[str, str] = {}
    for kid in getattr(info, "kids", []):
        for table in getattr(kid, "kids", []):
            for entry in getattr(table, "kids", []):
                name = getattr(entry, "name", None)
                value = getattr(entry, "val", None)
                if isinstance(name, str) and isinstance(value, str):
                    strings[name] = value
    return strings


def main(argv: list[str] | None = None) -> int:
    """Print the fields; with a path, read that executable back and compare."""
    args = list(sys.argv[1:] if argv is None else argv)
    declared = fields()
    for name, value in declared.items():
        print(f"{name}: {value}")
    if not args:
        return 0
    print()
    found = read_version_info(args[0])
    print(f"{args[0]}:")
    if not found:
        print("  no version resource")
        return 1
    for name in declared:
        print(f"  {name}: {found.get(name, '<missing>')}")
    missing = [name for name in declared if name not in found]
    if missing:
        print(f"  missing: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
