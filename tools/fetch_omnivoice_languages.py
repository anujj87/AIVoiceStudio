"""Regenerate ``ai_voice_studio/omnivoice/languages.py`` from upstream.

The OmniVoice language table is data, not hand-written code: it is a verbatim
copy of the model's own ``docs/languages.md`` (646 languages, 581k hours).  Run
this after upstream publishes a new table:

    python tools/fetch_omnivoice_languages.py

It downloads the markdown, rebuilds the module (name, OmniVoice id, ISO 639-3
code per row) and reports the row count, so the diff shows exactly which
languages moved.  Nothing else in the project reads the markdown, and the
module itself stays importable with no network access.

Pass ``--check`` in CI to fail when the shipped table no longer matches
upstream: the file is regenerated in memory and compared with the one on disk.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "ai_voice_studio" / "omnivoice" / "languages.py"
SOURCE_URL = ("https://raw.githubusercontent.com/k2-fsa/OmniVoice/master/"
              "docs/languages.md")

HEADER = '''"""The language ids OmniVoice understands (the manual language picker).

OmniVoice detects the language from the text on its own, and most of the time
that is exactly right.  On short lines it is not: a one-word sentence in a
language close to a neighbour (Spanish and Portuguese, Hindi and Marathi, a
Chinese line carrying a Latin brand name) can come back with the wrong accent
- or as gibberish.  OmniVoice also accepts an explicit language hint, and this
module is the list of hints it was trained on: the 646 languages of
``k2-fsa/OmniVoice`` (581k hours of training audio).

* **Auto** (:data:`AUTO_LABEL`) is always the first entry of the picker.  Its
  value is :data:`AUTO` (``""``), which normalises to "no hint at all" and
  therefore keeps OmniVoice's own detection - the behaviour of every release
  before the picker existed.
* Every other entry carries the upstream **OmniVoice language id** (``en``,
  ``zh``, ``hi`` ...), which is exactly the value ``spec.clean_language``
  passes to the engine, plus the English name for the label.  The ISO 639-3
  code is kept alongside so that a code typed or pasted by hand (``eng``,
  ``cmn``, ``deu``) still lands on the right language.

The table below is a verbatim copy of upstream ``docs/languages.md``
(:data:`SOURCE_URL`), in the file's own alphabetical-by-name order, which is
also the order the combo box shows.  It is code rather than a data file on
purpose: the dialogs, the frozen build and the tests then need no extra
resource and no packaging rule, and a wrong hint fails loudly at import time
against :func:`_check` instead of silently at synthesis time.

Regenerate it with ``python tools/fetch_omnivoice_languages.py``.

Nothing here imports the engine, so it is safe in the GUI thread and in tests.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

#: The picker's first entry - no hint, OmniVoice detects the language itself.
AUTO = ""
AUTO_LABEL = "Auto (detect from the text)"

#: Where the table came from, so a future upgrade has an obvious starting point.
SOURCE_URL = "https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md"

#: ``(English name, OmniVoice language id, ISO 639-3 code)`` for every
#: language OmniVoice was trained on, as published upstream.
LANGUAGES: Tuple[Tuple[str, str, str], ...] = (
'''

FOOTER = ''')


def _check() -> None:
    """Fail at import time if the generated table was ever edited by hand."""
    seen_ids: Dict[str, str] = {}
    for name, omni_id, iso in LANGUAGES:
        if not name or not omni_id or not iso:
            raise ValueError(f"incomplete language row: {(name, omni_id, iso)!r}")
        if not omni_id.islower() or not iso.islower():
            raise ValueError(f"language ids must be lower case: {omni_id!r}")
        if omni_id in seen_ids:
            raise ValueError(
                f"duplicate OmniVoice id {omni_id!r} ({seen_ids[omni_id]!r} and {name!r})"
            )
        seen_ids[omni_id] = name


_check()

#: Number of languages OmniVoice was trained on (upstream: 646, 581k hours).
COUNT = len(LANGUAGES)

_BY_ID: Dict[str, Tuple[str, str]] = {
    omni_id: (name, iso) for name, omni_id, iso in LANGUAGES
}
_BY_ISO: Dict[str, str] = {iso: omni_id for _name, omni_id, iso in LANGUAGES}
_BY_NAME: Dict[str, str] = {name.lower(): omni_id for name, omni_id, _iso in LANGUAGES}


def language_ids() -> Tuple[str, ...]:
    """Every OmniVoice language id, in the picker's order."""
    return tuple(omni_id for _name, omni_id, _iso in LANGUAGES)


def names() -> Tuple[str, ...]:
    """Every English language name, in the picker's order."""
    return tuple(name for name, _omni_id, _iso in LANGUAGES)


def count() -> int:
    """How many languages the table holds (``COUNT``, as a function)."""
    return len(LANGUAGES)


def is_known(value: Optional[str]) -> bool:
    """True when ``value`` is a language id this table lists."""
    return bool(value) and str(value).strip().lower() in _BY_ID


def display_name(value: Optional[str]) -> str:
    """The English name of a language id, or the raw text when unknown.

    Used by the dialogs to describe a stored hint without a lookup table of
    their own; an unknown value is returned unchanged so nothing is lost.
    """
    text = ("" if value is None else str(value)).strip()
    if not text:
        return ""
    entry = _BY_ID.get(text.lower())
    return entry[0] if entry else text


def label(value: Optional[str]) -> str:
    """The combo-box label of a language: ``"English (en)"``.

    The auto entry gets its own wording; an unknown value is shown as typed so
    a hint from a newer engine version is never hidden from the user.
    """
    text = ("" if value is None else str(value)).strip()
    if not text:
        return AUTO_LABEL
    lower = text.lower()
    entry = _BY_ID.get(lower)
    if entry:
        return f"{entry[0]} ({lower})"
    return text


def normalise(value: Optional[str]) -> Optional[str]:
    """Map free text to a canonical OmniVoice language id, or ``None``.

    Accepts what a user may reasonably type or paste:

    * ``""`` / ``"auto"`` / ``"automatic"`` / ``"any"`` / the picker's own
      :data:`AUTO_LABEL` -> ``None`` (no hint),
    * a language id (``en``, ``zh``, ``kbt``) -> itself, lower-cased,
    * an ISO 639-3 code (``eng``, ``cmn``, ``deu``) -> its OmniVoice id,
    * an English name (``English``, ``Chinese``) -> its OmniVoice id,
    * a picker label (``English (en)``) -> its OmniVoice id,
    * anything else -> the trimmed text, lower-cased, untouched.

    The last rule matters: an engine newer than this table may know languages
    it does not, and refusing that hint here would break a working project.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in ("auto", "automatic", "any") or lower.startswith("auto ("):
        return None
    if lower in _BY_ID:
        return lower
    if lower in _BY_ISO:
        return _BY_ISO[lower]
    if lower in _BY_NAME:
        return _BY_NAME[lower]
    # "English (en)" / "English - en" / "en (English)" -> the bracketed code.
    for part in re.split(r"[()\\[\\],/]| - ", lower):
        part = part.strip()
        if part in _BY_ID:
            return part
        if part in _BY_ISO:
            return _BY_ISO[part]
        if part in _BY_NAME:
            return _BY_NAME[part]
    return lower


def choices() -> List[Tuple[str, str]]:
    """``(label, value)`` pairs for the picker: auto first, then alphabetical.

    The list is built once per call and is cheap; callers that fill a combo box
    call it exactly once.
    """
    out: List[Tuple[str, str]] = [(AUTO_LABEL, AUTO)]
    out.extend((f"{name} ({omni_id})", omni_id) for name, omni_id, _iso in LANGUAGES)
    return out


def selection_index(value: Optional[str]) -> int:
    """The :func:`choices` index of a stored hint (0 = auto, 0 when unknown)."""
    normalised = normalise(value)
    if not normalised:
        return 0
    for index, (_label, item_value) in enumerate(choices()):
        if item_value == normalised:
            return index
    return 0
'''


def fetch(url: str = SOURCE_URL) -> str:
    """Download the upstream markdown table."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def parse(markdown: str) -> list:
    """``[(name, omni_id, iso)]`` from the upstream markdown table."""
    rows = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 4 or not cells[0].isdigit():
            continue
        rows.append((cells[1], cells[2], cells[3]))
    if not rows:
        raise SystemExit("no rows parsed - did the upstream table change format?")
    return rows


def render(rows: list) -> str:
    """The full module text for a parsed table."""
    body = "".join(
        f'    ("{name}", "{omni_id}", "{iso}"),\n' for name, omni_id, iso in rows
    )
    return HEADER + body + FOOTER


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="fail when the shipped table differs from upstream")
    parser.add_argument("--url", default=SOURCE_URL,
                        help="fetch the table from another URL")
    args = parser.parse_args(argv)

    rows = parse(fetch(args.url))
    text = render(rows)
    current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
    if args.check:
        if text != current:
            print(f"{TARGET.relative_to(ROOT)} is out of date "
                  f"({len(rows)} languages upstream)")
            return 1
        print(f"{TARGET.relative_to(ROOT)} matches upstream ({len(rows)} languages)")
        return 0
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET.relative_to(ROOT)} with {len(rows)} languages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
