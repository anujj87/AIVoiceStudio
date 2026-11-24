"""The OmniVoice Language box, shared by every place that offers one.

OmniVoice detects the language of the text by itself, and on short lines that
detection can come back wrong - a one-word line in a language close to a
neighbour (Spanish/Portuguese, Hindi/Marathi, a Chinese line with a Latin brand
name) can be read with the wrong accent, or as gibberish.  The engine also
accepts an explicit *language hint*, and pinning one is the fix.

Every OmniVoice **Language** box in the application therefore offers the same
thing: **Auto** first, then all 646 languages of the model's own table
(:mod:`ai_voice_studio.omnivoice.languages`, upstream ``docs/languages.md``),
and every one of those boxes means the same thing - a hint to the engine, never
a filter on the available voices.  An OmniVoice voice can speak any of the
model's languages, so the voice list stays complete while the language is
pinned (see the Recording window and Settings > Available TTS).

Three details keep the boxes consistent and quick:

* :func:`pairs` is the single source of the list, so no dialog keeps a second,
  hand-maintained copy of the languages.
* :func:`set_pairs` hands the whole list to wx in **one** call.  Appending 647
  entries one at a time is measurably slower (roughly a third of a second per
  language box, on every switch), so the combo is filled in bulk and the values
  are kept in a parallel list on the widget - :func:`code_at` returns them, and
  falls back to the combo's own client data for the ordinary cascades.
* the value of every entry is the *catalog language key*: ``"auto"`` for the
  auto entry (the code the OmniVoice catalog entries themselves use) and the
  OmniVoice language id (``"hi"``, ``"zh"`` ...) for the rest.  Nothing here
  imports the engine, so it is safe in the GUI thread and in tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import wx

from ..omnivoice import languages, spec

#: The language code the OmniVoice catalog entries carry; ``"auto"`` means
#: "no hint - let the engine detect the language".
AUTO_KEY = "auto"

#: The engines whose Language box is a hint to the model rather than a key
#: into the downloaded-voice list.
OMNIVOICE_ENGINES = frozenset(spec.ENGINE_IDS)

#: How many languages the box offers besides Auto (``languages.COUNT``).
COUNT = languages.COUNT

#: The attributes :func:`set_pairs` leaves on a combo box with the values and
#: the labels of its items (wx has no bulk "append with client data" call).
#: The labels are kept so :func:`code_at` can tell a box that still holds the
#: bulk fill from one that was refilled item by item afterwards.
_CODES_ATTR = "_aivs_codes"
_LABELS_ATTR = "_aivs_code_labels"


def is_omnivoice(tts_id: str | None) -> bool:
    """True when ``tts_id`` names one of the OmniVoice engines."""
    return bool(tts_id) and str(tts_id) in OMNIVOICE_ENGINES


def pairs() -> list[tuple[str, str]]:
    """``(label, value)`` for the OmniVoice language box: auto, then all 646."""
    return [(languages.AUTO_LABEL, AUTO_KEY)] + [
        (label, value) for label, value in languages.choices()[1:]
    ]


def labels() -> list[str]:
    """The 647 combo-box labels, without the values."""
    return [label for label, _value in pairs()]


def label(value: str | None) -> str:
    """How a stored hint is shown in a summary (``"Hindi (hi)"``)."""
    if not value or str(value) == AUTO_KEY:
        return languages.AUTO_LABEL
    return languages.label(value)


def hint_of(combo: wx.ComboBox) -> str | None:
    """The language pinned in ``combo``, or ``None`` for auto.

    Works whether the box was filled by :func:`set_pairs` (values in the
    parallel list) or the ordinary per-item ``Append(label, data)`` cascade.
    """
    return languages.normalise(selected_code(combo))


def set_pairs(combo: wx.ComboBox, items: Sequence[tuple[str, str]]) -> None:
    """Fill a combo box from ``(label, value)`` pairs in one native call.

    The values are kept on the widget (:data:`_CODES_ATTR`) and read back with
    :func:`code_at`, which keeps the fast path interchangeable with the classic
    ``Append(label, value)`` loop used by the other cascades.
    """
    combo.Clear()
    pairs_list = list(items)
    setattr(combo, _CODES_ATTR, [value for _label, value in pairs_list])
    list_labels = [item for item, _value in pairs_list]
    setattr(combo, _LABELS_ATTR, list_labels)
    if list_labels:
        combo.AppendItems(list_labels)


def code_at(combo: wx.ComboBox, index: int) -> Any:
    """The value of the item at ``index`` (parallel list, else client data).

    The parallel list only describes the box while it still holds the bulk
    fill: the Recording window refills the same box item by item when an
    ordinary (non-OmniVoice) engine lists its own catalog languages, and
    reading the stale list then returned ``"auto"`` for every entry - the
    engine's variants and voices came out empty.
    """
    if index < 0 or index >= combo.GetCount():
        return None
    codes = getattr(combo, _CODES_ATTR, None)
    stored = getattr(combo, _LABELS_ATTR, None)
    if (codes is not None and stored is not None
            and index < len(codes) and index < len(stored)
            and combo.GetString(index) == stored[index]):
        return codes[index]
    return combo.GetClientData(index)


def selected_code(combo: wx.ComboBox) -> Any:
    """The value of the combo's current selection (``None`` when empty)."""
    return code_at(combo, combo.GetSelection())


def select(combo: wx.ComboBox, value: str | None) -> int:
    """Select the entry whose value matches ``value``; return its index.

    ``None`` (or the auto key) selects the first entry - Auto - which is also
    what an unknown value falls back to.
    """
    wanted = value if value not in (None, "") else AUTO_KEY
    index = 0
    for candidate in range(combo.GetCount()):
        if code_at(combo, candidate) == wanted:
            index = candidate
            break
    combo.SetSelection(index)
    if combo.GetCount():
        combo.SetValue(combo.GetString(index))
    return index


def fill(combo: wx.ComboBox, hint: str | None = None) -> None:
    """Fill a Language box with Auto + all 646 languages, ``hint`` selected."""
    set_pairs(combo, pairs())
    select(combo, hint or AUTO_KEY)
