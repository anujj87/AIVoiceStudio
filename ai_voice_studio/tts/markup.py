"""Expression markup an engine reads as director's notes, kept out of harm's way.

Kitten TTS 2.0 (``kitten_tts_2``, the ``kittenml`` package) accepts three kinds
of control in the text it is given::

    [joyful] We won the grant <laugh> I can (((hardly))) believe it!

A leading ``[emotion]`` tag, inline ``<event>`` tags, and ``(((emphasis)))``
spans.  They are *not* text to pronounce - the model conditions on them - so the
engine's own pipeline protects every span before its normalizer runs and restores
it verbatim afterwards (``kittenml.kittentts2.text``).  What the engine cannot
protect them from is *this* application: the punctuation modes expand every mark
into its spoken word ("All") or delete marks outright ("None", "Math"), and both
turn ``[joyful]`` into ``left bracket joyful right bracket`` or into the bare word
``joyful``, and ``(((hardly)))`` into ``hardly``.  The markup then either loses
its conditioning or is read out loud.

So the same trick is applied one layer up, before the mode runs: every span is
swapped for a sentinel-wrapped index, the mode sees ordinary text, and the spans
are written back exactly as typed.  A mode is therefore free to mangle
punctuation anywhere else - the tags come out of it unchanged, in order, and
byte-for-byte (private-use sentinels hold no punctuation, so no mode can match
them).

The vocabulary below is a mirror of the engine's own (``EXPRESSION_TAGS``,
``VOCAL_EVENT_TAGS`` and the emphasis span in ``kittenml.kittentts2.text``),
because the application process must recognise a tag without importing the
engine's environment.  ``tests/test_markup.py`` compares the two whenever the
engine is installed, so the mirror cannot drift unnoticed.  A bracketed span
that is *not* in the vocabulary - a citation like ``section [3]``, a comparison
like ``x < 5`` - is left to the mode and to the engine, exactly as before.
"""

from __future__ import annotations

import re

#: A leading expression tag: how the delivery of what follows should sound.
EXPRESSION_TAGS = frozenset({
    "mundane", "nervous", "tender", "angry", "excited",
    "stern", "sad", "contemplative", "surprised", "joyful",
})

#: A vocal event, inline where it happens: an audible breath, laugh, sigh.
VOCAL_EVENT_TAGS = frozenset({
    "pause", "sigh", "gasp", "laugh", "giggle",
    "sob", "scoff", "growl", "um", "gulp",
})

#: Both kinds of tag, in one pattern (the engine matches them case-insensitively).
_TAG_SPAN = re.compile(
    r"\[(?:" + "|".join(re.escape(tag) for tag in sorted(EXPRESSION_TAGS)) + r")\]"
    r"|<(?:" + "|".join(re.escape(tag) for tag in sorted(VOCAL_EVENT_TAGS)) + r")>",
    re.IGNORECASE,
)

#: An emphasised span: up to 80 characters, no nested parens, no line break -
#: the engine's own limits, so the two agree on what is inside a span.
_EMPHASIS_SPAN = re.compile(r"\(\(\(([^()\n]{1,80})\)\)\)")

#: Where a protected span sits while the mode runs.  Private-use characters
#: cannot occur in real text and are neither punctuation nor whitespace, so no
#: punctuation mode can match, split or reorder them; the closing sentinel is
#: what keeps ``\ue0001\ue001`` from ever matching inside ``\ue00010\ue001``.
_PLACEHOLDER = "\ue000{}\ue001"


def has_markup(text: str) -> bool:
    """True when ``text`` uses any expression, vocal-event or emphasis control."""
    value = str(text or "")
    return bool(_TAG_SPAN.search(value) or _EMPHASIS_SPAN.search(value))


def supports(engine_id: str | None) -> bool:
    """True when this engine reads the markup as director's notes, not as words.

    Asked of the engine registry rather than kept as a list of ids here, so a
    future engine that adopts the same markup is covered the moment it says so
    (``voicelab.engines.expression_markup``).
    """
    if not engine_id:
        return False
    try:
        from ..voicelab import engines as voicelab_engines

        return bool(voicelab_engines.expression_markup(engine_id))
    except Exception:
        return False


def protect_for(engine_id: str | None, text: str) -> tuple[str, list[str]]:
    """``(text to transform, spans to write back)`` for one engine.

    An engine that does not read the markup gets its text back untouched (and an
    empty list), so the modes keep their old meaning for it: it pronounces no
    tags, and a bracketed span it is handed is punctuation like any other.
    """
    if not supports(engine_id):
        return str(text or ""), []
    return protect(text)


def protect(text: str) -> tuple[str, list[str]]:
    """Swap every markup span for a sentinel, keeping the spans in order.

    Tags first, then emphasis spans, so a tag inside an emphasis span keeps its
    own entry in the list and is restored with it.
    """
    spans: list[str] = []

    def _swap(match):
        spans.append(match.group(0))
        return _PLACEHOLDER.format(len(spans) - 1)

    return _EMPHASIS_SPAN.sub(_swap, _TAG_SPAN.sub(_swap, str(text or ""))), spans


def restore(text: str, spans: list[str]) -> str:
    """Write the protected spans back, exactly as they were given.

    Newest first.  A span captured over another span's placeholder - a
    ``<laugh>`` inside an ``(((emphasis)))``, which is exactly what the two-pass
    protection produces - holds that placeholder in its own text, so the outer
    span has to go back before the inner one can be found inside it.
    """
    for index in range(len(spans or ()) - 1, -1, -1):
        text = text.replace(_PLACEHOLDER.format(index), spans[index])
    return text
