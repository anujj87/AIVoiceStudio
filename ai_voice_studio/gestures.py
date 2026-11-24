"""Configurable keyboard gestures (after NVDA's *Input Gestures*).

NVDA models every user action as a *script* that belongs to a **category** and
carries a human-readable name; a *gesture* is a normalized key combination
(``kb:control+nvda+f1``) bound to that script.  A script's own defaults are
consulted first, then the locale map, then the **user map** saved to
``gestures.ini``; the Input Gestures dialog lists the commands by category and
lets the user add or remove those bindings.

AI Voice Studio has no braille and no per-object scriptable surface, so this
module keeps the parts that matter for a screen-reader-first desktop app:

* every command is ``(id, category, label, default gestures)``;
* a gesture is a normalized ``kb:<modifier+key>`` string;
* a user binding **replaces** the defaults for its command (an entry of ``[]``
  therefore unbinds it), and lives in ``settings['gestures.bindings']``;
* two commands that end up asking for the same key are reported as a
  **conflict**, never silently resolved.

The heavy lifting that needs wx (turning a key press into a gesture string,
the dialog itself) lives in :mod:`ai_voice_studio.gui.settings_dialog`; this
module stays importable without a GUI so the model can be unit tested.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: The gesture source prefix, exactly as NVDA writes it (``kb`` = keyboard).
SOURCE = "kb"

#: Canonical modifier order.  A gesture's chunks are reordered to this, so
#: ``control+shift+n`` and ``shift+control+n`` are the same gesture.
MODIFIER_ORDER = ("control", "alt", "shift", "windows")

#: Spellings users and other tools may write instead of the canonical one.
_ALIASES = {
    "ctrl": "control",
    "ctl": "control",
    "altgr": "alt",
    "win": "windows",
    "cmd": "windows",
    "super": "windows",
    "meta": "windows",
    "esc": "escape",
    "del": "delete",
    "ins": "insert",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "return": "enter",
}

#: Pretty names for the display text (NVDA shows the same kind of label).
_DISPLAY = {
    "control": "Ctrl",
    "alt": "Alt",
    "shift": "Shift",
    "windows": "Windows",
    "+": "Plus",
    " ": "Space",
    "space": "Space",
    ",": "Comma",
    "comma": "Comma",
    ".": "Period",
    "period": "Period",
    "-": "Minus",
    "=": "Equals",
    "[": "LeftBracket",
    "]": "RightBracket",
    ";": "Semicolon",
    "'": "Apostrophe",
    "/": "Slash",
    "\\": "Backslash",
    "`": "Grave",
}


# ---------------------------------------------------------------------------
# Gesture identifiers
# ---------------------------------------------------------------------------
def normalize(identifier: str) -> str:
    """Normalize a gesture to its canonical ``kb:<mods+key>`` form.

    ``KB:Control+Shift+N`` and ``kb:shift+control+n`` both become
    ``kb:control+shift+n``; modifiers are put in a fixed order and everything
    is lower-cased, which is what makes a user binding match the same key
    press whatever order the user happened to hold the keys in.
    """
    text = str(identifier or "").strip()
    if not text:
        return ""
    if ":" in text:
        source, _, body = text.partition(":")
        source = source.strip().lower()
    else:
        source, body = SOURCE, text
    parts = [chunk.strip().lower() for chunk in body.split("+") if chunk.strip()]
    if not parts:
        return ""
    parts = [_ALIASES.get(part, part) for part in parts]
    modifiers = [p for p in MODIFIER_ORDER if p in parts]
    keys = [p for p in parts if p not in MODIFIER_ORDER]
    if len(keys) != 1:
        # A gesture must end in exactly one non-modifier key.
        return ""
    if source and source != SOURCE:
        return f"{source}:{'+'.join(modifiers + keys)}"
    return f"{SOURCE}:{'+'.join(modifiers + keys)}"


def is_valid(identifier: str) -> bool:
    """True when ``identifier`` is a well-formed keyboard gesture."""
    text = str(identifier or "")
    if not text.lower().startswith(f"{SOURCE}:"):
        return False
    return bool(normalize(text))


def display_name(identifier: str) -> str:
    """A human-readable name for a gesture (``kb:control+shift+n`` -> ``Ctrl+Shift+N``)."""
    normalized = normalize(identifier)
    if not normalized:
        return str(identifier or "")
    body = normalized.split(":", 1)[1]
    labels = []
    for chunk in body.split("+"):
        if chunk in _DISPLAY:
            labels.append(_DISPLAY[chunk])
        elif len(chunk) == 1 and chunk.isalpha():
            labels.append(chunk.upper())
        else:
            labels.append(chunk.capitalize())
    return "+".join(labels)


def source_display(identifier: str) -> str:
    """The gesture's source, as the dialog shows it in a column."""
    return "Keyboard"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Command:
    """One action a gesture can trigger."""

    id: str
    category: str
    label: str
    defaults: tuple[str, ...] = field(default_factory=tuple)

    @property
    def description(self) -> str:
        return self.label


#: Categories, in the order the dialog shows them.
CATEGORIES: tuple[str, ...] = (
    "Projects",
    "Recording",
    "Help",
    "Application",
)

#: Every command the application exposes to the gesture manager.
COMMANDS: tuple[Command, ...] = (
    Command("project.new", "Projects", "Create a new project",
            ("kb:control+shift+n",)),
    Command("project.open", "Projects", "Open a project...",
            ("kb:control+o",)),
    Command("project.resume", "Projects",
            "Resume the selected project"),
    Command("project.restart", "Projects",
            "Restart the selected project..."),
    Command("project.restart_all", "Projects",
            "Restart all files of the selected project"),
    Command("project.start_selected", "Projects",
            "Start selected files of the selected project..."),
    Command("project.remove", "Projects",
            "Remove the selected project..."),
    Command("project.show_folder", "Projects",
            "Show the selected project's folder"),
    Command("record.start", "Recording", "Open the recording window",
            ("kb:control+shift+r",)),
    Command("help.readme", "Help", "Open the Read Me"),
    Command("help.user_guide", "Help", "Open the User Guide"),
    Command("help.whats_new", "Help", "Open What's New"),
    Command("help.check_updates", "Help", "Check for updates..."),
    Command("help.about", "Help", "About AI Voice Studio"),
    Command("app.settings", "Application", "Open Settings...",
            ("kb:control+comma",)),
)

_COMMAND_BY_ID = {command.id: command for command in COMMANDS}


def command(command_id: str) -> Command | None:
    return _COMMAND_BY_ID.get(command_id)


def command_ids() -> list[str]:
    return [c.id for c in COMMANDS]


def commands_in(category: str) -> list[Command]:
    return [c for c in COMMANDS if c.category == category]


def categories_present() -> list[str]:
    """Categories that actually hold a command, in the declared order."""
    used = {c.category for c in COMMANDS}
    ordered = [name for name in CATEGORIES if name in used]
    ordered += sorted(used - set(ordered))
    return ordered


# ---------------------------------------------------------------------------
# The user binding map
# ---------------------------------------------------------------------------
class GestureMap:
    """The effective bindings: catalog defaults, then the user's overrides.

    ``bindings`` is the user map (``{command_id: [gesture, ...]}``).  A
    command that appears in it uses exactly those gestures - including an
    empty list, which unbinds it - and every other command falls back to the
    defaults in :data:`COMMANDS`.
    """

    def __init__(self, bindings: dict[str, Sequence[str]] | None = None):
        self._user: dict[str, list[str]] = {}
        if bindings:
            self.load(bindings)

    # -- loading and saving -------------------------------------------------
    def load(self, bindings: dict[str, Sequence[str]]) -> None:
        """Replace the user map, ignoring commands and gestures we do not know."""
        self._user = {}
        for command_id, gestures in (bindings or {}).items():
            if command_id not in _COMMAND_BY_ID:
                log.debug("Ignoring gesture binding for unknown command %r",
                          command_id)
                continue
            cleaned: list[str] = []
            for gesture in _as_list(gestures):
                normalized = normalize(gesture)
                if not normalized:
                    log.debug("Ignoring malformed gesture %r for %s",
                              gesture, command_id)
                    continue
                if normalized not in cleaned:
                    cleaned.append(normalized)
            self._user[command_id] = cleaned

    def export(self) -> dict[str, list[str]]:
        """The user map, ready for the settings file."""
        return {command_id: list(gestures)
                for command_id, gestures in self._user.items()}

    # -- reading ------------------------------------------------------------
    def defaults_for(self, command_id: str) -> list[str]:
        entry = command(command_id)
        return [normalize(g) for g in (entry.defaults if entry else ())]

    def bindings_for(self, command_id: str) -> list[str]:
        """The gestures that trigger ``command_id`` (user map first)."""
        if command_id in self._user:
            return list(self._user[command_id])
        return self.defaults_for(command_id)

    def is_customized(self, command_id: str) -> bool:
        return command_id in self._user

    #: Backwards-friendly alias used by the dispatcher and the dialog.
    gestures_for = bindings_for

    def owner(self, gesture: str) -> str | None:
        """The command ``gesture`` triggers, or ``None``.

        When two commands claim the same key the first declared wins, so the
        dispatcher is deterministic; :meth:`conflicts` names the clash for the
        dialog to report.
        """
        normalized = normalize(gesture)
        if not normalized:
            return None
        for entry in COMMANDS:
            if normalized in self.bindings_for(entry.id):
                return entry.id
        return None

    def conflicts(self) -> dict[str, list[str]]:
        """``{gesture: [command ids]}`` for every key claimed more than once."""
        owners: dict[str, list[str]] = {}
        for entry in COMMANDS:
            for gesture in self.bindings_for(entry.id):
                owners.setdefault(gesture, [])
                if entry.id not in owners[gesture]:
                    owners[gesture].append(entry.id)
        return {gesture: ids for gesture, ids in owners.items() if len(ids) > 1}

    # -- editing ------------------------------------------------------------
    def set_bindings(self, command_id: str, gestures: Iterable[str]) -> None:
        if command_id not in _COMMAND_BY_ID:
            raise KeyError(command_id)
        cleaned: list[str] = []
        for gesture in gestures:
            normalized = normalize(gesture)
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        self._user[command_id] = cleaned

    def add(self, command_id: str, gesture: str, *, replace_owner: bool = False) -> None:
        """Give ``command_id`` the key ``gesture``.

        With ``replace_owner`` the same key is taken away from any other
        command that currently holds it (that is what the dialog offers when
        it finds a conflict); without it the clash is left for
        :meth:`conflicts` to report.
        """
        normalized = normalize(gesture)
        if not normalized:
            raise ValueError(f"not a keyboard gesture: {gesture!r}")
        if replace_owner:
            for entry in COMMANDS:
                if entry.id == command_id:
                    continue
                if normalized in self.bindings_for(entry.id):
                    self.set_bindings(
                        entry.id,
                        [g for g in self.bindings_for(entry.id)
                         if g != normalized],
                    )
        current = self.bindings_for(command_id)
        if normalized not in current:
            current.append(normalized)
        self.set_bindings(command_id, current)

    def remove(self, command_id: str, gesture: str) -> None:
        normalized = normalize(gesture)
        self.set_bindings(
            command_id,
            [g for g in self.bindings_for(command_id) if g != normalized],
        )

    def reset(self, command_id: str) -> None:
        """Forget the user's override so the command's defaults apply again."""
        self._user.pop(command_id, None)

    def reset_all(self) -> None:
        self._user.clear()


def _as_list(value) -> list[str]:
    """Coerce a stored binding value (str, list, tuple or None) to a list."""
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


__all__ = [
    "CATEGORIES",
    "COMMANDS",
    "MODIFIER_ORDER",
    "SOURCE",
    "Command",
    "GestureMap",
    "categories_present",
    "command",
    "command_ids",
    "commands_in",
    "display_name",
    "is_valid",
    "normalize",
    "source_display",
]
