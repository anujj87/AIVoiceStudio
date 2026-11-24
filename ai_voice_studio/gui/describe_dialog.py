"""Audio description dialog.

The final step of an audio description project (and what *Open* on a recent
audio description project reopens).  The New Project wizard already asked which
model describes the picture, which variant of it, on what device, at what frame
rate, in what chunk length and into what documents, so this window repeats none
of them as a question: it reports them, and asks only what is decided at run
time - how the descriptions are placed, whether the pauses are fitted with the
speech to text model, and which installed voice speaks the narration, chosen as
the same cascade the recording side uses (a TTS model, then one of its variants,
then one of that variant's voices).  Then it estimates how long the run will take
- and, for a long video, warns before anything starts, offering shorter chunks.

*Start audio description* runs the job in the background and writes the
description documents; *Open project folder* shows them; *Close* ends the
dialog.  A run that is closed early keeps every chunk it had already described,
because the job rewrites the script after each one.

A run produces **three** things per video, and the window says so rather than
leaving the user to discover which of them appeared:

* the description document in the format the project chose (the script, HTML,
  SRT, WebVTT or JSON);
* the **spoken track**, when the narration option is on - an installed offline
  voice speaks every description (see :mod:`ai_voice_studio.describe.narration`)
  into an audio file next to the script; and
* the **described film** - a new video file, never the source: with narration on,
  the original picture and sound with the description mixed over the pauses
  (standard), or the film stopped for each description (extended).

Screen-reader follows the rest of the application: every control is labelled with
an accessible name, the status line is a real ``StaticText``, and the progress is
a gauge.
"""

from __future__ import annotations

import logging
import os
import threading

import wx

from .. import project
from ..constants import (
    AD_CHUNK_CHOICES,
    AD_DEFAULT_FPS,
    AD_DEFAULT_OUTPUT,
    AD_DEFAULT_PLACEMENT,
    AD_DEFAULT_TRACK_PRIORITY,
    AD_EXTENDED_SECONDS_PER_DESCRIPTION,
    AD_FPS_CHOICES,
    AD_MAX_DESCRIPTION_SECONDS,
    AD_MAX_WINDOW_SECONDS,
    AD_MIN_GAP_SECONDS,
    AD_NARRATION_VOLUME_DEFAULT,
    AD_OUTPUT_CHOICES,
    AD_PLACEMENT_CHOICES,
    AD_PLACEMENT_EXTENDED,
    AD_TRACK_PRIORITY_CHOICES,
    AD_TRACK_PRIORITY_GAIN,
    AD_WARN_SECONDS,
    AD_WINDOW_SECONDS,
    COMPUTE_CPU,
    VOLUME_MAX,
    VOLUME_MIN,
)
from ..describe import catalog, engine, extended, media, narration, pipeline
from ..describe.job import DescribeJob
from ..settings import Settings
from ..tts.models import ModelStore
from . import announce, dpi
from .a11y import (
    add_labeled,
    finalize_accessibility,
    set_accessible_name,
    update_accessible_name,
)
from .dialogs import open_folder, run_action_dialog
from .progress import TaskProgressDialog

log = logging.getLogger(__name__)

#: The shorter chunk length the long-video warning offers.
_SHORT_CHUNK = 60.0


class DescribeDialog(wx.Dialog):
    """Choose how to describe a project's videos and run the job."""

    #: What *Track priority* means, written once: it is the tool tip of the
    #: enabled combo and, in extended mode, replaced by the reason it cannot
    #: apply there (see :meth:`_on_placement`).
    _PRIORITY_TOOLTIP = (
        "Which of the two tracks gives way when the description is heard over "
        "the film's own soundtrack. Same volume leaves both tracks as they "
        "are. The 25% and 50% choices keep the description at the level your "
        "narration settings gave it and make the film's own sound quieter by a "
        "quarter or by half for the whole film, so the description stands out "
        "from a film whose music and dialogue never stop. Duck, don't lower "
        "does not lower the film at all: it dips the film's own sound to a "
        "quarter of its own level while a description is actually being spoken "
        "and lets it back up between descriptions, so a film scored wall to "
        "wall keeps its own sound everywhere the description is not talking."
    )

    #: The narration volume row's label, written once: the slider's own value is
    #: part of this text ("Narration voice volume: 1.00"), so a screen reader
    #: reads the word and the number together instead of a bare "1.00".
    _NARRATION_VOLUME_NAME = "Narration voice volume"

    def __init__(self, parent, project_dir: str, settings: Settings,
                 store: ModelStore, sources: list | None = None):
        super().__init__(
            parent,
            title="Audio description",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(820, 640),
        )
        self.project_dir = project_dir
        self.settings = settings
        self.store = store
        self.project = project.load_project(project_dir)
        self.describe_settings = dict(self.project.get("describe", {}) or {})
        # ``sources`` restricts the run to a subset (Edit > Start Selected
        # Descriptions); ``None`` describes every source of the project.
        self.sources = list(sources) if sources else self._collect_sources()
        self.job: DescribeJob | None = None
        self._progress_dlg: TaskProgressDialog | None = None
        # The voices the recording side already has installed, and the audio
        # format the narration track is written in (the recording setting, so a
        # project's narration and its recordings sound and store the same way).
        self.voices = self._installed_voices()
        self.narration_format = str(
            self.settings.get("recording.output_format", "wav") or "wav"
        ).lower()

        name = self.project.get("name") or os.path.basename(project_dir)
        self.SetTitle(f"Audio description - {name}")
        self._build_ui(name)
        self._populate_narration()
        finalize_accessibility(self)
        dpi.prepare(self)
        dpi.follow(self)
        wx.CallAfter(self.placement_combo.SetFocus)

    # -- data --------------------------------------------------------------
    def _collect_sources(self) -> list[str]:
        sources = self.describe_settings.get("sources") or []
        if sources:
            return [s for s in sources if s]
        found = [seg.get("source") for seg in self.project.get("segments", [])]
        return [s for s in found if s]

    # -- construction ------------------------------------------------------
    def _build_ui(self, name: str) -> None:
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label=f"Project: {name}"), 0, wx.ALL, 8)
        count = len(self.sources)
        sizer.Add(
            wx.StaticText(
                self,
                label=f"{count} video file(s) will be described.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8,
        )

        # What the project was created with, stated rather than asked again.
        self.run_label = wx.StaticText(self, label="")
        set_accessible_name(self.run_label, "Description settings")
        self.run_label.Wrap(760)
        sizer.Add(self.run_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        self._refresh_run_label()

        grid = wx.FlexGridSizer(cols=2, vgap=8, hgap=10)
        grid.AddGrowableCol(1)

        # Where a description is spoken: into the pauses of the soundtrack, or
        # while the film is stopped.  The two are alternatives, so choosing one
        # takes the other's controls away instead of letting them contradict.
        self.placement_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                           name="How to place the descriptions")
        for value, label in AD_PLACEMENT_CHOICES:
            self.placement_combo.Append(label, value)
        self._select_data(
            self.placement_combo,
            self.describe_settings.get("placement", AD_DEFAULT_PLACEMENT),
            fallback=0,
        )
        self.placement_combo.SetToolTip(
            "Standard audio description speaks into the pauses of the "
            "soundtrack. Extended audio description stops the film for each "
            "description and goes on afterwards, which is the only option for "
            "a video whose dialogue never leaves a pause; it makes the film "
            "longer by the spoken time of every description.")
        self.placement_combo.Bind(wx.EVT_COMBOBOX,
                                  lambda _: self._on_placement())
        add_labeled(self, grid, "Placement", self.placement_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        # Track priority: which of the two tracks gives way when a description is
        # spoken over the film's own soundtrack.  It matters only in standard
        # placement (extended mode never overlaps the two), so choosing extended
        # switches it off rather than leaving a choice on screen that would not
        # be honoured.
        self.priority_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                          name="Track priority")
        for value, label in AD_TRACK_PRIORITY_CHOICES:
            self.priority_combo.Append(label, value)
        self._select_data(
            self.priority_combo,
            self.describe_settings.get("track_priority",
                                       AD_DEFAULT_TRACK_PRIORITY),
            # The list runs from the gentlest answer to the film to the most
            # forceful, and it opens on its first entry: a project that has
            # never chosen one - written before any of these choices existed -
            # starts on *Same volume*, which is exactly what those runs
            # produced.
            fallback=self._choice_index(AD_TRACK_PRIORITY_CHOICES,
                                        AD_DEFAULT_TRACK_PRIORITY),
        )
        self.priority_combo.SetToolTip(self._PRIORITY_TOOLTIP)
        self.priority_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._on_change())
        add_labeled(self, grid, "Track priority", self.priority_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 8)

        # Gap fitting: place each description into the silence that follows it.
        self.gap_check = wx.CheckBox(
            self,
            label="Fit descriptions into the pauses (needs a speech to text model)",
        )
        self.gap_check.SetName(
            "Fit descriptions into the pauses (needs a speech to text model)")
        self.gap_check.SetValue(
            bool(self.describe_settings.get("gap_fitting", False)))
        self.gap_check.SetToolTip(
            "Use the speech to text model to find the silent gaps and speak the "
            "description there instead of over the dialogue")
        self.gap_check.Bind(wx.EVT_CHECKBOX, lambda _: self._on_change())
        sizer.Add(self.gap_check, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Spoken narration: the same descriptions, read aloud by an installed
        # voice, written next to the script as a playable audio file.
        self.narrate_check = wx.CheckBox(
            self,
            label="Also speak the descriptions into a narration audio file",
        )
        self.narrate_check.SetName(
            "Also speak the descriptions into a narration audio file")
        # Speaking the descriptions is what makes the *audio* and the *video* of
        # the project; it was off unless the project had said otherwise, which
        # left a run that wrote only the text looking like a run that had failed
        # at the other two.  A project that recorded no choice now opens on
        # "speak them" whenever a voice is installed, and an explicit choice - in
        # either direction - is still the project's own.
        remembered_narration = self.describe_settings.get("narration")
        self.narrate_check.SetValue(
            bool(self.voices) if remembered_narration is None
            else bool(remembered_narration))
        self.narrate_check.SetToolTip(
            "Writes one spoken narration file per video next to the description "
            "script, using an offline voice. The audio format is the one chosen "
            "in Settings > Recording settings.")
        self.narrate_check.Bind(wx.EVT_CHECKBOX, lambda _: self._on_narration())
        sizer.Add(self.narrate_check, 0, wx.LEFT | wx.RIGHT, 8)

        # The voice is chosen the way the recording window chooses one - the TTS
        # model, then one of its downloaded variants, then one of that variant's
        # voices - so a machine with several engines behind the descriptions is
        # not a single flat list of unrelated voices.
        voice_grid = wx.FlexGridSizer(cols=2, vgap=8, hgap=10)
        voice_grid.AddGrowableCol(1)
        self.narration_model_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                                 name="Narration TTS model")
        self.narration_model_combo.Bind(wx.EVT_COMBOBOX,
                                        lambda _: self._on_narration_model())
        add_labeled(self, voice_grid, "Narration TTS model",
                    self.narration_model_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        self.narration_variant_combo = wx.ComboBox(
            self, style=wx.CB_READONLY, name="Narration model variant")
        self.narration_variant_combo.Bind(
            wx.EVT_COMBOBOX, lambda _: self._on_narration_variant())
        add_labeled(self, voice_grid, "Narration model variant",
                    self.narration_variant_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        self.voice_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                       name="Narration voice")
        self.voice_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._on_narration())
        add_labeled(self, voice_grid, "Narration voice", self.voice_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        # How loud the description is spoken.  It belongs to the voice chosen
        # just above, so it sits directly under it, and it is the one narration
        # level the describe window asks for on its own: *Track priority* moves
        # the film's track, never the description.
        self.narration_volume_slider, self.narration_volume_label = (
            self._narration_volume_row(voice_grid))
        sizer.Add(voice_grid, 0, wx.EXPAND | wx.ALL, 8)

        self.narration_note = wx.StaticText(self, label="")
        set_accessible_name(self.narration_note, "Narration")
        sizer.Add(self.narration_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.model_note = wx.StaticText(self, label="")
        self.model_note.Wrap(760)
        sizer.Add(self.model_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.estimate_label = wx.StaticText(self, label="")
        self.estimate_label.Wrap(760)
        set_accessible_name(self.estimate_label, "Time estimate")
        sizer.Add(self.estimate_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.status = wx.StaticText(self, label="Ready to describe.")
        set_accessible_name(self.status, "Status")
        sizer.Add(self.status, 0, wx.LEFT | wx.RIGHT | wx.TOP, 8)

        self.gauge = wx.Gauge(self, range=100, name="Audio description progress")
        sizer.Add(self.gauge, 0, wx.EXPAND | wx.ALL, 8)

        # Every chunk, as it finishes: the gauge says how far the run is, this
        # says what it has done so far - which chunk was described, which was
        # skipped because its picture had not changed, and what the description
        # says.  It is read-only and bounded, so a long film cannot flood it.
        self.chunk_log = wx.TextCtrl(
            self, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP,
            name="Chunk progress", size=wx.Size(-1, 120),
        )
        set_accessible_name(self.chunk_log, "Chunk progress")
        sizer.Add(self.chunk_log, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(self, label="&Start audio description")
        self.start_btn.SetName("Start audio description")
        self.folder_btn = wx.Button(self, label="Open project &folder")
        self.folder_btn.SetName("Open project folder")
        self.close_btn = wx.Button(self, label="&Close")
        self.close_btn.SetName("Close")
        buttons.Add(self.start_btn, 0, wx.ALL, 4)
        buttons.Add(self.folder_btn, 0, wx.ALL, 4)
        buttons.Add(self.close_btn, 0, wx.ALL, 4)
        sizer.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALL, 4)

        self.SetSizer(sizer)
        self.start_btn.Bind(wx.EVT_BUTTON, self._on_start)
        self.folder_btn.Bind(wx.EVT_BUTTON, lambda _: self._open_folder())
        self.close_btn.Bind(wx.EVT_BUTTON, self._on_close)
        self.Bind(wx.EVT_CLOSE, self._on_close)

        self._on_placement()
        self._sync_start_state()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _select_data(combo: wx.ComboBox, value, fallback: int = 0) -> None:
        for index in range(combo.GetCount()):
            if combo.GetClientData(index) == value:
                combo.SetSelection(index)
                return
        if 0 <= fallback < combo.GetCount():
            combo.SetSelection(fallback)

    # -- what the project already decided ----------------------------------
    def _current_model(self) -> str:
        """The describing model chosen when the project was created."""
        model_id = str(self.describe_settings.get("model") or "")
        if model_id in catalog.model_ids():
            return model_id
        return catalog.default_model_id()

    def _current_variant(self) -> str:
        variant = str(self.describe_settings.get("variant") or "")
        if variant:
            return variant
        return catalog.default_variant(self._current_model())

    def _selected_compute(self) -> str:
        return str(self.describe_settings.get("compute") or COMPUTE_CPU)

    def _selected_fps(self) -> float:
        try:
            return float(self.describe_settings.get("fps", AD_DEFAULT_FPS))
        except (TypeError, ValueError):
            return AD_DEFAULT_FPS

    def _selected_chunk(self) -> float:
        try:
            return float(self.describe_settings.get(
                "chunk_seconds", AD_CHUNK_CHOICES[1][0]))
        except (TypeError, ValueError):
            return AD_CHUNK_CHOICES[1][0]

    def _window_seconds(self) -> float:
        """How much video one description covers.

        A window is what decides how much a listener is told, so it is a
        first-class number here rather than a hidden one: one description per two
        minutes was measured to reduce a whole lecture to "a man sits in a chair
        beside a whiteboard".  The project may carry its own value; otherwise the
        catalogue's default (twenty seconds, the broadcast cadence) is used.
        """
        value = self.describe_settings.get("window_seconds")
        if not value:
            try:
                value = self.settings.get("describe.window_seconds",
                                          AD_WINDOW_SECONDS)
            except Exception:
                value = AD_WINDOW_SECONDS
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = AD_WINDOW_SECONDS
        return max(1.0, min(number, AD_MAX_WINDOW_SECONDS))

    def _selected_output(self) -> str:
        return str(self.describe_settings.get("output_format",
                                               AD_DEFAULT_OUTPUT))

    def _selected_track_priority(self) -> str:
        """The chosen track priority, or the default when nothing is chosen.

        A project from before this choice existed has none, and ``same`` is
        exactly what those runs did, so the fallback is honest rather than a
        silent change of behaviour.
        """
        sel = self.priority_combo.GetSelection()
        if sel < 0:
            return AD_DEFAULT_TRACK_PRIORITY
        return str(self.priority_combo.GetClientData(sel) or "")

    def _track_priority_gain(self) -> float:
        """The film soundtrack's gain for the chosen priority (1.0 for same)."""
        return float(AD_TRACK_PRIORITY_GAIN.get(
            self._selected_track_priority(), 1.0))

    def _track_priority_duck(self) -> float:
        """The level the film dips to while the description speaks.

        1.0 means "this film never dips", which every choice but *Duck, don't
        lower* returns - the one place that answers the question, so the window
        and the mixer cannot disagree about it.
        """
        return extended.track_priority_duck(self._selected_track_priority())

    def _track_priority_label(self) -> str:
        return self._choice_label(
            AD_TRACK_PRIORITY_CHOICES, self._selected_track_priority(),
            self._selected_track_priority())

    @staticmethod
    def _choice_label(choices, value, default: str) -> str:
        """The printed label of a ``(value, label)`` choice list."""
        for option, label in choices:
            if option == value:
                return label
        return default

    @staticmethod
    def _choice_index(choices, value, default: int = 0) -> int:
        """Where a value sits in a ``(value, label)`` choice list."""
        for index, (option, _label) in enumerate(choices):
            if option == value:
                return index
        return default

    def _refresh_run_label(self) -> None:
        """State the run the project was created for, without asking again.

        Every one of these was collected by the New Project wizard, so showing
        them as editable combos here duplicated that page and left two places to
        change the same setting - and only one of them was the project's own
        record.  They are read from the project and reported.
        """
        model_id = self._current_model()
        variant = self._current_variant()
        entry = catalog.model(model_id)
        variant_label = ""
        for value, label in catalog.variants(model_id):
            if value == variant:
                variant_label = label
                break
        device = self._choice_label(
            [(option["value"], option["label"])
             for option in engine.compute_options(model_id, variant)],
            self._selected_compute(), self._selected_compute().upper(),
        )
        output = self._choice_label(AD_OUTPUT_CHOICES, self._selected_output(),
                                   self._selected_output())
        if self.describe_settings.get("chunk_seconds") is None:
            chunk_label = self._choice_label(
                AD_CHUNK_CHOICES, AD_CHUNK_CHOICES[1][0], "")
        else:
            chunk_label = self._choice_label(
                AD_CHUNK_CHOICES, self._selected_chunk(),
                f"{self._selected_chunk():g} seconds",
            )
        fps_label = self._choice_label(AD_FPS_CHOICES, self._selected_fps(),
                                      f"{self._selected_fps():g}")
        state = engine.variant_state_label(model_id, variant)
        self.run_label.SetLabel(
            f"Describer: {entry['name']}"
            + (f" - {variant_label}" if variant_label else "")
            + f" ({state})\n"
            f"Run: {device} - {fps_label} - {chunk_label} - {output}\n"
            "Chosen when this project was created (New Project wizard); "
            "change the model there, and install models in Settings > Manage "
            "the video describer."
        )
        self.run_label.Wrap(760)
        self.Layout()

    def _use_shorter_chunks(self) -> None:
        """The long-video offer: describe this project in 1 minute chunks."""
        self.describe_settings["chunk_seconds"] = _SHORT_CHUNK
        self._refresh_run_label()
        self._refresh_estimate()

    # -- the narration voice -----------------------------------------------
    def _narration_models(self) -> list:
        """``[(tts_id, label)]`` for the TTS models behind the installed voices."""
        models: dict = {}
        for entry in self.voices:
            tts = str(entry.get("tts") or "")
            if tts and tts not in models:
                models[tts] = str(entry.get("tts_name") or tts)
        return list(models.items())

    def _narration_variants(self, tts_id: str) -> list:
        """``[variant_id]`` of ``tts_id`` that have a voice installed."""
        variants: list = []
        for entry in self.voices:
            if str(entry.get("tts") or "") != tts_id:
                continue
            variant = str(entry.get("variant") or "")
            if variant and variant not in variants:
                variants.append(variant)
        return variants

    def _selected_narration_model(self) -> str:
        sel = self.narration_model_combo.GetSelection()
        if sel < 0:
            return ""
        return self.narration_model_combo.GetClientData(sel) or ""

    def _selected_narration_variant(self) -> str:
        sel = self.narration_variant_combo.GetSelection()
        if sel < 0:
            return ""
        return self.narration_variant_combo.GetClientData(sel) or ""

    def _populate_voices(self, tts_id: str, variant: str, wanted: dict) -> None:
        """List the voices of one model/variant, selecting ``wanted`` if it is
        among them."""
        self.voice_combo.Clear()
        for entry in self.voices:
            if str(entry.get("tts") or "") != tts_id:
                continue
            if variant and str(entry.get("variant") or "") != variant:
                continue
            self.voice_combo.Append(self._voice_label(entry), entry)
        if not self.voice_combo.GetCount():
            return
        wanted_voice = str(wanted.get("voice") or "")
        if wanted_voice:
            for index in range(self.voice_combo.GetCount()):
                entry = self.voice_combo.GetClientData(index) or {}
                if str(entry.get("voice") or "") == wanted_voice:
                    self.voice_combo.SetSelection(index)
                    return
        self.voice_combo.SetSelection(0)

    def _populate_narration(self) -> None:
        """Fill the model -> variant -> voice cascade from the installed voices.

        The remembered voice drives every level, so a project that was created
        with Kokoro opens on Kokoro rather than on whichever engine the catalog
        happens to list first; a remembered voice that has since been removed
        falls back to the first installed one.
        """
        remembered = narration.resolve_voice(
            self.voices, self.describe_settings.get("narration_voice") or {}
        ) or (dict(self.voices[0]) if self.voices else {})
        self.narration_model_combo.Clear()
        for tts_id, label in self._narration_models():
            self.narration_model_combo.Append(label, tts_id)
        wanted_tts = str(remembered.get("tts") or "")
        index = next(
            (i for i in range(self.narration_model_combo.GetCount())
             if self.narration_model_combo.GetClientData(i) == wanted_tts), 0
        )
        if self.narration_model_combo.GetCount():
            self.narration_model_combo.SetSelection(index)
        self._on_narration_model(remembered)

    def _on_narration_model(self, remembered: dict | None = None) -> None:
        """Refill the variant and voice combos for the chosen TTS model."""
        tts_id = self._selected_narration_model()
        variants = self._narration_variants(tts_id)
        self.narration_variant_combo.Clear()
        for variant in variants:
            self.narration_variant_combo.Append(variant, variant)
        wanted = dict(remembered or self._selected_voice())
        wanted_variant = str(wanted.get("variant") or "")
        index = next(
            (i for i, variant in enumerate(variants)
             if variant == wanted_variant), 0
        )
        if self.narration_variant_combo.GetCount():
            self.narration_variant_combo.SetSelection(index)
        self._on_narration_variant(wanted)

    def _on_narration_variant(self, wanted: dict | None = None) -> None:
        """Refill the voice combo for the chosen variant."""
        self._populate_voices(
            self._selected_narration_model(),
            self._selected_narration_variant(),
            dict(wanted or self._selected_voice()),
        )
        self._sync_narration_state()

    # -- the narration voice -----------------------------------------------
    def _installed_voices(self) -> list:
        """The voices the recording side already downloaded, newest state first."""
        try:
            voices = list(self.store.installed_voices())
        except Exception:
            log.debug("could not list the installed voices", exc_info=True)
            return []
        return [dict(entry) for entry in voices if entry.get("voice")]

    @staticmethod
    def _voice_label(entry: dict) -> str:
        """``Piper - Amy (medium)``: engine, voice and variant in one line."""
        engine_name = str(entry.get("tts_name") or entry.get("tts") or "voice")
        name = str(entry.get("voice_name") or entry.get("voice") or "")
        label = f"{engine_name} - {name}" if name else engine_name
        variant = str(entry.get("variant") or "")
        return f"{label} ({variant})" if variant else label

    def _selected_voice(self) -> dict:
        """The voice entry the narration will speak with, or ``{}``."""
        sel = self.voice_combo.GetSelection()
        if sel < 0:
            return {}
        return dict(self.voice_combo.GetClientData(sel) or {})

    # -- the narration voice's volume --------------------------------------
    def _remembered_narration_volume(self) -> float:
        """The volume the narration voice should open on.

        The project's own record first (written by the wizard), then the last
        run's choice, then the narration voice the project carries, and finally
        the same recording setting the rest of the application speaks with -
        so a project is never mixed at a level nobody chose.
        """
        candidates = (
            self.describe_settings.get("narration_volume"),
            self.settings.get("describe.narration_volume", None),
            (self.project.get("tts") or {}).get("volume"),
            self.settings.get("recording.volume", None),
        )
        for value in candidates:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number == number:  # not NaN
                return max(float(VOLUME_MIN), min(number, float(VOLUME_MAX)))
        return float(AD_NARRATION_VOLUME_DEFAULT)

    def _narration_volume(self) -> float:
        """The chosen volume: the slider's value once it exists, its default
        before that."""
        slider = getattr(self, "narration_volume_slider", None)
        if slider is not None:
            return slider.GetValue() / 100.0
        return self._remembered_narration_volume()

    def _narration_volume_row(self, grid):
        """A labelled slider for how loud the narration voice speaks.

        The row is the ``(slider, label)`` pair so the label can be rewritten
        as the slider moves; the value lives in the label's own text, the shape
        the Recording window uses for its own Volume row.
        """
        value = self._remembered_narration_volume()
        label = wx.StaticText(
            self, label=f"{self._NARRATION_VOLUME_NAME}: {value:.2f}")
        label.SetName(self._NARRATION_VOLUME_NAME + " value")
        grid.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        slider = wx.Slider(
            self, minValue=int(VOLUME_MIN * 100), maxValue=int(VOLUME_MAX * 100),
            value=round(value * 100),
            name=self._NARRATION_VOLUME_NAME,
        )
        set_accessible_name(
            slider, f"{self._NARRATION_VOLUME_NAME}: {value:.2f}")
        slider.SetToolTip(
            "How loud the description is spoken, as a proportion of the voice "
            "as it is: 1.00 is the voice as downloaded, 0.50 half of it and "
            "2.00 twice it. Only the description moves here; the film's own "
            "track is set by Track priority.")
        slider.Bind(wx.EVT_SLIDER, lambda _: self._on_narration_volume())
        grid.Add(slider, 1, wx.EXPAND)
        return slider, label

    def _on_narration_volume(self) -> None:
        """The slider moved: keep the label, the accessible name and the
        run's own record in step with it."""
        value = self.narration_volume_slider.GetValue() / 100.0
        self.narration_volume_label.SetLabel(
            f"{self._NARRATION_VOLUME_NAME}: {value:.2f}")
        update_accessible_name(
            self.narration_volume_slider,
            f"{self._NARRATION_VOLUME_NAME}: {value:.2f}")
        self._on_change()

    def _selected_placement(self) -> str:
        sel = self.placement_combo.GetSelection()
        if sel < 0:
            return AD_DEFAULT_PLACEMENT
        return extended.normalise_placement(
            self.placement_combo.GetClientData(sel) or AD_DEFAULT_PLACEMENT)

    def _is_extended(self) -> bool:
        return extended.is_extended(self._selected_placement())

    def _log_line(self, message: str) -> None:
        """Append one line to the chunk report, keeping it to the last 400.

        The report is what tells the user, chunk by chunk, that the run is
        moving: with the picture-change gate on, a chunk with nothing new is
        reported as skipped rather than looking like lost work, and every
        described chunk contributes its own text.
        """
        text = str(message or "").strip()
        if not text:
            return
        try:
            lines = self.chunk_log.GetValue().splitlines()
        except Exception:
            return
        if lines and lines[-1] == text:
            return
        lines.append(text)
        del lines[:-400]
        try:
            self.chunk_log.SetValue("\n".join(lines))
            self.chunk_log.ShowPosition(self.chunk_log.GetLastPosition())
        except Exception:
            log.debug("could not write the chunk report", exc_info=True)

    def _on_chunk_done(self, segment: dict, done: int, total: int) -> None:
        """One chunk produced a description: report it, with its text."""
        text = str(segment.get("text") or "").strip()
        number = segment.get("chunk")
        prefix = f"Chunk {done} of {total}"
        if number is not None:
            prefix += f" (chunk {number})"
        self._log_line(f"{prefix} described: {text}" if text else prefix)

    def _sync_narration_state(self) -> None:
        """Enable the narration row only where it can actually be honoured.

        Extended audio description *is* spoken: the film is stopped so the voice
        can be heard, so the spoken track is switched on and locked there -
        there would be nothing to hear otherwise.
        """
        stopping = self._is_extended()
        has_voices = bool(self.voices)
        if stopping:
            self.narrate_check.SetValue(True)
        elif not has_voices and self.narrate_check.GetValue():
            # A remembered yes from a machine that no longer has the voice must
            # not promise a file that cannot be written.
            self.narrate_check.SetValue(False)
        self.narrate_check.Enable(has_voices and not stopping)
        if not has_voices:
            self.narrate_check.SetToolTip(
                "No offline voice is installed yet. Download one in Settings > "
                "Download and remove ONNX TTS, then come back to speak the descriptions.")
        on = has_voices and self.narrate_check.GetValue()
        self.narration_model_combo.Enable(on)
        self.narration_variant_combo.Enable(on)
        self.voice_combo.Enable(on)
        self.narration_volume_slider.Enable(on)
        if not has_voices:
            self.narration_note.SetLabel(
                "Extended audio description needs an offline voice and none is "
                "installed; download one in Settings > Download and remove ONNX TTS."
                if stopping else ""
            )
        elif not on:
            self.narration_note.SetLabel("")
        else:
            voice = self._selected_voice()
            label = self._voice_label(voice) if voice else "(no voice chosen)"
            if stopping:
                self.narration_note.SetLabel(
                    f"Extended audio description: the film is stopped for every "
                    f"description, spoken by {label} as a "
                    f"{self.narration_format.upper()} track, and written as a "
                    "described video next to the script."
                )
            else:
                self.narration_note.SetLabel(
                    f"Narration: {label}, written as a "
                    f"{self.narration_format.upper()} file next to the script, "
                    "and mixed into the described video (the original picture "
                    "and sound with the description over the pauses)."
                )
        self.Layout()

    def _on_placement(self) -> None:
        """The film is either fitted into or stopped: the other choice's own
        controls are switched off, because they cannot both be true."""
        stopping = self._is_extended()
        self.gap_check.Enable(not stopping)
        if stopping and self.gap_check.GetValue():
            # The pauses are what stopping the film replaces, so the tick is
            # cleared rather than left ticked and ignored.
            self.gap_check.SetValue(False)
        self.gap_check.SetToolTip(
            "The film is stopped for each description, so there is no pause to "
            "fit into. Choose standard audio description to fit the pauses "
            "instead."
            if stopping else
            "Use the speech to text model to find the silent gaps and speak the "
            "description there instead of over the dialogue")
        # Track priority balances the film's own soundtrack against the
        # description, which only happens when the two are heard together.  In
        # extended mode the film is silent while the description speaks, so
        # there is nothing to balance and the choice is switched off.
        self.priority_combo.Enable(not stopping)
        self.priority_combo.SetToolTip(
            "Extended audio description is heard while the film is stopped "
            "and its soundtrack is silent, so nothing needs rebalancing. "
            "Choose standard audio description to set the track priority."
            if stopping else
            self._PRIORITY_TOOLTIP)
        self._sync_narration_state()
        self._refresh_estimate()

    def _on_narration(self) -> None:
        self._sync_narration_state()

    def _tts_params(self) -> dict:
        """Rate, pitch, volume and punctuation for the narration voice.

        An audio description project is created without recording settings, so
        the Recording window's own defaults stand in until the project is
        recorded into as well.
        """
        tts = self.project.get("tts") or {}

        def _number(value, default):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        return {
            "rate": _number(tts.get("rate"),
                            self.settings.get("recording.rate", 1.0)),
            "pitch": _number(tts.get("pitch"),
                             self.settings.get("recording.pitch", 1.0)),
            # The volume is the slider's, not the project's: it is the one
            # narration level this window asks for on its own.
            "volume": self._narration_volume(),
            "punctuation": str(tts.get("punctuation", "default")),
            "compute": str(tts.get("compute") or self.settings.compute),
        }

    # -- events ------------------------------------------------------------
    def _total_duration(self) -> float:
        """Known total length of the sources (0 when nothing could be probed)."""
        return media.total_duration(self.sources)

    def _on_change(self) -> None:
        """Refresh the notes, the estimate and the Start button."""
        self._refresh_notes()
        self._refresh_estimate()
        self._sync_start_state()

    def _compute_problem(self) -> str:
        """Why the chosen device cannot run here, or ``""`` when it can.

        The device is the project's, chosen in the wizard, so the way out is
        named: create the project again with the CPU, or install the model's GPU
        runtime - not a combo this window no longer offers.
        """
        if self._selected_compute() != "cuda" or engine.cuda_available():
            return ""
        if catalog.gpu_only(self._current_model(), self._current_variant()):
            return (
                "This model was chosen for an NVIDIA GPU and none was detected. "
                "Create the project again with a CPU-capable model (New Project "
                "wizard), or run it on a machine with the card it needs."
            )
        return (
            "No NVIDIA GPU was detected, so the GPU this project was created "
            "for cannot run here. Install the model's GPU runtime in Settings > "
            "Manage the video describer, or create the project again with the "
            "CPU (New Project wizard)."
        )

    def _refresh_notes(self) -> None:
        model_id = self._current_model()
        variant = self._current_variant()
        entry = catalog.model(model_id)
        lines = [
            f"{entry['name']}: {entry['license']}. "
            f"Download {catalog.variant_size_gb(model_id, variant):.1f} GB, "
            f"needs about {catalog.variant_vram_gb(model_id, variant):.1f} GB "
            "of video memory. "
            f"{entry['notes']}",
            "Hallucination: " + catalog.hallucination_note(model_id),
            engine.availability_note(model_id),
        ]
        self.model_note.SetLabel("\n".join(lines))
        self.model_note.Wrap(760)
        self.Layout()

    def _refresh_estimate(self) -> None:
        duration = self._total_duration()
        if duration <= 0:
            self.estimate_label.SetLabel(
                "The length of the video could not be read, so no time "
                "estimate is available. It will still be described in chunks."
            )
            self.estimate_label.Wrap(760)
            return
        info = pipeline.estimate(
            duration,
            model_id=self._current_model(),
            variant=self._current_variant(),
            fps=self._selected_fps(),
            device=self._selected_compute(),
            chunk_seconds=self._selected_chunk(),
            window_seconds=self._window_seconds(),
        )
        taken = pipeline.format_duration(info["seconds"])
        lines = [
            f"Estimated time: about {taken} for "
            f"{pipeline.format_duration(duration)} of video, in "
            f"{info['chunks']} chunk(s) at {info['fps']:g} frames per second "
            f"({info['frames']} frames)."
        ]
        if pipeline.needs_warning(duration):
            lines.append(
                "This is a long video: describing it takes far longer than "
                "transcribing it. Shorter chunks keep memory use low and let "
                "you stop and continue."
            )
        # What is about to be produced, stated rather than implied: the text, the
        # spoken track and the film itself.  A run that quietly wrote two of the
        # three looked like a run that had failed at one of them.
        lines.append(
            "Each one is described in pieces of about "
            f"{self._window_seconds():g} seconds - {info['descriptions']} "
            "description(s) for this video - so a change in the picture is "
            "reported instead of being summarised away."
        )
        if self._is_extended():
            # The honest part of the extended bargain: the film gets longer by
            # the spoken time of every description.
            spoken = info["descriptions"] * AD_EXTENDED_SECONDS_PER_DESCRIPTION
            lines.append(
                "Extended audio description stops the film for every "
                f"description: about {pipeline.format_duration(spoken)} of "
                "speaking, so the described film becomes about "
                f"{pipeline.format_duration(duration + spoken)} long. The "
                "spoken track is written on the described film's clock, and the "
                "described video next to the script."
            )
        elif self.narrate_check.GetValue():
            lines.append(
                "Three files per video: the description script, the spoken "
                "narration track, and a described video - the original picture "
                "with the original sound and the description heard together. "
                "The source video is never changed."
            )
            gain = self._track_priority_gain()
            dip = self._track_priority_duck()
            line = "Track priority: " + self._track_priority_label() + "."
            if dip < 1.0:
                # The ducking is the one choice that says *when* the film's own
                # sound moves, so the estimate says when as well: a fraction for
                # the length of a description reads as "quieter film", which is
                # the choice the user did not make.
                line += (
                    " The description keeps the level your narration settings "
                    "gave it, and the film's own sound is mixed at "
                    f"{dip:.0%} of its own level only while a description is "
                    "actually being spoken - between descriptions it is left "
                    "exactly as it is."
                )
            elif gain < 1.0:
                line += (
                    " The description keeps the level your narration settings "
                    f"gave it and the film's own soundtrack is mixed at "
                    f"{gain:.0%} of it."
                )
            lines.append(line)
        self.estimate_label.SetLabel("\n".join(lines))
        self.estimate_label.Wrap(760)
        self.Layout()

    def _sync_start_state(self) -> None:
        if self.job is not None and self.job.is_alive():
            return
        if not self.sources:
            announce.status_error(self.status, "This project has no video files.")
            self.start_btn.Disable()
            return
        problem = self._compute_problem()
        if problem:
            announce.status_error(self.status, problem)
            self.start_btn.Disable()
            return
        self.start_btn.Enable()
        announce.status_text(self.status, "Ready to describe.")

    # -- options -----------------------------------------------------------
    def _options(self) -> dict:
        model_id = self._current_model()
        stt_model = ""
        stt_variant = ""
        if self.gap_check.GetValue():
            try:
                stt_model = str(self.settings.get("stt.model", "") or "")
                stt_variant = str(self.settings.get("stt.variant", "") or "")
            except Exception:
                stt_model = ""
        return {
            "model": model_id,
            "variant": self._current_variant(),
            "compute": self._selected_compute(),
            "fps": self._selected_fps(),
            "chunk_seconds": self._selected_chunk(),
            "window_seconds": self._window_seconds(),
            "output_format": self._selected_output(),
            "gap_fitting": bool(self.gap_check.GetValue()),
            "min_gap": float(self.describe_settings.get(
                "min_gap", AD_MIN_GAP_SECONDS)),
            "max_description": float(self.describe_settings.get(
                "max_description", AD_MAX_DESCRIPTION_SECONDS)),
            "language": "en",
            "speech_model": stt_model,
            "speech_variant": stt_variant,
            "narration": bool(self.narrate_check.GetValue()),
            "voice": self._selected_voice(),
            "narration_format": self.narration_format,
            "tts": self._tts_params(),
            "placement": self._selected_placement(),
            "track_priority": self._selected_track_priority(),
        }

    def _remember(self, options: dict) -> None:
        try:
            self.settings.set("describe.model", options["model"])
            self.settings.set("describe.variant", options["variant"])
            self.settings.set("describe.compute", options["compute"])
            self.settings.set("describe.fps", options["fps"])
            self.settings.set("describe.chunk_seconds", options["chunk_seconds"])
            self.settings.set("describe.window_seconds",
                              options.get("window_seconds", AD_WINDOW_SECONDS))
            self.settings.set("describe.output_format", options["output_format"])
            self.settings.set("describe.gap_fitting", options["gap_fitting"])
            self.settings.set("describe.narration", bool(options.get("narration")))
            self.settings.set("describe.narration_volume",
                              self._narration_volume())
            self.settings.set("describe.placement", options["placement"])
            self.settings.set("describe.track_priority",
                              options.get("track_priority",
                                          AD_DEFAULT_TRACK_PRIORITY))
            voice = options.get("voice") or {}
            self.settings.set("describe.narration_voice", {
                "tts": voice.get("tts"),
                "language": voice.get("language"),
                "variant": voice.get("variant"),
                "voice": voice.get("voice"),
            })
            self.settings.save()
        except Exception:
            log.debug("could not remember audio description choices", exc_info=True)

    # -- running -----------------------------------------------------------
    def _long_video_gate(self, options: dict) -> bool:
        """Warn about a long video; True when the run may proceed.

        The warning offers *shorter chunks*, because that is the one setting
        that changes what the run costs and how well it can be stopped - so the
        offer is an action, not a paragraph.
        """
        duration = self._total_duration()
        if not pipeline.needs_warning(duration, AD_WARN_SECONDS):
            return True
        info = pipeline.estimate(
            duration,
            model_id=self._current_model(),
            variant=self._current_variant(),
            fps=self._selected_fps(),
            device=self._selected_compute(),
            chunk_seconds=self._selected_chunk(),
            window_seconds=self._window_seconds(),
        )
        chunked = options["chunk_seconds"] <= _SHORT_CHUNK
        message = pipeline.warning_message(duration, info, chunked=chunked)
        actions = [("go", "Continue as it is"), ("cancel", "Cancel")]
        if not chunked:
            actions.insert(0, ("chunk", "Use 1 minute chunks and continue"))
        kept = str(self.settings.get("stt.model", "") or "").strip()
        if options["gap_fitting"] and not kept:
            message += (
                "\n\nGap fitting needs a speech to text model; none is chosen "
                "yet, so descriptions will go at their chunk boundaries."
            )
        key = run_action_dialog(
            self, "This video is long", message, actions, default_key="chunk",
        )
        if key is None or key == "cancel":
            return False
        if key == "chunk":
            self._use_shorter_chunks()
            options["chunk_seconds"] = self._selected_chunk()
        return True

    def _on_start(self, _) -> None:
        if self.job is not None and self.job.is_alive():
            return
        if not self.sources:
            wx.MessageBox("This project has no video files.", "Audio description",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        model_id = self._current_model()
        if not self._ensure_ffmpeg():
            return
        if not engine.model_available(model_id):
            wx.MessageBox(
                f"{catalog.model(model_id)['name']} is not installed yet.\n\n"
                "Open Settings > Manage the video describer and install it "
                "there (each model gets its own Python environment), then come "
                "back and press Start audio description.",
                "Audio description", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        if self.narrate_check.GetValue() and not self._selected_voice():
            if self._is_extended():
                message = (
                    "Extended audio description stops the film for every "
                    "description and speaks it, so it needs an offline voice "
                    "and none is installed yet.\n\nDownload one in Settings > "
                    "Download and remove ONNX TTS, then try again - or choose standard "
                    "audio description and speak into the pauses instead."
                )
            else:
                message = (
                    "Speaking the descriptions needs an offline voice, and none "
                    "is installed yet.\n\nDownload one in Settings > Download "
                    "and remove, then try again - or clear the narration option "
                    "to write only the text."
                )
            wx.MessageBox(message, "Audio description",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        options = self._options()
        if not self._long_video_gate(options):
            return
        self._remember(options)
        self.start_btn.Disable()
        self.gauge.SetValue(0)
        # A fresh report per run: the previous run's chunks are its own record.
        self.chunk_log.SetValue("")
        self._log_line(
            f"Starting: {len(self.sources)} video file(s), "
            f"{catalog.model(model_id)['name']} as the describer."
        )
        announce.status_text(self.status, "Starting...")
        self._show_progress()

        self.job = DescribeJob(
            self.project_dir,
            self.sources,
            options,
            on_progress=lambda message, fraction: wx.CallAfter(
                self._update_progress, message, fraction),
            on_chunk=lambda segment, done, total: wx.CallAfter(
                self._on_chunk_done, segment, done, total),
            on_done=lambda outputs: wx.CallAfter(self._on_done, outputs),
            on_error=lambda exc: wx.CallAfter(self._on_error, exc),
        )
        self.job.start()

    # -- modal progress ----------------------------------------------------
    # -- FFmpeg ------------------------------------------------------------
    def _ensure_ffmpeg(self) -> bool:
        """Make sure FFmpeg is installed, offering to fetch it; True when it is.

        FFmpeg is what decodes the video to sample its frames, what reads its
        length, and what writes the described video, so a run without it cannot
        start at all - and the answer is an offer, not an error message: the
        application can download it into the user's folder, which is what the
        recording side already offers for MP3 and FLAC.  The fetch runs on a
        thread and the user presses Start again when it lands, exactly as it does
        there, so this window is never frozen while it downloads.
        """
        from ..audio import ffmpeg as ffmpeg_mod

        try:
            if ffmpeg_mod.ensure_ffmpeg():
                return True
        except Exception:
            log.debug("could not look for FFmpeg", exc_info=True)
        answer = wx.MessageBox(
            "Describing a video needs FFmpeg. It is not installed yet.\n\n"
            "FFmpeg decodes the video so its frames can be described, reads how "
            "long the video is, and writes the described video next to the "
            "script.\n\nDownload FFmpeg now? It will be saved to your user "
            "folder (AppData\\Roaming\\AIVoiceStudio\\ffmpeg) and installed for "
            "every part of the application.",
            "FFmpeg required",
            style=wx.YES_NO | wx.ICON_QUESTION,
        )
        if answer != wx.YES:
            announce.status_text(self.status,
                "FFmpeg is required to describe a video and to write the "
                "described video."
            )
            return False
        self.start_btn.Disable()
        announce.status_text(self.status, "Downloading FFmpeg...")
        self._log_line("Downloading FFmpeg...")
        threading.Thread(target=self._download_ffmpeg_job, daemon=True,
                         name="aivs-ffmpeg").start()
        return False

    def _download_ffmpeg_job(self) -> None:
        from ..audio import ffmpeg as ffmpeg_mod

        try:
            ffmpeg_mod.download_ffmpeg()
            wx.CallAfter(self._ffmpeg_done, None)
        except Exception as exc:
            wx.CallAfter(self._ffmpeg_done, f"FFmpeg download failed: {exc}")

    def _ffmpeg_done(self, error: str | None) -> None:
        self.start_btn.Enable()
        if error:
            announce.status_error(self.status, error)
            wx.MessageBox(error, "FFmpeg", style=wx.OK | wx.ICON_ERROR)
            return
        announce.status_text(self.status, "FFmpeg installed. Press Start audio description again.")
        self._log_line("FFmpeg installed. Press Start audio description again.")
        wx.MessageBox(
            "FFmpeg installed. Press Start audio description again.",
            "FFmpeg", style=wx.OK | wx.ICON_INFORMATION,
        )

    def _show_progress(self) -> None:
        if self._progress_dlg is not None:
            return
        self._progress_dlg = TaskProgressDialog(
            self,
            title="Describing the video",
            message="Starting...",
            cancel_label="Cancel audio description",
            on_cancel=self._cancel_job,
        )
        self._progress_dlg.Show()
        self._progress_dlg.start_pulse()

    def _finish_progress(self) -> None:
        dlg = self._progress_dlg
        self._progress_dlg = None
        if dlg is not None:
            dlg.finish()

    def _cancel_job(self) -> None:
        if self.job is not None and self.job.is_alive():
            self.job.cancel()
            announce.status_text(self.status,
                "Cancelling... the current chunk will finish first; chunks "
                "already described are kept."
            )

    def _update_progress(self, message: str, fraction: float) -> None:
        announce.status_text(self.status, message or "Describing...")
        # The chunk-by-chunk report: every progress message the run produces
        # lands here, so a long film shows which chunk it is on and which chunks
        # were skipped because their picture had not changed.
        self._log_line(message)
        try:
            value = int(max(0.0, min(1.0, fraction)) * 100)
            self.gauge.SetValue(value)
        except Exception:
            return
        dlg = self._progress_dlg
        if dlg is not None:
            dlg.stop_pulse()
            dlg.update(value, message)

    def _on_done(self, outputs) -> None:
        # The job's warnings are read before it is forgotten: a narration track
        # that could not be spoken has to be said out loud, not swallowed.
        warnings = list(getattr(self.job, "warnings", []) or [])
        self.job = None
        self._finish_progress()
        self.gauge.SetValue(100)
        total = sum(int(item.get("described", 0)) for item in outputs or [])
        written = len(outputs or [])
        narrated = [item for item in (outputs or []) if item.get("narration")]
        films = [item for item in (outputs or []) if item.get("film")]
        status = f"Finished. {total} description(s) written for {written} file(s)."
        if films:
            status += " Described video: " + ", ".join(
                item["film"] for item in films) + "."
        elif narrated:
            files = ", ".join(item["narration"]["file"] for item in narrated)
            status += f" Narration written: {files}."
        announce.status_text(self.status, status)
        self.start_btn.Enable()
        self._open_folder_offer(total, written, narrated, warnings, films)

    def _open_folder_offer(self, total: int, written: int,
                           narrated: list | None = None,
                           warnings: list | None = None,
                           films: list | None = None) -> None:
        lines = [
            "Audio description finished.",
            f"{total} description(s) for {written} file(s) were written to:",
            self.project_dir,
            "",
        ]
        if films:
            stopped = any(item.get("placement") == AD_PLACEMENT_EXTENDED
                          for item in films)
            lines.append(
                "Described video (the film with a pause for every "
                "description):" if stopped else
                "Described video (the original picture and sound with the "
                "description over the pauses):"
            )
            for item in films:
                lines.append(f"  {item['film']}")
            lines.append("")
        if narrated:
            lines.append("Spoken narration:")
            for item in narrated:
                info = item["narration"]
                extended_run = info.get("placement") == AD_PLACEMENT_EXTENDED
                note = " on the described film's clock" if extended_run else ""
                lines.append(
                    f"  {info['file']} - {info['spoken']} spoken "
                    f"description(s), {info['seconds']:.0f} second(s){note}"
                )
            lines.append("")
        lines.append(
            "The description script is an AI draft: review and edit it as needed."
        )
        if warnings:
            lines.append("")
            lines.append("Warnings and notes:")
            lines.extend(f"  {warning}" for warning in warnings)
        key = run_action_dialog(
            self,
            "Audio description complete",
            "\n".join(lines),
            [("open", "Open project folder"), ("ok", "OK")],
            default_key="ok",
        )
        if key == "open":
            self._open_folder()

    def _open_folder(self) -> None:
        open_folder(self.project_dir)

    def _on_error(self, exc: BaseException) -> None:
        self.job = None
        self._finish_progress()
        self.start_btn.Enable()
        announce.status_error(self.status, f"Failed: {exc}")
        wx.MessageBox(str(exc), "Audio description failed",
                      style=wx.OK | wx.ICON_ERROR)

    def _on_close(self, _) -> None:
        if self.job is not None and self.job.is_alive():
            self.job.cancel()
        self._finish_progress()
        self.EndModal(wx.ID_CLOSE)
