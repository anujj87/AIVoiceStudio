"""Speech-to-Text transcribe dialog.

The final step of a transcription project (and what *Open* on a recent
transcription project reopens).  It offers the compute back-end, the model and
its variant, the spoken language, the decode accuracy, and - when the model can
translate - the translation target.  *Start transcribe* runs the job in the
background and writes the documents; *Close* ends the dialog.

Screen-reader follows SPEC 3.2: every control is labelled with an accessible
name, the status line is a real ``StaticText``, and the progress is a gauge.
"""

from __future__ import annotations

import logging
import os

import wx

from .. import project
from ..constants import COMPUTE_CPU
from ..settings import Settings
from ..stt import catalog, engine
from ..stt.job import TranscriptionJob
from ..tts.models import ModelStore
from . import announce, dpi
from .a11y import add_labeled, finalize_accessibility, set_accessible_name
from .dialogs import open_folder, run_action_dialog
from .progress import TaskProgressDialog

log = logging.getLogger(__name__)


class STTTranscribeDialog(wx.Dialog):
    """Choose how to transcribe a project and run the job."""

    def __init__(self, parent, project_dir: str, settings: Settings,
                 store: ModelStore, sources: list | None = None):
        super().__init__(
            parent,
            title="Transcribe",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(760, 560),
        )
        self.project_dir = project_dir
        self.settings = settings
        self.store = store
        self.project = project.load_project(project_dir)
        self.stt_settings = dict(self.project.get("stt", {}) or {})
        # ``sources`` restricts the run to a subset (Edit > Start Selected
        # Files); ``None`` transcribes every source of the project.
        self.sources = list(sources) if sources else self._collect_sources()
        self.job: TranscriptionJob | None = None
        self._progress_dlg: TaskProgressDialog | None = None

        name = self.project.get("name") or os.path.basename(project_dir)
        self.SetTitle(f"Transcribe - {name}")
        self._build_ui(name)
        self._populate_models()
        finalize_accessibility(self)
        dpi.prepare(self)
        dpi.follow(self)
        wx.CallAfter(self.model_combo.SetFocus)

    # -- data --------------------------------------------------------------
    def _collect_sources(self) -> list[str]:
        sources = self.stt_settings.get("sources") or []
        if sources:
            return [s for s in sources if s]
        found = [seg.get("source") for seg in self.project.get("segments", [])]
        return [s for s in found if s]

    # -- construction ------------------------------------------------------
    def _build_ui(self, name: str) -> None:
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label=f"Project: {name}"),
            0, wx.ALL, 8,
        )
        count = len(self.sources)
        sizer.Add(
            wx.StaticText(
                self,
                label=f"{count} source file(s) will be transcribed.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8,
        )

        grid = wx.FlexGridSizer(cols=2, vgap=8, hgap=10)
        grid.AddGrowableCol(1)

        # Compute back-end.  The choices follow the chosen model (see
        # ``_refresh_compute``): a model is offered exactly the device it is
        # for, so this combo is filled in by ``_on_model``.
        self.compute_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Compute")
        self.compute_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._on_compute())
        add_labeled(self, grid, "Compute", self.compute_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        # Model.
        self.model_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                       name="Speech to text model")
        self.model_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._on_model())
        add_labeled(self, grid, "Speech to text model", self.model_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        # Variant.
        self.variant_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Model variant")
        self.variant_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._on_variant())
        add_labeled(self, grid, "Variant", self.variant_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        # Language.
        self.language_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                          name="Language")
        add_labeled(self, grid, "Language", self.language_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        # Accuracy.
        self.accuracy_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                          name="Accuracy")
        for value, label in catalog.ACCURACY_CHOICES:
            self.accuracy_combo.Append(label, value)
        self._select_data(
            self.accuracy_combo,
            self.stt_settings.get("accuracy", catalog.DEFAULT_ACCURACY),
            fallback=1,
        )
        self.accuracy_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._refresh_notes())
        add_labeled(self, grid, "Accuracy", self.accuracy_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        # Translation.
        self.translate_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                           name="Translation")
        add_labeled(self, grid, "Translation", self.translate_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 8)

        self.model_note = wx.StaticText(self, label="")
        self.model_note.Wrap(700)
        sizer.Add(self.model_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.accuracy_note = wx.StaticText(self, label="")
        self.accuracy_note.Wrap(700)
        sizer.Add(self.accuracy_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.status = wx.StaticText(self, label="Ready to transcribe.")
        set_accessible_name(self.status, "Status")
        sizer.Add(self.status, 0, wx.LEFT | wx.RIGHT | wx.TOP, 8)

        self.gauge = wx.Gauge(self, range=100, name="Transcription progress")
        sizer.Add(self.gauge, 0, wx.EXPAND | wx.ALL, 8)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(self, label="&Start transcribe")
        self.start_btn.SetName("Start transcribe")
        self.close_btn = wx.Button(self, label="&Close")
        self.close_btn.SetName("Close")
        buttons.Add(self.start_btn, 0, wx.ALL, 4)
        buttons.Add(self.close_btn, 0, wx.ALL, 4)
        sizer.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALL, 4)

        self.SetSizer(sizer)
        self.start_btn.Bind(wx.EVT_BUTTON, self._on_start)
        self.close_btn.Bind(wx.EVT_BUTTON, self._on_close)
        self.Bind(wx.EVT_CLOSE, self._on_close)

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

    def _populate_models(self) -> None:
        current = self.stt_settings.get("model", "whisper_large_v3")
        for model_id in catalog.model_ids():
            label = catalog.model(model_id)["name"]
            # Make it obvious which models still have to be installed (from
            # Settings > Manage the STT) instead of presenting every catalog
            # entry as if it were already downloaded.
            if not engine.model_available(model_id):
                label += " (not installed)"
            self.model_combo.Append(label, model_id)
        self._select_data(self.model_combo, current, fallback=0)
        self._on_model()

    def _current_model(self) -> str:
        sel = self.model_combo.GetSelection()
        return (self.model_combo.GetClientData(sel) if sel >= 0
                else catalog.model_ids()[0])

    def _current_variant(self) -> str:
        sel = self.variant_combo.GetSelection()
        if sel < 0:
            return catalog.default_variant(self._current_model())
        return self.variant_combo.GetClientData(sel) or ""

    def _selected_compute(self) -> str:
        sel = self.compute_combo.GetSelection()
        if sel < 0:
            return COMPUTE_CPU
        return self.compute_combo.GetClientData(sel) or COMPUTE_CPU

    def _on_model(self) -> None:
        model_id = self._current_model()
        # Variants first: a variant decides whether the model can run on a CPU
        # at all (Canary-Qwen's 2.5B cannot, its 1B can), so the Compute combo
        # is built from the variant that is then selected.
        self.variant_combo.Clear()
        for value, label in catalog.variants(model_id):
            self.variant_combo.Append(label, value)
        self._select_data(
            self.variant_combo,
            self.stt_settings.get("variant") or catalog.default_variant(model_id),
            fallback=0,
        )
        self._refresh_compute(model_id)
        # Languages.
        self.language_combo.Clear()
        for value, label in catalog.languages(model_id):
            self.language_combo.Append(label, value)
        self._select_data(
            self.language_combo,
            self.stt_settings.get("language", "auto"),
            fallback=0,
        )
        # Translation.
        self.translate_combo.Clear()
        if catalog.supports_translation(model_id):
            self.translate_combo.Append("Don't translate", "")
            for value, label in catalog.translation_targets(model_id):
                self.translate_combo.Append(label, value)
            self.translate_combo.Enable(True)
        else:
            self.translate_combo.Append(
                "Not applicable - this model does not translate", "")
            self.translate_combo.Enable(False)
        self.translate_combo.SetSelection(0)
        self._refresh_notes()

    def _on_variant(self) -> None:
        """A variant can change which devices the model may run on."""
        self._refresh_compute(self._current_model())
        self._refresh_notes()

    def _on_compute(self) -> None:
        self._sync_start_state()
        self._refresh_notes()

    def _refresh_compute(self, model_id: str) -> None:
        """Fill the compute combo with the devices this model/variant can use.

        Only a model that genuinely cannot run without a GPU is offered the
        GPU alone; every other model offers both the CPU and the GPU (the
        preferred one first) whenever the machine has an NVIDIA card, and the
        CPU alone when it does not.  ``Auto`` is never offered, because it
        would quietly pick the device instead of the user.
        """
        remembered = self.stt_settings.get("compute", "")
        self.compute_combo.Clear()
        for option in engine.compute_options(model_id, self._current_variant()):
            self.compute_combo.Append(option["label"], option["value"])
        if not self.compute_combo.GetCount():
            self.compute_combo.Append("CPU", COMPUTE_CPU)
        self._select_data(self.compute_combo, remembered, fallback=0)
        self._sync_start_state()

    def _compute_problem(self) -> str:
        """Why the chosen device cannot run here, or ``""`` when it can.

        A GPU-only model on a machine with no NVIDIA card has exactly one
        option and it cannot work; saying so on the status line is kinder than
        starting a run that is certain to fail.
        """
        if self._selected_compute() != "cuda" or engine.cuda_available():
            return ""
        if catalog.gpu_only(self._current_model(), self._current_variant()):
            return (
                "This model needs an NVIDIA GPU and none was detected. "
                "Choose a CPU-capable model to transcribe on this machine."
            )
        return "No NVIDIA GPU was detected, so a GPU run is not possible."

    def _sync_start_state(self) -> None:
        """Enable Start only when there is work and the device is usable."""
        if self.job is not None and self.job.is_alive():
            return
        if not self.sources:
            announce.status_text(self.status, "This project has no source files.")
            self.start_btn.Disable()
            return
        problem = self._compute_problem()
        if problem:
            announce.status_text(self.status, problem, key="transcribe-state")
            self.start_btn.Disable()
            return
        self.start_btn.Enable()
        announce.status_text(self.status, "Ready to transcribe.")

    def _refresh_notes(self) -> None:
        model_id = self._current_model()
        entry = catalog.model(model_id)
        lines = [
            f"{entry['name']}: {entry['fit_label']} - {entry['notes']}. "
            f"{engine.availability_note(model_id)}"
        ]
        missing = engine.gpu_runtime_missing(model_id)
        if missing and self._selected_compute() == "cuda":
            lines.append(
                "The CUDA runtime this model needs is not installed in its "
                "environment (" + ", ".join(missing) + "), so a GPU run falls "
                "back to the CPU. Install or repair this model from Settings "
                "> Manage the STT to use the GPU."
            )
        self.model_note.SetLabel("\n".join(lines))
        self.model_note.Wrap(700)
        hint = catalog.accuracy_hint(model_id)
        self.accuracy_note.SetLabel(f"Model accuracy: {hint}" if hint else "")
        self.accuracy_note.Wrap(700)
        self.Layout()

    def _options(self) -> dict:
        variant_sel = self.variant_combo.GetSelection()
        language_sel = self.language_combo.GetSelection()
        translate_sel = self.translate_combo.GetSelection()
        compute_sel = self.compute_combo.GetSelection()
        accuracy_sel = self.accuracy_combo.GetSelection()
        return {
            "model": self._current_model(),
            "variant": (self.variant_combo.GetClientData(variant_sel)
                        if variant_sel >= 0 else ""),
            "language": (self.language_combo.GetClientData(language_sel)
                         if language_sel >= 0 else "auto"),
            "accuracy": (self.accuracy_combo.GetClientData(accuracy_sel)
                         if accuracy_sel >= 0 else catalog.DEFAULT_ACCURACY),
            "translate_to": (self.translate_combo.GetClientData(translate_sel)
                             if translate_sel >= 0 else ""),
            "compute": (self.compute_combo.GetClientData(compute_sel)
                        if compute_sel >= 0 else COMPUTE_CPU),
            "output_format": self.stt_settings.get("output_format", "txt"),
            "include_timestamp": self.stt_settings.get("include_timestamp", False),
            "split_mode": self.stt_settings.get("split_mode", "per_source"),
            "minutes": self.stt_settings.get("minutes", 30),
        }

    # -- events ------------------------------------------------------------
    def _on_start(self, _) -> None:
        if self.job is not None and self.job.is_alive():
            return
        if not self.sources:
            wx.MessageBox("This project has no source files.", "Transcribe",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        options = self._options()
        self._remember(options)
        self.start_btn.Disable()
        self.gauge.SetValue(0)
        announce.status_text(self.status, "Starting...", key="transcribe")
        self._show_progress()

        self.job = TranscriptionJob(
            self.project_dir,
            self.sources,
            options,
            on_progress=lambda message, fraction: wx.CallAfter(
                self._update_progress, message, fraction),
            on_done=lambda outputs: wx.CallAfter(self._on_done, outputs),
            on_error=lambda exc: wx.CallAfter(self._on_error, exc),
        )
        self.job.start()

    def _remember(self, options: dict) -> None:
        try:
            self.settings.set("stt.model", options["model"])
            self.settings.set("stt.variant", options["variant"])
            self.settings.set("stt.language", options["language"])
            self.settings.set("stt.accuracy", options["accuracy"])
            self.settings.set("stt.translate_to", options["translate_to"])
            # The device belongs to the *model* (a GPU model is transcribed on
            # the GPU, a CPU model on the CPU), so it is remembered under
            # ``stt.compute`` and never written into the global ``compute``
            # setting - that one decides what the speech *engines* use and
            # must not be changed by a transcription.
            self.settings.set("stt.compute", options["compute"])
            self.settings.save()
        except Exception:
            log.debug("could not remember STT choices", exc_info=True)

    # -- modal progress ----------------------------------------------------
    def _show_progress(self) -> None:
        """Show the modal progress dialog for the running transcription.

        A long model load plus a multi-file run needs a visible, cancellable
        window of its own (the recording flow has one); the in-dialog gauge
        stays as the accessible status line for screen readers.
        """
        if self._progress_dlg is not None:
            return
        self._progress_dlg = TaskProgressDialog(
            self,
            title="Transcribing",
            message="Starting...",
            cancel_label="Cancel transcription",
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
            announce.status_text(
            self.status, "Cancelling... the current file will finish first.",
            key="transcribe",
        )

    def _update_progress(self, message: str, fraction: float) -> None:
        announce.status_text(self.status, message or "Transcribing...",
                             key="transcribe")
        try:
            value = int(max(0.0, min(1.0, fraction)) * 100)
            self.gauge.SetValue(value)
        except Exception:
            return
        dlg = self._progress_dlg
        if dlg is not None:
            # Real progress replaces the indeterminate pulse as soon as the
            # job reports a fraction.
            dlg.stop_pulse()
            dlg.update(value, message)

    def _on_done(self, outputs) -> None:
        self.job = None
        self._finish_progress()
        self.gauge.SetValue(100)
        written = len(outputs or [])
        announce.status_text(self.status,
                             f"Finished. {written} document(s) written.")
        self.start_btn.Enable()
        key = run_action_dialog(
            self,
            "Transcription complete",
            f"Transcription finished.\n{written} document(s) were written to:\n"
            f"{self.project_dir}",
            [("open", "Open project folder"), ("ok", "OK")],
            default_key="ok",
        )
        if key == "open":
            open_folder(self.project_dir)

    def _on_error(self, exc: BaseException) -> None:
        self.job = None
        self._finish_progress()
        self.start_btn.Enable()
        announce.status_error(self.status, f"Failed: {exc}")
        wx.MessageBox(str(exc), "Transcription failed",
                      style=wx.OK | wx.ICON_ERROR)

    def _on_close(self, _) -> None:
        if self.job is not None and self.job.is_alive():
            self.job.cancel()
        self._finish_progress()
        self.EndModal(wx.ID_CLOSE)
