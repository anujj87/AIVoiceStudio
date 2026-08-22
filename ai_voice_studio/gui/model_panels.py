"""Settings panels that manage TTS models (SPEC 3.1).

* ``DownloadPanel``      -- Download and remove tab (TTS -> language -> variant).
* ``AvailablePanel``     -- Available TTS tab (TTS -> language -> variant -> voice).
* ``VoiceClonePanel``    -- Voice clone tab (TTS -> language -> variant + upload).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import wx

from ..qwen import QWEN_LANGUAGES
from ..tts import catalog
from ..tts.downloader import (
    DownloadCancelled,
    DownloadError,
    ModelDownloader,
)
from ..tts.models import ModelStore
from .a11y import add_labeled
from .events import (
    DownloadFinishedEvent,
    DownloadProgressEvent,
    EVT_DOWNLOAD_FINISHED,
    EVT_DOWNLOAD_PROGRESS,
)

log = logging.getLogger(__name__)


class _ManagerPanel(wx.Panel):
    """Base for the model-manager panels used as Settings categories.

    These panels act immediately (they manage downloads, lists and cloned
    voices) so ``apply_to_settings`` is a no-op; they still provide the
    ``title`` / ``description`` / show-hide lifecycle the settings dialog
    expects (NVDA ``SettingsPanel`` pattern).
    """

    title = ""
    description = ""

    def on_activated(self):
        self.Show()

    def on_deactivated(self):
        self.Hide()

    def apply_to_settings(self):
        pass

    def isValid(self) -> bool:
        """Validate this panel (NVDA SettingsPanel.isValid). Always OK for model panels."""
        return True


# ---------------------------------------------------------------------------
# Combo-box helpers (cascading TTS -> language -> variant -> voice)
# ---------------------------------------------------------------------------
def populate_tts(combo: wx.ComboBox, tts_list, include_none: bool = False) -> None:
    combo.Clear()
    if include_none:
        combo.Append("Select a TTS engine", None)
    for tts in tts_list:
        combo.Append(tts["name"], tts["id"])
    if combo.GetCount():
        combo.SetSelection(0)


def populate_languages(combo: wx.ComboBox, tts) -> None:
    combo.Clear()
    if not tts:
        return
    for lang in tts.get("languages", []):
        combo.Append(f"{lang['name']} ({lang['code']})", lang["code"])
    if combo.GetCount():
        combo.SetSelection(0)


def populate_variants(combo: wx.ComboBox, tts, lang_code: str) -> None:
    combo.Clear()
    if not tts:
        return
    lang = catalog.find_language(tts, lang_code)
    if not lang:
        return
    variants = lang.get("variants", [])
    if len(variants) > 1:
        # "Download all variants" convenience (e.g. Kitten's four quality
        # levels): one click fetches every variant of this engine/language.
        combo.Append(f"Download all variants ({len(variants)})", "__all__")
    for variant in variants:
        combo.Append(variant["name"], variant["id"])
    if combo.GetCount():
        combo.SetSelection(0)


def populate_voices(combo: wx.ComboBox, variant, voices_only: bool = False) -> None:
    combo.Clear()
    if not variant:
        return
    for voice in variant.get("voices", []):
        label = f"{voice.get('name', voice['id'])} ({voice['id']})"
        combo.Append(label, voice["id"])
    if combo.GetCount():
        combo.SetSelection(0)


# ---------------------------------------------------------------------------
# Download and remove tab
# ---------------------------------------------------------------------------
class DownloadPanel(_ManagerPanel):
    title = "Download and remove"
    description = "Download or remove neural TTS voices from the official open-source releases."

    def __init__(self, parent, store: ModelStore, downloader: ModelDownloader,
                 on_models_changed=None):
        super().__init__(parent)
        self.store = store
        self.downloader = downloader
        self.on_models_changed = on_models_changed
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label="Choose a TTS engine, language, and variant, "
                                      "then press Download or Remove."),
            0, wx.ALL, 6,
        )
        self._build_selector(sizer)
        sizer.AddSpacer(8)

        # Progress
        self.gauge = wx.Gauge(self, range=100, size=(-1, 22))
        self.gauge.SetName("Download progress")
        sizer.Add(self.gauge, 0, wx.EXPAND | wx.ALL, 6)
        self.progress_label = wx.StaticText(self, label="No download in progress.")
        sizer.Add(self.progress_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        self.download_btn = wx.Button(self, label="Download selected variant")
        self.remove_btn = wx.Button(self, label="Remove selected variant")
        self.cancel_btn = wx.Button(self, label="Cancel download")
        self.cancel_btn.Disable()
        self.download_btn.SetName("Download selected variant")
        self.remove_btn.SetName("Remove selected variant")
        self.cancel_btn.SetName("Cancel download")
        btns.Add(self.download_btn, 0, wx.ALL, 4)
        btns.Add(self.remove_btn, 0, wx.ALL, 4)
        btns.Add(self.cancel_btn, 0, wx.ALL, 4)
        sizer.Add(btns, 0, wx.LEFT, 2)

        self.note = wx.StaticText(
            self,
            label="Models are downloaded from the official open-source releases "
                  "(k2-fsa/sherpa-onnx; rhasspy piper). You only download the "
                  "voices you actually use.",
        )
        sizer.Add(self.note, 0, wx.ALL, 6)
        self.SetSizer(sizer)

        self.tts_combo.Bind(wx.EVT_COMBOBOX, self._on_tts)
        self.lang_combo.Bind(wx.EVT_COMBOBOX, self._on_lang)
        self.variant_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._refresh_buttons())
        self.download_btn.Bind(wx.EVT_BUTTON, self._on_download)
        self.remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)
        self.cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel)
        self.Bind(EVT_DOWNLOAD_PROGRESS, self._on_progress)
        self.Bind(EVT_DOWNLOAD_FINISHED, self._on_finished)

        populate_tts(self.tts_combo, catalog.get_tts_list())
        self._on_tts(None)

    def _build_selector(self, sizer: wx.BoxSizer):
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.tts_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                     name="TTS engine to download")
        self.lang_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                      name="Language to download")
        self.variant_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Variant to download")
        add_labeled(self, grid, "TTS engine", self.tts_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Language", self.lang_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Variant", self.variant_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

    # -- selection state ----------------------------------------------------
    def _selected(self):
        tts_id = self.tts_combo.GetClientData(self.tts_combo.GetSelection()) if self.tts_combo.GetSelection() >= 0 else None
        lang_code = self.lang_combo.GetClientData(self.lang_combo.GetSelection()) if self.lang_combo.GetSelection() >= 0 else None
        variant_id = self.variant_combo.GetClientData(self.variant_combo.GetSelection()) if self.variant_combo.GetSelection() >= 0 else None
        return tts_id, lang_code, variant_id

    def _on_tts(self, _):
        tts_id, _, _ = self._selected()
        tts = catalog.find_tts(tts_id) if tts_id else None
        populate_languages(self.lang_combo, tts)
        self._on_lang(None)

    def _on_lang(self, _):
        tts_id, lang_code, _ = self._selected()
        tts = catalog.find_tts(tts_id) if tts_id else None
        populate_variants(self.variant_combo, tts, lang_code or "")
        self._refresh_buttons()

    def _refresh_buttons(self):
        tts_id, lang_code, variant_id = self._selected()
        if not all((tts_id, lang_code, variant_id)):
            self.download_btn.Disable()
            self.remove_btn.Disable()
            return
        if variant_id == "__all__":
            # "Download all variants": enabled until every variant is present.
            tts = catalog.find_tts(tts_id)
            lang = catalog.find_language(tts, lang_code) if tts else None
            variants = [v["id"] for v in (lang.get("variants", []) if lang else [])]
            missing = [
                v for v in variants
                if not self.store.is_variant_installed(tts_id, lang_code, v)
            ]
            self.download_btn.Enable(bool(missing) and self._thread is None)
            self.remove_btn.Disable()
            return
        installed = self.store.is_variant_installed(tts_id, lang_code, variant_id)
        self.download_btn.Enable(not installed and self._thread is None)
        self.remove_btn.Enable(installed and self._thread is None)

    # -- actions ------------------------------------------------------------
    def _on_download(self, _):
        tts_id, lang_code, variant_id = self._selected()
        if not all((tts_id, lang_code, variant_id)):
            return
        if variant_id == "__all__":
            tts = catalog.find_tts(tts_id)
            lang = catalog.find_language(tts, lang_code) if tts else None
            variants = [v["id"] for v in (lang.get("variants", []) if lang else [])]
        else:
            variants = [variant_id]
        self._cancel_event = threading.Event()
        self.download_btn.Disable()
        self.remove_btn.Disable()
        self.cancel_btn.Enable()
        self.gauge.SetValue(0)
        self.progress_label.SetLabel(
            f"Downloading {tts_id} / {lang_code} / {variant_id} ..."
        )
        self._thread = threading.Thread(
            target=self._download_job, args=(tts_id, lang_code, variants), daemon=True
        )
        self._thread.start()

    def _download_job(self, tts_id, lang_code, variants):
        def progress(name, done, total):
            wx.PostEvent(self, DownloadProgressEvent(name, done, total))

        try:
            for variant_id in variants:
                self.downloader.download_variant(
                    tts_id, lang_code, variant_id, progress=progress,
                    cancel_event=self._cancel_event,
                )
            names = ", ".join(variants) if len(variants) > 1 else variants[0]
            wx.PostEvent(self, DownloadFinishedEvent(True, f"Installed {tts_id} / {lang_code} / {names}."))
        except DownloadCancelled:
            wx.PostEvent(self, DownloadFinishedEvent(False, "Download cancelled."))
        except (DownloadError, OSError) as exc:
            wx.PostEvent(self, DownloadFinishedEvent(False, f"Download failed: {exc}"))

    def _on_cancel(self, _):
        self._cancel_event.set()
        self.cancel_btn.Disable()
        self.progress_label.SetLabel("Cancelling...")

    def _on_progress(self, evt: DownloadProgressEvent):
        name = evt.name
        done, total = evt.done, evt.total
        if total > 0:
            pct = int(min(100, done * 100 / total))
            self.gauge.SetValue(pct)
        self.progress_label.SetLabel(f"Downloading {name}: {done / (1024 * 1024):.1f} MB"
                                     + (f" of {total / (1024 * 1024):.1f} MB" if total else ""))

    def _on_finished(self, evt: DownloadFinishedEvent):
        self._thread = None
        self.cancel_btn.Disable()
        self.progress_label.SetLabel(evt.message)
        self.gauge.SetValue(100 if evt.success else 0)
        self._refresh_buttons()
        if self.on_models_changed:
            self.on_models_changed()

    def _on_remove(self, _):
        tts_id, lang_code, variant_id = self._selected()
        if not all((tts_id, lang_code, variant_id)):
            return
        name = f"{tts_id} / {lang_code} / {variant_id}"
        if wx.MessageBox(
            f"Remove the downloaded voice \"{name}\"? Its files will be deleted "
            "from your computer.",
            "Remove voice",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) == wx.YES:
            self.downloader.remove_variant(tts_id, lang_code, variant_id)
            self.progress_label.SetLabel(f"Removed {name}.")
            self._refresh_buttons()
            if self.on_models_changed:
                self.on_models_changed()


# ---------------------------------------------------------------------------
# Available TTS tab
# ---------------------------------------------------------------------------
class AvailablePanel(_ManagerPanel):
    """Lists downloaded voices (incl. cloned ones) with a voice combo."""
    title = "Available TTS"
    description = "Voices you have downloaded or cloned, with a preview button."

    def __init__(self, parent, store: ModelStore):
        super().__init__(parent)
        self.store = store
        self._voices: list = []  # parallel list of voice entries

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label="Voices you have downloaded. Select a voice "
                                      "to preview it."),
            0, wx.ALL, 6,
        )
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.tts_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                     name="TTS engine")
        self.lang_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                      name="Language")
        self.variant_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Variant")
        self.voice_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                       name="Voice")
        add_labeled(self, grid, "TTS engine", self.tts_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Language", self.lang_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Variant", self.variant_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Voice", self.voice_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        self.preview_btn = wx.Button(self, label="Preview selected voice")
        self.preview_btn.SetName("Preview selected voice")
        btns.Add(self.preview_btn, 0, wx.ALL, 4)
        sizer.Add(btns, 0, wx.LEFT, 2)

        self.detail = wx.StaticText(self, label="")
        sizer.Add(self.detail, 0, wx.ALL, 6)
        self.SetSizer(sizer)

        self.tts_combo.Bind(wx.EVT_COMBOBOX, self._on_tts)
        self.lang_combo.Bind(wx.EVT_COMBOBOX, self._on_lang)
        self.variant_combo.Bind(wx.EVT_COMBOBOX, self._on_variant)
        self.preview_btn.Bind(wx.EVT_BUTTON, self._on_preview)
        self.refresh()

    def refresh(self):
        self._voices = self.store.installed_voices()
        custom = self.store.custom_voices()
        for voice in custom:
            self._voices.append(
                {
                    "tts": voice["tts"],
                    "tts_name": voice["tts"],
                    "language": "custom",
                    "variant": "custom",
                    "voice": voice["name"],
                    "voice_name": f"Cloned voice: {voice['name']}",
                    "sid": 0,
                    "engine": voice.get("engine", "vits"),
                    "dir": voice["dir"],
                    "custom": True,
                    "sample": voice.get("sample", ""),
                    "reference": voice.get("reference", ""),
                    "xtts_lang": voice.get("language", "en"),
                    "qwen_lang": voice.get("qwen_lang") or voice.get("language", "english"),
                    "ref_text": voice.get("ref_text", ""),
                    "model_dir": voice.get("model_dir", ""),
                }
            )
        groups: dict = {}
        for v in self._voices:
            key = (v["tts"], v["language"], v["variant"])
            groups.setdefault(key, []).append(v)

        self.tts_combo.Clear()
        # One entry per TTS engine (a TTS may have several downloaded
        # variants - e.g. Kitten's four quality levels - which must not
        # appear as separate "Kitten TTS" rows).
        seen: set = set()
        for key, items in sorted(groups.items()):
            if key[0] in seen:
                continue
            seen.add(key[0])
            self.tts_combo.Append(f"{items[0]['tts_name']}", key[0])
        if self.tts_combo.GetCount():
            self.tts_combo.SetSelection(0)
            self._on_tts(None)
        else:
            self._clear_all()

    def _clear_all(self):
        for combo in (self.lang_combo, self.variant_combo, self.voice_combo):
            combo.Clear()
        self.detail.SetLabel("No voices downloaded yet. Use the 'Download and remove' tab.")

    def _groups(self):
        groups: dict = {}
        for v in self._voices:
            key = (v["tts"], v["language"], v["variant"])
            groups.setdefault(key, []).append(v)
        return groups

    def _on_tts(self, _):
        groups = self._groups()
        if not groups:
            self._clear_all()
            return
        sel = self.tts_combo.GetSelection()
        tts_id = self.tts_combo.GetClientData(sel) if sel >= 0 else None
        matching = {k: v for k, v in groups.items() if k[0] == tts_id}
        self.lang_combo.Clear()
        for key, items in sorted(matching.items()):
            tts = catalog.find_tts(key[0])
            label = catalog.language_display_name(tts, key[1]) if tts else key[1]
            self.lang_combo.Append(label, key)
        if self.lang_combo.GetCount():
            self.lang_combo.SetSelection(0)
        self._on_lang(None)

    def _on_lang(self, _):
        key = self.lang_combo.GetClientData(self.lang_combo.GetSelection()) if self.lang_combo.GetSelection() >= 0 else None
        if not key:
            self.variant_combo.Clear()
            self.voice_combo.Clear()
            return
        groups = self._groups()
        matching = {k: v for k, v in groups.items() if k[0] == key[0] and k[1] == key[1]}
        self.variant_combo.Clear()
        for vkey, items in sorted(matching.items()):
            tts = catalog.find_tts(vkey[0])
            variant = catalog.find_variant(tts, vkey[1], vkey[2]) if tts else None
            label = variant["name"] if variant else vkey[2]
            self.variant_combo.Append(label, vkey)
        if self.variant_combo.GetCount():
            self.variant_combo.SetSelection(0)
        self._on_variant(None)

    def _on_variant(self, _):
        key = self.variant_combo.GetClientData(self.variant_combo.GetSelection()) if self.variant_combo.GetSelection() >= 0 else None
        self.voice_combo.Clear()
        if not key:
            self.detail.SetLabel("")
            return
        for v in self._voices:
            if (v["tts"], v["language"], v["variant"]) == key:
                self.voice_combo.Append(v["voice_name"], v)
        if self.voice_combo.GetCount():
            self.voice_combo.SetSelection(0)
        self._update_detail()

    def selected_voice(self):
        sel = self.voice_combo.GetSelection()
        if sel < 0:
            return None
        return self.voice_combo.GetClientData(sel)

    def _update_detail(self):
        voice = self.selected_voice()
        if voice:
            self.detail.SetLabel(
                f"Engine: {voice.get('engine', 'vits')}  |  "
                f"Folder: {voice.get('dir', '')}"
            )

    def _on_preview(self, _):
        voice = self.selected_voice()
        if not voice:
            wx.MessageBox("Select a voice first.", "Preview", style=wx.OK | wx.ICON_INFORMATION)
            return
        self.preview_btn.Disable()
        self.detail.SetLabel("Synthesizing preview...")
        # Lazy import to avoid GUI<->engine coupling at import time.
        threading.Thread(target=self._preview_job, args=(voice,), daemon=True).start()

    def _preview_job(self, voice):
        from .. import compute
        from ..audio.output import write_wav
        from ..tts.engine import EngineUnavailableError, get_engine, process_punctuation

        try:
            engine = get_engine(voice, provider=compute.provider_for(compute.resolve_compute("cpu")))
            text = process_punctuation(
                "This is a preview of the selected voice. Hello, welcome to AI Voice Studio.",
                "default",
            )
            samples = engine.synthesize(text, sid=voice.get("sid", 0), speed=1.0)
            import tempfile  # noqa: PLC0415

            tmp = os.path.join(tempfile.gettempdir(), "aivs_preview.wav")
            write_wav(samples, engine.sample_rate, tmp)
            wx.CallAfter(self._preview_done, tmp, None)
        except EngineUnavailableError as exc:
            wx.CallAfter(self._preview_done, None, str(exc))
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._preview_done, None, f"Preview failed: {exc}")

    def _preview_done(self, tmp, error):
        self.preview_btn.Enable()
        if error:
            self.detail.SetLabel(error)
            wx.MessageBox(error, "Preview unavailable", style=wx.OK | wx.ICON_ERROR)
            return
        self._play_wav(tmp)

    def _play_wav(self, path: str) -> None:
        import wx.adv  # noqa: PLC0415

        sound = getattr(self, "_preview_sound", None)
        if sound is not None:
            try:
                sound.Stop()
            except Exception:  # noqa: BLE001
                pass
            self._preview_sound = None
        sound = wx.adv.Sound(path)
        if sound.IsOk():
            # Keep a reference: wxSound must outlive Play(SOUND_ASYNC) or the
            # preview is cut off before it is heard (garbage collection bug).
            self._preview_sound = sound
            sound.Play(wx.adv.SOUND_ASYNC)
            self.detail.SetLabel("Playing preview.")
        else:
            self.detail.SetLabel("Preview file could not be played.")


# ---------------------------------------------------------------------------
# Voice clone category (Qwen3-TTS - clone from a short audio sample)
# ---------------------------------------------------------------------------
class VoiceClonePanel(_ManagerPanel):
    """Clone a voice using Qwen3-TTS (Apache-2.0, commercial OK).

    Qwen3-TTS runs as pure ONNX in a separate worker subprocess. It clones
    a voice from a short reference recording and speaks any of 10 languages.
    """

    title = "Voice clone"
    description = (
        "Clone a voice from a short recording using Qwen3-TTS (Apache-2.0, "
        "commercial use allowed). Provide a WAV sample, choose its language, "
        "and the app creates a new voice that speaks any of 10 languages."
    )

    def __init__(self, parent, store: ModelStore, downloader: ModelDownloader,
                 on_models_changed=None):
        super().__init__(parent)
        self.store = store
        self.downloader = downloader
        self.on_models_changed = on_models_changed
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._progress_dlg: wx.ProgressDialog | None = None
        self.sample_path: str = ""

        # Wrap all content in a ScrolledPanel so the Create buttons
        # are always reachable even when the settings dialog is small.
        from wx.lib.scrolledpanel import ScrolledPanel
        scroll = ScrolledPanel(self, style=wx.VSCROLL)
        scroll.SetupScrolling(scroll_x=False, scroll_y=True)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # -- Qwen3-TTS clone ------------------------------------------------
        qwen_box = wx.StaticBox(scroll, label="Qwen3-TTS voice cloning (10 languages, high quality, Apache-2.0)")
        qwen_sizer = wx.StaticBoxSizer(qwen_box, wx.VERTICAL)
        self.qwen_status = wx.StaticText(qwen_box, label="")
        self.qwen_status.SetName("Qwen3-TTS clone status")
        self.qwen_status.Wrap(640)
        qwen_sizer.Add(self.qwen_status, 0, wx.ALL, 6)
        qwen_sizer.Add(
            wx.StaticText(
                qwen_box,
                label="Qwen3-TTS is Alibaba's open-source multilingual TTS "
                      "(Apache-2.0, pure ONNX). It clones a voice from a short "
                      "reference recording and speaks any of 10 languages. "
                      "Download the Qwen3-TTS model first in Settings > "
                      "Download and remove (about 5.6 GB), then install the "
                      "engine below. An NVIDIA GPU is strongly recommended - "
                      "CPU synthesis is very slow.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )
        qwen_btns = wx.BoxSizer(wx.HORIZONTAL)
        self.qwen_engine_dl_btn = wx.Button(qwen_box, label="Install runtime dependency")
        self.qwen_engine_rm_btn = wx.Button(qwen_box, label="Remove dependency")
        self.qwen_engine_cancel_btn = wx.Button(qwen_box, label="Cancel install")
        self.qwen_engine_dl_btn.SetName("Install Qwen3 runtime dependency")
        self.qwen_engine_dl_btn.SetToolTip("Download and install the Qwen3-TTS runtime (onnxruntime, librosa)")
        self.qwen_engine_rm_btn.SetName("Remove Qwen3 dependency")
        self.qwen_engine_rm_btn.SetToolTip("Delete the Qwen3 runtime files from your user folder")
        self.qwen_engine_cancel_btn.SetName("Cancel Qwen3 install")
        self.qwen_engine_cancel_btn.SetToolTip("Cancel the Qwen3 runtime installation")
        self.qwen_engine_cancel_btn.Disable()
        qwen_btns.Add(self.qwen_engine_dl_btn, 0, wx.ALL, 4)
        qwen_btns.Add(self.qwen_engine_rm_btn, 0, wx.ALL, 4)
        qwen_btns.Add(self.qwen_engine_cancel_btn, 0, wx.ALL, 4)
        qwen_sizer.Add(qwen_btns, 0, wx.LEFT, 2)

        qwen_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        qwen_grid.AddGrowableCol(1)
        self.qwen_name_ctrl = wx.TextCtrl(qwen_box)
        self.qwen_name_ctrl.SetName("Qwen3 voice name")
        add_labeled(qwen_box, qwen_grid, "Voice name", self.qwen_name_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        self.qwen_lang_combo = wx.ComboBox(qwen_box, style=wx.CB_READONLY)
        for code, label in QWEN_LANGUAGES:
            self.qwen_lang_combo.Append(f"{label}", code)
        self.qwen_lang_combo.SetSelection(0)
        add_labeled(qwen_box, qwen_grid, "Language", self.qwen_lang_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        self.qwen_reftext_ctrl = wx.TextCtrl(qwen_box)
        self.qwen_reftext_ctrl.SetName("Qwen3 reference transcript")
        self.qwen_reftext_ctrl.SetValue("")
        add_labeled(qwen_box, qwen_grid, "Transcript of the sample (optional)",
                    self.qwen_reftext_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)
        self.qwen_sample_ctrl = wx.TextCtrl(qwen_box, style=wx.TE_READONLY)
        self.qwen_sample_ctrl.SetName("Qwen3 sample recording")
        self.qwen_sample_ctrl.SetValue("No sample chosen.")
        qwen_sample_row = wx.BoxSizer(wx.HORIZONTAL)
        qwen_sample_row.Add(self.qwen_sample_ctrl, 1, wx.EXPAND)
        self.qwen_browse_btn = wx.Button(qwen_box, label="Browse...")
        self.qwen_browse_btn.SetName("Browse for a Qwen3-TTS WAV sample")
        qwen_sample_row.Add(
            self.qwen_browse_btn, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 6
        )
        label = wx.StaticText(qwen_box, label="Sample recording:")
        qwen_grid.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        qwen_grid.Add(qwen_sample_row, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)
        qwen_sizer.Add(qwen_grid, 0, wx.EXPAND | wx.ALL, 6)
        self.qwen_create_btn = wx.Button(qwen_box, label="Create Qwen3 voice")
        self.qwen_create_btn.SetName("Create Qwen3-TTS voice")
        qwen_sizer.Add(self.qwen_create_btn, 0, wx.LEFT, 4)
        sizer.Add(qwen_sizer, 0, wx.EXPAND | wx.ALL, 6)

        # -- OmniVoice clone -----------------------------------------------
        omni_box = wx.StaticBox(scroll, label="OmniVoice TTS (600+ languages, voice cloning & design, Apache-2.0)")
        omni_sizer = wx.StaticBoxSizer(omni_box, wx.VERTICAL)
        self.omni_status = wx.StaticText(omni_box, label="")
        self.omni_status.SetName("OmniVoice clone status")
        self.omni_status.Wrap(640)
        omni_sizer.Add(self.omni_status, 0, wx.ALL, 6)
        omni_sizer.Add(
            wx.StaticText(
                omni_box,
                label="OmniVoice (k2-fsa) is a state-of-the-art multilingual TTS "
                      "supporting 600+ languages with voice cloning and voice "
                      "design.\n"
                      "ONNX runtime is bundled with the app (works on CPU immediately). "
                      "GPU variant requires NVIDIA GPU + download (~2.5 GB).\n\n"
                      "Download the OmniVoice model first in Settings > Download "
                      "and remove, then create a voice below.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )
        omni_btns = wx.BoxSizer(wx.HORIZONTAL)
        self.omni_gpu_dl_btn = wx.Button(omni_box, label="Install GPU dependency (optional)")
        self.omni_engine_rm_btn = wx.Button(omni_box, label="Remove GPU dependency")
        self.omni_engine_cancel_btn = wx.Button(omni_box, label="Cancel install")
        self.omni_gpu_dl_btn.SetName("Install OmniVoice GPU dependency")
        self.omni_gpu_dl_btn.SetToolTip("Download and install PyTorch + CUDA for OmniVoice GPU synthesis (~2.5 GB)")
        self.omni_engine_rm_btn.SetName("Remove OmniVoice GPU dependency")
        self.omni_engine_rm_btn.SetToolTip("Delete the OmniVoice GPU runtime files from your user folder")
        self.omni_engine_cancel_btn.SetName("Cancel OmniVoice install")
        self.omni_engine_cancel_btn.SetToolTip("Cancel the OmniVoice GPU installation")
        self.omni_engine_cancel_btn.Disable()
        omni_btns.Add(self.omni_gpu_dl_btn, 0, wx.ALL, 4)
        omni_btns.Add(self.omni_engine_rm_btn, 0, wx.ALL, 4)
        omni_btns.Add(self.omni_engine_cancel_btn, 0, wx.ALL, 4)
        omni_sizer.Add(omni_btns, 0, wx.LEFT, 2)

        omni_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        omni_grid.AddGrowableCol(1)
        self.omni_name_ctrl = wx.TextCtrl(omni_box)
        self.omni_name_ctrl.SetName("OmniVoice voice name")
        add_labeled(omni_box, omni_grid, "Voice name", self.omni_name_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        self.omni_variant_combo = wx.ComboBox(omni_box, style=wx.CB_READONLY)
        self.omni_variant_combo.Append("ONNX (CPU-capable)", "onnx")
        self.omni_variant_combo.Append("GPU (requires NVIDIA CUDA)", "gpu")
        self.omni_variant_combo.SetSelection(0)
        add_labeled(omni_box, omni_grid, "Variant", self.omni_variant_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        self.omni_lang_combo = wx.ComboBox(omni_box, style=wx.CB_READONLY)
        from ..omnivoice import OMNIVOICE_LANGUAGES
        for code, label in OMNIVOICE_LANGUAGES:
            self.omni_lang_combo.Append(label, code)
        self.omni_lang_combo.SetSelection(0)
        add_labeled(omni_box, omni_grid, "Language", self.omni_lang_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        self.omni_reftext_ctrl = wx.TextCtrl(omni_box)
        self.omni_reftext_ctrl.SetName("OmniVoice reference transcript")
        self.omni_reftext_ctrl.SetValue("")
        add_labeled(omni_box, omni_grid, "Transcript of the sample (optional)",
                    self.omni_reftext_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)
        self.omni_instruct_ctrl = wx.TextCtrl(omni_box)
        self.omni_instruct_ctrl.SetName("OmniVoice voice design instruct")
        self.omni_instruct_ctrl.SetValue("")
        self.omni_instruct_ctrl.SetToolTip(
            "Optional: describe the desired voice (e.g. 'female, low pitch, British accent'). "
            "Leave empty for voice cloning mode."
        )
        add_labeled(omni_box, omni_grid, "Voice design instruct (optional)",
                    self.omni_instruct_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)
        self.omni_sample_ctrl = wx.TextCtrl(omni_box, style=wx.TE_READONLY)
        self.omni_sample_ctrl.SetName("OmniVoice sample recording")
        self.omni_sample_ctrl.SetValue("No sample chosen (auto voice if empty).")
        omni_sample_row = wx.BoxSizer(wx.HORIZONTAL)
        omni_sample_row.Add(self.omni_sample_ctrl, 1, wx.EXPAND)
        self.omni_browse_btn = wx.Button(omni_box, label="Browse...")
        self.omni_browse_btn.SetName("Browse for an OmniVoice WAV sample")
        omni_sample_row.Add(
            self.omni_browse_btn, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 6
        )
        label = wx.StaticText(omni_box, label="Sample recording:")
        omni_grid.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        omni_grid.Add(omni_sample_row, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)
        omni_sizer.Add(omni_grid, 0, wx.EXPAND | wx.ALL, 6)
        self.omni_create_btn = wx.Button(omni_box, label="Create OmniVoice voice")
        self.omni_create_btn.SetName("Create OmniVoice voice")
        omni_sizer.Add(self.omni_create_btn, 0, wx.LEFT, 4)
        sizer.Add(omni_sizer, 0, wx.EXPAND | wx.ALL, 6)

        # -- cloned voices list --------------------------------------------
        sizer.Add(wx.StaticText(scroll, label="Cloned voices:"), 0, wx.ALL, 4)
        self.clones_list = wx.ListBox(scroll, style=wx.LB_SINGLE)
        self.clones_list.SetName("Cloned voices list")
        sizer.Add(self.clones_list, 1, wx.EXPAND | wx.ALL, 6)

        rm_btns = wx.BoxSizer(wx.HORIZONTAL)
        self.remove_btn = wx.Button(scroll, label="Remove selected voice")
        self.remove_btn.SetName("Remove selected voice")
        rm_btns.Add(self.remove_btn, 0, wx.ALL, 4)
        sizer.Add(rm_btns, 0, wx.LEFT, 2)

        self.status = wx.StaticText(scroll, label="")
        sizer.Add(self.status, 0, wx.ALL, 6)
        scroll.SetSizer(sizer)
        scroll.SetupScrolling(scroll_x=False, scroll_y=True)

        outer_sizer = wx.BoxSizer(wx.VERTICAL)
        outer_sizer.Add(scroll, 1, wx.EXPAND)
        self.SetSizer(outer_sizer)

        self.qwen_engine_dl_btn.Bind(wx.EVT_BUTTON, self._on_download_qwen_engine)
        self.qwen_engine_rm_btn.Bind(wx.EVT_BUTTON, self._on_remove_qwen_engine)
        self.qwen_engine_cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel_qwen_engine)
        self.qwen_browse_btn.Bind(wx.EVT_BUTTON, self._on_qwen_browse)
        self.qwen_create_btn.Bind(wx.EVT_BUTTON, self._on_qwen_create)
        self.omni_gpu_dl_btn.Bind(wx.EVT_BUTTON, self._on_download_omni_gpu_engine)
        self.omni_engine_rm_btn.Bind(wx.EVT_BUTTON, self._on_remove_omni_engine)
        self.omni_engine_cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel_omni_engine)
        self.omni_browse_btn.Bind(wx.EVT_BUTTON, self._on_omni_browse)
        self.omni_create_btn.Bind(wx.EVT_BUTTON, self._on_omni_create)
        self.remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)
        self.Bind(EVT_DOWNLOAD_FINISHED, self._on_download_finished)
        self.Bind(EVT_DOWNLOAD_PROGRESS, self._on_progress)

        self.refresh_clones()
        self._refresh_qwen_state()
        self._refresh_omni_state()

    @staticmethod
    def _wav_info(path: str):
        import wave  # noqa: PLC0415

        try:
            with wave.open(path, "rb") as w:
                framerate = w.getframerate()
                seconds = w.getnframes() / framerate if framerate else 0.0
                return framerate, seconds
        except Exception:  # noqa: BLE001
            return None

    # -- Qwen3-TTS clone ---------------------------------------------------
    _QWEN_TTS = "qwen3"
    _QWEN_LANG = "multi"
    _QWEN_VARIANT = "v1_7b_fp16"

    def _refresh_qwen_state(self):
        from ..qwen import engine_installed  # noqa: PLC0415

        busy = self._thread is not None
        model_ok = self.store.is_variant_installed(
            self._QWEN_TTS, self._QWEN_LANG, self._QWEN_VARIANT
        )
        if engine_installed():
            self.qwen_status.SetLabel(
                "Runtime dependency installed."
                + (" Qwen3-TTS model is installed - create a voice below."
                   if model_ok else
                   " Download the Qwen3-TTS model first (Settings > Download "
                   "and remove, about 5.6 GB).")
            )
        else:
            self.qwen_status.SetLabel(
                "Runtime dependency not installed. Install it below, then download "
                "the Qwen3-TTS model (Settings > Download and remove, about "
                "5.6 GB), then create a voice."
            )
        self.qwen_engine_dl_btn.Enable(not engine_installed() and not busy)
        self.qwen_engine_rm_btn.Enable(engine_installed() and not busy)
        self.qwen_engine_cancel_btn.Enable(busy)
        self.qwen_create_btn.Enable(model_ok and engine_installed())

    def _on_download_qwen_engine(self, _):
        from .. import qwen  # noqa: PLC0415

        self._cancel_event = threading.Event()
        self._progress_dlg = wx.ProgressDialog(
            "Downloading Qwen3 engine",
            "Starting...",
            maximum=100, parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE,
        )
        self._progress_dlg.Update(0)
        self._thread = threading.Thread(
            target=self._qwen_engine_job, args=(qwen,), daemon=True
        )
        self.qwen_engine_dl_btn.Disable()
        self.qwen_engine_rm_btn.Disable()
        self.qwen_engine_cancel_btn.Enable()
        self.qwen_status.SetLabel("Installing the Qwen3 runtime dependency...")
        self._thread.start()

    def _qwen_engine_job(self, qwen):
        def progress(name, done, total):
            wx.PostEvent(self, DownloadProgressEvent(name, done, total))

        try:
            qwen.ensure_engine(progress=progress, cancel_event=self._cancel_event)
            wx.PostEvent(self, DownloadFinishedEvent(
                True,
                "Runtime dependency installed. Download the Qwen3-TTS model from "
                "Settings > Download and remove, then create a voice below.",
            ))
        except qwen.Qwen3Error as exc:
            wx.PostEvent(self, DownloadFinishedEvent(False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            wx.PostEvent(self, DownloadFinishedEvent(False, f"Download failed: {exc}"))

    def _on_cancel_qwen_engine(self, _):
        self._cancel_event.set()
        self.qwen_engine_cancel_btn.Disable()
        self.qwen_status.SetLabel("Cancelling...")

    def _on_remove_qwen_engine(self, _):
        from .. import qwen  # noqa: PLC0415

        if wx.MessageBox(
            "Remove the Qwen3 runtime dependency? Its files will be deleted from your "
            "user folder. Existing Qwen3 voices are kept, but they cannot be "
            "used until the dependency is installed again.",
            "Remove Qwen3 dependency",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) == wx.YES:
            qwen.remove_engine()
            self.status.SetLabel("Qwen3 dependency removed.")
            self._refresh_qwen_state()

    def _on_qwen_browse(self, _):
        with wx.FileDialog(
            self, "Choose a WAV reference recording of the voice",
            wildcard="WAV audio (*.wav)|*.wav|All files|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        info = self._wav_info(path)
        if info is None:
            wx.MessageBox(
                "This is not a valid WAV file. Export the recording as WAV "
                "(for example with Audacity) and try again.",
                "Sample recording", style=wx.OK | wx.ICON_WARNING,
            )
            return
        framerate, seconds = info
        self.sample_path = path
        self.qwen_sample_ctrl.SetValue(f"{os.path.basename(path)} ({seconds:.1f} s)")
        self.qwen_sample_ctrl.SetToolTip(path)
        if seconds < 1.0:
            wx.MessageBox("The sample is very short - at least 1 second is "
                          "recommended, 4-5 seconds gives the best result.",
                          "Sample recording", style=wx.OK | wx.ICON_WARNING)
        elif seconds > 60.0:
            wx.MessageBox("The sample is longer than one minute. Only the first "
                          "seconds are used; a 4-5 second clip is enough.",
                          "Sample recording", style=wx.OK | wx.ICON_WARNING)

    def _on_qwen_create(self, _):
        from ..qwen import (  # noqa: PLC0415
            Qwen3Error,
            create_qwen_voice,
            engine_installed,
        )

        name = self.qwen_name_ctrl.GetValue().strip()
        if not name:
            wx.MessageBox("Give the cloned voice a name first.", "Create Qwen3 voice",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        if not self.sample_path:
            wx.MessageBox("Choose a WAV reference recording first.",
                          "Create Qwen3 voice", style=wx.OK | wx.ICON_INFORMATION)
            return
        if not engine_installed():
            wx.MessageBox(
                "The Qwen3 engine is not installed yet. Download it above "
                "(section 4) first.",
                "Create Qwen3 voice", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        info = self.store.variant_info(
            self._QWEN_TTS, self._QWEN_LANG, self._QWEN_VARIANT
        )
        model_dir = (info or {}).get("dir") or ""
        if not model_dir or not os.path.isdir(model_dir):
            wx.MessageBox(
                "The Qwen3-TTS model is not downloaded yet. Download it from "
                "Settings > Download and remove first (about 5.6 GB).",
                "Create Qwen3 voice", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        lang_sel = self.qwen_lang_combo.GetSelection()
        language = self.qwen_lang_combo.GetClientData(lang_sel) if lang_sel >= 0 else "english"
        ref_text = self.qwen_reftext_ctrl.GetValue().strip()
        try:
            entry = create_qwen_voice(
                name, self.sample_path, language, model_dir, ref_text, self.store
            )
        except (Qwen3Error, OSError) as exc:
            wx.MessageBox(f"Could not create the voice: {exc}", "Create Qwen3 voice",
                          style=wx.OK | wx.ICON_ERROR)
            return
        self.qwen_name_ctrl.SetValue("")
        self.qwen_reftext_ctrl.SetValue("")
        self.sample_path = ""
        self.qwen_sample_ctrl.SetValue("No sample chosen.")
        self.status.SetLabel(
            f"Voice \"{entry['name']}\" created with Qwen3-TTS. Find it in "
            "Available TTS and in the Recording window."
        )
        self.refresh_clones()
        if self.on_models_changed:
            self.on_models_changed()

    # -- OmniVoice clone --------------------------------------------------
    _OMNI_TTS = "omnivoice"
    _OMNI_LANG = "multi"
    _OMNI_VARIANT_GPU = "gpu"
    _OMNI_VARIANT_ONNX = "onnx"

    def _refresh_omni_state(self):
        from ..omnivoice import engine_installed  # noqa: PLC0415

        busy = self._thread is not None
        onnx_ok = self.store.is_variant_installed(
            self._OMNI_TTS, self._OMNI_LANG, self._OMNI_VARIANT_ONNX
        ) or engine_installed("onnx")  # bundled ONNX counts as installed
        gpu_ok = self.store.is_variant_installed(
            self._OMNI_TTS, self._OMNI_LANG, self._OMNI_VARIANT_GPU
        )
        gpu_engine = engine_installed("gpu")

        # ONNX is always bundled — no download needed
        status_parts = ["ONNX runtime is bundled and ready."]
        if gpu_engine:
            status_parts.append(" GPU dependency also installed.")
        if onnx_ok or gpu_ok:
            status_parts.append(" Model installed - create a voice below.")
        else:
            status_parts.append(" Download the OmniVoice model first (Settings > Download and remove).")
        self.omni_status.SetLabel(" ".join(status_parts))
        self.omni_gpu_dl_btn.Enable(not gpu_engine and not busy)
        self.omni_engine_rm_btn.Enable(not busy)  # Always allow remove
        self.omni_engine_cancel_btn.Enable(busy)
        self.omni_create_btn.Enable((onnx_ok or gpu_ok))

    def _on_download_omni_gpu_engine(self, _):
        from .. import omnivoice  # noqa: PLC0415

        answer = wx.MessageBox(
            "⚠️ The OmniVoice GPU variant requires:\n\n"
            "• An NVIDIA GPU with CUDA 12 support\n"
            "• ~2.5 GB download (PyTorch + CUDA)\n"
            "• ~8 GB disk space\n\n"
            "CPU-only systems should use the ONNX variant instead.\n\n"
            "Continue with GPU engine download?",
            "GPU Engine Warning",
            style=wx.YES_NO | wx.ICON_WARNING,
        )
        if answer != wx.YES:
            return

        self._cancel_event = threading.Event()
        self._progress_dlg = wx.ProgressDialog(
            "Installing OmniVoice GPU dependency",
            "Starting...",
            maximum=100, parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE,
        )
        self._progress_dlg.Update(0)
        self._thread = threading.Thread(
            target=self._omni_engine_job, args=(omnivoice, "gpu"), daemon=True
        )
        self.omni_gpu_dl_btn.Disable()
        self.omni_engine_rm_btn.Disable()
        self.omni_engine_cancel_btn.Enable()
        self.omni_status.SetLabel("Installing the OmniVoice GPU dependency (~2.5 GB)...")
        self._thread.start()

    def _omni_engine_job(self, omnivoice, variant):
        def progress(name, done, total):
            wx.PostEvent(self, DownloadProgressEvent(name, done, total))

        try:
            log.info("OmniVoice GPU install: starting ensure_engine(%s)", variant)
            omnivoice.ensure_engine(
                variant=variant, progress=progress,
                cancel_event=self._cancel_event,
            )
            log.info("OmniVoice GPU install: ensure_engine(%s) succeeded", variant)
            wx.PostEvent(self, DownloadFinishedEvent(
                True,
                f"OmniVoice {variant.upper()} dependency installed. Download the "
                f"OmniVoice model from Settings > Download and remove, then "
                "create a voice below.",
            ))
        except omnivoice.OmniVoiceError as exc:
            log.error("OmniVoice GPU install failed: %s", exc)
            wx.PostEvent(self, DownloadFinishedEvent(False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            log.exception("OmniVoice GPU install: unexpected error")
            wx.PostEvent(self, DownloadFinishedEvent(False, f"Download failed: {exc}"))

    def _on_cancel_omni_engine(self, _):
        self._cancel_event.set()
        self.omni_engine_cancel_btn.Disable()
        self.omni_status.SetLabel("Cancelling...")

    def _on_remove_omni_engine(self, _):
        from .. import omnivoice  # noqa: PLC0415

        gpu_engine = omnivoice.engine_installed("gpu")
        if not gpu_engine:
            wx.MessageBox(
                "No GPU dependency is currently installed.\n\n"
                "To install the GPU dependency later, click \"Install GPU dependency\" "
                "in the Voice Clone tab. Note: this requires downloading ~2.5 GB.",
                "GPU dependency not installed",
                style=wx.OK | wx.ICON_INFORMATION,
            )
            return

        if wx.MessageBox(
            "Remove the OmniVoice GPU dependency? Its files (~2.5 GB) will be deleted "
            "from your user folder. Existing OmniVoice voices are kept, but they "
            "cannot be used with GPU until the dependency is installed again.\n\n"
            "To reinstall later, click \"Install GPU dependency\" in the Voice Clone tab.",
            "Remove OmniVoice GPU dependency",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) == wx.YES:
            omnivoice.remove_engine("gpu")
            log.info("OmniVoice GPU dependency removed by user")
            self.status.SetLabel("OmniVoice GPU dependency removed.")
            self._refresh_omni_state()

    def _on_omni_browse(self, _):
        with wx.FileDialog(
            self, "Choose a WAV reference recording of the voice",
            wildcard="WAV audio (*.wav)|*.wav|All files|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        info = self._wav_info(path)
        if info is None:
            wx.MessageBox(
                "This is not a valid WAV file. Export the recording as WAV "
                "(for example with Audacity) and try again.",
                "Sample recording", style=wx.OK | wx.ICON_WARNING,
            )
            return
        framerate, seconds = info
        self.omni_sample_path = path
        self.omni_sample_ctrl.SetValue(f"{os.path.basename(path)} ({seconds:.1f} s)")
        self.omni_sample_ctrl.SetToolTip(path)
        if seconds < 1.0:
            wx.MessageBox("The sample is very short - at least 1 second is "
                          "recommended, 3-10 seconds gives the best result.",
                          "Sample recording", style=wx.OK | wx.ICON_WARNING)
        elif seconds > 60.0:
            wx.MessageBox("The sample is longer than one minute. Only the first "
                          "few seconds are used; a 3-10 second clip is enough.",
                          "Sample recording", style=wx.OK | wx.ICON_WARNING)

    def _on_omni_create(self, _):
        from ..omnivoice import (  # noqa: PLC0415
            OmniVoiceError,
            create_omnivoice_voice,
            engine_installed,
        )

        name = self.omni_name_ctrl.GetValue().strip()
        if not name:
            wx.MessageBox("Give the voice a name first.", "Create OmniVoice voice",
                          style=wx.OK | wx.ICON_INFORMATION)
            return

        variant = self.omni_variant_combo.GetClientData(
            self.omni_variant_combo.GetSelection()
        ) or "onnx"

        if not engine_installed(variant):
            wx.MessageBox(
                f"The OmniVoice {variant.upper()} engine is not installed yet. "
                "Download it above first.",
                "Create OmniVoice voice", style=wx.OK | wx.ICON_INFORMATION,
            )
            return

        # Find the model directory
        model_ok = self.store.is_variant_installed(
            self._OMNI_TTS, self._OMNI_LANG, variant
        )
        if not model_ok:
            wx.MessageBox(
                "The OmniVoice model is not downloaded yet. Download it from "
                "Settings > Download and remove first.",
                "Create OmniVoice voice", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        info = self.store.variant_info(self._OMNI_TTS, self._OMNI_LANG, variant)
        model_dir = (info or {}).get("dir") or ""

        instruct = self.omni_instruct_ctrl.GetValue().strip()
        ref_text = self.omni_reftext_ctrl.GetValue().strip()
        sample = getattr(self, "omni_sample_path", "")

        if not instruct and not sample:
            wx.MessageBox(
                "Provide either a reference recording (for voice cloning) or "
                "a voice design instruction (e.g. 'female, low pitch, British accent'), "
                "or both. Leave sample empty for auto voice.",
                "Create OmniVoice voice", style=wx.OK | wx.ICON_INFORMATION,
            )
            return

        # Language for the cloned voice (default: English)
        lang_sel = getattr(self, 'omni_lang_combo', None)
        if lang_sel is not None:
            lang_code = lang_sel.GetClientData(lang_sel.GetSelection()) if lang_sel.GetSelection() >= 0 else 'en'
        else:
            lang_code = 'en'

        try:
            entry = create_omnivoice_voice(
                name, sample if sample else None, lang_code, model_dir,
                ref_text, instruct, variant, self.store,
            )
        except (OmniVoiceError, OSError) as exc:
            wx.MessageBox(f"Could not create the voice: {exc}", "Create OmniVoice voice",
                          style=wx.OK | wx.ICON_ERROR)
            return
        self.omni_name_ctrl.SetValue("")
        self.omni_reftext_ctrl.SetValue("")
        self.omni_instruct_ctrl.SetValue("")
        self.omni_sample_path = ""
        self.omni_sample_ctrl.SetValue("No sample chosen (auto voice if empty).")
        self.status.SetLabel(
            f"Voice \"{entry['name']}\" created with OmniVoice. Find it in "
            "Available TTS and in the Recording window."
        )
        self.refresh_clones()
        if self.on_models_changed:
            self.on_models_changed()

    def refresh_clones(self):
        self.clones_list.Clear()
        for voice in self.store.custom_voices():
            self.clones_list.Append(f"{voice['name']}  ({voice['tts']})")
        self.remove_btn.Enable(self.clones_list.GetCount() > 0)

    # -- remove ------------------------------------------------------------
    def _on_remove(self, _):
        sel = self.clones_list.GetSelection()
        if sel < 0:
            return
        name = self.store.custom_voices()[sel]["name"]
        if wx.MessageBox(
            f"Remove the cloned voice \"{name}\"? Its files will be deleted.",
            "Remove cloned voice",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) == wx.YES:
            self.store.remove_custom_voice(name)
            self.refresh_clones()
            if self.on_models_changed:
                self.on_models_changed()

    # -- progress / finished ------------------------------------------------
    def _on_progress(self, evt: DownloadProgressEvent):
        dlg = getattr(self, "_progress_dlg", None)
        if dlg is None:
            return
        if evt.total > 0:
            dlg.Update(
                int(min(100, evt.done * 100 / evt.total)),
                f"{evt.name}: {evt.done / (1024 * 1024):.0f} of "
                f"{evt.total / (1024 * 1024):.0f} MB",
            )
        else:
            dlg.Pulse(evt.name)

    def _on_download_finished(self, evt: DownloadFinishedEvent):
        log.info("Download finished: success=%s message=%s", evt.success, evt.message)
        self._thread = None
        self.qwen_engine_cancel_btn.Disable()
        self.omni_engine_cancel_btn.Disable()
        self.status.SetLabel(evt.message)
        dlg = getattr(self, "_progress_dlg", None)
        if dlg is not None:
            try:
                dlg.Destroy()
            except Exception:  # noqa: BLE001
                pass
            self._progress_dlg = None
        self._refresh_qwen_state()
        self._refresh_omni_state()
        if evt.success and self.on_models_changed:
            self.on_models_changed()


