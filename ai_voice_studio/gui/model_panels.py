"""Settings panels that manage TTS models (SPEC 3.1).

* ``DownloadPanel``      -- Download and remove tab (TTS -> language -> variant).
* ``AvailablePanel``     -- Available TTS tab (TTS -> language -> variant -> voice).

"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import wx

from ..tts import catalog
from ..tts.downloader import (
    DownloadCancelled,
    DownloadError,
    ModelDownloader,
)
from ..tts.models import ModelStore
from . import dialogs
from .a11y import add_labeled, finalize_accessibility
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

    def __init__(self, parent, store: ModelStore, downloader: ModelDownloader,
                 on_models_changed=None):
        super().__init__(parent)
        self.store = store
        self.downloader = downloader
        self.on_models_changed = on_models_changed
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

        sizer = wx.BoxSizer(wx.VERTICAL)
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

        # Exclude TTS engines that are installed via pip (e.g. omnivoice-triton)
        # and don't use the artifact download system.
        downloadable = [
            tts for tts in catalog.get_tts_list()
            if not tts.get("requires_package")
        ]
        populate_tts(self.tts_combo, downloadable)
        self._on_tts(None)
        finalize_accessibility(self)

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

    def __init__(self, parent, store: ModelStore):
        super().__init__(parent)
        self.store = store
        self._voices: list = []  # parallel list of voice entries

        sizer = wx.BoxSizer(wx.VERTICAL)
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
        finalize_accessibility(self)

    def on_activated(self):
        """Refresh the voice lists when the category is opened so voices
        created in other categories (e.g. the OmniVoice voice library in
        Settings -> OmniVoice engines) appear without closing the dialog."""
        prev = self._snapshot_selection()
        super().on_activated()
        self.refresh()
        self._restore_selection(prev)

    def _snapshot_selection(self):
        sel = self.voice_combo.GetSelection()
        voice = self.voice_combo.GetClientData(sel) if sel >= 0 else None
        if not voice:
            return None
        return (voice.get("tts"), voice.get("language"),
                voice.get("variant"), voice.get("voice"))

    def _restore_selection(self, prev):
        if not prev:
            return
        tts_id, lang, variant, voice_id = prev

        def _cd(combo, index):
            return combo.GetClientData(index) if index >= 0 else None

        for i in range(self.tts_combo.GetCount()):
            if _cd(self.tts_combo, i) == tts_id:
                self.tts_combo.SetSelection(i)
                self._on_tts(None)
                break
        else:
            return
        for i in range(self.lang_combo.GetCount()):
            key = _cd(self.lang_combo, i)
            if key and key[0] == tts_id and key[1] == lang:
                self.lang_combo.SetSelection(i)
                self._on_lang(None)
                break
        else:
            return
        for i in range(self.variant_combo.GetCount()):
            key = _cd(self.variant_combo, i)
            if key and key[0] == tts_id and key[1] == lang and key[2] == variant:
                self.variant_combo.SetSelection(i)
                self._on_variant(None)
                break
        else:
            return
        for i in range(self.voice_combo.GetCount()):
            voice = _cd(self.voice_combo, i)
            if voice and voice.get("voice") == voice_id:
                self.voice_combo.SetSelection(i)
                break
        self._update_detail()

    def refresh(self):
        self._voices = self.store.installed_voices()
        # Inject pip-installed TTS voices (e.g. OmniVoice) that don't go
        # through the artifact download system and thus never appear in
        # installed_voices().  The voices are shown when the package is
        # installed, even though the model file is only downloaded on first
        # synthesis.
        self._inject_pip_installed_voices()
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
                    "ref_text": voice.get("ref_text", ""),
                    "model_dir": voice.get("model_dir", ""),
                }
            )
        # Universal OmniVoice voice library: created voices are engine
        # agnostic, so register them under every installed OmniVoice engine.
        try:
            from ..omnivoice import voice_store  # noqa: PLC0415
            if voice_store.omni_custom_voices(self.store):
                installed = voice_store.engine_ids_installed()
                self._voices.extend(
                    voice_store.consumer_entries(self.store, installed)
                )
        except Exception:  # noqa: BLE001
            pass
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

    def _inject_pip_installed_voices(self):
        """Inject voices for TTS engines installed via pip (e.g. OmniVoice).

        These engines don't use the artifact download system, so their
        voices never appear in ``store.installed_voices()``.  We check if
        the package is installed in the *managed venv* (not the main
        process) and, if so, create voice entries from the catalog.
        """
        try:
            from ..python_runtime import get_runtime  # noqa: PLC0415
            rt = get_runtime()
            if not rt.is_created:
                return
            for tts_entry in catalog.get_tts_list():
                pkg = tts_entry.get("requires_package")
                if not pkg:
                    continue
                # Check in the managed venv, not the main process.
                result = rt.run_in_env(
                    f"import importlib.metadata; "
                    f"print(importlib.metadata.version('{pkg}'))"
                )
                if result.returncode != 0 or not result.stdout.strip():
                    continue
                # Package is installed — add all catalog voices for this
                # TTS engine so the user can see and select them.
                for lang in tts_entry.get("languages", []):
                    for variant in lang.get("variants", []):
                        for voice in variant.get("voices", []):
                            self._voices.append(
                                {
                                    "tts": tts_entry["id"],
                                    "tts_name": tts_entry["name"],
                                    "language": lang["code"],
                                    "variant": variant["id"],
                                    "voice": voice["id"],
                                    "voice_name": voice.get("name", voice["id"]),
                                    "sid": voice.get("sid", 0),
                                    "engine": tts_entry.get("engine", "vits"),
                                    "dir": "",
                                    "requires_gpu": tts_entry.get("requires_gpu", False),
                                    "requires_package": pkg,
                                }
                            )
        except Exception:  # noqa: BLE001
            pass

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

            fd, tmp = tempfile.mkstemp(prefix="aivs_preview_", suffix=".wav")
            os.close(fd)
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
            dialogs.notify_engine_error(self, "Preview unavailable", error)
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

