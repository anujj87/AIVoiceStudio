"""Per-project OmniVoice options dialog.

Exposes the full OmniVoice feature set for the Recording window:

* **Voice mode** - Auto voice / Voice design / Voice clone.
* **Voice design** - build an instruction from the canonical attribute
  vocabulary (gender, age, pitch, whisper style, English accent, Chinese
  dialect) or type a free-form description; both engines use the same
  instruction language.
* **Voice clone** - pick a 3-15 s reference sample and an optional
  transcript (leave blank -> the engine auto-transcribes it via Whisper).
* **Generation knobs** - diffusion steps, CFG guidance scale, token
  sampling temperature, voice-diversity temperature, denoise, fixed
  duration, optional RNG seed and an optional language hint.
* **Inline controls** - non-verbal symbols and pronunciation hints are
  passed through untouched, so the supported tags are listed here.

Parameters that a given engine/package version cannot apply are skipped
and reported by the engines (see ``omnivoice/spec.py`` and the worker).

The dialog itself never talks to an engine: it edits and returns the
canonical ``omni`` dict (``omnivoice/spec.build_omni``) that the Recording
window stores in ``project.json`` under ``tts.omni``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import wx

from ..omnivoice import spec
from .a11y import add_labeled

_NONE = "(not set)"

_MODE_CHOICES = [
    ("auto", "Auto voice (let OmniVoice pick)"),
    ("design", "Voice design (describe the voice)"),
    ("clone", "Voice clone (copy a reference sample)"),
]


def _combo_value(combo: wx.ComboBox) -> str:
    """Return the client data of the current selection, or \"\"."""
    sel = combo.GetSelection()
    if sel < 0:
        return ""
    data = combo.GetClientData(sel)
    return str(data) if data else ""


class OmniVoiceOptionsDialog(wx.Dialog):
    """Full-feature OmniVoice voice options for one project."""

    def __init__(
        self,
        parent,
        engine_label: str = "OmniVoice",
        omni: Optional[Dict[str, Any]] = None,
        project_name: str = "",
    ):
        super().__init__(
            parent,
            title=f"OmniVoice options - {project_name}" if project_name else "OmniVoice options",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            size=(620, 720),
        )
        self._engine_label = engine_label
        self._omni = spec.build_omni(**(omni or {}))
        self._build_ui()
        self._load(self._omni)
        self._on_mode(None)
        wx.CallAfter(self.mode_box.SetFocus)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(
            wx.StaticText(
                self,
                label=f"{self._engine_label} gives you three ways to choose a "
                      "voice plus full control over how the audio is generated. "
                      "These options are saved with this project only.",
            ),
            0, wx.ALL, 6,
        )

        # -- Voice mode ---------------------------------------------------
        self.mode_box = wx.RadioBox(
            self, label="Voice mode", choices=[label for _, label in _MODE_CHOICES],
        )
        self.mode_box.SetName("OmniVoice voice mode")
        sizer.Add(self.mode_box, 0, wx.ALL, 6)

        # -- Design group -------------------------------------------------
        self.design_group = wx.StaticBoxSizer(
            wx.StaticBox(self, label="Voice design"), wx.VERTICAL
        )
        design_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        design_grid.AddGrowableCol(1)

        # name -> combo (values are the canonical attribute strings)
        self.attr_combos: Dict[str, wx.ComboBox] = {}
        attribute_rows = [
            ("Gender", "gender", ["male", "female"]),
            ("Age", "age", [
                "child", "teenager", "young adult", "middle-aged", "elderly",
            ]),
            ("Pitch", "pitch", [
                "very low pitch", "low pitch", "moderate pitch",
                "high pitch", "very high pitch",
            ]),
            ("Style", "style", ["whisper"]),
            ("English accent", "accent_en", [
                "american accent", "british accent", "australian accent",
                "chinese accent", "canadian accent", "indian accent",
                "korean accent", "portuguese accent", "russian accent",
                "japanese accent",
            ]),
            ("Chinese dialect", "dialect_zh", [
                "河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话",
                "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话",
            ]),
        ]
        for label, key, values in attribute_rows:
            combo = wx.ComboBox(self.design_group.GetStaticBox(), style=wx.CB_READONLY)
            combo.Append(_NONE, None)
            for value in values:
                combo.Append(value, value)
            combo.SetSelection(0)
            combo.SetName(f"{label} attribute")
            combo.Bind(wx.EVT_COMBOBOX, lambda _e, c=combo: self._compose_instruct(c))
            self.attr_combos[key] = combo
            add_labeled(self.design_group.GetStaticBox(), design_grid, label, combo,
                        flag=wx.LEFT | wx.RIGHT, border=2)

        self.design_group.Add(design_grid, 0, wx.EXPAND | wx.ALL, 4)
        self.design_group.Add(
            wx.StaticText(
                self.design_group.GetStaticBox(),
                label="Instructions (comma separated):",
            ),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 6,
        )
        self.instruct_ctrl = wx.TextCtrl(
            self.design_group.GetStaticBox(), style=wx.TE_MULTILINE,
            size=(-1, 44),
        )
        self.instruct_ctrl.SetName("Instructions (comma separated)")
        self.instruct_ctrl.SetToolTip(
            "Describe the voice freely, e.g. 'female, low pitch, british accent, "
            "whisper'. Pick attributes above or type your own - what you type is "
            "sent to the model."
        )
        self.design_group.Add(self.instruct_ctrl, 0, wx.EXPAND | wx.ALL, 6)
        sizer.Add(self.design_group, 0, wx.EXPAND | wx.ALL, 6)

        # -- Clone group --------------------------------------------------
        self.clone_group = wx.StaticBoxSizer(
            wx.StaticBox(self, label="Voice clone"), wx.VERTICAL
        )
        clone_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        clone_grid.AddGrowableCol(1)

        self.ref_audio_ctrl = wx.TextCtrl(self.clone_group.GetStaticBox())
        self.ref_audio_ctrl.SetName("Reference audio")
        browse_btn = wx.Button(self.clone_group.GetStaticBox(), label="Browse...")
        browse_btn.SetName("Browse reference audio")
        browse_btn.Bind(wx.EVT_BUTTON, lambda _e: self._on_browse())
        ref_box = wx.BoxSizer(wx.HORIZONTAL)
        ref_box.Add(self.ref_audio_ctrl, 1, wx.EXPAND)
        ref_box.Add(browse_btn, 0, wx.LEFT, 4)
        ref_label = wx.StaticText(
            self.clone_group.GetStaticBox(), label="Reference audio:"
        )
        ref_label.SetName("Reference audio label")
        clone_grid.Add(
            ref_label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2
        )
        clone_grid.Add(ref_box, 1, wx.EXPAND)

        clone_grid.Add(
            wx.StaticText(
                self.clone_group.GetStaticBox(),
                label="Reference text (optional):",
            ),
            0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2,
        )
        self.ref_text_ctrl = wx.TextCtrl(
            self.clone_group.GetStaticBox(), style=wx.TE_MULTILINE, size=(-1, 48),
        )
        self.ref_text_ctrl.SetName("Reference text (optional)")
        self.ref_text_ctrl.SetToolTip(
            "The transcript of the sample. Leave empty to let the engine "
            "transcribe it automatically (Whisper)."
        )
        clone_grid.Add(self.ref_text_ctrl, 1, wx.EXPAND)
        self.clone_group.Add(clone_grid, 0, wx.EXPAND | wx.ALL, 4)
        self.clone_group.Add(
            wx.StaticText(
                self.clone_group.GetStaticBox(),
                label="Tip: use a clean 3-15 second sample of one speaker. For "
                      "standard pronunciation, use a sample in the same language "
                      "as the text you record.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )
        sizer.Add(self.clone_group, 0, wx.EXPAND | wx.ALL, 6)

        # -- Advanced generation box -------------------------------------
        adv = wx.StaticBoxSizer(
            wx.StaticBox(self, label="Advanced generation settings"), wx.VERTICAL
        )
        adv_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        adv_grid.AddGrowableCol(1)

        self.language_ctrl = wx.TextCtrl(adv.GetStaticBox())
        self.language_ctrl.SetName("Language hint")
        self.language_ctrl.SetToolTip(
            "Optional language hint for pronunciation. Examples: "
            + ", ".join(spec.LANGUAGE_EXAMPLES)
            + ". Leave empty to auto-detect from the text."
        )
        add_labeled(adv.GetStaticBox(), adv_grid, "Language hint (optional)",
                    self.language_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)

        self.num_step_ctrl = wx.SpinCtrl(
            adv.GetStaticBox(), min=spec.NUM_STEP_MIN, max=spec.NUM_STEP_MAX,
            initial=spec.DEFAULT_NUM_STEP, name="Diffusion steps",
        )
        self.num_step_ctrl.SetToolTip(
            "Diffusion steps (16 = faster, 32 = balanced, 64 = best quality)."
        )
        add_labeled(adv.GetStaticBox(), adv_grid, "Diffusion steps",
                    self.num_step_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)

        self.guidance_ctrl = wx.SpinCtrlDouble(
            adv.GetStaticBox(), min=spec.GUIDANCE_MIN, max=spec.GUIDANCE_MAX,
            initial=spec.DEFAULT_GUIDANCE_SCALE, inc=0.1, name="Guidance scale",
        )
        self.guidance_ctrl.SetToolTip(
            "Classifier-free guidance strength (0-10). Higher = stronger voice "
            "conditioning."
        )
        add_labeled(adv.GetStaticBox(), adv_grid, "Guidance scale",
                    self.guidance_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)

        self.class_temp_ctrl = wx.SpinCtrlDouble(
            adv.GetStaticBox(), min=spec.CLASS_TEMP_MIN, max=spec.CLASS_TEMP_MAX,
            initial=spec.DEFAULT_CLASS_TEMPERATURE, inc=0.1,
            name="Class temperature",
        )
        self.class_temp_ctrl.SetToolTip(
            "Token sampling temperature (0 = greedy/deterministic, higher = more "
            "random)."
        )
        add_labeled(adv.GetStaticBox(), adv_grid, "Class temperature",
                    self.class_temp_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)

        self.position_temp_ctrl = wx.SpinCtrlDouble(
            adv.GetStaticBox(), min=spec.POSITION_TEMP_MIN, max=spec.POSITION_TEMP_MAX,
            initial=spec.DEFAULT_POSITION_TEMPERATURE, inc=0.1,
            name="Position temperature",
        )
        self.position_temp_ctrl.SetToolTip(
            "Voice-diversity temperature (server mode; 0 = deterministic, "
            "higher = more variation)."
        )
        add_labeled(adv.GetStaticBox(), adv_grid, "Position temperature",
                    self.position_temp_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)

        self.denoise_cb = wx.CheckBox(adv.GetStaticBox(), label="Enable denoising")
        self.denoise_cb.SetName("Denoising")
        self.denoise_cb.SetToolTip(
            "Post-process the audio with the model's denoising token "
            "(recommended). Server mode only."
        )
        adv_grid.Add((1, 1))
        adv_grid.Add(self.denoise_cb, 0, wx.ALL, 2)

        self.duration_ctrl = wx.SpinCtrlDouble(
            adv.GetStaticBox(), min=0.0, max=spec.DURATION_MAX, initial=0.0,
            inc=0.5, name="Fixed duration",
        )
        self.duration_ctrl.SetToolTip(
            "Generate audio of a fixed length in seconds (0 = automatic). "
            "Overrides the speed slider when set. Server mode only."
        )
        add_labeled(adv.GetStaticBox(), adv_grid, "Fixed duration (seconds, 0 = off)",
                    self.duration_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)

        self.seed_ctrl = wx.TextCtrl(adv.GetStaticBox())
        self.seed_ctrl.SetName("Random seed")
        self.seed_ctrl.SetToolTip(
            "Optional RNG seed: the same text and seed reproduce the same "
            "audio. Leave empty for fresh randomness."
        )
        add_labeled(adv.GetStaticBox(), adv_grid, "Random seed (optional)",
                    self.seed_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)

        adv.Add(adv_grid, 0, wx.EXPAND | wx.ALL, 4)
        adv.Add(
            wx.StaticText(
                adv.GetStaticBox(),
                label="Only parameters supported by the installed engine are "
                      "applied; the others are skipped and reported. On the "
                      "direct engine, speed uses the Rate slider (approximate "
                      "resampling); the server applies it natively.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )
        sizer.Add(adv, 0, wx.EXPAND | wx.ALL, 6)

        # -- Inline symbols help ------------------------------------------
        help_box = wx.StaticBoxSizer(
            wx.StaticBox(self, label="Inline sounds and pronunciation (text level)"),
            wx.VERTICAL,
        )
        help_box.Add(
            wx.StaticText(
                help_box.GetStaticBox(),
                label="OmniVoice reads these markers straight from the text - "
                      "just type them in your document:",
            ),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 6,
        )
        help_box.Add(
            wx.StaticText(
                help_box.GetStaticBox(),
                label="    " + ", ".join(f"[{t}]" for t in spec.NONVERBAL_TAGS),
            ),
            0, wx.LEFT | wx.RIGHT, 6,
        )
        help_box.Add(
            wx.StaticText(
                help_box.GetStaticBox(),
                label="Also supported: pinyin tone hints for Chinese (e.g. "
                      "'打ZHE2') and CMU dictionary hints for English (e.g. "
                      "'[B EY1 S]'). Give each tag about 8-10 words of text so "
                      "the model does not produce a drone.",
            ),
            0, wx.ALL, 6,
        )
        sizer.Add(help_box, 0, wx.EXPAND | wx.ALL, 6)

        btns = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btns, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(sizer)
        self.SetMinSize((560, 620))

        self.mode_box.Bind(wx.EVT_RADIOBOX, self._on_mode)

    # -- behaviour ---------------------------------------------------------
    def _on_browse(self):
        with wx.FileDialog(
            self,
            "Choose the reference voice sample (3-15 seconds)",
            wildcard="Audio files (*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac)|"
                     "*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.ref_audio_ctrl.SetValue(dlg.GetPath())

    def _on_mode(self, _):
        mode = _MODE_CHOICES[self.mode_box.GetSelection()][0]
        self._show_group(self.design_group, mode == "design")
        self._show_group(self.clone_group, mode == "clone")
        self.Layout()
        if self.GetSizer():
            self.GetSizer().Fit(self)

    @staticmethod
    def _show_group(group_sizer: wx.StaticBoxSizer, visible: bool):
        """Show or hide a whole StaticBox group (box + every descendant)."""
        box = group_sizer.GetStaticBox()

        def set_tree(win: wx.Window):
            win.Show(visible)
            for child in win.GetChildren():
                if isinstance(child, wx.Window) and child is not win:
                    set_tree(child)

        set_tree(box)

    def _compose_instruct(self, _changed: wx.ComboBox):
        parts = [v for v in (_combo_value(c) for c in self.attr_combos.values()) if v]
        text = ", ".join(parts)
        self.instruct_ctrl.SetValue(text)

    # -- state -------------------------------------------------------------
    def _load(self, omni: Dict[str, Any]):
        mode = omni.get("mode", "auto")
        mode_idx = next(
            (i for i, (key, _label) in enumerate(_MODE_CHOICES) if key == mode), 0
        )
        self.mode_box.SetSelection(mode_idx)

        instruct = (omni.get("instruct") or "").strip()
        self.instruct_ctrl.SetValue(instruct)
        # Only pre-select combos whose values appear in the instruction text;
        # instructions typed by hand may not map cleanly back.
        for key, combo in self.attr_combos.items():
            combo.SetSelection(0)
        if instruct:
            for key, combo in self.attr_combos.items():
                for i in range(1, combo.GetCount()):
                    value = combo.GetClientData(i)
                    if value and str(value) in instruct:
                        combo.SetSelection(i)
                        break

        self.ref_audio_ctrl.SetValue(omni.get("ref_audio") or "")
        self.ref_text_ctrl.SetValue(omni.get("ref_text") or "")
        self.language_ctrl.SetValue(omni.get("language") or "")

        self.num_step_ctrl.SetValue(int(omni.get("num_step") or spec.DEFAULT_NUM_STEP))
        self.guidance_ctrl.SetValue(float(
            omni.get("guidance_scale") if omni.get("guidance_scale") is not None
            else spec.DEFAULT_GUIDANCE_SCALE
        ))
        self.class_temp_ctrl.SetValue(float(
            omni.get("class_temperature") if omni.get("class_temperature") is not None
            else spec.DEFAULT_CLASS_TEMPERATURE
        ))
        self.position_temp_ctrl.SetValue(float(
            omni.get("position_temperature")
            if omni.get("position_temperature") is not None
            else spec.DEFAULT_POSITION_TEMPERATURE
        ))
        self.denoise_cb.SetValue(
            omni.get("denoise") if omni.get("denoise") is not None else spec.DEFAULT_DENOISE
        )
        duration = omni.get("duration")
        self.duration_ctrl.SetValue(float(duration) if duration else 0.0)
        seed = omni.get("seed")
        self.seed_ctrl.SetValue(str(seed) if seed is not None else "")

    def get_omni(self) -> Dict[str, Any]:
        """Return the canonical ``omni`` dict the Recording window saves."""
        mode = _MODE_CHOICES[self.mode_box.GetSelection()][0]

        instruct = ""
        if mode == "design":
            instruct = self.instruct_ctrl.GetValue().strip()
        elif mode == "auto":
            instruct = ""

        duration = self.duration_ctrl.GetValue()
        seed_text = self.seed_ctrl.GetValue().strip()
        try:
            seed = int(seed_text) if seed_text else None
        except ValueError:
            seed = None

        return spec.build_omni(
            mode=mode,
            instruct=instruct,
            ref_audio=self.ref_audio_ctrl.GetValue().strip() if mode == "clone" else "",
            ref_text=self.ref_text_ctrl.GetValue().strip() if mode == "clone" else "",
            language=self.language_ctrl.GetValue().strip() or None,
            num_step=int(self.num_step_ctrl.GetValue()),
            guidance_scale=float(self.guidance_ctrl.GetValue()),
            class_temperature=float(self.class_temp_ctrl.GetValue()),
            position_temperature=float(self.position_temp_ctrl.GetValue()),
            denoise=bool(self.denoise_cb.GetValue()),
            duration=duration if duration > 0 else None,
            seed=seed,
        )
