"""Per-category compute choice for the Preview buttons.

Every settings category that has a **Preview** button offers a small
**Compute** combo right next to it: CPU, GPU (when an NVIDIA GPU with a
working driver is detected) and Auto when there is a choice.  The choice is
remembered per category in Settings (``preview_compute.<category>``), and the
preview uses it:

* the pip-installed engines (Voice Lab: Pocket TTS, Bark, F5-TTS) run
  on their own PyTorch, so "GPU" means CUDA there without any extra runtime;
* the ONNX engines (Piper, Kokoro, ...) use the sherpa-onnx GPU back-end,
  which additionally needs the CUDA runtime downloaded from Settings > Compute
  (the preview reports that, it never silently falls back).

Nothing here is widget-specific, so the tests can drive the resolution
without wx.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

#: Settings key prefix.  It must *not* be ``compute.preview``: ``compute`` is
#: a plain string setting (the global back-end), so nesting under it would
#: make ``Settings.set`` fail and silently drop every category's choice.
SETTINGS_PREFIX = "preview_compute"


def choices() -> List[Tuple[str, str]]:
    """``(value, label)`` pairs for the combo (CPU first, GPU when present)."""
    from .. import compute  # noqa: PLC0415

    options: List[Tuple[str, str]] = [("cpu", "CPU")]
    if compute.has_nvidia_gpu():
        options.append(("cuda", "GPU (CUDA)"))
    if len(options) > 1:
        options.append(("auto", "Auto (GPU when possible)"))
    return options


def settings_key(category: str) -> str:
    """Settings key that stores one category's preview compute choice."""
    return f"{SETTINGS_PREFIX}.{category}"


def _stored_choice(settings, category: str):
    """The category's saved choice, or ``None`` when the user never chose."""
    try:
        value = str(settings.get(settings_key(category), "") or "").lower()
    except Exception:  # noqa: BLE001
        return None
    valid = {value for value, _label in choices()}
    return value if value in valid else None


def saved_choice(settings, category: str) -> str:
    """The stored choice for a category (``cpu`` until the user changes it)."""
    return _stored_choice(settings, category) or "cpu"


def provider_for_preview(choice: str, engine_id: Optional[str] = None) -> str:
    """Turn the combo value into the provider string ``get_engine`` expects.

    ``engine_id`` is the voice's engine: the Voice Lab engines resolve their
    own device (CUDA via their own PyTorch), every other engine goes through
    the ONNX compute detection (which needs the downloaded GPU runtime).
    """
    from .. import compute  # noqa: PLC0415

    if engine_id:
        from ..voicelab import engines as voice_lab  # noqa: PLC0415

        if voice_lab.is_engine(engine_id):
            from ..voicelab import resolve_device  # noqa: PLC0415

            return resolve_device(choice)
    return compute.provider_for(compute.resolve_compute(choice))


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
def _combo_value(combo) -> str:
    selection = combo.GetSelection()
    if selection < 0:
        return "cpu"
    return str(combo.GetClientData(selection) or "cpu")


def make_compute_row(parent, settings, category: str, default: str = "cpu"):
    """Build ``(label, combo)`` for a category's Preview row.

    Every settings category that has a Preview button gets one of these, so
    the back-end can be chosen wherever a voice is heard.  The combo lists
    CPU, GPU (when an NVIDIA GPU is detected) and Auto, and remembers the
    category's choice in Settings immediately, so the next preview (in this
    session or the next) uses the same back-end.  ``default`` is used until
    the user picks something (an engine that *needs* the GPU, like OmniVoice,
    passes ``"cuda"``).
    """
    import wx  # noqa: PLC0415

    from .a11y import set_accessible_name  # noqa: PLC0415

    label = wx.StaticText(parent, label="Compute:")
    combo = wx.ComboBox(parent, style=wx.CB_READONLY, name="Compute")
    set_accessible_name(combo, "Compute")
    combo.SetToolTip(
        "Which back-end the Preview uses: CPU, GPU (CUDA, when an NVIDIA GPU "
        "with a driver is detected) or Auto. The ONNX engines need the GPU "
        "runtime downloaded from Settings > Compute."
    )
    options = choices()
    for value, text in options:
        combo.Append(text, value)
    available = {value for value, _text in options}
    stored = _stored_choice(settings, category)
    current = stored or (default if default in available else "cpu")
    if current not in available:
        current = "cpu"
    for index in range(combo.GetCount()):
        if combo.GetClientData(index) == current:
            combo.SetSelection(index)
            break
    else:
        combo.SetSelection(0)

    def _persist(_evt):
        if settings is None:
            return
        try:
            settings.set(settings_key(category), _combo_value(combo))
        except Exception:  # noqa: BLE001
            pass

    combo.Bind(wx.EVT_COMBOBOX, _persist)
    return label, combo
