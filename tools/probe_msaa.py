"""Probe MSAA accName of the wizard page-1 controls (what NVDA reads)."""
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wx  # noqa: E402

from ai_voice_studio.settings import Settings  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402
from ai_voice_studio.gui.new_project_wizard import NewProjectWizard  # noqa: E402

OBJID_CLIENT = 0xFFFFFFFC
CHILDID_SELF = 0
VT_I4 = 3

# IID_IAccessible = {618736e0-3c3d-11cf-810c-00aa00389b71}
GUID_IAccessible = bytes.fromhex("E03687613D3CCF11810C00AA00389B71")


class VARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("r1", ctypes.c_ushort),
        ("r2", ctypes.c_ushort),
        ("r3", ctypes.c_ushort),
        ("lVal", ctypes.c_long),
        ("l2", ctypes.c_long),
    ]


def acc_name(hwnd):
    iid = ctypes.create_string_buffer(GUID_IAccessible, 16)
    ptr = ctypes.c_void_p()
    hr = ctypes.windll.oleacc.AccessibleObjectFromWindow(
        hwnd, OBJID_CLIENT, ctypes.byref(iid), ctypes.byref(ptr)
    )
    if hr != 0 or not ptr.value:
        return f"<hr={hr:#x}>"
    vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))
    func_addr = vtable[0] and ctypes.cast(vtable[0], ctypes.POINTER(ctypes.c_void_p))[10]
    fn = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, VARIANT, ctypes.POINTER(ctypes.c_wchar_p)
    )(func_addr)
    name = ctypes.c_wchar_p()
    var = VARIANT(vt=VT_I4, lVal=CHILDID_SELF)
    hr = fn(ptr, var, ctypes.byref(name))
    out = name.value or ""
    if name.value:
        ctypes.windll.oleaut32.SysFreeString(name)
    return f"hr={hr:#x} name={out!r}"


app = wx.App(False)
wiz = NewProjectWizard(None, Settings(), ModelStore())
wiz.Show()
wiz.Raise()
page = wiz.page_details
for _ in range(15):
    wx.SafeYield()
    time.sleep(0.3)

for ctrl, tag in (
    (page.name_ctrl, "edit(Project name)"),
    (page.project_type_combo, "combo(Project type)"),
    (page.punct_combo, "combo(Punctuation)"),
):
    print(tag, "->", acc_name(ctrl.GetHandle()), flush=True)

for st in page.GetChildren():
    if isinstance(st, wx.StaticText) and st.GetLabel() in (
        "Project name:", "Project type:", "Punctuation:",
    ):
        print("static", repr(st.GetLabel()), "->",
              acc_name(st.GetHandle()), flush=True)

wiz.Destroy()
print("done", flush=True)