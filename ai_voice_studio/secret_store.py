"""Secrets at rest: the Hugging Face token and the OmniVoice API key.

``settings.json`` is a per-user file in ``%APPDATA%\\AIVoiceStudio``, and two of
its values are credentials the user typed in: the Hugging Face **read token**
(what makes a gated model downloadable) and the OmniVoice server's ``api_key``.
Storing them as plain JSON means any process running as the user - a screensaver
helper, a backup agent, a sync client, a support bundle the user zips and sends -
reads them without asking.  For a desktop application the honest fix is the one
Windows already offers: **DPAPI**, ``CryptProtectData`` with
``CRYPTPROTECT_UI_FORBIDDEN``, which encrypts the value so that only the same
user on the same machine can decrypt it, with no password to prompt for and no
key file to protect.

The shape of the file stays readable
------------------------------------
Only the values are encrypted; the JSON structure, the key names and every other
setting stay as they were, so a corrupted token can still be found and replaced
by hand, and a settings file from an older version keeps working:

* a value written by this module is the string ``dpapi:<base64 blob>``;
* a value without that prefix is legacy plain text - it is read as-is and
  re-written encrypted the next time the file is saved (see
  ``Settings.load``).

Off Windows - or if DPAPI itself fails - :func:`protect` returns the text
unchanged and :func:`available` says so.  A token that cannot be encrypted must
not become a token that cannot be *used*: the application then behaves exactly
as it did before this module existed, and the log says why.
"""

from __future__ import annotations

import base64
import logging
import sys

log = logging.getLogger(__name__)

#: What an encrypted value starts with, in ``settings.json``.
PREFIX = "dpapi:"

#: Settings keys whose value is a credential.  Matched on the last dotted part
#: as well as on the whole key, because ``api_key`` sits inside a section.
SECRET_KEYS = frozenset({"hf.token", "api_key"})

#: ``CRYPTPROTECT_UI_FORBIDDEN``: never show a Windows prompt from a save.
_UI_FORBIDDEN = 0x1

# Set by _load_api(): the ctypes prototypes, or None when DPAPI is unavailable.
_api = None
_api_loaded = False


class _Api:
    """The ctypes pieces DPAPI needs, resolved once per process."""

    __slots__ = ("blob", "crypt32", "ctypes", "kernel32")

    def __init__(self, ctypes_module, blob_type, crypt32, kernel32):
        self.ctypes = ctypes_module
        self.blob = blob_type
        self.crypt32 = crypt32
        self.kernel32 = kernel32


def _load_api():
    """Build the ctypes bindings for ``CryptProtectData``/``CryptUnprotectData``."""
    global _api, _api_loaded
    if _api_loaded:
        return _api
    _api_loaded = True
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _DataBlob(ctypes.Structure):
            """``DATA_BLOB`` from ``dpapi.h``."""

            _fields_ = [
                ("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_char)),
            ]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        _api = _Api(ctypes, _DataBlob, crypt32, kernel32)
    except Exception as exc:  # pragma: no cover - depends on the interpreter
        log.debug("DPAPI is not available: %s", exc)
        _api = None
    return _api


def available() -> bool:
    """True when secrets written here are really encrypted."""
    return _load_api() is not None


def _blob_to_bytes(blob) -> bytes:
    api = _load_api()
    try:
        return api.ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        if blob.pbData:
            api.kernel32.LocalFree(blob.pbData)


def _bytes_to_blob(payload: bytes):
    api = _load_api()
    buffer = api.ctypes.create_string_buffer(payload, len(payload))
    return api.blob(
        len(payload),
        api.ctypes.cast(buffer, api.ctypes.POINTER(api.ctypes.c_char)),
    ), buffer


def protect(text: str) -> str:
    """Encrypt ``text`` for this user; returns the value to store.

    An already-protected value is returned unchanged, so a save of an
    unmodified settings file cannot encrypt an encrypted string.
    """
    if not text or is_protected(text):
        return text
    api = _load_api()
    if api is None:
        return text
    try:
        payload = text.encode("utf-8")
        in_blob, _buffer = _bytes_to_blob(payload)
        out_blob = api.blob()
        if not api.crypt32.CryptProtectData(
            api.ctypes.byref(in_blob), "AI Voice Studio", None, None, None,
            _UI_FORBIDDEN, api.ctypes.byref(out_blob),
        ):
            raise OSError(api.ctypes.get_last_error())
        return PREFIX + base64.b64encode(_blob_to_bytes(out_blob)).decode("ascii")
    except Exception as exc:
        log.warning("Could not encrypt a stored secret (%s); it is saved unencrypted", exc)
        return text


def unprotect(value: str) -> str:
    """Decrypt a stored value; plain text is returned unchanged.

    A value that cannot be decrypted (written by another Windows account, or
    corrupted) is reported as empty rather than raised: the application then
    behaves as if no token were configured, which is a state it already handles
    with a message telling the user to set one.
    """
    if not isinstance(value, str) or not is_protected(value):
        return value if isinstance(value, str) else ""
    api = _load_api()
    if api is None:
        log.warning("A stored secret is encrypted but DPAPI is unavailable here")
        return ""
    try:
        blob = base64.b64decode(value[len(PREFIX):])
        in_blob, _buffer = _bytes_to_blob(blob)
        out_blob = api.blob()
        if not api.crypt32.CryptUnprotectData(
            api.ctypes.byref(in_blob), None, None, None, None,
            _UI_FORBIDDEN, api.ctypes.byref(out_blob),
        ):
            raise OSError(api.ctypes.get_last_error())
        return _blob_to_bytes(out_blob).decode("utf-8", errors="replace")
    except Exception as exc:
        log.warning("Could not decrypt a stored secret: %s", exc)
        return ""


def is_protected(value) -> bool:
    """True when ``value`` is one of this module's encrypted strings."""
    return isinstance(value, str) and value.startswith(PREFIX)


def is_secret_key(key: str) -> bool:
    """True when a settings key names a credential."""
    return key in SECRET_KEYS or key.rsplit(".", 1)[-1] in SECRET_KEYS


def _walk(node, transform, prefix: str = ""):
    """Apply ``transform`` to every secret value in a nested dict, in place.

    The dotted path is carried down, so ``hf.token`` is recognised as a whole
    even though the leaf key is only ``"token"``.
    """
    if not isinstance(node, dict):
        return node
    for key, value in list(node.items()):
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _walk(value, transform, path)
        elif is_secret_key(path) and isinstance(value, str):
            node[key] = transform(value)
    return node


def protect_settings(data: dict) -> dict:
    """A copy of ``data`` with every secret replaced by its stored form."""
    from copy import deepcopy

    return _walk(deepcopy(data), protect)


def unprotect_settings(data: dict) -> dict:
    """A copy of ``data`` with every secret replaced by its usable form."""
    from copy import deepcopy

    return _walk(deepcopy(data), unprotect)


def has_plaintext_secrets(data: dict) -> bool:
    """True when any secret in ``data`` is stored unencrypted."""
    found = []

    def note(value: str) -> str:
        if value and not is_protected(value):
            found.append(value)
        return value

    _walk(data, note)
    return bool(found)


__all__ = [
    "PREFIX",
    "SECRET_KEYS",
    "available",
    "has_plaintext_secrets",
    "is_protected",
    "is_secret_key",
    "protect",
    "protect_settings",
    "unprotect",
    "unprotect_settings",
]
