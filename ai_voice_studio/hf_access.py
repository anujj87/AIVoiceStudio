"""Hugging Face access: the token a *gated* model needs, and why.

Almost every model in this application downloads with no account at all.  A few
do not: Google's Gemma 3 lives in a **gated** repository, so ``transformers``
and ``huggingface_hub`` answer a bare ``401`` for it -

    Cannot access gated repo for url https://huggingface.co/google/gemma-3-12b-it/
    Access to model google/gemma-3-12b-it is restricted. You must have access to
    it and be authenticated to access it. Please log in.

- which tells the user nothing they can do from inside this application.  A
person who installed the setup file has no ``huggingface-cli`` to log in with,
and the two things that actually fix it (accepting the model's terms on its own
Hugging Face page, and handing this application a token for that account) are
nowhere in that message.

This module is the single answer to "which token, from where, and is it any
good":

* :func:`token` looks for one in three places, in this order - the
  application's own settings (Settings > Hugging Face account), then the
  ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variables, then the file
  ``huggingface-cli login`` writes;
* :func:`worker_environment` hands it to every model worker as ``HF_TOKEN``, so
  the library that does the download sees it;
* :func:`access_message` turns a gated/401/403 failure into the two steps that
  fix it, and :func:`gated_help` is what a run says *before* it starts when the
  chosen checkpoint is gated and no token is configured yet;
* :func:`verify` asks huggingface.co who the token belongs to, which is the only
  way to tell a working token from a mistyped one without starting a download;
* :func:`check_access` asks the harder question - *this* token, *this*
  repository, right now - and separates the two failures that share one error
  message: a token that is not accepted, and a licence that has not been
  accepted by the account the token belongs to.  Only the second one is fixed
  on the model's page, and only the first one in this panel.

The token is the user's own *read* token.  It is kept in this user's
``settings.json`` (the same file, and the same plain-text footing, as the
``huggingface-cli login`` cache) and is never written anywhere else.

One token serves every gated model - there is no per-model token: what makes a
model downloadable is accepting *its* licence, with the account the token
belongs to.  :data:`GATED_REPOS` is the list of repositories this application
offers that are gated, so the Settings panel can offer the page of the one being
set up (and check that it is in place) instead of making the user find it.
"""

from __future__ import annotations

import os
import re

import requests

#: The settings keys this module reads (``Settings`` is a dotted-key store).
TOKEN_SETTING = "hf.token"
ACCOUNT_SETTING = "hf.account"

#: Environment variables ``huggingface_hub`` itself honours, in its own order.
ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

#: Where a token can be created, and where an account's tokens are listed.
TOKEN_PAGE = "https://huggingface.co/settings/tokens"

#: The endpoint that answers "whose token is this?" (goes with the token only).
WHOAMI_URL = "https://huggingface.co/api/whoami-v2"

#: Where a model's repository page is (accept its terms here).
MODEL_PAGE = "https://huggingface.co/{repo}"

#: The repositories this application offers whose download is **gated**: their
#: licence has to be accepted on the model's own page, with the account the
#: token belongs to, before a single byte can be fetched.  Google's three Gemma 3
#: checkpoints are gated ``manual`` (access is granted on request), Kyutai's
#: voice-cloning Pocket TTS weights are gated ``auto`` (accepting is instant),
#: AI4Bharat's Indic Parler-TTS asks for contact details the same way, and
#: everything else in the catalog downloads anonymously.
#:
#: A Voice Lab engine says which repository it is one of through its own
#: ``gated_repo`` metadata; ``tests/test_hf_access.py`` holds this list and those
#: engines to each other, and ``tools/check_hf_access.py`` holds both to what
#: huggingface.co answers for each repository.
GATED_REPOS: tuple[dict[str, str], ...] = (
    {"repo": "google/gemma-3-4b-it", "label": "Gemma 3 4B (video describer)"},
    {"repo": "google/gemma-3-12b-it", "label": "Gemma 3 12B (video describer)"},
    {"repo": "google/gemma-3-27b-it", "label": "Gemma 3 27B (video describer)"},
    {"repo": "kyutai/pocket-tts", "label": "Pocket TTS voice cloning (Voice Lab)"},
    {
        "repo": "ai4bharat/indic-parler-tts",
        "label": "Indic Parler-TTS (Indic voices)",
    },
)

#: The public API: what a repository is (``gated``) and which files it holds.
MODEL_API = "https://huggingface.co/api/models/{repo}"

#: One file inside a repository.  A gated one answers ``401`` here until access
#: is in place, which is exactly what an access check needs to ask - the file is
#: chosen as the smallest one the API lists, so a *successful* check downloads a
#: few hundred bytes rather than a checkpoint.
FILE_URL = "https://huggingface.co/{repo}/resolve/main/{path}"

#: What :func:`check_access` found.
ACCESS_OK = "ok"              # this token's account can download it
ACCESS_NO_TOKEN = "no-token"  # gated, and no token is configured at all
ACCESS_REFUSED = "refused"    # the licence is not accepted, or the token is not
ACCESS_OPEN = "open"          # not gated: no account is needed
ACCESS_UNKNOWN = "unknown"    # Hugging Face could not be asked

#: What a gated download says; matched case-insensitively against a failure.
#: The last three are Kyutai's wording for Pocket TTS's voice-cloning weights
#: ("... go to https://huggingface.co/kyutai/pocket-tts and accept the terms,
#: then make sure you're logged in locally with `uvx hf auth login`"), which
#: names a command a person who installed the setup file does not have.
_GATED_MARKERS = (
    "gated",
    "gatedrepoerror",
    "must be authenticated",
    "cannot access",
    "unauthorized",
    "you must have access",
    "access to model",
    "please log in",
    "repository not found",  # what a private/gated repo answers anonymously
    "accept the terms",
    "logged in locally",
    "hf auth login",
)

#: ``401`` / ``403`` as whole tokens: a request id in the middle of the text must
#: not be mistaken for an HTTP status.
_STATUS_RE = re.compile(r"(?<!\d)40[13](?!\d)")


def model_page(repo: str) -> str:
    """The Hugging Face page of a repository (``""`` in, ``""`` out)."""
    repo = (repo or "").strip().strip("/")
    return MODEL_PAGE.format(repo=repo) if repo else ""


def gated_repos() -> list[dict[str, str]]:
    """Copies of the gated-repository registry, for a picker."""
    return [dict(item) for item in GATED_REPOS]


def gated_entry(repo: str) -> dict[str, str]:
    """The registry entry for ``repo`` (``{}`` when it is not one we ship)."""
    wanted = _clean(repo).lower().strip("/")
    for item in GATED_REPOS:
        if item["repo"].lower() == wanted:
            return dict(item)
    return {}


def gated_label(repo: str) -> str:
    """A human name for a gated repository (its id when it is not in ours)."""
    return gated_entry(repo).get("label") or _clean(repo)


def cli_token_path() -> str:
    """The file ``huggingface-cli login`` writes, honouring HF's own overrides."""
    explicit = os.environ.get("HF_TOKEN_PATH")
    if explicit:
        return explicit
    home = os.environ.get("HF_HOME")
    if home:
        return os.path.join(home, "token")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                        "token")


def _clean(value) -> str:
    return str(value or "").strip()


#: The settings store the application handed over, if any (see
#: :func:`use_settings`).
_LIVE_SETTINGS = None


def use_settings(settings) -> None:
    """Hand this module the application's live settings store.

    The GUI's store is the same file the Hugging Face panel writes, but it is
    the *instance* it writes to: reading that instance means a token pasted a
    second ago is in force for the next run, with no re-read and no chance of a
    second copy of the file falling out of step with the first.
    """
    global _LIVE_SETTINGS
    _LIVE_SETTINGS = settings


def settings_store():
    """The store to read a token from: the live one, or a fresh read of it.

    Without a live store (the engines are also used from scripts and tests) the
    application's settings file is read directly; it is a few kilobytes of JSON
    and is read at most once per worker launch.
    """
    if _LIVE_SETTINGS is not None:
        return _LIVE_SETTINGS
    from .settings import Settings

    return Settings()


def stored_token(settings=None) -> str:
    """The token the user typed into Settings (``""`` when there is none)."""
    if settings is None:
        try:
            settings = settings_store()
        except Exception:
            return ""
    return _clean(settings.get(TOKEN_SETTING, ""))


def env_token() -> str:
    """The token from ``HF_TOKEN`` (or its older spelling)."""
    for name in ENV_VARS:
        value = _clean(os.environ.get(name))
        if value:
            return value
    return ""


def cli_token() -> str:
    """The token saved by ``huggingface-cli login``, when there is one."""
    try:
        with open(cli_token_path(), encoding="utf-8") as handle:
            return _clean(handle.read())
    except OSError:
        return ""


def token(settings=None) -> str:
    """The token to use, from Settings, the environment, or the CLI cache."""
    return stored_token(settings) or env_token() or cli_token()


def token_source(settings=None) -> str:
    """Where :func:`token` would take a token from, in the user's words.

    Empty when there is no token at all, which is what the preflight and the
    panel both need to know.
    """
    if stored_token(settings):
        return "Settings > Hugging Face account"
    if env_token():
        return f"the {ENV_VARS[0]} environment variable"
    if cli_token():
        return "a previous huggingface-cli login"
    return ""


def has_token(settings=None) -> bool:
    return bool(token(settings))


def account(settings=None) -> str:
    """The account name last verified with the token (``""`` if never)."""
    if settings is None:
        try:
            settings = settings_store()
        except Exception:
            return ""
    return _clean(settings.get(ACCOUNT_SETTING, ""))


def worker_environment(settings=None) -> dict[str, str]:
    """The environment a model worker is started with.

    Everything the application itself runs in, plus ``HF_TOKEN`` when a token is
    configured, so the download a worker performs is authenticated.  Passing the
    environment explicitly (rather than relying on inheritance) is what makes a
    token typed into Settings reach a worker started later in the same session.
    """
    env = dict(os.environ)
    value = token(settings)
    if value:
        # Both spellings: huggingface_hub reads HF_TOKEN, older transformers
        # releases only HUGGING_FACE_HUB_TOKEN.
        for name in ENV_VARS:
            env[name] = value
    return env


def is_access_error(text: str) -> bool:
    """True when a failure is a missing/refused Hugging Face access, not a bug."""
    lowered = (text or "").lower()
    if any(marker in lowered for marker in _GATED_MARKERS):
        return True
    return bool(_STATUS_RE.search(text or ""))


def gated_help(repo: str = "", model_name: str = "") -> str:
    """The two steps that make a gated repository downloadable.

    Deliberately names the *page* and the *panel*: the failure itself names
    neither, and both are needed before a single byte can be fetched.
    """
    what = _clean(model_name) or "This model"
    if repo:
        what = f"{what} ({repo})"
    page = model_page(repo) or "its page on huggingface.co"
    return (
        f"{what} downloads from a gated Hugging Face repository, so it needs "
        "your own Hugging Face account. One token serves every gated model: "
        "what this needs is accepting *this* model's licence with the same "
        "account.\n\n"
        f"1. Open {page} and accept the model's licence (its terms of use) - a "
        "one-time click, with the account that will download it.\n"
        f"2. Create a read token at {TOKEN_PAGE}, then paste it into Settings > "
        "Hugging Face account in this application (the Check token button says "
        "whether it works, and Check access says whether the licence is in "
        "place).\n\n"
        "Then run this again: the download starts authenticated, and nothing is "
        "fetched before the access is in place."
    )


def access_message(text: str, repo: str = "", model_name: str = "") -> str:
    """``gated_help`` when ``text`` is an access failure, and ``""`` when not.

    Callers use the empty answer as "leave the original message alone", so a
    genuine crash is never dressed up as a missing licence.
    """
    if not is_access_error(text):
        return ""
    return gated_help(repo, model_name)


def verify(token_value: str, timeout: float = 20.0) -> tuple[bool, str]:
    """Ask huggingface.co whose token this is.

    Returns ``(True, account name)`` for a token that works, and
    ``(False, reason)`` for one that does not - which is the only way to tell
    "the token is wrong" from "the licence is not accepted" before a
    multi-gigabyte download is attempted.
    """
    value = _clean(token_value)
    if not value:
        return False, "Enter a token first."
    try:
        response = requests.get(
            WHOAMI_URL, headers={"Authorization": f"Bearer {value}"},
            timeout=timeout,
        )
    except Exception as exc:
        return False, f"Could not reach huggingface.co: {exc}"
    if response.status_code == 401:
        return False, ("Hugging Face did not accept this token (it may be "
                       "mistyped, or the token was revoked).")
    if response.status_code >= 400:
        return False, f"Hugging Face answered {response.status_code}."
    try:
        data = response.json()
    except ValueError:
        return False, "Hugging Face answered something that is not an account."
    name = _clean(data.get("name") or data.get("fullname")) or "your account"
    return True, name


def _repo_info(repo: str, timeout: float) -> tuple[bool, str, str]:
    """``(gated, smallest file, error)`` for a repository, from the public API.

    The API answers for a gated repository too (that is how the catalog's flags
    are held to Hugging Face by ``tools/check_hf_access.py``), and ``siblings``
    carries every file's size, so the smallest file can be used for the probe.
    """
    try:
        response = requests.get(MODEL_API.format(repo=repo),
                                params={"blobs": True}, timeout=timeout)
    except Exception as exc:
        return False, "", f"Could not reach huggingface.co: {exc}"
    if response.status_code == 404:
        return False, "", f"Hugging Face knows no repository called {repo}."
    if response.status_code >= 400:
        return False, "", f"Hugging Face answered {response.status_code} for {repo}."
    try:
        data = response.json()
    except ValueError:
        return False, "", "Hugging Face answered something that is not a repository."
    sizes = [
        (int(item["size"]), str(item.get("rfilename") or ""))
        for item in (data.get("siblings") or [])
        if isinstance(item, dict) and isinstance(item.get("size"), int)
        and item.get("rfilename")
    ]
    return bool(data.get("gated")), (min(sizes)[1] if sizes else ""), ""


def check_access(repo: str, token_value: str = "", timeout: float = 30.0,
                 verify_token: bool = True) -> tuple[str, str]:
    """Can the configured token download ``repo`` **right now**?

    This is the question the failure itself leaves unanswered - ``Cannot access
    gated repo ... Please log in`` is the same message for two different
    problems, and they are fixed in two different places:

    * the token is not accepted (or there is none) - fixed in this panel;
    * the token is fine but its account has not accepted the repository's
      licence - fixed on the model's page, and nowhere else.

    So on a refusal the token is also checked (``verify_token``), and the answer
    says which of the two it is.  Returns ``(state, sentence)`` with the state
    one of :data:`ACCESS_OK`, :data:`ACCESS_NO_TOKEN`, :data:`ACCESS_REFUSED`,
    :data:`ACCESS_OPEN` or :data:`ACCESS_UNKNOWN`, and a sentence ready to show a
    user.  A clean answer needs a network; nothing else does.
    """
    repo = _clean(repo).strip("/")
    if not repo:
        return ACCESS_UNKNOWN, "No repository was given."
    value = _clean(token_value) or token()
    gated, sample, error = _repo_info(repo, timeout)
    if error:
        return ACCESS_UNKNOWN, error
    if not gated:
        return ACCESS_OPEN, (f"{repo} is not gated: it downloads with no "
                             "account at all.")
    where = model_page(repo)
    if not value:
        return ACCESS_NO_TOKEN, (
            f"{repo} is gated, and no token is configured. Open {where}, accept "
            f"its licence with your account, then create a read token at "
            f"{TOKEN_PAGE} and paste it into Settings > Hugging Face account."
        )
    if not sample:
        return ACCESS_UNKNOWN, (
            f"{repo} is gated and Hugging Face listed no files for it, so its "
            "access could not be tested."
        )
    try:
        response = requests.get(
            FILE_URL.format(repo=repo, path=sample),
            headers={"Authorization": f"Bearer {value}"}, timeout=timeout,
        )
    except Exception as exc:
        return ACCESS_UNKNOWN, f"Could not reach huggingface.co: {exc}"
    if response.status_code < 400:
        return ACCESS_OK, (f"Access is in place: this token can download "
                           f"{repo}.")
    if response.status_code not in (401, 403):
        return ACCESS_UNKNOWN, (
            f"Hugging Face answered {response.status_code} for {repo}, which is "
            "neither an allowance nor a refusal."
        )
    if not verify_token:
        return ACCESS_REFUSED, (
            f"Hugging Face refused {repo} (HTTP {response.status_code}): accept "
            f"its licence at {where} with the account this token belongs to."
        )
    works, detail = verify(value, timeout)
    if not works:
        return ACCESS_REFUSED, (
            f"Hugging Face refused {repo}, and it did not accept the token "
            f"either. {detail} Fix the token first (Check token), then accept "
            f"the licence at {where} if the repository is still refused."
        )
    return ACCESS_REFUSED, (
        f"The token works (signed in as {detail}), but that account has not "
        f"been granted {repo} yet. Open {where} and accept its licence - the "
        "button on the page is called \"Agree and access repository\" - then "
        "press Check access again."
    )


def store_token(settings, token_value: str, account_name: str | None = None
                ) -> None:
    """Save (or clear) the token, and the account it was verified with."""
    value = _clean(token_value)
    settings.set(TOKEN_SETTING, value)
    settings.set(ACCOUNT_SETTING, _clean(account_name))
    settings.save()
