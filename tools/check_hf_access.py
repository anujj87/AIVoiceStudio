"""Check the gated-model support against the live Hugging Face API.

Run this by hand (``python tools/check_hf_access.py``); the test suite never
touches the network.  It answers the four questions a change to this feature can
get wrong, each against huggingface.co rather than against a mock:

1. **Are the catalog's ``gated`` flags still true?** Every describing and
   speech-to-text repository is asked for its own ``gated`` field and compared
   with what the catalog claims.  A repository that becomes gated (or is
   un-gated) without the flag following it is the failure that puts a bare ``401``
   in front of a user again.
2. **Is the user's failure reproduced, and recognised?** An anonymous request for
   a gated checkpoint's ``config.json`` is expected to answer ``401``; the body is
   then passed through :func:`ai_voice_studio.hf_access.access_message`, which has
   to turn it into the two steps that fix it.
3. **Does the run stop before it downloads?** With a throwaway settings file and
   the token sources cleared, the engine's own download path is asked for
   Gemma 3 12B; it must refuse with the page and the Settings panel named, and
   must not have started a worker process.
4. **Does ``Check token`` really talk to Hugging Face?** A deliberately invalid
   token is sent to the whoami endpoint and must be refused with a readable
   reason rather than raising.
5. **Does ``Check access`` really talk to Hugging Face?** Every gated repository
   this application ships - Gemma 3's three and Kyutai's voice-cloning Pocket
   TTS weights - is asked for with no token (must answer "no token") and with a
   token that does not exist (must be refused, and must say whether the token or
   the licence is the problem); an open model must be recognised as one that
   needs no account at all.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_voice_studio import hf_access
from ai_voice_studio.describe import catalog as describe_catalog
from ai_voice_studio.describe import engine as describe_engine
from ai_voice_studio.settings import Settings
from ai_voice_studio.stt import engine as stt_engine
from ai_voice_studio.voicelab import engines as voicelab_engines

API = "https://huggingface.co/api/models/{repo}"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/config.json"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def repositories() -> list[tuple[str, bool, str]]:
    """``(repo, claimed gated, where it is used)`` for every catalog entry."""
    found: dict[str, tuple[bool, str]] = {}
    for model_id in describe_catalog.model_ids():
        for value, _label in describe_catalog.variants(model_id):
            repo = describe_catalog.hf_model(model_id, value)
            if repo:
                found[repo] = (describe_catalog.gated(model_id, value),
                               f"describe:{model_id}/{value}")
    for model_id, variants in stt_engine.HF_MODELS.items():
        for value, repo in variants.items():
            # The STT catalog refuses gated entries outright, so it claims False.
            found.setdefault(repo, (False, f"stt:{model_id}/{value}"))
    # The Voice Lab's gated entry: Pocket TTS clones from Kyutai's repository,
    # which is gated, while the pre-made voices do not need it at all.
    for engine_id in voicelab_engines.engine_ids():
        repo = voicelab_engines.gated_repo(engine_id)
        if repo:
            found[repo] = (True, f"voicelab:{engine_id}")
    return [(repo, gated, where) for repo, (gated, where) in sorted(found.items())]


def live_gated(repo: str) -> object:
    with urllib.request.urlopen(API.format(repo=repo), timeout=30) as handle:
        return json.load(handle).get("gated")


def check_catalog_flags() -> None:
    print("== 1. the catalog's gated flags, against the live API ==")
    for repo, claimed, where in repositories():
        try:
            actual = bool(live_gated(repo))
        except urllib.error.HTTPError as exc:
            check(f"{repo} ({where})", False, f"HTTP {exc.code}")
            continue
        except Exception as exc:
            check(f"{repo} ({where})", False, f"{type(exc).__name__}: {exc}")
            continue
        check(f"{repo}: catalog says gated={claimed}, API says gated={actual} "
              f"({where})", claimed == actual)


def check_failure_text() -> None:
    print("\n== 2. the anonymous 401, and what the application makes of it ==")
    repo = describe_catalog.hf_model("gemma3", "12b")
    try:
        urllib.request.urlopen(RESOLVE.format(repo=repo), timeout=30)
        check(f"an anonymous fetch of {repo} is refused", False,
              "the request succeeded, so the repository is no longer gated")
        return
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        check(f"an anonymous fetch of {repo} answers {exc.code}", exc.code == 401)
    except Exception as exc:
        check(f"an anonymous fetch of {repo} is refused", False,
              f"{type(exc).__name__}: {exc}")
        return
    check("the refusal is recognised as an access failure",
          hf_access.is_access_error(body))
    message = hf_access.access_message(body, repo, "Gemma 3")
    check("it becomes the two steps that unblock it",
          hf_access.model_page(repo) in message
          and hf_access.TOKEN_PAGE in message
          and "Settings > Hugging Face account" in message)


def check_preflight() -> None:
    print("\n== 3. a gated run stops before any download ==")
    folder = tempfile.mkdtemp(prefix="aivs_check_hf_")
    settings = Settings(path=os.path.join(folder, "settings.json"))
    hf_access.use_settings(settings)
    saved = {name: os.environ.pop(name, None) for name in hf_access.ENV_VARS}
    os.environ.pop("HF_TOKEN_PATH", None)
    started: list = []

    def _refuse(*args, **kwargs):
        started.append(args)
        raise AssertionError("a worker process was started")

    original = describe_engine.subprocess.Popen
    describe_engine.subprocess.Popen = _refuse
    try:
        result = describe_engine.download_variant("gemma3", "12b")
    finally:
        describe_engine.subprocess.Popen = original
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
        hf_access.use_settings(None)
    check("a token is really absent in this check", not hf_access.has_token())
    check("no worker process was started", not started)
    check("the refusal names the model page, the licence and the panel",
          bool(result.get("error"))
          and hf_access.model_page("google/gemma-3-12b-it") in result["error"]
          and "licence" in result["error"]
          and "Settings > Hugging Face account" in result["error"],
          " | ".join(str(result.get("error") or "").splitlines()[:2]))
    print("   the message the user would read:")
    for line in str(result.get("error") or "").splitlines():
        print(f"     {line}")


def check_token_check() -> None:
    print("\n== 4. Check token, against the real endpoint ==")
    ok, detail = hf_access.verify("hf_this_token_does_not_exist_0000")
    check("an invalid token is refused with a readable reason", not ok, detail)
    ok, detail = hf_access.verify("")
    check("an empty field is refused before any request", not ok, detail)


def check_access_check() -> None:
    print("\n== 5. Check access, against the real endpoint ==")
    folder = tempfile.mkdtemp(prefix="aivs_check_hf_access_")
    settings = Settings(path=os.path.join(folder, "settings.json"))
    hf_access.use_settings(settings)
    saved = {name: os.environ.pop(name, None) for name in hf_access.ENV_VARS}
    os.environ.pop("HF_TOKEN_PATH", None)
    try:
        for row in hf_access.gated_repos():
            repo = row["repo"]
            state, detail = hf_access.check_access(repo)
            check(f"{repo}: with no token the answer is \"no token\"",
                  state == hf_access.ACCESS_NO_TOKEN, f"{state} - {detail}")

        repo = hf_access.gated_repos()[1]["repo"]
        state, detail = hf_access.check_access(repo, "hf_a_token_that_does_not_exist")
        check(f"{repo}: a token that does not exist is refused",
              state == hf_access.ACCESS_REFUSED, f"{state} - {detail}")
        check("the refusal says whether the token or the licence is the problem",
              "token" in detail.lower() and "licence" in detail.lower())
        print("   what the panel would say:")
        for line in detail.splitlines():
            print(f"     {line}")

        state, detail = hf_access.check_access(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        check("an open model is recognised as needing no account",
              state == hf_access.ACCESS_OPEN, f"{state} - {detail}")
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
        hf_access.use_settings(None)


def main() -> int:
    check_catalog_flags()
    check_failure_text()
    check_preflight()
    check_token_check()
    check_access_check()
    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
