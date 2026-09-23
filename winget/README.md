# winget package for AI Voice Studio

[winget](https://learn.microsoft.com/windows/package-manager/) is the Windows
Package Manager. Installing with it is one line:

```
winget install AnujSharma.AIVoiceStudio
```

and it is what makes the application discoverable in
`winget search voice`, in the Microsoft Store's package search, and in
`winget upgrade` — which is how most users will receive new versions.

## What is in here

```
winget/manifests/a/AnujSharma/AIVoiceStudio/<version>/
    AnujSharma.AIVoiceStudio.yaml                 version manifest
    AnujSharma.AIVoiceStudio.installer.yaml       installer manifest
    AnujSharma.AIVoiceStudio.locale.en-US.yaml    metadata for people
```

The layout (`manifests/<first letter>/<Publisher>/<Package>/<Version>/`) is
exactly the one the public
[`microsoft/winget-pkgs`](https://github.com/microsoft/winget-pkgs)
repository requires, so the folder can be copied there unchanged.

The manifests are **generated, not hand-written**:

```
.venv/Scripts/python.exe tools/make_winget.py
```

`tools/winget_validate.py` checks the shipped files against winget's rules and
is also run by `tests/test_winget.py`, so a manifest that stops matching the
application fails the test suite instead of a submission review.

## How the package is described

| winget field | value | why |
| --- | --- | --- |
| `PackageIdentifier` | `AnujSharma.AIVoiceStudio` | `Publisher.PackageName`, no spaces |
| `InstallerType` | `inno` | built with Inno Setup (`packaging/*.iss`) |
| `Scope` | `user` | the installer defaults to a per-user install |
| `InstallerSwitches.Silent` | `/VERYSILENT` | Inno's own switch; the installer's `[Run]` entries are `skipifsilent`, so a silent install never launches the app |
| `ProductCode` | `{D9B4E3A0-…}_is1` | the Inno `AppId` plus `_is1`, i.e. the real "Apps and features" entry |
| `UpgradeBehavior` | `install` | a newer version replaces the installed one |
| `UpgradeCode` | not used | this is not an MSI |

The `AppId` in `packaging/installer_common.iss` is deliberately **not**
changed: it is the key Windows uses to recognise an installed copy, so
changing it would leave every existing installation behind as an orphan.

## Publishing a new version

1. Bump `constants.APP_VERSION` and the version lines in
   `packaging/installer_common.iss` (see `DEV_PLAN.md`).
2. Build: `powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Arch x64`
   (add `-Arch x86` when the 32-bit installer is wanted too).
3. Generate the manifests — they pick up the SHA256 of the built installers:
   `.venv/Scripts/python.exe tools/make_winget.py`
4. Run the checker: `.venv/Scripts/python.exe tools/winget_validate.py`.
5. Publish the GitHub release whose tag is `v<version>`, attaching the
   installer named exactly `AI-Voice-Studio-v-<dashed-version>-Setup-x64.exe`
   (the name `packaging/build.ps1` already produces). The manifests' download
   URLs point at that asset, so the release must exist before the manifests
   are submitted.
6. Copy the generated `<version>` folder into a fork of `microsoft/winget-pkgs`
   at `manifests/a/AnujSharma/AIVoiceStudio/<version>/`, then open a pull
   request. The first submission additionally needs the repository to be
   public and the publisher name to match the GitHub account.

Until the package is accepted into `winget-pkgs`, users can still install it
from a local copy of this repository with

```
winget install --manifest winget\manifests\a\AnujSharma\AIVoiceStudio\<version>
```
