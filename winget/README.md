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

Each generated file starts with the `yaml-language-server` schema line that
`microsoft/winget-pkgs` requires as the first line of a submitted manifest.

`tools/winget_validate.py` checks the shipped files against winget's rules and
is also run by `tests/test_winget.py`, so a manifest that stops matching the
application fails the test suite instead of a submission review.

`SUBMISSION-STEPS.txt` in this folder is the full runbook for getting the
package (and each later version of it) into `winget-pkgs`.

## How the package is described

| winget field | value | why |
| --- | --- | --- |
| `PackageIdentifier` | `AnujSharma.AIVoiceStudio` | `Publisher.PackageName`, no spaces |
| `InstallerType` | `inno` | built with Inno Setup (`packaging/*.iss`) |
| `Scope` | `user` | the default install is per-user: `PrivilegesRequired=lowest`, so no administrator rights are needed and {autopf} is `%LOCALAPPDATA%\Programs`. The mode dialog can still elevate to an all-users install, which registers per-machine and is outside what winget models |
| `InstallerSwitches.Silent` | `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART` | Inno's own switch; the installer's `[Run]` entries are `skipifsilent`, so a silent install never launches the app |
| `InstallerSwitches.SilentWithProgress` | `/SILENT /SUPPRESSMSGBOXES /NORESTART` | same flags, keeping the progress window |
| `ProductCode` | `{D9B4E3A0-…}_is1` | the Inno `AppId` plus `_is1`, i.e. the real "Apps and features" entry |
| `UpgradeBehavior` | `install` | a newer version replaces the installed one |
| `UpgradeCode` | not used | this is not an MSI |

The `AppId` in `packaging/installer_common.iss` is deliberately **not**
changed: it is the key Windows uses to recognise an installed copy, so
changing it would leave every existing installation behind as an orphan.

### Why the silent switches are spelled out in full

winget knows Inno's defaults (`/VERYSILENT /SUPPRESSMSGBOXES /NORESTART` and
`/SILENT /SUPPRESSMSGBOXES /NORESTART`) and uses them when a manifest declares
no `InstallerSwitches`. Declaring `Silent` and `SilentWithProgress` replaces
those two values outright, so whatever we write there has to carry **every**
flag the installer needs — a bare `/VERYSILENT` is not a shorter way of saying
the default, it is a different, quieter-but-still-blocking install.

`/SUPPRESSMSGBOXES` is the flag that matters. `/VERYSILENT` hides the wizard
but, as the Inno documentation says, "error messages during installation are
displayed" and the startup prompt is still shown unless `DisableStartupPrompt`
or `/SP-` is used. That is what failed the first submission: with a bare
`/VERYSILENT`, Setup's "This will install ... Do you wish to continue?" prompt
and the install-mode dialog (added by the `PrivilegesRequiredOverridesAllowed
=dialog` the script keeps) both appeared, and winget-pkgs' automated
installation test had nobody there to answer them — hence its *"this
application failed to install without user input"*.

**Measured, not assumed:** the published 2026.3.7 installer run with
`/VERYSILENT /NORESTART` is still waiting for a click after 60 seconds and
installs nothing, while the same file with `/SUPPRESSMSGBOXES` installs and
exits 0.

The script keeps the *Select Setup Install Mode* dialog on purpose, so the flag
is the thing that makes an unattended install work: winget-pkgs' step 08 passed
on the published 2026.3.7 bytes with exactly these switches. Setup's own
"Do you wish to continue?" prompt is a different, wholly choiceless box and is
turned off by `DisableStartupPrompt=yes` in the script, so no build waits on it
even if the flag were ever dropped. Spelling the switches out in full also means
a later change to the script cannot silently alter what winget runs.
`ai_voice_studio/updates.py` (`launch_installer`) passes the same set when the
application updates itself from Help > Check for updates.

### Install mode

The installer is per-user by default, which is what the manifest's `Scope: user`
records: `PrivilegesRequired=lowest` puts {autopf} at
`%LOCALAPPDATA%\Programs` and needs no administrator rights. Inno's install-mode
dialog is kept (`PrivilegesRequiredOverridesAllowed=dialog`), so an interactive
user can still choose *install for all users* — that elevates, writes to
`C:\Program Files` and registers a per-machine ARP entry, which winget does not
track because the manifest describes the per-user path. Nothing waits on the
dialog in a scripted install: `/SUPPRESSMSGBOXES` makes Setup take the
recommended choice, which without elevation is *for me only*.

Because the override is allowed, Inno's `UsePreviousPrivileges` applies: an
existing installation of this `AppId` is found and updated in place, with the
mode it was installed with preselected. So a user who chose all users earlier
sees that same choice again and the copy in `C:\Program Files` is upgraded —
not duplicated.

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
6. Add the `<version>` folder to `microsoft/winget-pkgs` at
   `manifests/a/AnujSharma/AIVoiceStudio/<version>/` and open a pull request -
   see `SUBMISSION-STEPS.txt` for the branch, the PR title and the CLA. The
   first submission additionally needs the repository to be public and the
   publisher name to match the GitHub account.

Until the package is accepted into `winget-pkgs`, users can still install it
from a local copy of this repository with

```
winget install --manifest winget\manifests\a\AnujSharma\AIVoiceStudio\<version>
```
