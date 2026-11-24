; AI Voice Studio - shared Inno Setup script.
; Included by installer_64.iss / installer_32.iss which define
; MyAppArch (x64|x86), ArchAllowed, Arch64BitMode and DistDir.

; [Files] below ships everything under DistDir, including the bundled Python
; runtime staged by stage_python_runtime.py.  Without it the installed
; application cannot create a single per-engine virtualenv (python -m venv
; needs a full interpreter), so a dist that has no runtime is refused here as
; well as in build.ps1 - this script can also be compiled directly with iscc.
#if !FileExists(AddBackslash(SourcePath) + DistDir + "\python-runtime\python.exe")
  #error The bundled Python runtime is missing from DistDir. Run packaging\build.ps1 for this architecture (it stages python-runtime) before compiling this setup.
#endif

[Setup]
AppId={{D9B4E3A0-7A1E-4E8B-9C2F-AIVOICESTUDIO01}
AppName=AI Voice Studio
AppVersion=2027.1.0
AppPublisher=Anuj Sharma
; The three ARP links winget and "Apps and features" read back.  They must
; agree with the locale manifest (winget/PackageUrl, PublisherSupportUrl,
; ReleaseNotesUrl) for the package to be recognisable as the same product.
AppPublisherURL=https://github.com/anujj87
AppSupportURL=https://github.com/anujj87/AIVoiceStudio/issues
AppUpdatesURL=https://github.com/anujj87/AIVoiceStudio/releases
DefaultDirName={autopf}\AI Voice Studio
DefaultGroupName=AI Voice Studio
; The "install for all users / install for me only" choice stays: an
; interactive install can still be elevated for everybody or kept per-user.
; PrivilegesRequired=lowest is the default, so {autopf} resolves to
; {localappdata}\Programs unless the person picks otherwise, which is what the
; winget manifest's Scope: user records.  Because the override is allowed,
; Inno's UsePreviousPrivileges applies and an earlier install of this AppId is
; found and updated in place, with its mode preselected.
; The dialog is not a problem for unattended installs: /SUPPRESSMSGBOXES makes
; Setup take the recommended choice instead of waiting (measured on the
; published 2026.3.7 build).  The winget manifest therefore carries that flag
; in both silent modes - see winget/README.md.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; The mode dialog above is a real choice and is kept; this one is not a
; question at all, just a "This will install ... Do you wish to continue?"
; box, so it is switched off.  A scripted install (winget, an unattended
; deployment, Help > Check for updates) must never wait for a click, and
; /VERYSILENT alone does not suppress it.  Scripted installs run through
; packaging\silent_install.ps1, which hands these switches over untouched (a
; POSIX shell would rewrite a leading /VERYSILENT into a path) and waits for
; the uninstaller, which returns before it has finished.
DisableStartupPrompt=yes
; Start Menu folder choice page with a "don't create a folder" checkbox.
AllowNoIcons=yes
ArchitecturesAllowed={#ArchAllowed}
ArchitecturesInstallIn64BitMode={#Arch64BitMode}
LicenseFile=..\LICENSE
Compression=lzma2
SolidCompression=yes
OutputDir=..\dist
OutputBaseFilename=AI-Voice-Studio-v-2027-1-0-Setup-{#MyAppArch}
UninstallDisplayName=AI Voice Studio
UninstallDisplayIcon={app}\AI-Voice-Studio.exe
WizardStyle=modern
DisableProgramGroupPage=no
VersionInfoVersion=2027.1.0
VersionInfoCompany=Anuj Sharma
VersionInfoDescription=AI Voice Studio installer
VersionInfoProductName=AI Voice Studio
VersionInfoProductVersion=2027.1.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "readme"; Description: "Open the &Read Me after installation"; GroupDescription: "Other:"
Name: "launch"; Description: "&Launch AI Voice Studio after setup"; GroupDescription: "Other:"

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; The bundled copy of README.md lives inside _internal; ship a top-level copy
; so the "View the Read Me" shortcut and installer work with {app}\README.md.
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
; HTML documentation (Help menu opens these from {app}\docs).
Source: "..\docs\README.html"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\UserGuide.html"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\AddonDevelopmentGuide.html"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\AccessibilityGuide.html"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\THIRD-PARTY-LICENSES.html"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\WHATS-NEW.html"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AI Voice Studio"; Filename: "{app}\AI-Voice-Studio.exe"
Name: "{autodesktop}\AI Voice Studio"; Filename: "{app}\AI-Voice-Studio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AI-Voice-Studio.exe"; Description: "{cm:LaunchProgram,AI Voice Studio}"; Flags: nowait postinstall skipifsilent; Tasks: launch
Filename: "{app}\docs\README.html"; Description: "View the Read Me"; Flags: shellexec postinstall skipifsilent; Tasks: readme

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\AIVoiceStudio\logs"
