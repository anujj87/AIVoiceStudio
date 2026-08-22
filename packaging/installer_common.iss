; AI Voice Studio - shared Inno Setup script.
; Included by installer_64.iss / installer_32.iss which define
; MyAppArch (x64|x86), ArchAllowed, Arch64BitMode and DistDir.

[Setup]
AppId={{D9B4E3A0-7A1E-4E8B-9C2F-AIVOICESTUDIO01}
AppName=AI Voice Studio
AppVersion=0.1.0
AppPublisher=Anuj Sharma
AppPublisherURL=https://github.com/anujj87
AppSupportURL=https://github.com/
DefaultDirName={autopf}\AI Voice Studio
DefaultGroupName=AI Voice Studio
; "Install for all users / for me only" radio button on the directory page.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Start Menu folder choice page with a "don't create a folder" checkbox.
AllowNoIcons=yes
ArchitecturesAllowed={#ArchAllowed}
ArchitecturesInstallIn64BitMode={#Arch64BitMode}
LicenseFile=..\LICENSE
Compression=lzma2
SolidCompression=yes
OutputDir=..\dist
OutputBaseFilename=AI-Voice-Studio-Setup-{#MyAppArch}
UninstallDisplayName=AI Voice Studio
UninstallDisplayIcon={app}\AI-Voice-Studio.exe
WizardStyle=modern
DisableProgramGroupPage=no
VersionInfoVersion=0.1.0
VersionInfoCompany=Anuj Sharma
VersionInfoDescription=AI Voice Studio installer
VersionInfoProductName=AI Voice Studio
VersionInfoProductVersion=0.1.0

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
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AI Voice Studio"; Filename: "{app}\AI-Voice-Studio.exe"
Name: "{autodesktop}\AI Voice Studio"; Filename: "{app}\AI-Voice-Studio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AI-Voice-Studio.exe"; Description: "{cm:LaunchProgram,AI Voice Studio}"; Flags: nowait postinstall skipifsilent; Tasks: launch
Filename: "{app}\docs\README.html"; Description: "View the Read Me"; Flags: shellexec postinstall skipifsilent; Tasks: readme

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\AIVoiceStudio\logs"
