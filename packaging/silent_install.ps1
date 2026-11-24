# AI Voice Studio - unattended install / uninstall of the built setup file.
#
# Why this script exists
# ----------------------
# Inno's switches begin with "/", and a shell that translates POSIX paths
# rewrites them before Setup ever sees them: in Git Bash / MSYS2
#
#     Setup.exe /VERYSILENT /SUPPRESSMSGBOXES      # becomes
#     Setup.exe "C:/Program Files/Git/VERYSILENT" "C:/Program Files/Git/SUPPRESSMSGBOXES"
#
# Setup then ignores the switches and opens its wizard, so a scripted install
# waits on a dialog that nobody is there to click - it never exits.  PowerShell
# passes its arguments through untouched, which is what this script uses.  (A
# bare `/VERYSILENT` is also not enough on its own: `/SUPPRESSMSGBOXES` is the
# switch that answers Inno's install-mode dialog - see winget/README.md.)
#
# It also waits for the uninstaller: Inno's unins000.exe copies itself to a
# temporary file, exits 0 immediately and finishes the removal in that copy, so
# a script that only checks the exit code sees success while the files are
# still there.
#
# Usage (from the repository root, any shell):
#   powershell -ExecutionPolicy Bypass -File packaging\silent_install.ps1
#   powershell -ExecutionPolicy Bypass -File packaging\silent_install.ps1 `
#       -Dir "$env:TEMP\aivs-e2e" -Log "$env:TEMP\aivs-install.log"
#   powershell -ExecutionPolicy Bypass -File packaging\silent_install.ps1 `
#       -Uninstall -Dir "$env:TEMP\aivs-e2e"
#
# Exit code: 0 only when Setup or the uninstaller really succeeded, and after an
# uninstall only when the files are actually gone.

param(
    # The setup file to run.  Empty means the newest built x64 installer.
    [string]$Setup = "",
    # Install directory, passed to Setup as /DIR= (its own default is
    # %LOCALAPPDATA%\Programs\AI Voice Studio).  For -Uninstall this locates
    # unins000.exe; empty reads the install location from the registry.
    [string]$Dir = "",
    # Inno setup/uninstall log.  Empty leaves logging to the installer.
    [string]$Log = "",
    # Install for every user (needs elevation) instead of the per-user default.
    [switch]$AllUsers,
    # Remove an existing installation instead of installing.
    [switch]$Uninstall,
    # How long to wait for install, and for the uninstaller to finish.
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# The Inno AppId, so the uninstaller can be found without being told the
# directory.  It is deliberately never changed between versions.
$AppId = "{D9B4E3A0-7A1E-4E8B-9C2F-AIVOICESTUDIO01}_is1"
$DefaultDir = Join-Path $env:LOCALAPPDATA "Programs\AI Voice Studio"

function Get-Switches {
    # The complete unattended switch set.  Passed as an array so PowerShell
    # hands each one to Setup as written.
    $switches = @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-")
    if ($AllUsers) { $switches += "/ALLUSERS" } else { $switches += "/CURRENTUSER" }
    if ($Dir) { $switches += "/DIR=$Dir" }
    if ($Log) { $switches += "/LOG=$Log" }
    return $switches
}

function Resolve-Setup {
    if ($Setup) {
        if (-not (Test-Path $Setup)) { throw "Setup file not found: $Setup" }
        return (Resolve-Path $Setup).Path
    }
    $found = Get-ChildItem (Join-Path $root "dist\*-Setup-x64.exe") -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $found) {
        throw "No installer in dist\. Build one first: powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Arch x64"
    }
    return $found.FullName
}

function Get-Uninstaller {
    if ($Dir) {
        $exe = Join-Path $Dir "unins000.exe"
        if (-not (Test-Path $exe)) { throw "No uninstaller at $exe (was that the install directory?)" }
        return $exe
    }
    foreach ($hive in @("HKCU:", "HKLM:")) {
        $key = "$hive\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$AppId"
        if (Test-Path $key) {
            $entry = Get-ItemProperty $key
            $candidate = ""
            if ($entry.InstallLocation) { $candidate = Join-Path $entry.InstallLocation "unins000.exe" }
            if (-not (Test-Path $candidate) -and $entry.UninstallString) {
                $candidate = $entry.UninstallString.Trim('"')
            }
            if ($candidate -and (Test-Path $candidate)) { return $candidate }
        }
    }
    throw "AI Voice Studio is not installed for this account or for all users; pass -Dir to point at the installation."
}

function Get-RuntimeCache {
    # ``__pycache__`` files the application's own Python wrote next to the
    # runtime's modules the first time it ran.  Inno removes the files *it*
    # installed; these are not among them, and they are what made the wait
    # below run out its clock on any machine where the app had been started -
    # reported as "timed out with 47 file(s) left" for a folder that was in
    # fact uninstalled.  They are regenerable, so the driver clears them.
    param([string]$Path)
    if (-not $Path -or -not (Test-Path $Path)) { return @() }
    return @(Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue |
             Where-Object { $_.Extension -eq ".pyc" -or $_.Extension -eq ".pyo" })
}

function Wait-ForRemoval {
    # The uninstaller finishes in a copy of itself, after it has already
    # returned.  Wait for that copy rather than trusting the exit code.
    param([string]$Path)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        Start-Sleep -Seconds 2
        $busy = Get-Process -Name "unins000", "_iu*" -ErrorAction SilentlyContinue
        if ($Path -and (Test-Path $Path)) {
            $files = @(Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue)
            $caches = @(Get-RuntimeCache -Path $Path)
        } else {
            $files = @()
            $caches = @()
        }
        $installed = $files.Count - $caches.Count
    } while (($busy -or $installed -gt 0) -and (Get-Date) -lt $deadline)
    return @{
        Left = $files.Count
        Caches = $caches.Count
        Installed = $installed
        TimedOut = ((Get-Date) -ge $deadline) -and ($busy -or $installed -gt 0)
    }
}

if ($Uninstall) {
    $uninstaller = Get-Uninstaller
    $target = if ($Dir) { (Resolve-Path $Dir).Path } else { Split-Path -Parent $uninstaller }
    Write-Host "== Uninstalling from $target ==" -ForegroundColor Cyan
    Write-Host "   using $uninstaller" -ForegroundColor DarkGray
    $proc = Start-Process -FilePath $uninstaller -ArgumentList (Get-Switches) -Wait -PassThru
    Write-Host "   uninstaller exited $($proc.ExitCode) (it finishes in a copy of itself)" -ForegroundColor DarkGray
    $wait = Wait-ForRemoval -Path $target
    if ($wait.TimedOut) {
        Write-Host "   timed out after $TimeoutSeconds s with $($wait.Installed) file(s) left" -ForegroundColor Red
        exit 1
    }
    if ($wait.Left -gt 0 -and $wait.Caches -eq $wait.Left) {
        # Only the runtime's bytecode caches: the uninstall itself is complete.
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "   removed $target, its Start Menu entry and $($wait.Left) bytecode cache file(s) the runtime wrote" -ForegroundColor Green
        exit 0
    }
    if ($wait.Left -gt 0) {
        Write-Host "   $($wait.Left) file(s) could not be removed (a running copy of the app locks them)" -ForegroundColor Red
        exit 1
    }
    if ($proc.ExitCode -ne 0) {
        Write-Host "   uninstaller reported failure: exit $($proc.ExitCode)" -ForegroundColor Red
        exit 1
    }
    Write-Host "   removed $target and its Start Menu entry" -ForegroundColor Green
    exit 0
}

$installer = Resolve-Setup
$switches = Get-Switches
Write-Host "== Installing unattended ==" -ForegroundColor Cyan
Write-Host "   $installer" -ForegroundColor DarkGray
Write-Host "   $($switches -join ' ')" -ForegroundColor DarkGray
$proc = Start-Process -FilePath $installer -ArgumentList $switches -Wait -PassThru
if ($proc.ExitCode -ne 0) {
    Write-Host "   Setup failed: exit $($proc.ExitCode)" -ForegroundColor Red
    if ($Log) { Write-Host "   see $Log" -ForegroundColor DarkGray }
    exit 1
}
$installed = if ($Dir) { $Dir } else { $DefaultDir }
if (-not (Test-Path $installed)) {
    Write-Host "   Setup exited 0 but nothing is at $installed" -ForegroundColor Red
    exit 1
}
$count = @(Get-ChildItem $installed -Recurse -File -ErrorAction SilentlyContinue).Count
Write-Host "   installed $count file(s) into $installed" -ForegroundColor Green
Write-Host "   uninstall with: powershell -ExecutionPolicy Bypass -File packaging\silent_install.ps1 -Uninstall -Dir `"$installed`"" -ForegroundColor DarkGray
exit 0
