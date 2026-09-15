# AI Voice Studio - installer build pipeline (Windows).
#
# Prerequisites:
#   * 64-bit Python 3.13 venv with requirements.txt installed (for x64 build)
#   * 32-bit Python 3.13 venv with requirements.txt installed (for x86 build)
#   * Inno Setup 6 (ISCC.exe) on PATH
#
# Usage (from the repository root):
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Arch x64
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Arch x86
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Arch both
#
# Output: dist\AI-Voice-Studio-Setup-x64.exe / -x86.exe

param(
    [ValidateSet("x64", "x86", "both")]
    [string]$Arch = "x64",
    # Optional explicit compiler path, e.g.
    #   -IsccPath 'C:\Program Files\Inno Setup 7\ISCC.exe'
    [string]$IsccPath = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# Locate the Inno Setup compiler (ISCC.exe): PATH first, then an explicit
# -IsccPath, then the standard Inno Setup 6/7 install locations for both
# Program Files flavours.  (A bare string literal here would be emitted to the
# output stream and get captured by the caller, hence real comments.)
function Find-Iscc {
    $candidates = @()
    $onPath = Get-Command iscc -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += $onPath.Source }
    if ($IsccPath) { $candidates += $IsccPath }
    foreach ($base in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $base) { continue }
        foreach ($version in @("Inno Setup 7", "Inno Setup 6")) {
            $candidates += (Join-Path $base "$version\ISCC.exe")
        }
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return $candidate }
    }
    return $null
}

function Build-Arch {
    param([string]$a, [string]$venv, [string]$iss)

    Write-Host "== Building $a ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        $py = Join-Path $venv "Scripts\python.exe"
        if (-not (Test-Path $py)) {
            throw "Python venv not found: $py (create it and install requirements.txt)"
        }
        $distArch = if ($a -eq "x64") { "64" } else { "32" }
        & $py -m PyInstaller packaging\ai_voice_studio.spec --noconfirm `
            --distpath "dist\AIVS-$distArch" --workpath "build\pyinstaller-$a"
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed for $a" }

        Write-Host "== Compiling $a installer ==" -ForegroundColor Cyan
        $iscc = Find-Iscc
        if (-not $iscc) {
            throw ("Inno Setup ISCC.exe was not found. Install Inno Setup 6 or 7, " +
                   "put iscc on PATH, or pass -IsccPath 'C:\Program Files\Inno Setup 7\ISCC.exe'.")
        }
        Write-Host "   using $iscc" -ForegroundColor DarkGray
        & $iscc $iss
        if ($LASTEXITCODE -ne 0) { throw "ISCC failed for $a" }
    } finally {
        Pop-Location
    }
}

switch ($Arch) {
    "x64" {
        Build-Arch "x64" "$root\.venv" "$root\packaging\installer_64.iss"
    }
    "x86" {
        # Point to your 32-bit venv, e.g. .venv32
        Build-Arch "x86" "$root\.venv32" "$root\packaging\installer_32.iss"
    }
    "both" {
        Build-Arch "x64" "$root\.venv" "$root\packaging\installer_64.iss"
        Build-Arch "x86" "$root\.venv32" "$root\packaging\installer_32.iss"
    }
}

Write-Host "Done. Installers are in dist\." -ForegroundColor Green
