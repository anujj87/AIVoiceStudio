; 64-bit AI Voice Studio installer.
; Compile with: iscc installer_64.iss   (after building the 64-bit dist)

#define MyAppArch "x64"
#define ArchAllowed "x64compatible"
#define Arch64BitMode "x64compatible"
#define DistDir "..\dist\AIVS-64\AI-Voice-Studio"

#include "installer_common.iss"
