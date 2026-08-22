; 32-bit AI Voice Studio installer.
; Requires a 32-bit Python 3.13 build of the app in dist\AIVS-32.
; Compile with: iscc installer_32.iss

#define MyAppArch "x86"
#define ArchAllowed "x86"
#define Arch64BitMode ""
#define DistDir "..\dist\AIVS-32\AI-Voice-Studio"

#include "installer_common.iss"
