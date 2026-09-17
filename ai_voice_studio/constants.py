"""Application-wide constants."""

APP_NAME = "AI Voice Studio"
APP_ID = "AIVoiceStudio"
APP_VERSION = "2026.3.1"

# Version of the first-launch terms/acceptance dialog. Bump this on every
# release so the dialog shows again after an update (SPEC: user must re-accept
# the terms once per program version; it never repeats within one version).
TERMS_VERSION = "2026.3.1"

# ---------------------------------------------------------------------------
# User data layout (%APPDATA%/AIVoiceStudio by default)
# ---------------------------------------------------------------------------
USER_DATA_DIR_NAME = "AIVoiceStudio"
MODELS_DIR_NAME = "models"
PROJECTS_DIR_NAME = "projects"
FFMPEG_DIR_NAME = "ffmpeg"
LOGS_DIR_NAME = "logs"
SETTINGS_FILE_NAME = "settings.json"
MODELS_STATE_FILE_NAME = "models.json"
PROJECT_FILE_NAME = "project.json"

# JSON log of the exact text that was sent to the TTS engine, written next to
# the audio files as recording progresses (one entry per segment/file).
PROCESSED_TEXT_FILE_NAME = "processed_text.json"

# Where FFmpeg is downloaded when the user accepts the offer (SPEC 3.6).
FFMPEG_DOWNLOAD_URL = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
)
FFMPEG_EXE_NAME = "ffmpeg.exe"

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
THEME_SYSTEM = "system"
THEME_LIGHT = "light"
THEME_DARK = "dark"

# ---------------------------------------------------------------------------
# Recording / synthesis defaults
# ---------------------------------------------------------------------------
PUNCTUATION_DEFAULT = "default"  # pass text through unchanged (TTS decides)
PUNCTUATION_NONE = "none"
PUNCTUATION_MATH = "math"
PUNCTUATION_ALL = "all"
PUNCTUATION_CHOICES = [
    (PUNCTUATION_DEFAULT, "Default (TTS decides)"),
    (PUNCTUATION_NONE, "None (strip punctuation)"),
    (PUNCTUATION_MATH, "Math (keep math symbols, strip sentence punctuation)"),
    (PUNCTUATION_ALL, "All (speak every punctuation mark as a word)"),
]

DEFAULT_RATE = 1.0
DEFAULT_PITCH = 1.0
DEFAULT_VOLUME = 1.0
RATE_MIN, RATE_MAX = 0.5, 2.0
PITCH_MIN, PITCH_MAX = 0.5, 2.0
VOLUME_MIN, VOLUME_MAX = 0.0, 2.0

# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------
FORMAT_WAV = "wav"
FORMAT_MP3 = "mp3"
FORMAT_FLAC = "flac"
OUTPUT_FORMAT_CHOICES = [
    (FORMAT_WAV, "WAV (lossless, always available)"),
    (FORMAT_MP3, "MP3 (needs FFmpeg)"),
    (FORMAT_FLAC, "FLAC (needs FFmpeg)"),
]

# ---------------------------------------------------------------------------
# Project types
# ---------------------------------------------------------------------------
PROJECT_TYPE_CLIPBOARD = "clipboard"
PROJECT_TYPE_AUDIO_PLAYLIST = "audio_playlist"
PROJECT_TYPE_DAISY_AUDIO = "daisy_audio"
# Legacy alias kept for reading older projects (project_type no longer
# selectable in the wizard; such projects keep working and build books).
PROJECT_TYPE_DAISY_AUDIO_TEXT = "daisy_audio_text"
PROJECT_TYPE_DAISY3_AUDIO_TEXT = "daisy3_audio_text"
PROJECT_TYPES = [
    (PROJECT_TYPE_AUDIO_PLAYLIST, "Audio files with playlist"),
    (PROJECT_TYPE_CLIPBOARD, "Clipboard"),
    (PROJECT_TYPE_DAISY3_AUDIO_TEXT, "DAISY audio and text with images book (3)"),
    (PROJECT_TYPE_DAISY_AUDIO, "DAISY audio book (2.02)"),
]
PROJECT_TYPE_DESCRIPTIONS = {
    PROJECT_TYPE_CLIPBOARD: (
        "Paste text from the clipboard and record it directly. "
        "No document splitting or project setup needed."
    ),
    PROJECT_TYPE_AUDIO_PLAYLIST: (
        "Open a document, split it into audio segments, and record "
        "each segment as a separate audio file."
    ),
    PROJECT_TYPE_DAISY_AUDIO: (
        "Create a DAISY 2.02 audio-only book. The document is analyzed "
        "and split into chapters; each chapter is recorded as a separate "
        "audio file. SMIL navigation and NCC metadata are generated "
        "automatically."
    ),
    # Legacy type (no longer offered in the wizard) still builds correctly.
    PROJECT_TYPE_DAISY_AUDIO_TEXT: (
        "Create a DAISY 2.02 audio + text book. Like audio-only, but each "
        "chapter also stores the full text, enabling synchronized text+audio "
        "playback in DAISY readers."
    ),
    PROJECT_TYPE_DAISY3_AUDIO_TEXT: (
        "Create a DAISY 3 (Z39.86-2005) audio + text book. The whole book "
        "text is stored as DTBook XML with an NCX navigation file and SMIL "
        "files that synchronize every paragraph with the recorded audio. "
        "Images found in the source document are embedded too."
    ),
}

# ---------------------------------------------------------------------------
# DAISY 2.02 constants
# ---------------------------------------------------------------------------
DAISY_OUTPUT_DIR_NAME = "DAISY"
DAISY_AUDIO_DIR_NAME = "audio"
DAISY_SMIL_DIR_NAME = "smil"
DAISY_TEXT_DIR_NAME = "text"
DAISY_NCC_FILE = "ncc.html"
DAISY_PACKAGE_FILE = "package.opf"
DAISY_MASTER_SMIL_FILE = "master.smil"

# DAISY 3 (Z39.86-2005) output location.  The builder writes a flat fileset
# like the DAISY 2.02 one (book.xml + ncx.xml + package.opf + SMILs + audio).
DAISY3_OUTPUT_DIR_NAME = "DAISY3"
DAISY3_DTBOOK_FILE = "book.xml"
DAISY3_NCX_FILE = "ncx.xml"
DAISY3_PACKAGE_FILE = "package.opf"

# DAISY chapter splitting modes (used by the wizard DAISY page and Settings)
DAISY_SPLIT_H1 = "h1"
DAISY_SPLIT_ALL_HEADINGS = "all"
DAISY_SPLIT_CHOICES = [
    (DAISY_SPLIT_H1, "Heading style 1 only"),
    (DAISY_SPLIT_ALL_HEADINGS, "Break on every heading"),
]
DAISY_SPLIT_DESCRIPTIONS = {
    DAISY_SPLIT_H1: (
        "Each heading style 1 starts a new DAISY chapter; the text after it up "
        "to the next heading style 1 is recorded into that chapter's audio "
        "file. Chapters that stay too long are split into parts."
    ),
    DAISY_SPLIT_ALL_HEADINGS: (
        "Any heading (styles 1 to 6) starts a new DAISY chapter. The heading "
        "and the text that follows it up to the next heading are recorded "
        "into that chapter's audio file."
    ),
}

# ---------------------------------------------------------------------------
# Audio file creation modes (SPEC 3.4)
# ---------------------------------------------------------------------------
MODE_PAGE_WITH_H1 = "page_with_h1"
MODE_PAGE_ONLY = "page_only"
MODE_H1_ONLY = "h1_only"
MODE_ALL_HEADINGS = "all_headings"
MODE_ONE_FILE = "one_file"

# "Page by page only" can group several pages into one audio file.
PAGES_PER_FILE_MIN = 1
PAGES_PER_FILE_MAX = 50

AUDIO_MODE_CHOICES = [
    (MODE_PAGE_WITH_H1, "Page by page with heading style 1"),
    (MODE_PAGE_ONLY, "Page by page only"),
    (MODE_H1_ONLY, "Heading style 1 only"),
    (MODE_ALL_HEADINGS, "Break on every heading"),
    (MODE_ONE_FILE, "One audio file (whole document, no separation)"),
]
AUDIO_MODE_DESCRIPTIONS = {
    MODE_PAGE_WITH_H1: (
        "Text is sent to the TTS engine page by page. Whenever a heading style 1 "
        "is found, the text before the heading is sent first; then the heading is "
        "sent together with the rest of that page, so the heading is spoken at the "
        "start of its own audio file. Files are named '01 page 1', '02 <heading>', "
        "'03 page 2', ..."
    ),
    MODE_PAGE_ONLY: (
        "Text is sent page by page and one audio file is created per page, named "
        "'01 page 1', '02 page 2', ... When it is selected you can also choose how "
        "many pages go into one audio file (1 to 50); several pages are then "
        "recorded together into one file, named e.g. '01 pages 1 to 3'. Recording "
        "can be resumed from any unrecorded group."
    ),
    MODE_H1_ONLY: (
        "Audio files are created per heading style 1, each containing the heading "
        "and the text that follows it up to the next heading. Files are named "
        "'01 <heading>', '02 <heading>', ..."
    ),
    MODE_ALL_HEADINGS: (
        "Audio files are created per heading of any style (1 to 6), each containing "
        "the heading and the text that follows it up to the next heading - the same "
        "pattern as 'Heading style 1 only', but breaking on every heading level. "
        "Files are named '01 <heading>', '02 <heading>', ..."
    ),
    MODE_ONE_FILE: (
        "The whole document is synthesized into one single audio file with no "
        "separation. A recording that is stopped or interrupted before the file "
        "is finished cannot be resumed from the middle - it starts from the "
        "beginning again, so keep this window open until recording finishes."
    ),
}

# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------
# Approximate page size used to split free-flowing formats (txt/md/html/docx)
# into "pages" when the source has no real pages.
TEXT_PAGE_CHARS = 1800
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".html", ".htm",
    ".pdf", ".doc", ".docx", ".epub",
}

# ---------------------------------------------------------------------------
# Compute back-ends (SPEC 2.3)
# ---------------------------------------------------------------------------
COMPUTE_AUTO = "auto"
COMPUTE_CPU = "cpu"
COMPUTE_CUDA = "cuda"
COMPUTE_DML = "dml"

# ---------------------------------------------------------------------------
# Shortcuts (SPEC 3.5 / 3.6)
# ---------------------------------------------------------------------------
ACCEL_NEW_PROJECT = "Ctrl+Shift+N"
ACCEL_RECORD = "Ctrl+Shift+R"
ACCEL_SETTINGS = "Ctrl+,"

# ---------------------------------------------------------------------------
# Addons
# ---------------------------------------------------------------------------
ADDONS_DIR_NAME = "addons"
ADDON_MANIFEST_FILE = "manifest.json"
ADDON_ENV_DIR_NAME = "addon_env"

# ---------------------------------------------------------------------------
# Developer Mode
# ---------------------------------------------------------------------------
HEAVY_LOG_SUBSYSTEMS = ["gui", "tts", "engine", "recording", "addons", "download"]

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
DOWNLOAD_TIMEOUT = 60
DOWNLOAD_CHUNK = 256 * 1024
