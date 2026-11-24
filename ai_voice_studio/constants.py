"""Application-wide constants."""

APP_NAME = "AI Voice Studio"
APP_ID = "AIVoiceStudio"
APP_VERSION = "2027.1.0"

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
# Project base: text-to-speech or speech-to-text (New Project wizard)
# ---------------------------------------------------------------------------
PROJECT_BASE_TTS = "tts"
PROJECT_BASE_STT = "stt"
# Audio description: describe the picture of a video, offline, into a script (and
# optionally an SRT/VTT track) that a blind or low-vision listener can follow.
PROJECT_BASE_AD = "ad"
PROJECT_BASE_CHOICES = [
    (PROJECT_BASE_TTS, "Text to speech"),
    (PROJECT_BASE_STT, "Speech to text"),
    (PROJECT_BASE_AD, "Audio description"),
]
DEFAULT_PROJECT_BASE = PROJECT_BASE_TTS

# Project type written into project.json for transcription projects.
PROJECT_TYPE_STT = "stt_transcription"

# Project type written into project.json for audio description projects.
PROJECT_TYPE_AD = "audio_description"

# ---------------------------------------------------------------------------
# Audio description: chunking and estimates
# ---------------------------------------------------------------------------
#: Above this much video the wizard and the dialog stop being quiet about the
#: compute and warn how long a description run will take.
AD_WARN_SECONDS = 300.0

#: Default chunk length used when a long video is cut into pieces.  A chunk is
#: described on its own and the partial descriptions are merged, which is the
#: only way the strongest off-the-shelf models reach a long video at all.
#:
#: A chunk bounds **memory, progress and resume**, not the length of a
#: description; that is :data:`AD_WINDOW_SECONDS`.
AD_CHUNK_SECONDS = 120.0

#: How much video **one description** covers.
#:
#: This is the setting that decides how much a listener is told.  A chunk is
#: described window by window, so a two-minute chunk still yields a description
#: every twenty seconds - the cadence broadcast audio description uses.  One
#: summary per chunk was measured to swallow exactly what a viewer needs: a line
#: of on-screen text appearing, a gesture, a slide advancing.  Twenty seconds is
#: short enough to report a change and long enough that the descriptions do not
#: step on each other.
AD_WINDOW_SECONDS = 20.0

#: The longest a window may be.  A very large window setting still describes a
#: chunk in pieces, so the detail does not silently collapse back to one summary
#: per chunk.
AD_MAX_WINDOW_SECONDS = 60.0
AD_CHUNK_CHOICES = [
    (60.0, "1 minute per chunk"),
    (120.0, "2 minutes per chunk (recommended)"),
    (180.0, "3 minutes per chunk"),
    (300.0, "5 minutes per chunk"),
    (600.0, "10 minutes per chunk"),
]

#: Frames sampled per second of video.  Video is far more expensive than audio
#: per second, so every "long video" claim is really a claim about frame rate.
AD_DEFAULT_FPS = 0.5
AD_FPS_CHOICES = [
    (1.0, "1 frame per second (most detail, slowest)"),
    (0.5, "1 frame every 2 seconds (recommended)"),
    (0.25, "1 frame every 4 seconds"),
    (0.2, "1 frame every 5 seconds (fastest)"),
]

#: Gap fitting: the shortest silence worth speaking into, and the longest a
#: single description may run so it never talks over the dialogue.
AD_MIN_GAP_SECONDS = 1.5
AD_MAX_DESCRIPTION_SECONDS = 12.0

# ---------------------------------------------------------------------------
# Audio description: where a description is spoken
# ---------------------------------------------------------------------------
#: Standard audio description speaks into the pauses of the existing soundtrack;
#: extended audio description *stops the film*, speaks, and goes on, which is the
#: only option for a film whose dialogue never leaves a pause long enough.
AD_PLACEMENT_STANDARD = "standard"
AD_PLACEMENT_EXTENDED = "extended"
AD_PLACEMENT_CHOICES = [
    (AD_PLACEMENT_STANDARD,
     "Speak into the pauses (standard audio description)"),
    (AD_PLACEMENT_EXTENDED,
     "Pause the video for each description (extended audio description)"),
]
AD_DEFAULT_PLACEMENT = AD_PLACEMENT_STANDARD

# ---------------------------------------------------------------------------
# Audio description: track priority (which of the two tracks gives way)
# ---------------------------------------------------------------------------
#: Standard audio description speaks the description *over* the film's own
#: soundtrack, so one of the two tracks has to give way somewhere.  Which one is
#: the user's call, because it depends on the film: a lecture with one quiet
#: speaker needs nothing, while a film scored wall to wall buries the
#: description unless the film's own track comes down.
AD_TRACK_PRIORITY_SAME = "same"
AD_TRACK_PRIORITY_25 = "higher25"
AD_TRACK_PRIORITY_50 = "higher50"
#: The one choice that does not lower the film at all: the film's own soundtrack
#: is left at its own level everywhere and comes down **only while a description
#: is actually being spoken** (see ``AD_DUCK_LEVEL``), so a film with music and
#: dialogue that never stop keeps its own sound between the descriptions instead
#: of playing quietly for its whole length.
AD_TRACK_PRIORITY_DUCK = "duck"
#: Ordered from the gentlest answer to the film to the most forceful: *same*
#: leaves both tracks alone, *duck* touches the film only under the voice, and
#: the two percentage choices lower the film for its whole length.  The first
#: entry is also the default (``AD_DEFAULT_TRACK_PRIORITY``), so a project
#: written before any of these choices existed opens on exactly what it did.
AD_TRACK_PRIORITY_CHOICES = [
    (AD_TRACK_PRIORITY_SAME,
     "Same volume (both tracks as they are)"),
    (AD_TRACK_PRIORITY_DUCK,
     "Duck, don't lower (the film's own sound dips while a description speaks)"),
    (AD_TRACK_PRIORITY_25,
     "Description 25% higher (the film's own sound is quieter by 25%)"),
    (AD_TRACK_PRIORITY_50,
     "Description 50% higher (the film's own sound is quieter by 50%)"),
]

#: What one description is worth by default: both tracks stay as they are.  A
#: project written before any of these choices existed carries no value at all,
#: and this is exactly what those runs did, so an old project is never mixed
#: differently because a newer version added choices.
AD_DEFAULT_TRACK_PRIORITY = AD_TRACK_PRIORITY_SAME

#: The film soundtrack's own gain for each choice.  The description is **always**
#: left at its own level - the voice the user tuned in Settings, spoken by the
#: narration settings - and the film's track is what moves: *25% higher* plays it
#: at 0.75, *50% higher* at 0.5.  Raising the description instead would clip
#: (full-scale speech already peaks near full scale) and would change a level the
#: user had already chosen by ear, so only one direction is applied.  *Duck, don't
#: lower* is 1.0 here on purpose: it does not lower the film at all, it dips it
#: while the voice is heard (``AD_DUCK_LEVEL``).
AD_TRACK_PRIORITY_GAIN = {
    AD_TRACK_PRIORITY_SAME: 1.0,
    AD_TRACK_PRIORITY_25: 0.75,
    AD_TRACK_PRIORITY_50: 0.5,
    AD_TRACK_PRIORITY_DUCK: 1.0,
}

#: How far *Duck, don't lower* brings the film's own soundtrack down **while a
#: description is being spoken**: a quarter of its own level, about -12 dB.  A
#: duck is deep and short by nature - the film is normal between descriptions and
#: steps out of the way under the voice - which is why it is not one of the
#: quieter-for-the-whole-film gains above.
AD_DUCK_LEVEL = 0.25

#: How long the film takes to come down and to come back up, either side of a
#: spoken description.  A step would be a click on a music track; a few tens of
#: milliseconds is the ramp a broadcast ducker uses and is inaudible.
AD_DUCK_EDGE_SECONDS = 0.06

#: Two descriptions closer together than this are one dip.  Coming back up for a
#: fraction of a second between two back-to-back descriptions would be heard as
#: the film's own sound flapping, which is worse than either extreme.
AD_DUCK_MERGE_SECONDS = 0.4

#: The ceiling on the ducking expression, in characters.  The dips are written as
#: one FFmpeg expression, and a command line has a length limit (about 32,000
#: characters on Windows, which is the shortest of them).  A film long enough to
#: reach this - hundreds of descriptions - is mixed as it always was rather than
#: failing to write at all; the ducking is a refinement, not a reason to lose the
#: described film.
AD_DUCK_MAX_EXPRESSION = 24000

#: The narration voice's own volume - how loud the spoken description is, as a
#: multiplier of what the voice engine produces.  1.0 is the voice as it came;
#: the slider in the Audio description window runs from ``VOLUME_MIN`` to
#: ``VOLUME_MAX``, the same range the recording settings use, and it is the
#: *description's* level - the film's track is what *Track priority* moves.
AD_NARRATION_VOLUME_DEFAULT = 1.0

#: Extended mode holds the film on the last frame while the description speaks,
#: then a beat of silence before the picture moves again, so the description is
#: never clipped by the film restarting.
AD_PAUSE_PADDING = 0.6

#: Nobody wants a pause shorter than this, however short the description is.
AD_MIN_PAUSE_SECONDS = 0.8

#: Rough spoken length of one description, used only by the pre-run estimate of
#: how much longer an extended description makes a film.  Taken from a real run's
#: measured pauses (8.8 - 15.2 s, about 11 s on average), because that is exactly
#: how long the film is held for.
AD_EXTENDED_SECONDS_PER_DESCRIPTION = 11.0

#: Output documents an audio description project writes.
AD_FORMAT_SCRIPT = "script"
AD_FORMAT_SCRIPT_HTML = "script_html"
AD_FORMAT_SRT = "srt"
AD_FORMAT_VTT = "vtt"
AD_FORMAT_JSON = "json"
AD_OUTPUT_CHOICES = [
    (AD_FORMAT_SCRIPT, "Description script (plain text)"),
    (AD_FORMAT_SCRIPT_HTML, "Description script (web page)"),
    (AD_FORMAT_SRT, "SubRip subtitles (.srt)"),
    (AD_FORMAT_VTT, "WebVTT subtitles (.vtt)"),
    (AD_FORMAT_JSON, "JSON with timings (.json)"),
]
AD_DEFAULT_OUTPUT = AD_FORMAT_SCRIPT

# ---------------------------------------------------------------------------
# Speech-to-text: output document types
# ---------------------------------------------------------------------------
STT_FORMAT_TXT = "txt"
STT_FORMAT_HTML = "html"
STT_FORMAT_MARKDOWN = "markdown"
STT_FORMAT_SRT = "srt"
STT_FORMAT_VTT = "vtt"
STT_FORMAT_JSON = "json"
STT_FORMAT_CSV = "csv"
# Markdown, WebVTT, JSON and CSV are the "popular types for this purpose"
# added beyond the requested .txt/.html/.srt: VTT is the open subtitle format
# used alongside SRT, and JSON/CSV keep the per-segment timings for tooling.
STT_OUTPUT_CHOICES = [
    (STT_FORMAT_TXT, "Plain text (.txt)"),
    (STT_FORMAT_HTML, "Web page (.html)"),
    (STT_FORMAT_MARKDOWN, "Markdown (.md)"),
    (STT_FORMAT_SRT, "SubRip subtitles (.srt)"),
    (STT_FORMAT_VTT, "WebVTT subtitles (.vtt)"),
    (STT_FORMAT_JSON, "JSON with timings (.json)"),
    (STT_FORMAT_CSV, "Spreadsheet (.csv)"),
]
STT_DEFAULT_OUTPUT = STT_FORMAT_TXT

# ---------------------------------------------------------------------------
# Speech-to-text: source selection
# ---------------------------------------------------------------------------
STT_SOURCE_SINGLE = "single"
STT_SOURCE_MULTIPLE = "multiple"
STT_SOURCE_FOLDER = "folder"
STT_SOURCE_CHOICES = [
    (STT_SOURCE_SINGLE, "A single file"),
    (STT_SOURCE_MULTIPLE, "Multiple files"),
    (STT_SOURCE_FOLDER, "A complete folder"),
]
STT_DEFAULT_SOURCE = STT_SOURCE_SINGLE

# ---------------------------------------------------------------------------
# Speech-to-text: file creation mode
# ---------------------------------------------------------------------------
STT_SPLIT_PER_SOURCE = "per_source"
STT_SPLIT_BY_TIME = "by_time"
STT_SPLIT_CHOICES = [
    (STT_SPLIT_PER_SOURCE, "A document file for each source file"),
    (STT_SPLIT_BY_TIME, "Time by time separation"),
]
STT_DEFAULT_SPLIT = STT_SPLIT_PER_SOURCE
STT_TIME_CHOICES = [10, 20, 30, 40, 50, 60]
STT_DEFAULT_MINUTES = 30

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

#: The OmniVoice server's bind addresses.  ``LOOPBACK`` is the default; the
#: "network access" checkbox switches to ``ALL_INTERFACES``, which exposes an
#: unauthenticated HTTP server on the LAN (see docs/UserGuide.html).
OMNIVOICE_BIND_LOOPBACK = "127.0.0.1"
OMNIVOICE_BIND_ALL_INTERFACES = "0.0.0.0"  # noqa: S104 - see above: opt-in only
