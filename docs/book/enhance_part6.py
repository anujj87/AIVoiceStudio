#!/usr/bin/env python3
"""
Enhanced line-by-line description engine with AIVoiceStudio-specific context.
Processes Part VI chapters (43-60) to replace generic descriptions with
project-specific, meaningful explanations.
"""

import re, html, os, glob

CHAPTERS_DIR = os.path.join(os.path.dirname(__file__), "chapters")
PART6_FILES = [f"ch{i:02d}" for i in range(43, 61)]

# ── AIVoiceStudio Knowledge Base ──────────────────────────────────────────

AIVS_KB = {
    # === Paths module ===
    "user_data_dir()": "Returns the per-user application data directory. On Windows this is %APPDATA%/AIVoiceStudio; on Linux/macOS it's ~/.aivoicestudio. The directory is created automatically if it doesn't exist.",
    "models_dir()": "Returns the directory where downloaded TTS voice models are stored. Respects any custom path set in Settings > Paths. Creates the directory if needed.",
    "recordings_dir()": "Returns the directory where project recordings (audio files) are saved. Respects custom path settings.",
    "projects_dir()": "Alias for recordings_dir(). Projects hold the recorded audio files.",
    "ffmpeg_dir()": "Returns the directory where the bundled FFmpeg binary lives inside the user data folder.",
    "settings_file()": "Returns the full path to settings.json, the persistent configuration file.",
    "log_file()": "Returns the full path to app.log, the application log file.",
    "sanitize_filename": "Removes or replaces characters that are illegal in filenames (Windows: <>:\"/\\|?*). Truncates to max_length to avoid filesystem limits.",

    # === Settings module ===
    "Settings()": "Loads the persistent application settings from settings.json. Thread-safe; defaults are defined in _DEFAULTS. Provides get/set/reset methods.",
    "Settings.get(key, default)": "Retrieves a setting by dotted key path (e.g. 'recording.rate'). Returns default if the key doesn't exist.",
    "Settings.set(key, value)": "Stores a setting by dotted key path. Thread-safe via internal lock.",
    "Settings.reset()": "Restores all settings to factory defaults. Useful when the user wants to start fresh.",
    "apply_theme": "Applies the selected theme (light/dark/system) to the entire wxPython application by updating system colours and font settings.",

    # === TTS Engine ===
    "SherpaEngine": "The primary TTS engine using sherpa-onnx. Runs in-process (no subprocess). Supports VITS, Kokoro, and Matcha voice models. No voice cloning.",
    "XtTsCloneEngine": "Voice cloning engine using Coqui TTS (XTTS v2). Runs in a separate subprocess for isolation. Requires a reference audio sample.",
    "OmniVoiceEngine": "Zero-shot voice cloning engine using OmniVoice-Triton. Runs in a subprocess. Can clone any voice from a short audio clip.",
    "OmniVoiceServerEngine": "Voice cloning via a remote OmniVoice HTTP server. Communicates over HTTP API instead of subprocess.",
    "TtsEngine": "Abstract base class that all TTS engines must implement. Defines the synthesize() contract: takes text + parameters, returns int16 audio samples.",
    "TtsEngine.synthesize()": "Core synthesis method. Takes text string, voice ID (sid), speed/pitch/volume floats. Returns a NumPy array of int16 audio samples at the engine's sample_rate.",
    "TtsEngine.close()": "Releases engine resources (loaded models, subprocess handles). Called when switching voices or shutting down.",
    "get_engine(voice_entry)": "Factory function that creates the right engine for a voice. Inspects voice_entry['engine'] to decide between Sherpa, XTTS, OmniVoice, or OmniVoiceServer.",
    "PiperEngine": "Piper TTS engine (community fork). Uses ONNX models. Fast CPU inference with good quality.",
    "KokoroEngine": "Kokoro TTS engine. Neural TTS with natural prosody. Supports multiple languages.",
    "MatchaEngine": "Matcha-TTS engine. Two-stage: acoustic model + neural vocoder. Good for expressive speech.",

    # === TTS Catalog ===
    "load_catalog()": "Loads the voice catalog from catalog.json. Returns a list of voice entries, each with id, name, language, engine, and download metadata.",
    "catalog.json": "JSON file containing all available TTS voices. Each entry has: id, name, language, engine type, download URL, file size, and sample rate.",
    "ModelStore": "Manages downloaded voice models on disk. Tracks which voices are installed, their file paths, and total disk usage.",
    "ModelStore.is_installed(voice_id)": "Checks whether a voice model's files exist on disk and are complete.",
    "ModelStore.voice_path(voice_id)": "Returns the filesystem path where a specific voice model's files are stored.",

    # === Download ===
    "download_variant": "Downloads a voice model variant (e.g. 'en_US/medium') from the catalog URL to the local models directory. Shows progress in the GUI.",
    "Downloader": "HTTP download manager. Handles chunked downloads, progress callbacks, and resume support for large model files.",

    # === GUI ===
    "MainFrame": "The main application window. Contains the menu bar, welcome panel, and status bar. Entry point for all user actions.",
    "MainFrame._build_menu()": "Constructs the menu bar with File (New/Open/Recent/Exit), Edit (Resume/Restart/Remove), Tools (Record/Settings), and Help menus.",
    "MainFrame._build_welcome()": "Creates the welcome panel shown when no project is open. Contains large action buttons and a recent projects list.",
    "MainFrame._build_statusbar()": "Creates the status bar at the bottom of the window. Shows current status messages and progress.",
    "MainFrame._on_new_project()": "Opens the New Project wizard dialog. The wizard guides the user through document import and voice selection.",
    "MainFrame._on_settings()": "Opens the Settings dialog. Allows the user to download voices, change compute backend, and adjust preferences.",
    "MainFrame._on_record()": "Opens the Recording dialog for the currently selected project. The dialog manages audio recording with TTS playback.",
    "MainFrame._on_about()": "Shows the About dialog with version info, credits, and license.",
    "SettingsDialog": "Multi-tab dialog for application settings. Tabs: Voice Downloads, Compute Backend, Recording Defaults, Paths, Developer Mode.",
    "NewProjectWizard": "Two-page wizard for creating a new TTS project. Page 1: import document (PDF/DOCX/TXT). Page 2: select voice and split settings.",
    "RecordingDialog": "Dialog for recording audio for a project. Shows the text to read, records microphone input, and plays TTS reference audio.",
    "apply_theme(frame, theme)": "Applies a visual theme (light/dark/system) to the entire wxPython frame and its children.",

    # === Document parsing ===
    "parse_document(path)": "Parses a document file (PDF, DOCX, or TXT) and extracts its text content. Returns a list of text segments.",
    "parse_pdf(path)": "Extracts text from a PDF file using pypdf. Returns text organized by pages.",
    "parse_docx(path)": "Extracts text from a Word document using python-docx. Returns text organized by paragraphs.",
    "parse_txt(path)": "Reads a plain text file. Returns the content as a single text segment.",
    "TextSplitter": "Splits parsed document text into segments suitable for TTS. Can split by headings (H1), page breaks, or custom rules.",
    "split_by_heading(text)": "Splits text at H1-level headings (lines starting with #). Each heading becomes the title of a new segment.",
    "split_by_page(pages)": "Groups parsed pages into logical segments. Each page becomes a segment unless a heading triggers a split.",

    # === DAISY ===
    "DaisyBuilder": "Builds a DAISY 2.02 audio book from a project's recorded audio files. Creates ncc.html, package.opf, and SMIL files.",
    "build_daisy(project_path)": "Main entry point for DAISY book generation. Reads project metadata, processes audio files, and outputs a standards-compliant DAISY book.",

    # === Clone / OmniVoice ===
    "CloneClient": "Manages the XTTS v2 subprocess. Starts the clone server process, sends synthesis requests, and receives audio back.",
    "CloneClient._ensure_proc()": "Checks if the clone subprocess is running. If not, starts it. If it crashed, restarts it automatically.",
    "OmniVoiceWorker": "Background worker that runs the OmniVoice-Triton inference server. Manages GPU memory and request queue.",
    "OmniVoiceServer": "HTTP API server that exposes voice cloning over REST endpoints. Allows multiple clients to clone voices simultaneously.",

    # === Compute ===
    "detect_compute()": "Probes the system for available compute backends. Returns a dict like {'cpu': True, 'cuda': False, 'dml': False}.",
    "ComputeBackend": "Enum-like constants for compute backends: AUTO, CPU, CUDA, DML (DirectML). The user selects one in Settings.",
    "has_cuda()": "Returns True if NVIDIA CUDA is available on this system (GPU detected and drivers installed).",
    "has_dml()": "Returns True if DirectML is available (Windows GPU acceleration without NVIDIA).",

    # === Jobs / Synthesis ===
    "SynthesisWorker": "Background thread that synthesizes text segments into audio files. Handles progress callbacks and cancellation.",
    "SynthesisWorker.run()": "Main synthesis loop. Iterates over segments, calls the TTS engine for each, and saves audio files.",
    "SynthesisWorker.cancel()": "Signals the worker to stop after the current segment. Used when the user cancels or switches projects.",

    # === wxPython specific ===
    "wx.Frame": "wxPython's top-level window class. MainFrame inherits from this to create the application's main window with title bar, menus, and status bar.",
    "wx.Panel": "A container window that sits inside a Frame. Used to group child controls and manage tab traversal.",
    "wx.BoxSizer": "A layout manager that arranges child windows in a horizontal or vertical line. The primary layout mechanism in wxPython.",
    "wx.StaticText": "A non-editable text label. Used for displaying information to the user.",
    "wx.Button": "A clickable push button. Used for actions like 'Create Project' or 'Download Voice'.",
    "wx.ListBox": "A list control that displays items and allows single selection. Used for voice selection and recent projects.",
    "wx.Choice": "A drop-down selection control. Used for choosing compute backend, output format, etc.",
    "wx.Slider": "A slider control for selecting a value within a range. Used for rate, pitch, and volume adjustment (0-200 where 100 = 1.0).",
    "wx.Gauge": "A progress bar control. Shows download progress and synthesis progress.",
    "wx.TextCtrl": "A text input control. Can be single-line (for file paths) or multi-line (for text display).",
    "wx.Notebook": "A tabbed container. Each tab is a separate panel. Used in the Settings dialog for tabbed pages.",
    "wx.FileDialog": "A native file open/save dialog. Lets the user browse and select files.",
    "wx.DirDialog": "A native directory selection dialog. Lets the user browse and select folders.",
    "wx.MessageDialog": "A dialog that displays a message and OK/Cancel buttons. Used for confirmations and error messages.",
    "wx.MenuBar": "The horizontal menu bar at the top of a Frame. Contains File, Edit, Tools, Settings, Help menus.",
    "wx.Menu": "A dropdown menu that appears when clicking a menu bar item. Contains menu items and separators.",
    "wx.StatusBar": "The status bar at the bottom of a Frame. Shows status text, progress, and context information.",
    "wx.NewIdRef()": "Creates a unique integer ID for use with menu items, controls, and event binding. Prevents ID collisions.",
    "wx.EVT_MENU": "Event type triggered when a menu item is clicked. Bind handlers with frame.Bind(wx.EVT_MENU, handler, id=item_id).",
    "wx.CallAfter(func, *args)": "Schedules a function to be called on the main thread after the current event handler completes. Essential for thread-safe GUI updates.",
    "wx.Font": "Represents a font face, size, and style. Used for consistent typography across the application.",
    "wx.Colour": "Represents an RGB colour. Used for theming and custom drawing.",
    "wx.Bitmap": "An image that can be drawn on a DC or displayed in a StaticBitmap control.",
    "wx.DC": "Device Context for drawing graphics. Used in custom painting and printing.",
    "ShowModal()": "Displays a dialog modally (blocks input to the parent window until the dialog is closed). Returns wx.ID_OK or wx.ID_CANCEL.",
    "Destroy()": "Destroys the dialog window and frees its resources. Must be called after ShowModal() returns.",

    # === Python runtime ===
    "PythonRuntime": "Manages the embedded Python interpreter used by addons. Provides isolation between the main app and addon code.",
    "subprocess.Popen": "Starts a new process. Used by CloneClient and OmniVoiceWorker to run TTS engines in isolated processes.",
    "threading.Thread": "Creates a background thread. Used for TTS synthesis, downloads, and other long-running operations that shouldn't block the GUI.",
    "threading.RLock": "A reentrant lock that allows the same thread to acquire it multiple times without deadlocking. Used for thread-safe settings access.",
    "json.load / json.dump": "Reads/writes JSON files. Used for settings, catalog, and project metadata persistence.",
    "logging.getLogger(__name__)": "Creates a module-level logger. The logger name matches the module path (e.g. 'ai_voice_studio.settings'), enabling per-module log level control.",
    "os.makedirs(path, exist_ok=True)": "Recursively creates directories. exist_ok=True prevents an error if the directory already exists.",
    "os.path.join()": "Joins path components using the OS-appropriate separator. Portable across Windows and Unix.",
    "os.environ.get(key, default)": "Reads an environment variable with a fallback default. Used for APPDATA on Windows.",
    "sys.platform": "String identifying the OS: 'win32', 'darwin', 'linux'. Used for platform-specific code paths.",
    "pathlib.Path": "Modern path handling using objects instead of strings. Preferred in new Python code for cleaner path manipulation.",
    "functools.lru_cache": "Decorator that caches function results. Same inputs always return the same cached output. Used for expensive lookups.",
    "typing.Optional[T]": "Type hint meaning the value can be T or None. Equivalent to T | None in Python 3.10+.",
    "typing.Dict[K, V]": "Type hint for a dictionary with keys of type K and values of type V.",
    "from __future__ import annotations": "Enables PEP 604 style type hints (X | Y) on Python 3.7+. Must be the first import in the file.",
}

# ── Description Engine ────────────────────────────────────────────────────

def is_ascii_art(line: str) -> bool:
    """Check if a line is part of an ASCII art diagram."""
    art_chars = set('│┌┐└┘├┤┬┴┼─═║╔╗╚╝╠╣╦╩╬══─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬')
    stripped = line.strip()
    if not stripped:
        return False
    # If most characters are box-drawing or spaces, it's art
    art_count = sum(1 for c in stripped if c in art_chars or c == ' ')
    return art_count / max(len(stripped), 1) > 0.6

def describe_line_aivs(line: str, chapter_num: int, context_lines=None, line_idx=0) -> str:
    """Generate an AIVoiceStudio-enhanced description for one line of code."""
    stripped = line.strip()

    if not stripped:
        return ""

    # Check for ASCII art - label as diagram
    if is_ascii_art(stripped):
        return ""

    # Comments
    if stripped.startswith('#'):
        inner = stripped.lstrip('#').strip()
        if inner:
            # Check if comment references a known file
            if 'from' in inner.lower() and '.py' in inner:
                return f"A comment indicating the source file: <em>{html.escape(inner)}</em>. This helps the reader locate the code in the AIVoiceStudio codebase."
            return f"A comment: <em>{html.escape(inner)}</em>. Comments are ignored by the Python interpreter and serve as documentation."
        return ""

    # Docstrings
    if stripped.startswith('"""') or stripped.startswith("'''"):
        inner = stripped.strip('"""').strip("'''").strip()
        if inner:
            return f"Docstring: <em>{html.escape(inner)}</em>. Documents the purpose of the following code block."
        return "Opens a docstring block for documentation."

    # Check against knowledge base first
    for pattern, desc in AIVS_KB.items():
        if pattern in stripped:
            return desc

    # Import statements
    m = re.match(r'^from\s+(\S+)\s+import\s+(.+)', stripped)
    if m:
        module, names = m.group(1), m.group(2)
        name_list = [n.strip().split(' as ')[0].strip() for n in names.split(',')]
        for name in name_list:
            if name in AIVS_KB:
                return f"Imports <code>{html.escape(name)}</code> from <code>{html.escape(module)}</code>: {AIVS_KB[name]}"
        return f"Imports <code>{html.escape(names)}</code> from the <code>{html.escape(module)}</code> module."

    m = re.match(r'^import\s+(.+)', stripped)
    if m:
        modules = m.group(1)
        return f"Imports the <code>{html.escape(modules)}</code> module."

    # Class definitions
    m = re.match(r'^class\s+(\w+)(?:\(([^)]*)\))?:', stripped)
    if m:
        name, bases = m.group(1), m.group(2) or ""
        if name in AIVS_KB:
            return f"Defines class <code>{html.escape(name)}</code>: {AIVS_KB[name]}"
        if bases:
            return f"Defines class <code>{html.escape(name)}</code> inheriting from <code>{html.escape(bases)}</code>."
        return f"Defines class <code>{html.escape(name)}</code>."

    # Function definitions
    m = re.match(r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:', stripped)
    if m:
        name, params, ret = m.group(2), m.group(3), m.group(4)
        param_list = [p.strip().split(':')[0].split('=')[0].strip() for p in params.split(',') if p.strip()]
        # Check KB for this function
        for kb_key, kb_desc in AIVS_KB.items():
            if name in kb_key and '(' in kb_key:
                return f"Defines function <code>{html.escape(name)}</code>: {kb_desc}"
        param_str = ', '.join(param_list[:4])
        if len(param_list) > 4:
            param_str += ', ...'
        desc = f"Defines function <code>{html.escape(name)}</code>"
        if param_list:
            desc += f" accepting parameters ({html.escape(param_str)})"
        if ret:
            desc += f", returning <code>{html.escape(ret.strip())}</code>"
        desc += "."
        return desc

    # Lambda
    m = re.match(r'^(\w+)\s*=\s*lambda\s+(.+?):', stripped)
    if m:
        return f"Creates an anonymous function and assigns it to <code>{html.escape(m.group(1))}</code>."

    # Return
    m = re.match(r'^\s*return\s+(.+)', stripped)
    if m:
        val = m.group(1)
        if 'None' in val:
            return "Returns <code>None</code> (no value)."
        return f"Returns <code>{html.escape(val[:80])}</code> to the caller."

    if stripped == 'return':
        return "Returns <code>None</code> implicitly (exits the function)."

    # Control flow
    m = re.match(r'^\s*if\s+(.+):', stripped)
    if m:
        cond = m.group(1)
        return f"Checks condition: <code>{html.escape(cond[:100])}</code>. The indented block below runs only when this is <code>True</code>."

    m = re.match(r'^\s*elif\s+(.+):', stripped)
    if m:
        return f"Alternative condition: <code>{html.escape(m.group(1)[:100])}</code>. Checked only if all previous conditions were <code>False</code>."

    if re.match(r'^\s*else:', stripped):
        return "Fallback: executes when none of the above conditions matched."

    # Loops
    m = re.match(r'^\s*for\s+(\w+)\s+in\s+(.+):', stripped)
    if m:
        return f"Iterates over <code>{html.escape(m.group(2)[:80])}</code>, assigning each element to <code>{html.escape(m.group(1))}</code>."

    m = re.match(r'^\s*while\s+(.+):', stripped)
    if m:
        return f"While loop: repeats as long as <code>{html.escape(m.group(1)[:80])}</code>."

    # Exception handling
    if re.match(r'^\s*try:', stripped):
        return "Begins exception handling. Errors in this block will be caught below."
    m = re.match(r'^\s*except\s+(\w+)', stripped)
    if m:
        return f"Catches <code>{html.escape(m.group(1))}</code> exceptions."
    if re.match(r'^\s*finally:', stripped):
        return "Always executes for cleanup, whether or not an exception occurred."
    if re.match(r'^\s*raise\s+', stripped):
        return f"Raises an exception: <code>{html.escape(stripped)}</code>."

    # Context managers
    m = re.match(r'^\s*with\s+(.+):', stripped)
    if m:
        return f"Context manager: <code>{html.escape(m.group(1)[:100])}</code>. Automatically manages resource cleanup."

    # self.xxx assignments
    m = re.match(r'^\s*self\.(\w+)\s*=\s*(.+)', stripped)
    if m:
        attr, val = m.group(1), m.group(2)
        if attr in AIVS_KB:
            return f"Sets instance attribute <code>{html.escape(attr)}</code>: {AIVS_KB[attr]}"
        return f"Stores <code>{html.escape(val[:60])}</code> in the instance attribute <code>{html.escape(attr)}</code>."

    # wx.xxx calls
    m = re.match(r'^\s*(wx\.\w+)\s*\(', stripped)
    if m:
        wxclass = m.group(1)
        if wxclass in AIVS_KB:
            return f"Creates a wxPython {wxclass.replace('wx.', '')}: {AIVS_KB[wxclass]}"
        return f"Creates a wxPython <code>{html.escape(wxclass)}</code> widget."

    # Method calls on self
    m = re.match(r'^\s*self\.(\w+)\(', stripped)
    if m:
        method = m.group(1)
        if method in AIVS_KB:
            return f"Calls <code>self.{method}()</code>: {AIVS_KB[method]}"
        return f"Calls <code>self.{html.escape(method)}()</code> to perform an operation."

    # Generic method calls
    m = re.match(r'^\s*(\w+)\(', stripped)
    if m:
        func = m.group(1)
        if func in AIVS_KB:
            return f"Calls <code>{html.escape(func)}()</code>: {AIVS_KB[func]}"
        return f"Calls function <code>{html.escape(func)}()</code>."

    # Augmented assignment
    m = re.match(r'^\s*(\w+)\s*(\+=|-=|\*=|/=|//=|%=)\s*(.+)', stripped)
    if m:
        var, op, val = m.group(1), m.group(2), m.group(3)
        return f"Updates <code>{html.escape(var)}</code> by {op} <code>{html.escape(val[:50])}</code>."

    # Generic assignment
    m = re.match(r'^(\w+(?:\.\w+)*)\s*=\s*(.+)', stripped)
    if m:
        var, val = m.group(1), m.group(2)
        if var in AIVS_KB:
            return f"Sets <code>{html.escape(var)}</code>: {AIVS_KB[var]}"
        return f"Assigns <code>{html.escape(val[:80])}</code> to <code>{html.escape(var)}</code>."

    # Decorators
    if stripped.startswith('@'):
        return f"Decorator: <code>{html.escape(stripped)}</code>. Wraps the following function/class to add behavior."

    # pass/break/continue
    if stripped == 'pass':
        return "Placeholder: does nothing. Used where Python requires a statement syntactically."
    if stripped == 'break':
        return "Exits the innermost loop immediately."
    if stripped == 'continue':
        return "Skips to the next iteration of the loop."

    # Ellipsis (abstract methods)
    if stripped == '...':
        return "Ellipsis placeholder: indicates an abstract method that subclasses must implement."

    # Fallback
    return f"<code>{html.escape(stripped[:120])}</code>"


def build_explanation_block_enhanced(code_text: str, block_index: int, chapter_num: int) -> str:
    """Build an enhanced explanation block for one code block."""
    raw_lines = code_text.split('\n')

    # Detect ASCII art blocks - treat as single item
    art_lines = [l for l in raw_lines if l.strip() and is_ascii_art(l.strip())]
    is_art = len(art_lines) > len([l for l in raw_lines if l.strip()]) * 0.5

    if is_art and art_lines:
        return f'''
<div class="line-explanation diagram">
  <h4>Architecture Diagram (Block {block_index})</h4>
  <p>This is an ASCII art diagram showing the AIVoiceStudio architecture. The boxes represent modules and the connections show data flow between layers. Refer to Chapter 43 for a detailed explanation of each component.</p>
</div>
'''

    code_lines = [(i, l) for i, l in enumerate(raw_lines) if l.strip()]
    if not code_lines:
        return ''

    items = []
    for display_num, (orig_idx, line) in enumerate(code_lines, 1):
        desc = describe_line_aivs(line, chapter_num, context_lines=code_lines, line_idx=orig_idx)
        escaped_code = html.escape(line.strip())
        if desc:
            items.append(
                f'      <li>\n'
                f'        <strong>Line {display_num}:</strong> <code>{escaped_code}</code>\n'
                f'        <br><span class="desc">{desc}</span>\n'
                f'      </li>'
            )
        else:
            items.append(
                f'      <li>\n'
                f'        <strong>Line {display_num}:</strong> <code>{escaped_code}</code>\n'
                f'      </li>'
            )

    items_html = '\n'.join(items)
    return f'''
<div class="line-explanation">
  <h4>Explain Line by Line Code Description (Block {block_index})</h4>
  <ol>
{items_html}
  </ol>
</div>
'''


def process_file_enhanced(filepath: str) -> bool:
    """Process a single HTML file with enhanced descriptions."""
    basename = os.path.basename(filepath)
    # Extract chapter number
    m = re.match(r'ch(\d+)', basename)
    chapter_num = int(m.group(1)) if m else 0

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove existing line-explanation blocks
    content = re.sub(
        r'\n<div class="line-explanation[^"]*">.*?</div>\n',
        '\n',
        content,
        flags=re.DOTALL
    )

    # Find all <pre><code>...</code></pre> blocks
    pattern = re.compile(r'(<pre><code>)(.*?)(</code></pre>)', re.DOTALL)

    block_index = 0
    def replacer(match):
        nonlocal block_index
        block_index += 1
        opening = match.group(1)
        code_html = match.group(2)
        closing = match.group(3)
        code_text = strip_html_tags(code_html)
        explanation = build_explanation_block_enhanced(code_text, block_index, chapter_num)
        return f'{opening}{code_html}{closing}\n{explanation}'

    new_content = pattern.sub(replacer, content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False


def strip_html_tags(text: str) -> str:
    """Remove HTML tags but keep the text content."""
    text = re.sub(r'<span[^>]*>', '', text)
    text = re.sub(r'</span>', '', text)
    text = html.unescape(text)
    return text


def main():
    files = sorted(glob.glob(os.path.join(CHAPTERS_DIR, 'ch4[3-9]_*.html')) +
                   glob.glob(os.path.join(CHAPTERS_DIR, 'ch5[0-9]_*.html')) +
                   glob.glob(os.path.join(CHAPTERS_DIR, 'ch60_*.html')))
    modified = 0
    for fp in files:
        if process_file_enhanced(fp):
            print(f'  [OK] {os.path.basename(fp)}')
            modified += 1
        else:
            print(f'  [--] {os.path.basename(fp)} (skipped)')
    print(f'\nDone. Enhanced {modified}/{len(files)} Part VI files.')


if __name__ == '__main__':
    main()
