#!/usr/bin/env python3
"""
Enhanced line-by-line description engine for Parts V, VII, and VIII.
Part V: NumPy, Requests, sherpa-onnx, PDF/DOCX, FFmpeg, Virtual Envs
Part VII: Testing, Mocking, Integration, PyInstaller, Inno Setup, CI/CD
Part VIII: Python Runtime, Subprocess, Logging, Performance, Security, Roadmap
"""

import re, html, os, glob

CHAPTERS_DIR = os.path.join(os.path.dirname(__file__), "chapters")

# ── NumPy Knowledge Base ──────────────────────────────────────────────────

NUMPY_KB = {
    "np.array(": "Creates a NumPy array from a list or tuple. The dtype parameter specifies the data type (e.g., np.int16 for 16-bit audio samples).",
    "np.zeros(": "Creates an array filled with zeros. Shape is specified as (rows, cols) or (length,).",
    "np.ones(": "Creates an array filled with ones.",
    "np.empty(": "Creates an array without initializing values (faster but undefined contents).",
    "np.linspace(": "Creates an array with evenly spaced values between start and stop. num specifies the count.",
    "np.arange(": "Creates an array with values from start to stop (exclusive) with a given step.",
    "np.full(": "Creates an array filled with a constant value.",
    "np.eye(": "Creates an identity matrix (2D array with 1s on the diagonal).",
    "np.random": "NumPy's random number generation module. Used for creating test data and noise.",
    "np.random.randn(": "Creates an array of random values from a standard normal distribution.",
    "np.random.randint(": "Creates an array of random integers within a range.",
    "np.random.seed(": "Seeds the random number generator for reproducible results.",
    "np.sin(": "Element-wise sine function. Used for generating audio waveforms.",
    "np.cos(": "Element-wise cosine function.",
    "np.pi": "Mathematical constant pi (3.14159...). Used in trigonometric calculations.",
    "np.int16": "16-bit integer type. The standard format for audio samples in TTS output.",
    "np.float32": "32-bit floating point type. Used for intermediate audio processing.",
    "np.float64": "64-bit floating point (double precision). Default float type in NumPy.",
    "np.uint8": "8-bit unsigned integer type. Used for image data and some audio formats.",
    ".astype(": "Converts an array to a different data type. Returns a new array (original is unchanged).",
    ".shape": "Returns the dimensions of the array as a tuple (e.g., (100,) for 1D, (100, 2) for 2D).",
    ".reshape(": "Returns a new array with the same data but different shape. Total elements must match.",
    ".flatten()": "Returns a 1-D copy of the array, collapsing all dimensions.",
    ".T": "Returns the transposed array (swaps rows and columns for 2D arrays).",
    ".mean()": "Returns the arithmetic mean of the array elements.",
    ".std()": "Returns the standard deviation of the array elements.",
    ".sum()": "Returns the sum of all array elements.",
    ".min()": "Returns the minimum value in the array.",
    ".max()": "Returns the maximum value in the array.",
    ".argmin()": "Returns the index of the minimum value.",
    ".argmax()": "Returns the index of the maximum value.",
    ".copy()": "Returns a deep copy of the array (independent data).",
    ".tolist()": "Converts a NumPy array back to a Python list.",
    "np.concatenate(": "Joins arrays along an existing axis. axis=0 stacks vertically, axis=1 horizontally.",
    "np.stack(": "Joins arrays along a new axis.",
    "np.split(": "Splits an array into equal-sized sub-arrays.",
    "np.where(": "Returns elements chosen from x or y depending on condition. Like vectorized if-else.",
    "np.clip(": "Clips array values to a range [min, max]. Essential for preventing audio clipping.",
    "np.abs(": "Element-wise absolute value.",
    "np.sqrt(": "Element-wise square root.",
    "np.log(": "Element-wise natural logarithm.",
    "np.exp(": "Element-wise exponential (e^x).",
    "np.dot(": "Dot product of two arrays. For matrices, this is matrix multiplication.",
    "np.convolve(": "Discrete, linear convolution of two 1-D sequences. Used in audio filtering.",
    "np.fft.fft(": "Fast Fourier Transform. Converts time-domain audio to frequency domain.",
    "np.fft.ifft(": "Inverse FFT. Converts frequency domain back to time domain.",
    "np.interp(": "One-dimensional linear interpolation. Used for resampling audio.",
    "np.unwrap(": "Unwraps discontinuities in an angle array (adds 2*pi jumps).",
    "np.frombuffer(": "Creates an array from a binary buffer. Used for decoding raw audio data.",
    "np.save(": "Saves a NumPy array to a .npy file.",
    "np.load(": "Loads a NumPy array from a .npy file.",
}

# ── Requests / HTTP Knowledge Base ────────────────────────────────────────

REQUESTS_KB = {
    "requests.get(": "Sends an HTTP GET request. Returns a Response object. Used for downloading voice models and checking API endpoints.",
    "requests.post(": "Sends an HTTP POST request. Used for submitting data to APIs.",
    "requests.head(": "Sends an HTTP HEAD request. Returns headers only (no body). Useful for checking file size.",
    "requests.put(": "Sends an HTTP PUT request. Used for updating resources.",
    "requests.delete(": "Sends an HTTP DELETE request.",
    "requests.Session(": "Creates a session with persistent settings (headers, cookies, auth). Reuses TCP connections.",
    ".status_code": "The HTTP status code of the response (200=OK, 404=Not Found, 500=Server Error).",
    ".text": "The response body as a decoded string.",
    ".content": "The response body as raw bytes.",
    ".json()": "Parses the response body as JSON and returns a Python dict/list.",
    ".headers": "Dictionary of response headers.",
    ".raise_for_status()": "Raises an HTTPError for bad status codes (4xx, 5xx).",
    "requests.exceptions.ConnectionError": "Raised when a network connection fails.",
    "requests.exceptions.Timeout": "Raised when the request times out.",
    "requests.exceptions.HTTPError": "Raised when raise_for_status() detects an error status code.",
    "stream=True": "Parameter for get() that enables chunked downloading. Essential for large files.",
    "chunk_size": "The number of bytes to read at a time when streaming downloads.",
    "timeout=": "Maximum seconds to wait for a response. Prevents hanging connections.",
    "headers=": "Dictionary of HTTP headers to send with the request.",
    "auth=": "Authentication credentials (username, password) or a custom auth handler.",
}

# ── sherpa-onnx Knowledge Base ────────────────────────────────────────────

SHERPA_KB = {
    "sherpa_onnx.OfflineTts(": "Creates an offline TTS engine from a configuration. The main entry point for text-to-speech synthesis.",
    "sherpa_onnx.OfflineTtsConfig": "Configuration container for the offline TTS engine. Specifies model, rule_fsts, and max_num_sentences.",
    "sherpa_onnx.OfflineTtsModelConfig": "Configuration for the TTS model. Specifies which model type (VITS, Kokoro, Matcha) to use.",
    "sherpa_onnx.OfflineTtsVitsModelConfig": "Configuration for VITS-style models. Requires model, tokens, and optionally lexicon, data_dir.",
    "sherpa_onnx.OfflineTtsKokoroModelConfig": "Configuration for Kokoro models. Requires model, voices, tokens, and data_dir for espeak-ng.",
    "sherpa_onnx.OfflineTtsMatchaModelConfig": "Configuration for Matcha models. Uses acoustic_model + vocoder for two-stage synthesis.",
    ".generate(": "Generates audio from text. Returns a GeneratedAudio object with .samples and .sample_rate.",
    "GeneratedAudio": "The result of TTS synthesis. Contains .samples (int16 array) and .sample_rate (e.g., 22050, 24000).",
    "rule_fsts=": "Path to compiled finite-state transducers for text normalization (e.g., expanding abbreviations).",
    "max_num_sentences=": "Maximum number of sentences to synthesize in one call. Higher values use more memory.",
    "num_threads=": "Number of CPU threads for inference. Higher values use more CPU but may be faster.",
    "provider=": "Compute backend: 'cpu', 'cuda', 'dml' (DirectML), or 'coreml' (macOS).",
    "lexicon=": "Path to a pronunciation lexicon file. Maps words to phoneme sequences.",
    "tokens=": "Path to the tokens file. Defines the vocabulary the model uses.",
    "data_dir=": "Path to espeak-ng data directory (for Kokoro models).",
    "model=": "Path to the ONNX model file. The main neural network weights.",
}

# ── PDF/DOCX Knowledge Base ──────────────────────────────────────────────

DOCX_KB = {
    "PdfReader(": "Creates a PDF reader object. Use len(reader) for page count; iterate for pages.",
    "PdfWriter(": "Creates a PDF writer for building new PDF files.",
    "Document(": "Opens a Word document (DOCX). Access paragraphs via doc.paragraphs.",
    ".paragraphs": "List of Paragraph objects in a Word document. Each paragraph has .text and .style.",
    ".text": "The text content of a paragraph or page.",
    ".style": "The style of a paragraph (e.g., 'Heading 1', 'Normal').",
    ".pages": "List of Page objects in a PDF reader. Each page has extract_text().",
    "page.extract_text()": "Extracts text content from a PDF page. Returns a string.",
    "page.page_number": "Zero-based page number in the PDF reader.",
    "doc.add_heading(": "Adds a heading to a Word document with the specified text and level.",
    "doc.add_paragraph(": "Adds a paragraph to a Word document.",
    "doc.save(": "Saves the Word document to a file.",
}

# ── FFmpeg Knowledge Base ────────────────────────────────────────────────

FFMPEG_KB = {
    "ffmpeg": "A command-line tool for audio/video processing. Used by AIVS for WAV-to-MP3/FLAC conversion.",
    "ffmpeg.exe": "The FFmpeg binary bundled with AIVS. Located in %APPDATA%/AIVoiceStudio/ffmpeg/.",
    "-i ": "Input file parameter. Specifies the source audio file.",
    "-y": "Overwrite output files without asking.",
    "-acodec ": "Audio codec parameter. 'libmp3lame' for MP3, 'flac' for FLAC.",
    "-ab ": "Audio bitrate parameter (e.g., -ab 192k for 192 kbps MP3).",
    "-ar ": "Audio sample rate parameter (e.g., -ar 24000 for 24 kHz).",
    "-ac ": "Audio channels parameter (-ac 1 for mono, -ac 2 for stereo).",
    "subprocess.run(": "Runs a command and waits for completion. Used to invoke FFmpeg.",
    "subprocess.Popen(": "Starts a process with more control. Used for FFmpeg with progress monitoring.",
}

# ── Testing Knowledge Base ────────────────────────────────────────────────

TESTING_KB = {
    "pytest": "The testing framework used by AIVS. Discovers test files named test_*.py.",
    "pytest.main(": "Runs pytest programmatically. Returns an ExitCode.",
    "def test_": "Test function. pytest discovers functions prefixed with 'test_'. No class needed.",
    "assert ": "Assertion that verifies a condition is True. If False, pytest raises AssertionError with the message.",
    "assertEqual(": "Asserts two values are equal. Provides a detailed diff on failure.",
    "assertNotEqual(": "Asserts two values are not equal.",
    "assertTrue(": "Asserts a value is True.",
    "assertFalse(": "Asserts a value is False.",
    "assertRaises(": "Asserts that a specific exception is raised. Use as a context manager.",
    "assertIn(": "Asserts that a value is in a container.",
    "assertIsNone(": "Asserts that a value is None.",
    "assertIsNotNone(": "Asserts that a value is not None.",
    "assertAlmostEqual(": "Asserts two floats are approximately equal (within 7 decimal places).",
    "setUp(": "Runs before each test method. Used to create test fixtures.",
    "tearDown(": "Runs after each test method. Used to clean up test fixtures.",
    "setUpClass(": "Runs once before all test methods in the class. Used for expensive one-time setup.",
    "tearDownClass(": "Runs once after all test methods in the class.",
    "mock.patch(": "Decorator/context manager that replaces a function/attribute with a Mock during testing.",
    "Mock(": "Creates a mock object. Any method call or attribute access returns a Mock.",
    "MagicMock(": "Like Mock but supports magic methods (__len__, __iter__, etc.).",
    "patch(": "Context manager that replaces a name with a Mock. Use 'as' to get the mock reference.",
    "mock.return_value": "The value that a mock returns when called.",
    "mock.side_effect": "A function or exception to raise when the mock is called. Use for simulating errors.",
    "mock.called": "Boolean: True if the mock was called at least once.",
    "mock.call_args": "The arguments the mock was last called with.",
    "mock.assert_called_with(": "Asserts the mock was called with specific arguments.",
    "mock.assert_not_called()": "Asserts the mock was never called.",
    "tmp_path": "pytest fixture that provides a temporary directory unique to each test.",
    "monkeypatch": "pytest fixture for modifying objects, environment variables, or attributes during testing.",
    "capsys": "pytest fixture that captures stdout/stderr output during a test.",
    "parametrize": "Decorator that runs a test function with different sets of arguments.",
    "fixture": "Decorator that marks a function as a test fixture. Provides setup/teardown for tests.",
    "mark.skip(": "Skips a test with a reason. Used for platform-specific or incomplete tests.",
    "mark.xfail(": "Marks a test as expected to fail. If it passes, it's reported as an unexpected success.",
    "mark.parametrize(": "Decorator that runs the test with multiple argument sets.",
    "conftest.py": "Special pytest file for shared fixtures and hooks. Auto-discovered by pytest.",
}

# ── Packaging Knowledge Base ──────────────────────────────────────────────

PACKAGING_KB = {
    "PyInstaller": "A tool that packages Python apps into standalone executables. Bundles Python interpreter and all dependencies.",
    "pyinstaller": "Command-line tool for building executables. Supports --onefile, --windowed, --add-data options.",
    ".spec": "PyInstaller specification file. Defines the build configuration: files, hidden imports, options.",
    "spec=": "PyInstaller spec parameter. Specifies the .spec file to use for building.",
    "--onefile": "PyInstaller flag: bundles everything into a single .exe file.",
    "--windowed": "PyInstaller flag: hides the console window for GUI applications.",
    "--noconfirm": "PyInstaller flag: overwrites output without asking.",
    "--add-data": "PyInstaller flag: includes additional files in the bundle (src;dest format).",
    "hiddenimports=": "PyInstaller parameter for modules that aren't automatically detected.",
    "Analysis(": "PyInstaller spec component that analyzes the script and collects dependencies.",
    "PYZ(": "PyInstaller spec component that creates the compressed Python archive.",
    "EXE(": "PyInstaller spec component that creates the final executable.",
    "COLLECT(": "PyInstaller spec component that gathers all files into the output directory.",
    "Inno Setup": "A Windows installer builder. Compiles .iss scripts into setup.exe files.",
    "ISCC.exe": "Inno Setup Compiler. Compiles .iss scripts into installer executables.",
    ".iss": "Inno Setup Script file. Defines the installer: files, icons, license, dialogs.",
    "[Setup]": "Inno Setup section for installer metadata (name, version, output file).",
    "[Files]": "Inno Setup section listing files to include in the installer.",
    "[Icons]": "Inno Setup section for Start Menu and desktop shortcuts.",
    "[Run]": "Inno Setup section for programs to run after installation.",
    "[UninstallDelete]": "Inno Setup section for files to remove during uninstall.",
    "AppVersion=": "Inno Setup directive setting the application version number.",
    "OutputBaseFilename=": "Inno Setup directive setting the output installer filename.",
    "DefaultDirName=": "Inno Setup directive setting the default installation directory.",
}

# ── Subprocess / Runtime Knowledge Base ───────────────────────────────────

RUNTIME_KB = {
    "subprocess.run(": "Runs a command, waits for completion. Returns CompletedProcess. Use capture_output=True to capture stdout/stderr.",
    "subprocess.Popen(": "Starts a process. More control than run() — can read stdout/stderr asynchronously.",
    "subprocess.Popen.stdout": "The stdout stream of the process. Use .readline() or iterate for line-by-line reading.",
    "subprocess.Popen.stderr": "The stderr stream of the process.",
    "subprocess.Popen.poll()": "Returns the exit code if the process has finished, or None if still running.",
    "subprocess.Popen.terminate()": "Sends SIGTERM to the process. Use for graceful shutdown.",
    "subprocess.Popen.kill()": "Sends SIGKILL to the process. Use for forced termination.",
    "subprocess.Popen.communicate()": "Sends input and waits for process to finish. Returns (stdout, stderr).",
    "subprocess.PIPE": "Special value that creates a new pipe for stdout/stderr. Used with Popen to capture output.",
    "subprocess.DEVNULL": "Special value that discards output. Used when you don't need stdout/stderr.",
    "check=False": "If True, raises CalledProcessError on non-zero exit code.",
    "shell=True": "Runs the command through the shell. Use only with trusted input (security risk).",
    "capture_output=True": "Captures stdout and stderr. Available in subprocess.run().",
    "cwd=": "Sets the working directory for the subprocess.",
    "env=": "Dictionary of environment variables for the subprocess.",
    "timeout=": "Maximum seconds to wait. Raises TimeoutExpired if exceeded.",
    "PythonRuntime": "Manages the embedded Python interpreter for addons. Creates isolated venvs.",
    "PythonRuntime.ensure_env()": "Creates the managed virtual environment if it doesn't exist yet.",
    "PythonRuntime.pip_install()": "Installs packages into the managed venv. Provides progress callbacks.",
    "PythonRuntime.run_in_env()": "Runs a Python script inside the managed venv's interpreter.",
    "PythonRuntime.is_package_installed()": "Checks if a package is installed in the managed venv.",
}

# ── Logging Knowledge Base ────────────────────────────────────────────────

LOGGING_KB = {
    "logging.getLogger(": "Creates or retrieves a logger by name. Hierarchical naming (a.b.c) enables per-module control.",
    "logging.basicConfig(": "Configures the root logger. Sets format, level, and output destination (file or stream).",
    "logging.FileHandler(": "Sends log output to a file. Supports rotation to prevent huge log files.",
    "logging.StreamHandler(": "Sends log output to a stream (default: stderr).",
    "logging.Formatter(": "Formats log records. Use %(name)s, %(levelname)s, %(message)s, %(asctime)s placeholders.",
    "log.debug(": "Logs a debug message. Only visible when the level is set to DEBUG. For detailed diagnostic info.",
    "log.info(": "Logs an informational message. The default level. For normal operation events.",
    "log.warning(": "Logs a warning. Something unexpected happened but the program continues.",
    "log.error(": "Logs an error. Something failed but the program can still run.",
    "log.critical(": "Logs a critical error. The program may not be able to continue.",
    "log.exception(": "Logs an error with the full traceback. Use inside except blocks.",
    "logging.DEBUG": "Level 10. Most verbose. Shows all diagnostic information.",
    "logging.INFO": "Level 20. Default level. Shows normal operation messages.",
    "logging.WARNING": "Level 30. Shows warnings and above.",
    "logging.ERROR": "Level 40. Shows errors and above.",
    "logging.CRITICAL": "Level 50. Shows only critical errors.",
}

# ── Performance / Security Knowledge Base ─────────────────────────────────

PERF_KB = {
    "functools.lru_cache": "Decorator that caches function results. Reduces repeated expensive computations. maxsize controls cache size.",
    "@lru_cache(maxsize=128)": "Caches the last 128 unique calls. Same inputs always return the cached output.",
    "time.perf_counter()": "Returns the current time in seconds (high resolution). Used for benchmarking.",
    "time.time()": "Returns the current Unix timestamp. Lower resolution than perf_counter().",
    "timeit.timeit(": "Measures execution time of a small code snippet. Runs it multiple times for accuracy.",
    "memory_profiler": "Third-party tool that measures memory usage line by line. Useful for finding leaks.",
    "cProfile": "Built-in profiler that records function call counts and timing. Use cProfile.run() or context manager.",
    "line_profiler": "Third-party profiler that measures time per line of code.",
    "hashlib": "Built-in module for secure hashing (SHA-256, MD5). Used for verifying file integrity.",
    "hashlib.sha256(": "Creates a SHA-256 hash object. Use .update() to feed data, .hexdigest() to get the hash.",
    "secrets.token_hex(": "Generates a cryptographically secure random hex string. Used for API keys and tokens.",
    "secrets.token_urlsafe(": "Generates a URL-safe random string. Used for session tokens.",
    "ssl.create_default_context()": "Creates a secure SSL context for HTTPS connections.",
    "shutil.which(": "Finds the full path of an executable in the system PATH. Returns None if not found.",
}

# ── Combine all knowledge bases ───────────────────────────────────────────

ALL_KB = {}
ALL_KB.update(NUMPY_KB)
ALL_KB.update(REQUESTS_KB)
ALL_KB.update(SHERPA_KB)
ALL_KB.update(DOCX_KB)
ALL_KB.update(FFMPEG_KB)
ALL_KB.update(TESTING_KB)
ALL_KB.update(PACKAGING_KB)
ALL_KB.update(RUNTIME_KB)
ALL_KB.update(LOGGING_KB)
ALL_KB.update(PERF_KB)

# Also include core Python patterns (imported from enhance_part1_to_4)
CORE_PYTHON_KB = {
    "print(": "Outputs text to the console. The primary way to display information during development.",
    "len(": "Returns the number of items in a sequence (string, list, dict, tuple).",
    "type(": "Returns the type/class of an object. Used for debugging.",
    "isinstance(": "Checks if an object is an instance of a class. Preferred over type() for type checking.",
    "int(": "Converts a value to an integer.",
    "float(": "Converts a value to a floating-point number.",
    "str(": "Converts a value to its string representation.",
    "bool(": "Converts a value to its boolean equivalent.",
    "range(": "Generates a sequence of integers. Commonly used in for loops.",
    "enumerate(": "Adds a counter to an iterable, returning (index, value) pairs.",
    "zip(": "Combines multiple iterables element-wise.",
    "sorted(": "Returns a new sorted list from any iterable.",
    "reversed(": "Returns a reverse iterator over a sequence.",
    "sum(": "Adds all items in an iterable.",
    "min(": "Returns the smallest item.",
    "max(": "Returns the largest item.",
    "abs(": "Returns the absolute value of a number.",
    "round(": "Rounds a number to the given decimal places.",
    "open(": "Opens a file and returns a file object.",
    "isinstance(": "Checks if an object is an instance of a class.",
    "super(": "Returns a proxy that delegates method calls to a parent class.",
    "any(": "Returns True if any element in the iterable is truthy.",
    "all(": "Returns True if all elements in the iterable are truthy.",
    "getattr(": "Gets an attribute value by name.",
    "setattr(": "Sets an attribute value by name.",
    "hasattr(": "Returns True if an object has the named attribute.",
    "callable(": "Returns True if an object can be called.",
    "with open(": "Opens a file using a context manager. Auto-closes when the block exits.",
    "os.path.join()": "Joins path components using the OS-appropriate separator.",
    "os.makedirs": "Recursively creates directories.",
    "os.environ.get(": "Reads an environment variable with a fallback default.",
    "sys.platform": "String identifying the OS: 'win32', 'darwin', 'linux'.",
    "json.load": "Deserializes JSON from a file into a Python object.",
    "json.dump": "Serializes a Python object to JSON and writes to a file.",
    "logging.getLogger": "Creates or retrieves a logger by name.",
    "threading.Thread": "Creates a background thread.",
    "threading.Lock": "A mutual exclusion lock.",
    "threading.RLock": "A reentrant lock that allows the same thread to acquire it multiple times.",
    "__init__": "Constructor method. Called when creating a new instance of a class.",
    "__str__": "Returns a human-readable string representation.",
    "__repr__": "Returns a developer-friendly string representation.",
    "__enter__": "Called when entering a 'with' block.",
    "__exit__": "Called when leaving a 'with' block.",
    "__call__": "Makes an instance callable like a function.",
    "yield ": "Pauses the function and sends a value to the caller.",
    "@property": "Decorator that converts a method into a managed attribute.",
    "@staticmethod": "Decorator that creates a method without 'self' or 'cls'.",
    "@classmethod": "Decorator that receives the class (cls) as the first argument.",
    "def test_": "Test function. pytest discovers functions prefixed with 'test_'.",
    "assert ": "Assertion that verifies a condition is True.",
    "try:": "Begins exception handling.",
    "except:": "Catches exceptions.",
    "finally:": "Always executes for cleanup.",
    "raise ": "Manually raises an exception.",
    "class ": "Defines a new class.",
    "def ": "Defines a function.",
    "import ": "Imports a module.",
    "from ": "Imports specific names from a module.",
    "if ": "Conditional check.",
    "elif ": "Alternative condition.",
    "else:": "Fallback branch.",
    "for ": "For loop iteration.",
    "while ": "While loop.",
    "return ": "Returns a value from a function.",
    "pass": "Placeholder: does nothing.",
    "break": "Exits the innermost loop.",
    "continue": "Skips to the next iteration.",
}
ALL_KB.update(CORE_PYTHON_KB)

# ── ASCII Art Detection ──────────────────────────────────────────────────

def is_ascii_art(line: str) -> bool:
    art_chars = set('│┌┐└┘├┤┬┴┼─═║╔╗╚╝╠╣╦╩╬')
    stripped = line.strip()
    if not stripped:
        return False
    art_count = sum(1 for c in stripped if c in art_chars or c == ' ')
    return art_count / max(len(stripped), 1) > 0.6


# ── Description Engine ───────────────────────────────────────────────────

def describe_line(line: str, chapter_num: int, context_lines=None, line_idx=0) -> str:
    stripped = line.strip()
    if not stripped:
        return ""

    if is_ascii_art(stripped):
        return ""

    # Comments
    if stripped.startswith('#'):
        inner = stripped.lstrip('#').strip()
        if inner:
            return f"A comment: <em>{html.escape(inner)}</em>."
        return ""

    # Docstrings
    if stripped.startswith('"""') or stripped.startswith("'''"):
        inner = stripped.strip('"""').strip("'''").strip()
        if inner:
            return f"Docstring: <em>{html.escape(inner)}</em>. Documents the following code."
        return "Opens a docstring block."

    # Check knowledge base (longest match first)
    best_match = None
    best_len = 0
    for pattern, desc in ALL_KB.items():
        if pattern in stripped and len(pattern) > best_len:
            best_match = desc
            best_len = len(pattern)
    if best_match:
        return best_match

    # Import statements
    m = re.match(r'^from\s+(\S+)\s+import\s+(.+)', stripped)
    if m:
        module, names = m.group(1), m.group(2)
        name_list = [n.strip().split(' as ')[0].strip() for n in names.split(',')]
        for name in name_list:
            if name in ALL_KB:
                return f"Imports <code>{html.escape(name)}</code>: {ALL_KB[name]}"
        return f"Imports <code>{html.escape(names)}</code> from <code>{html.escape(module)}</code>."

    m = re.match(r'^import\s+(.+)', stripped)
    if m:
        modules = m.group(1)
        return f"Imports the <code>{html.escape(modules)}</code> module."

    # Class definitions
    m = re.match(r'^class\s+(\w+)(?:\(([^)]*)\))?:', stripped)
    if m:
        name, bases = m.group(1), m.group(2) or ""
        if name in ALL_KB:
            return f"Defines class <code>{html.escape(name)}</code>: {ALL_KB[name]}"
        if bases:
            return f"Defines class <code>{html.escape(name)}</code> inheriting from <code>{html.escape(bases)}</code>."
        return f"Defines class <code>{html.escape(name)}</code>."

    # Function definitions
    m = re.match(r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:', stripped)
    if m:
        name, params, ret = m.group(2), m.group(3), m.group(4)
        param_list = [p.strip().split(':')[0].split('=')[0].strip() for p in params.split(',') if p.strip()]
        if name in ALL_KB:
            return f"Defines <code>{html.escape(name)}</code>: {ALL_KB[name]}"
        param_str = ', '.join(param_list[:4])
        if len(param_list) > 4:
            param_str += ', ...'
        desc = f"Defines function <code>{html.escape(name)}</code>"
        if param_list:
            desc += f" accepting ({html.escape(param_str)})"
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
        return f"Returns <code>{html.escape(val[:100])}</code> to the caller."
    if stripped == 'return':
        return "Returns <code>None</code> implicitly."

    # Control flow
    m = re.match(r'^\s*if\s+(.+):', stripped)
    if m:
        return f"Checks condition: <code>{html.escape(m.group(1)[:120])}</code>. Block runs only when True."

    m = re.match(r'^\s*elif\s+(.+):', stripped)
    if m:
        return f"Alternative condition: <code>{html.escape(m.group(1)[:120])}</code>."

    if re.match(r'^\s*else:', stripped):
        return "Fallback: runs when none of the above conditions matched."

    # Loops
    m = re.match(r'^\s*for\s+(\w+)\s+in\s+(.+):', stripped)
    if m:
        return f"Iterates over <code>{html.escape(m.group(2)[:100])}</code>."

    m = re.match(r'^\s*while\s+(.+):', stripped)
    if m:
        return f"While loop: repeats as long as <code>{html.escape(m.group(1)[:100])}</code>."

    # Exception handling
    if re.match(r'^\s*try:', stripped):
        return "Begins exception handling."
    m = re.match(r'^\s*except\s+(\w+)', stripped)
    if m:
        return f"Catches <code>{html.escape(m.group(1))}</code> exceptions."
    if re.match(r'^\s*finally:', stripped):
        return "Always executes for cleanup."
    if re.match(r'^\s*raise\s+', stripped):
        return f"Raises an exception: <code>{html.escape(stripped)}</code>."

    # Context managers
    m = re.match(r'^\s*with\s+(.+):', stripped)
    if m:
        return f"Context manager: <code>{html.escape(m.group(1)[:120])}</code>. Auto-manages cleanup."

    # self.xxx assignments
    m = re.match(r'^\s*self\.(\w+)\s*=\s*(.+)', stripped)
    if m:
        attr, val = m.group(1), m.group(2)
        if attr in ALL_KB:
            return f"Sets <code>{html.escape(attr)}</code>: {ALL_KB[attr]}"
        return f"Stores <code>{html.escape(val[:60])}</code> in instance attribute <code>{html.escape(attr)}</code>."

    # wx.xxx calls
    m = re.match(r'^\s*(wx\.\w+)\s*\(', stripped)
    if m:
        wxclass = m.group(1)
        if wxclass in ALL_KB:
            return f"Creates a {wxclass.replace('wx.', '')}: {ALL_KB[wxclass]}"
        return f"Creates a wxPython <code>{html.escape(wxclass)}</code> widget."

    # Method calls on self
    m = re.match(r'^\s*self\.(\w+)\(', stripped)
    if m:
        method = m.group(1)
        if method in ALL_KB:
            return f"Calls <code>self.{method}()</code>: {ALL_KB[method]}"
        return f"Calls <code>self.{html.escape(method)}()</code>."

    # Generic method calls
    m = re.match(r'^\s*(\w+)\(', stripped)
    if m:
        func = m.group(1)
        if func in ALL_KB:
            return f"Calls <code>{html.escape(func)}()</code>: {ALL_KB[func]}"
        return f"Calls function <code>{html.escape(func)}()</code>."

    # Augmented assignment
    m = re.match(r'^\s*(\w+)\s*(\+=|-=|\*=|/=|//=|%=)\s*(.+)', stripped)
    if m:
        var, op, val = m.group(1), m.group(2), m.group(3)
        return f"Updates <code>{html.escape(var)}</code> by {op} <code>{html.escape(val[:60])}</code>."

    # Generic assignment
    m = re.match(r'^(\w+(?:\.\w+)*)\s*=\s*(.+)', stripped)
    if m:
        var, val = m.group(1), m.group(2)
        if var in ALL_KB:
            return f"Sets <code>{html.escape(var)}</code>: {ALL_KB[var]}"
        return f"Assigns <code>{html.escape(val[:100])}</code> to <code>{html.escape(var)}</code>."

    # Decorators
    if stripped.startswith('@'):
        return f"Decorator: <code>{html.escape(stripped)}</code>. Wraps the following function/class."

    # Special keywords
    if stripped == 'pass':
        return "Placeholder: does nothing."
    if stripped == 'break':
        return "Exits the innermost loop."
    if stripped == 'continue':
        return "Skips to the next iteration."
    if stripped == '...':
        return "Ellipsis: abstract method placeholder."
    if stripped == 'super()':
        return "Returns a proxy that delegates to the parent class."

    return f"<code>{html.escape(stripped[:140])}</code>"


def build_explanation_block(code_text: str, block_index: int, chapter_num: int) -> str:
    raw_lines = code_text.split('\n')
    art_lines = [l for l in raw_lines if l.strip() and is_ascii_art(l.strip())]
    is_art = len(art_lines) > len([l for l in raw_lines if l.strip()]) * 0.5

    if is_art and art_lines:
        return f'''
<div class="line-explanation diagram">
  <h4>Diagram (Block {block_index})</h4>
  <p>This diagram illustrates a concept in the chapter. Refer to the surrounding text for explanation.</p>
</div>
'''

    code_lines = [(i, l) for i, l in enumerate(raw_lines) if l.strip()]
    if not code_lines:
        return ''

    items = []
    for display_num, (orig_idx, line) in enumerate(code_lines, 1):
        desc = describe_line(line, chapter_num, context_lines=code_lines, line_idx=orig_idx)
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


def strip_html_tags(text: str) -> str:
    text = re.sub(r'<span[^>]*>', '', text)
    text = re.sub(r'</span>', '', text)
    text = html.unescape(text)
    return text


def process_file(filepath: str) -> bool:
    basename = os.path.basename(filepath)
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

    pattern = re.compile(r'(<pre><code>)(.*?)(</code></pre>)', re.DOTALL)

    block_index = 0
    def replacer(match):
        nonlocal block_index
        block_index += 1
        opening = match.group(1)
        code_html = match.group(2)
        closing = match.group(3)
        code_text = strip_html_tags(code_html)
        explanation = build_explanation_block(code_text, block_index, chapter_num)
        return f'{opening}{code_html}{closing}\n{explanation}'

    new_content = pattern.sub(replacer, content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False


def main():
    # Part V (37-42), Part VII (61-66), Part VIII (67-72)
    patterns = (
        glob.glob(os.path.join(CHAPTERS_DIR, 'ch3[7-9]_*.html')) +
        glob.glob(os.path.join(CHAPTERS_DIR, 'ch4[0-2]_*.html')) +
        glob.glob(os.path.join(CHAPTERS_DIR, 'ch6[1-6]_*.html')) +
        glob.glob(os.path.join(CHAPTERS_DIR, 'ch6[7-9]_*.html')) +
        glob.glob(os.path.join(CHAPTERS_DIR, 'ch7[0-2]_*.html'))
    )
    files = sorted(set(patterns))
    modified = 0
    for fp in files:
        if process_file(fp):
            print(f'  [OK] {os.path.basename(fp)}')
            modified += 1
        else:
            print(f'  [--] {os.path.basename(fp)} (skipped)')
    print(f'\nDone. Enhanced {modified}/{len(files)} remaining files.')


if __name__ == '__main__':
    main()
