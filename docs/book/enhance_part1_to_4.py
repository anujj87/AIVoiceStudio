#!/usr/bin/env python3
"""
Enhanced line-by-line description engine for Parts I-IV (chapters 1-36).
Covers Python fundamentals, intermediate Python, advanced Python, and wxPython.
"""

import re, html, os, glob

CHAPTERS_DIR = os.path.join(os.path.dirname(__file__), "chapters")

# ── Python Fundamentals Knowledge Base ────────────────────────────────────

PYTHON_KB = {
    # Built-in functions
    "print(": "Outputs text to the console. The primary way to display information during development and debugging.",
    "len(": "Returns the number of items in a sequence (string, list, dict, tuple). Essential for loop bounds and validation.",
    "type(": "Returns the type/class of an object. Used for debugging and type checking.",
    "isinstance(": "Checks if an object is an instance of a class (or tuple of classes). Preferred over type() == for type checking because it supports inheritance.",
    "int(": "Converts a value to an integer. Truncates floats toward zero. Raises ValueError for invalid strings.",
    "float(": "Converts a value to a floating-point number. Used for decimal arithmetic.",
    "str(": "Converts a value to its string representation. Essential for string concatenation and formatting.",
    "bool(": "Converts a value to its boolean equivalent. Zero, empty strings, None, and empty collections are False; everything else is True.",
    "range(": "Generates a sequence of integers. Commonly used in for loops: range(n) produces 0 to n-1.",
    "enumerate(": "Adds a counter to an iterable, returning (index, value) pairs. Preferred over manual index tracking.",
    "zip(": "Combines multiple iterables element-wise. Stops at the shortest input. Used for parallel iteration.",
    "map(": "Applies a function to every item in an iterable. Returns an iterator. Often replaced by list comprehensions.",
    "filter(": "Returns items from an iterable where the function returns True. Returns an iterator.",
    "sorted(": "Returns a new sorted list from any iterable. Does not modify the original. Accepts a key= parameter for custom sorting.",
    "reversed(": "Returns a reverse iterator over a sequence. Does not modify the original.",
    "sum(": "Adds all items in an iterable. Supports an optional start= parameter.",
    "min(": "Returns the smallest item. Can compare strings lexicographically.",
    "max(": "Returns the largest item. Can compare strings lexicographically.",
    "abs(": "Returns the absolute value of a number. Works with int, float, and complex.",
    "round(": "Rounds a number to the given number of decimal places. Uses banker's rounding (round half to even).",
    "input(": "Reads a line of text from the user via the console. Always returns a string.",
    "open(": "Opens a file and returns a file object. Supports modes: 'r' (read), 'w' (write), 'a' (append), 'b' (binary).",
    "range(": "Generates a sequence of integers. Commonly used in for loops.",
    "dir(": "Returns a list of attributes and methods of an object. Useful for exploration and debugging.",
    "vars(": "Returns the __dict__ attribute of an object — its namespace as a dictionary.",
    "id(": "Returns the unique identity of an object (memory address in CPython). Used with 'is' for identity checks.",
    "hash(": "Returns the hash value of an object. Required for objects used as dictionary keys or in sets.",
    "callable(": "Returns True if an object can be called (function, method, class with __call__).",
    "getattr(": "Gets an attribute value by name. Supports a default value if the attribute doesn't exist.",
    "setattr(": "Sets an attribute value by name. Dynamically adds or modifies object attributes.",
    "hasattr(": "Returns True if an object has the named attribute. Useful before calling getattr.",
    "super(": "Returns a proxy object that delegates method calls to a parent class. Used in __init__ to call the parent constructor.",
    "repr(": "Returns the developer-friendly string representation of an object. Used in debugging and logging.",
    "format(": "Formats a value according to a format specification. Used in f-strings and .format() method.",
    "chr(": "Returns the character corresponding to a Unicode code point. chr(65) returns 'A'.",
    "ord(": "Returns the Unicode code point of a character. ord('A') returns 65.",
    "hex(": "Converts an integer to a hexadecimal string. hex(255) returns '0xff'.",
    "bin(": "Converts an integer to a binary string. bin(10) returns '0b1010'.",
    "oct(": "Converts an integer to an octal string. oct(8) returns '0o10'.",
    "any(": "Returns True if any element in the iterable is truthy. Short-circuits on first True.",
    "all(": "Returns True if all elements in the iterable are truthy. Short-circuits on first False.",
    "next(": "Gets the next item from an iterator. Raises StopIteration if exhausted.",
    "iter(": "Returns an iterator from an object. Used to create custom iterators.",
    "slice(": "Creates a slice object for sequence slicing. Useful for dynamic slice construction.",
    "property(": "Creates a managed attribute. Allows getter, setter, and deleter methods.",
    "staticmethod(": "Creates a static method that doesn't receive the instance or class as the first argument.",
    "classmethod(": "Creates a method that receives the class (cls) as the first argument instead of the instance (self).",
    "isinstance(": "Checks if an object is an instance of a class. Preferred over type() for type checking.",
    "exec(": "Executes dynamically created Python code. Use with caution — security risk.",
    "eval(": "Evaluates a Python expression and returns its result. Use with caution.",

    # String methods
    ".upper(": "Converts all characters to uppercase. Returns a new string (strings are immutable).",
    ".lower(": "Converts all characters to lowercase. Returns a new string.",
    ".strip(": "Removes leading and trailing whitespace (or specified characters). Returns a new string.",
    ".lstrip(": "Removes leading whitespace only.",
    ".rstrip(": "Removes trailing whitespace only.",
    ".split(": "Splits a string into a list by the given delimiter. Without arguments, splits on whitespace.",
    ".join(": "Joins an iterable of strings with the calling string as separator. ', '.join(['a','b']) returns 'a, b'.",
    ".replace(": "Returns a copy with all occurrences of old replaced by new. Strings are immutable.",
    ".find(": "Returns the index of the first occurrence, or -1 if not found.",
    ".index(": "Like find(), but raises ValueError instead of returning -1.",
    ".count(": "Returns the number of non-overlapping occurrences of a substring.",
    ".startswith(": "Returns True if the string starts with the given prefix.",
    ".endswith(": "Returns True if the string ends with the given suffix.",
    ".isdigit(": "Returns True if all characters are digits.",
    ".isalpha(": "Returns True if all characters are alphabetic.",
    ".isalnum(": "Returns True if all characters are alphanumeric.",
    ".zfill(": "Pads the string with zeros on the left to the given width. '42'.zfill(5) returns '00042'.",
    ".center(": "Centers the string in a field of given width, padded with spaces.",
    ".ljust(": "Left-justifies the string in a field of given width.",
    ".rjust(": "Right-justifies the string in a given width field.",
    ".encode(": "Encodes the string to bytes using the specified encoding (default UTF-8).",
    ".format(": "Formats the string with positional or keyword arguments. Superseded by f-strings.",
    ".title(": "Capitalizes the first letter of each word.",
    ".capitalize(": "Capitalizes the first letter of the string.",
    ".swapcase(": "Swaps case of each character.",
    ".expandtabs(": "Expands tab characters to spaces.",

    # List methods
    ".append(": "Adds an item to the end of the list. Modifies the list in-place.",
    ".extend(": "Adds all items from an iterable to the end of the list. Modifies in-place.",
    ".insert(": "Inserts an item at the given index. Shifts subsequent items right.",
    ".remove(": "Removes the first occurrence of a value. Raises ValueError if not found.",
    ".pop(": "Removes and returns the item at the given index (default: last).",
    ".clear(": "Removes all items from the list.",
    ".index(": "Returns the index of the first occurrence of a value.",
    ".count(": "Returns the number of occurrences of a value.",
    ".sort(": "Sorts the list in-place. Accepts key= and reverse= parameters.",
    ".copy(": "Returns a shallow copy of the list.",
    ".reverse(": "Reverses the list in-place.",
    ".extend(": "Extends the list by appending items from an iterable.",
    ".append(": "Adds a single item to the end of the list.",

    # Dict methods
    ".get(": "Returns the value for a key, or a default value if the key doesn't exist. Safer than direct access.",
    ".keys(": "Returns a view of the dictionary's keys.",
    ".values(": "Returns a view of the dictionary's values.",
    ".items(": "Returns a view of the dictionary's (key, value) pairs.",
    ".update(": "Updates the dictionary with key-value pairs from another dict or iterable.",
    ".pop(": "Removes and returns the value for a key, or a default if not found.",
    ".setdefault(": "Returns the value for a key, or inserts it with a default if not found.",
    ".copy(": "Returns a shallow copy of the dictionary.",
    ".clear(": "Removes all items from the dictionary.",
    ".fromkeys(": "Creates a new dictionary with all keys set to the same value.",

    # Set methods
    ".add(": "Adds an element to the set. Ignores duplicates.",
    ".remove(": "Removes an element. Raises KeyError if not found.",
    ".discard(": "Removes an element if present. Does nothing if not found.",
    ".union(": "Returns a new set with elements from both sets.",
    ".intersection(": "Returns a new set with elements common to both sets.",
    ".difference(": "Returns a new set with elements in the first set but not the second.",
    ".symmetric_difference(": "Returns a new set with elements in either set but not both.",
    ".issubset(": "Returns True if all elements are in another set.",
    ".issuperset(": "Returns True if the set contains all elements of another set.",
    ".isdisjoint(": "Returns True if the sets have no common elements.",

    # File I/O
    "with open(": "Opens a file using a context manager. The file is automatically closed when the block exits, even if an error occurs.",
    "read()": "Reads the entire file contents as a string.",
    "readline()": "Reads a single line from the file, including the newline character.",
    "readlines()": "Reads all lines and returns them as a list of strings.",
    "write(": "Writes a string to the file. In text mode, does not add a newline automatically.",
    "writelines(": "Writes a list of strings to the file without adding newlines.",
    "seek(": "Moves the file pointer to a specific position. seek(0) goes to the beginning.",
    "tell()": "Returns the current position of the file pointer.",
    "close()": "Closes the file and flushes the write buffer. Prefer using 'with' instead.",

    # Exception handling
    "try:": "Begins exception handling. Code in this block is monitored for errors.",
    "except:": "Catches exceptions. Use specific exception types when possible.",
    "except Exception:": "Catches all exceptions that inherit from Exception. Use as a last resort.",
    "except Exception as e:": "Catches all exceptions and assigns the exception object to 'e' for inspection.",
    "finally:": "Always executes for cleanup code, whether or not an exception occurred.",
    "raise ": "Manually raises an exception. Use to signal error conditions to callers.",
    "ValueError": "Raised when a function receives an argument of the right type but inappropriate value.",
    "TypeError": "Raised when an operation is applied to an object of inappropriate type.",
    "KeyError": "Raised when a dictionary key is not found.",
    "IndexError": "Raised when a sequence index is out of range.",
    "FileNotFoundError": "Raised when a file or directory is requested but doesn't exist.",
    "AttributeError": "Raised when an attribute reference or assignment fails.",
    "ImportError": "Raised when an import statement fails.",
    "RuntimeError": "Raised when an error doesn't fall into any other category.",
    "StopIteration": "Raised by iterators when there are no more values to yield.",

    # Iterators and generators
    "yield ": "Pauses the function and sends a value to the caller. The function can be resumed later. Makes the function a generator.",
    "yield from ": "Delegates to a sub-generator. Shorthand for iterating and yielding each item.",
    "__iter__": "Returns the iterator object itself. Called when starting a for loop.",
    "__next__": "Returns the next value from the iterator. Raises StopIteration when exhausted.",
    "__getitem__": "Allows bracket notation (obj[key]). Called by indexing and slicing.",

    # Decorators
    "@property": "Decorator that converts a method into a managed attribute. Enables getter/setter syntax.",
    "@staticmethod": "Decorator that creates a method without 'self' or 'cls' parameter. Called on the class, not an instance.",
    "@classmethod": "Decorator that receives the class (cls) as the first argument instead of the instance.",
    "@functools.wraps": "Decorator that preserves the wrapped function's name, docstring, and other metadata.",
    "@lru_cache": "Decorator that caches function results. Same inputs always return the cached output.",
    "@abstractmethod": "Decorator marking a method as abstract. Subclasses must override it.",

    # Type hints
    "Optional[": "Type hint meaning the value can be T or None. Equivalent to T | None in Python 3.10+.",
    "Union[": "Type hint meaning the value can be one of several types.",
    "List[": "Type hint for a list containing elements of type T.",
    "Dict[": "Type hint for a dictionary with specific key and value types.",
    "Tuple[": "Type hint for a tuple with specific element types.",
    "Set[": "Type hint for a set containing elements of type T.",
    "Callable[": "Type hint for a callable (function) with specific parameter and return types.",
    "Any": "Type hint meaning any type is acceptable. Use sparingly.",
    "NoReturn": "Type hint indicating the function never returns normally (always raises an exception).",

    # Threading and concurrency
    "threading.Thread": "Creates a daemon or non-daemon thread for concurrent execution. Use target= to specify the function.",
    "threading.Lock": "A mutual exclusion lock. Only one thread can hold it at a time. Prevents race conditions.",
    "threading.RLock": "A reentrant lock that allows the same thread to acquire it multiple times.",
    "threading.Event": "A flag that threads can wait on. Used for signaling between threads.",
    "threading.Semaphore": "Controls access to a resource with a limited number of permits.",
    "Queue(": "Thread-safe queue for passing data between threads. Use put() and get().",
    "Lock(": "Creates a new lock object. Use with 'with' statement for automatic acquire/release.",

    # Common patterns
    "__init__": "Constructor method. Called when creating a new instance of a class. Initializes the object's attributes.",
    "__str__": "Returns a human-readable string representation. Called by print() and str().",
    "__repr__": "Returns a developer-friendly string representation. Called by repr() and in the interactive console.",
    "__len__": "Returns the length of the object. Called by len().",
    "__eq__": "Defines equality comparison (==). Called when comparing two objects.",
    "__lt__": "Defines less-than comparison (<). Used for sorting.",
    "__hash__": "Returns a hash value. Required for objects used as dictionary keys.",
    "__enter__": "Called when entering a 'with' block. Returns the context manager's value.",
    "__exit__": "Called when leaving a 'with' block. Handles cleanup and exception suppression.",
    "__call__": "Makes an instance callable like a function. Useful for strategy patterns.",
    "__del__": "Destructor. Called when the object is about to be garbage collected. Avoid using.",
    "__copy__": "Defines behavior for shallow copy (copy.copy()).",
    "__deepcopy__": "Defines behavior for deep copy (copy.deepcopy()).",
    "__bool__": "Defines truth value for use in if statements and boolean operations.",
    "__contains__": "Defines behavior for 'in' operator. Called by 'x in obj'.",
    "__iter__": "Returns an iterator. Called by for loops.",
    "__next__": "Returns the next item from an iterator. Raises StopIteration when done.",
    "__await__": "Makes an object awaitable in async/await contexts.",

    # Common library modules
    "os.path": "OS-agnostic path manipulation. Use join(), exists(), dirname(), basename(), splitext().",
    "os.makedirs": "Recursively creates directories. exist_ok=True prevents errors if directory exists.",
    "os.environ": "Dictionary of environment variables. Use get() for safe access with defaults.",
    "sys.path": "List of directories Python searches for modules. Can be modified to add import paths.",
    "json.load": "Deserializes JSON from a file into a Python object.",
    "json.dump": "Serializes a Python object to JSON and writes to a file.",
    "json.loads": "Deserializes a JSON string into a Python object.",
    "json.dumps": "Serializes a Python object to a JSON string.",
    "re.match": "Checks for a match only at the beginning of the string.",
    "re.search": "Searches anywhere in the string for a match.",
    "re.findall": "Returns all non-overlapping matches as a list of strings.",
    "re.sub": "Replaces all matches with the replacement string.",
    "logging.getLogger": "Creates or retrieves a logger by name. Hierarchical naming enables per-module log levels.",
    "logging.basicConfig": "Configures the root logger with format, level, and output destination.",
    "collections.defaultdict": "A dict that automatically creates a default value for missing keys.",
    "collections.Counter": "A dict subclass for counting hashable objects. Most_common(n) returns top n items.",
    "collections.namedtuple": "Creates a tuple subclass with named fields. More readable than plain tuples.",
    "collections.OrderedDict": "A dict that remembers insertion order. (Regular dicts do this in Python 3.7+.)",
    "itertools.chain": "Combines multiple iterables into a single sequence.",
    "itertools.groupby": "Groups consecutive elements with the same key.",
    "itertools.product": "Cartesian product of input iterables.",
    "itertools.permutations": "All possible orderings of r elements from an iterable.",
    "itertools.combinations": "All possible r-length combinations from an iterable.",
    "pathlib.Path": "Object-oriented path handling. Preferred over os.path for modern Python code.",
    "pathlib.Path.mkdir": "Creates a directory. parents=True creates intermediate directories.",
    "pathlib.Path.exists": "Returns True if the path exists.",
    "pathlib.Path.read_text": "Reads the file contents as a string.",
    "pathlib.Path.write_text": "Writes a string to the file.",
    "functools.lru_cache": "Decorator that caches function results based on arguments.",
    "functools.wraps": "Decorator that copies the original function's metadata to the wrapper.",
    "typing.TypeVar": "Defines a generic type variable for use in type hints.",
    "typing.Protocol": "Defines a structural subtyping interface (duck typing for type checkers).",
    "abc.ABC": "Abstract Base Class. Use with @abstractmethod to define interfaces.",
    "abc.abstractmethod": "Decorator marking a method as abstract. Subclasses must override it.",
    "dataclasses.dataclass": "Decorator that auto-generates __init__, __repr__, __eq__, and more from class attributes.",
    "dataclasses.field": "Defines a field in a dataclass with custom default, repr, or comparison behavior.",
    "enum.Enum": "Base class for creating enumeration types. Each member has a name and value.",
    "enum.IntEnum": "Enumeration where members are also integers.",
    "contextlib.contextmanager": "Decorator that turns a generator function into a context manager.",
    "contextlib.suppress": "Context manager that suppresses specified exceptions.",
    "textwrap.dedent": "Removes common leading whitespace from all lines. Useful for multi-line strings.",
    "textwrap.fill": "Wraps text to a given width, inserting newlines.",
    "shutil.copy": "Copies a file or directory tree.",
    "shutil.rmtree": "Recursively deletes a directory tree.",
    "tempfile.NamedTemporaryFile": "Creates a temporary file that is deleted when closed.",
    "subprocess.run": "Runs a command and waits for it to complete. Returns a CompletedProcess.",
    "subprocess.Popen": "Starts a process. More control than run() — can read stdout/stderr asynchronously.",
    "threading.Thread": "Creates a background thread. Use daemon=True for threads that should not block app exit.",
    "multiprocessing.Process": "Creates a separate process with its own Python interpreter. Bypasses the GIL.",
    "multiprocessing.Queue": "A process-safe queue for inter-process communication.",
    "multiprocessing.Pool": "A pool of worker processes for parallel execution.",
    "asyncio.run": "Runs an async function in the event loop. Entry point for async programs.",
    "asyncio.gather": "Runs multiple coroutines concurrently and returns their results.",
    "asyncio.create_task": "Schedules a coroutine to run concurrently without blocking.",
    "asyncio.sleep": "Async version of time.sleep(). Yields control to the event loop.",
    "async def": "Defines an asynchronous function (coroutine). Can use 'await' inside.",
    "await": "Pauses the coroutine until the awaited operation completes. Yields control to the event loop.",
}

# ── wxPython Knowledge Base ───────────────────────────────────────────────

WX_KB = {
    # Core classes
    "wx.App(": "Creates the wxPython application object. Must be created before any GUI elements. False means don't redirect stdout.",
    "wx.Frame(": "Creates a top-level window with title bar, menus, and status bar. The main container for desktop apps.",
    "wx.Frame": "The top-level window class. MainFrame inherits from this. Provides the window frame, title bar, and system menu.",
    "wx.Panel(": "Creates a container window inside a Frame. Groups child controls and manages tab traversal order.",
    "wx.Panel": "A container that sits inside a Frame. Groups controls and provides a background for drawing.",
    "wx.Dialog(": "Creates a modal or modeless dialog window. Used for settings, file selection, and user input.",
    "wx.Dialog": "A window that typically requires user interaction before continuing. Can be modal (blocks parent) or modeless.",
    "wx.MDIChildFrame(": "Creates a child frame inside an MDI (Multiple Document Interface) parent.",
    "wx.SplitterWindow(": "Creates a window that can be split into two resizable panes.",

    # Layout
    "wx.BoxSizer(": "Creates a layout manager that arranges children horizontally (HORIZONTAL) or vertically (VERTICAL).",
    "wx.BoxSizer": "The primary layout manager. Arranges child windows in a line. Use proportion and flags to control sizing.",
    "wx.GridSizer(": "Creates a grid layout manager with equal-sized cells. Useful for button grids.",
    "wx.GridBagSizer(": "Creates a grid layout where items can span multiple rows and columns.",
    "wx.FlexGridSizer(": "Like GridSizer but rows/columns can have different sizes.",
    "wx.StaticBoxSizer(": "Creates a labeled border around a group of controls.",
    "wx.Notebook(": "Creates a tabbed container. Each page is a separate panel. Used in Settings dialog.",
    "wx.Notebook": "A tabbed container widget. Clicking a tab shows a different panel. Used for organizing settings pages.",
    "sizer.Add(": "Adds a child window to the sizer. Parameters: window, proportion, flags, border. Flags control alignment and padding.",
    "sizer.AddStretchSpacer(": "Adds a flexible spacer that pushes subsequent items to the opposite end.",
    "sizer.AddSpacer(": "Adds a fixed-size spacer between items.",
    "panel.SetSizer(": "Assigns a sizer to the panel. The sizer takes control of laying out the panel's children.",
    "panel.SetSizerAndFit(": "Sets the sizer and resizes the panel to fit its minimum size.",

    # Widgets
    "wx.StaticText(": "Creates a non-editable text label. Used for displaying information to the user.",
    "wx.StaticText": "A non-editable text label widget. The primary way to display text information in wxPython.",
    "wx.Button(": "Creates a clickable push button. The most common way to trigger user actions.",
    "wx.Button": "A clickable push button. Clicking it generates an EVT_BUTTON event.",
    "wx.ToggleButton(": "Creates a toggle button that stays pressed when clicked.",
    "wx.BitmapButton(": "Creates a button that displays an image instead of text.",
    "wx.TextCtrl(": "Creates a text input control. Can be single-line (for paths) or multi-line (for text display).",
    "wx.TextCtrl": "A text input control. Can be single-line, multi-line, read-only, or password-masked.",
    "wx.StaticText(": "Creates a non-editable text label.",
    "wx.ListBox(": "Creates a list of items that can be selected. Single or multiple selection modes.",
    "wx.ListBox": "A list control for displaying and selecting items. Used for voice selection and recent projects.",
    "wx.Choice(": "Creates a drop-down selection control. The user picks one item from a list.",
    "wx.Choice": "A drop-down selection control. Shows one item at a time; click to see all options.",
    "wx.ComboBox(": "Like Choice but also allows typing a custom value.",
    "wx.CheckBox(": "Creates a check box toggle. Returns True when checked, False when unchecked.",
    "wx.CheckBox": "A toggle control with a label. Returns True when checked, False when unchecked.",
    "wx.Slider(": "Creates a slider control for selecting a value within a range. Used for rate, pitch, volume.",
    "wx.Slider": "A slider control for numeric values. The range is set in the constructor; GetValue() returns the current position.",
    "wx.Gauge(": "Creates a progress bar control. Shows download and synthesis progress.",
    "wx.Gauge": "A progress bar. SetRange() sets the max value; SetValue() updates the current progress.",
    "wx.SpinCtrl(": "Creates a numeric spinner with up/down arrows.",
    "wx.RadioBox(": "Creates a group of radio buttons where only one can be selected.",
    "wx.TreeCtrl(": "Creates a tree view control for hierarchical data.",
    "wx.Notebook(": "Creates a tabbed container for organizing panels.",
    "wx.html.HtmlWindow(": "Creates an HTML rendering window.",
    "wx.StaticBitmap(": "Displays an image (bitmap) on the window.",
    "wx.CalendarCtrl(": "Displays a calendar for date selection.",
    "wx.DatePickerCtrl(": "Creates a date selection control.",
    "wx.FilePickerCtrl(": "Creates a file selection control with a browse button.",
    "wx.DirPickerCtrl(": "Creates a directory selection control with a browse button.",
    "wx.ColourPickerCtrl(": "Creates a colour selection control.",
    "wx.FontPickerCtrl(": "Creates a font selection control.",
    "wx.SplitterWindow(": "Creates a window that can be split into two resizable panes.",

    # Events
    "wx.EVT_BUTTON": "Event triggered when a button is clicked. Bind with: widget.Bind(wx.EVT_BUTTON, handler).",
    "wx.EVT_MENU": "Event triggered when a menu item is clicked. Bind with: frame.Bind(wx.EVT_MENU, handler, id=item_id).",
    "wx.EVT_CLOSE": "Event triggered when the window close button (X) is clicked.",
    "wx.EVT_COMBOBOX": "Event triggered when a ComboBox selection changes.",
    "wx.EVT_TEXT": "Event triggered when text in a TextCtrl changes.",
    "wx.EVT_TEXT_ENTER": "Event triggered when Enter is pressed in a TextCtrl.",
    "wx.EVT_CHECKBOX": "Event triggered when a CheckBox is toggled.",
    "wx.EVT_LISTBOX": "Event triggered when a ListBox selection changes.",
    "wx.EVT_LISTBOX_DCLICK": "Event triggered when a ListBox item is double-clicked.",
    "wx.EVT_SLIDER": "Event triggered when a Slider value changes.",
    "wx.EVT_NOTEBOOK_PAGE_CHANGED": "Event triggered when a Notebook tab is switched.",
    "wx.EVT_TREE_SEL_CHANGED": "Event triggered when a TreeCtrl selection changes.",
    "wx.EVT_SPINCTRL": "Event triggered when a SpinCtrl value changes.",
    "wx.EVT_RADIOBOX": "Event triggered when a RadioBox selection changes.",
    "wx.EVT_PAINT": "Event triggered when the window needs repainting.",
    "wx.EVT_SIZE": "Event triggered when the window is resized.",
    "wx.EVT_MOUSEWHEEL": "Event triggered when the mouse wheel is scrolled.",
    "wx.EVT_KEY_DOWN": "Event triggered when a key is pressed.",
    "wx.EVT_KEY_UP": "Event triggered when a key is released.",
    "wx.EVT_TIMER": "Event triggered when a timer fires.",
    "widget.Bind(": "Registers an event handler for a specific event on a widget. The handler receives an event object.",
    "self.Bind(": "Registers an event handler on the frame/window itself. Used for frame-level events.",
    "self.Unbind(": "Removes a previously registered event handler.",
    "event.Skip()": "Allows the event to propagate to the default handler. Essential for some system events.",
    "event.GetId()": "Returns the ID of the widget that triggered the event.",
    "event.GetString()": "Returns the string value of the event (e.g., selected item in a ListBox).",
    "event.GetInt()": "Returns the integer value of the event (e.g., slider position).",
    "event.IsChecked()": "Returns the boolean state of a CheckBox event.",

    # Sizer flags
    "wx.ALL": "Border flag: adds a border on all four sides of the widget.",
    "wx.LEFT": "Border flag: adds a border on the left side only.",
    "wx.RIGHT": "Border flag: adds a border on the right side only.",
    "wx.TOP": "Border flag: adds a border on the top only.",
    "wx.BOTTOM": "Border flag: adds a border on the bottom only.",
    "wx.EXPAND": "Flag: the widget stretches to fill the sizer's available space.",
    "wx.ALIGN_CENTER": "Flag: centers the widget within the sizer cell.",
    "wx.ALIGN_LEFT": "Flag: aligns the widget to the left of the sizer cell.",
    "wx.ALIGN_RIGHT": "Flag: aligns the widget to the right of the sizer cell.",
    "wx.ALIGN_TOP": "Flag: aligns the widget to the top of the sizer cell.",
    "wx.ALIGN_BOTTOM": "Flag: aligns the widget to the bottom of the sizer cell.",
    "wx.ALIGN_CENTER_HORIZONTAL": "Flag: centers the widget horizontally.",
    "wx.ALIGN_CENTER_VERTICAL": "Flag: centers the widget vertically.",
    "wx.SHAPED": "Flag: the widget maintains its aspect ratio when resized.",
    "wx.FIXED_MINSIZE": "Flag: prevents the widget from being smaller than its minimum size.",

    # Dialogs
    "wx.MessageDialog(": "Displays a message with OK/Cancel buttons. Used for confirmations and error messages.",
    "wx.MessageDialog": "A dialog that shows a message and waits for user acknowledgment. Returns wx.ID_OK or wx.ID_CANCEL.",
    "wx.FileDialog(": "Opens a native file open/save dialog. Lets the user browse and select files.",
    "wx.FileDialog": "A native file selection dialog. Supports filters for file types and multiple selection.",
    "wx.DirDialog(": "Opens a native directory selection dialog. Lets the user browse and select folders.",
    "wx.DirDialog": "A native directory selection dialog. Used for choosing output folders.",
    "wx.ColourDialog(": "Opens a colour selection dialog.",
    "wx.FontDialog(": "Opens a font selection dialog.",
    "wx.PrintDialog(": "Opens a print configuration dialog.",
    "wx.Dirdrop(": "Enables drag-and-drop of files onto a window.",
    "wx.FileDropTarget(": "Implements the file drop target interface for drag-and-drop.",
    "ShowModal()": "Displays a dialog modally (blocks input to the parent window until closed). Returns wx.ID_OK or wx.ID_CANCEL.",
    "ShowModal": "Shows a dialog modally. The user must close the dialog before interacting with the parent window.",
    "Destroy()": "Destroys the dialog window and frees its resources. Must be called after ShowModal() returns.",
    "EndModal(": "Ends a modal dialog with the given return code (e.g., wx.ID_OK).",
    "SetReturnCode(": "Sets the return code that ShowModal() will return.",

    # Frame methods
    "frame.Show(": "Makes the frame visible. Must be called after creating the frame.",
    "frame.Show()": "Makes the frame visible on screen. The frame is hidden by default.",
    "frame.Hide(": "Hides the frame without destroying it.",
    "frame.Close(": "Closes the frame. Sends EVT_CLOSE which can be intercepted.",
    "frame.Destroy(": "Destroys the frame and frees all its resources.",
    "frame.SetTitle(": "Changes the frame's title bar text.",
    "frame.GetTitle()": "Returns the current title bar text.",
    "frame.SetSize(": "Sets the frame's size in pixels (width, height).",
    "frame.GetSize()": "Returns the frame's current size as (width, height).",
    "frame.SetPosition(": "Sets the frame's screen position.",
    "frame.Centre()": "Centers the frame on the screen.",
    "frame.Center()": "American spelling of Centre(). Same behavior.",
    "frame.Maximize(": "Maximizes the frame to fill the screen.",
    "frame.Restore()": "Restores a maximized frame to its original size.",
    "frame.Iconize(": "Minimizes the frame to the taskbar.",
    "frame.SetStatusBar(": "Attaches a status bar to the frame.",
    "frame.CreateStatusBar(": "Creates and attaches a status bar to the frame.",
    "frame.SetStatusText(": "Sets the text displayed in the status bar.",
    "frame.GetStatusBar()": "Returns the frame's status bar object.",
    "frame.SetMenuBar(": "Attaches a menu bar to the frame.",
    "frame.GetMenuBar()": "Returns the frame's menu bar.",
    "frame.CreateMenuBar()": "Creates and attaches a new menu bar.",
    "frame.SetMinSize(": "Sets the minimum size the frame can be resized to.",
    "frame.SetMaxSize(": "Sets the maximum size the frame can be resized to.",
    "frame.GetClientSize()": "Returns the usable area size (excluding title bar, borders).",
    "frame.GetPosition()": "Returns the frame's screen position as (x, y).",
    "frame.Layout()": "Forces the frame to re-layout its children using the assigned sizer.",

    # Menu
    "wx.MenuBar(": "Creates a horizontal menu bar at the top of a Frame.",
    "wx.MenuBar": "The horizontal menu bar at the top of a Frame. Contains File, Edit, Tools, Help menus.",
    "wx.Menu(": "Creates a dropdown menu. Contains menu items, separators, and sub-menus.",
    "wx.Menu": "A dropdown menu that appears when clicking a menu bar item or right-clicking.",
    "menu.Append(": "Adds a menu item with an ID, label, and optional help text. Use \\t for accelerators.",
    "menu.AppendSeparator()": "Adds a visual separator line between menu items.",
    "menu.AppendSubMenu(": "Adds a sub-menu (dropdown within a dropdown).",
    "menubar.Append(": "Adds a menu to the menu bar with a label.",
    "wx.NewIdRef()": "Creates a unique integer ID for use with menu items and event binding.",
    "wx.NewIdRef": "Generates a unique ID to prevent conflicts between different menu items and controls.",

    # Status bar
    "wx.StatusBar(": "Creates a status bar at the bottom of a Frame.",
    "wx.StatusBar": "A bar at the bottom of the frame showing status messages, progress, and context info.",
    "SetStatusText(": "Sets the text in the specified status bar pane.",
    "SetStatusWidths(": "Sets the widths of the status bar panes. -1 means auto-size.",

    # Misc GUI
    "wx.CallAfter(": "Schedules a function to run on the main thread after the current event handler. Essential for thread-safe GUI updates.",
    "wx.CallAfter": "The safest way to update the GUI from a background thread. Schedules the call for the main event loop.",
    "wx.Timer(": "Creates a timer that fires events at regular intervals. Useful for polling and animations.",
    "wx.BeginBusyCursor(": "Changes the cursor to a busy indicator (hourglass).",
    "wx.EndBusyCursor(": "Restores the cursor to normal after BeginBusyCursor().",
    "wx.Yield(": "Processes pending GUI events. Use sparingly — can cause reentrancy issues.",
    "wx.SafeYield(": "Like Yield() but prevents the window from receiving additional events.",
    "wx.HelpController(": "Manages the display of help files (HTML Help, WinHelp).",
    "wx.Locale(": "Sets the application's locale for internationalization and translations.",
    "wx.ArtProvider(": "Provides stock bitmaps and icons (wx.ART_INFORMATION, wx.ART_ERROR, etc.).",
    "wx.Font(": "Represents a font face, size, and style. Used for consistent typography.",
    "wx.Colour(": "Represents an RGB colour. Used for theming and custom drawing.",
    "wx.Bitmap(": "An image that can be drawn on a DC or displayed in a control.",
    "wx.Image(": "An image that can be manipulated (resized, rotated) before converting to Bitmap.",
    "wx.DC(": "Device Context for drawing graphics. Used in custom painting and printing.",
    "wx.PaintDC(": "A DC for handling EVT_PAINT events. Auto-finishes painting when the context is destroyed.",
    "wx.ClientDC(": "A DC for drawing on a window outside of paint events.",
    "wx.MemoryDC(": "A DC for drawing onto a Bitmap in memory.",
    "wx.BufferedDC(": "A DC that buffers drawing operations to prevent flicker.",
    "wx.BufferedPaintDC(": "A buffered DC for flicker-free paint events.",
    "wx.StockGDI(": "Stock GDI objects (colours, pens, brushes) for system defaults.",
    "wx.STATIC_BORDER": "Window style: creates a static (sunken) border around a control.",
    "wx.SIMPLE_BORDER": "Window style: creates a simple border.",
    "wx.RESIZE_BORDER": "Window style: makes the window resizable.",
    "wx.CAPTION": "Window style: adds a title bar to the window.",
    "wx.CLOSE_BOX": "Window style: adds a close button to the title bar.",
    "wx.MINIMIZE_BOX": "Window style: adds a minimize button.",
    "wx.MAXIMIZE_BOX": "Window style: adds a maximize button.",
    "wx.SYSTEM_MENU": "Window style: adds a system menu (icon in the title bar).",
    "wx.STAY_ON_TOP": "Window style: keeps the window above all other windows.",
    "wx.FRAME_NO_TASKBAR": "Window style: prevents the frame from appearing in the taskbar.",
    "wx.FRAME_SHAPED": "Window style: allows the frame to have a non-rectangular shape.",
    "wx.TE_READONLY": "TextCtrl style: makes the text control read-only.",
    "wx.TE_MULTILINE": "TextCtrl style: enables multi-line text input.",
    "wx.TE_PROCESS_ENTER": "TextCtrl style: generates EVT_TEXT_ENTER when Enter is pressed.",
    "wx.TE_PASSWORD": "TextCtrl style: masks the input with asterisks.",
    "wx.TE_RICH2": "TextCtrl style: enables rich text formatting.",
    "wx.LB_SINGLE": "ListBox style: allows only one selection.",
    "wx.LB_MULTIPLE": "ListBox style: allows multiple selections.",
    "wx.LB_EXTENDED": "ListBox style: allows shift+click and ctrl+click selections.",
    "wx.CB_READONLY": "ComboBox style: prevents typing custom values.",
    "wx.CB_DROPDOWN": "ComboBox style: shows a drop-down list (default).",
    "wx.CB_SIMPLE": "ComboBox style: shows the list permanently below the text field.",
    "wx.GA_HORIZONTAL": "Gauge style: horizontal progress bar.",
    "wx.GA_VERTICAL": "Gauge style: vertical progress bar.",
    "wx.GA_SMOOTH": "Gauge style: smooth (non-segmented) progress.",
    "wx.SL_HORIZONTAL": "Slider style: horizontal slider.",
    "wx.SL_VERTICAL": "Slider style: vertical slider.",
    "wx.SL_LABELS": "Slider style: shows min/max/current labels.",
    "wx.SL_AUTOTICKS": "Slider style: shows tick marks.",
    "wx.ID_OK": "Standard ID for OK button in dialogs.",
    "wx.ID_CANCEL": "Standard ID for Cancel button in dialogs.",
    "wx.ID_YES": "Standard ID for Yes button.",
    "wx.ID_NO": "Standard ID for No button.",
    "wx.ID_EXIT": "Standard ID for Exit menu item.",
    "wx.ID_ABOUT": "Standard ID for About menu item.",
    "wx.ID_SAVE": "Standard ID for Save button.",
    "wx.ID_OPEN": "Standard ID for Open button.",
    "wx.ID_CLOSE": "Standard ID for Close button.",
    "wx.ID_RETRY": "Standard ID for Retry button.",
    "wx.ID_APPLY": "Standard ID for Apply button.",
    "wx.ID_HELP": "Standard ID for Help button.",

    # MainLoop
    "app.MainLoop()": "Starts the wxPython event loop. This blocks until the last window is closed. All GUI events are processed here.",
    "MainLoop()": "The heart of a wxPython application. Processes user clicks, key presses, paint events, and timers until the app exits.",
}

# ── Combine all knowledge bases ───────────────────────────────────────────

ALL_KB = {}
ALL_KB.update(PYTHON_KB)
ALL_KB.update(WX_KB)


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
            return f"Docstring: <em>{html.escape(inner)}</em>. Documents the purpose of the following code."
        return "Opens a docstring block for documentation."

    # Check knowledge base (longest match first for specificity)
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
        return "Returns <code>None</code> implicitly (exits the function)."

    # Control flow
    m = re.match(r'^\s*if\s+(.+):', stripped)
    if m:
        return f"Checks condition: <code>{html.escape(m.group(1)[:120])}</code>. The indented block runs only when this is <code>True</code>."

    m = re.match(r'^\s*elif\s+(.+):', stripped)
    if m:
        return f"Alternative condition: <code>{html.escape(m.group(1)[:120])}</code>. Checked only if all previous conditions were <code>False</code>."

    if re.match(r'^\s*else:', stripped):
        return "Fallback: executes when none of the above conditions matched."

    # Loops
    m = re.match(r'^\s*for\s+(\w+)\s+in\s+(.+):', stripped)
    if m:
        return f"Iterates over <code>{html.escape(m.group(2)[:100])}</code>, assigning each element to <code>{html.escape(m.group(1))}</code>."

    m = re.match(r'^\s*while\s+(.+):', stripped)
    if m:
        return f"While loop: repeats as long as <code>{html.escape(m.group(1)[:100])}</code>."

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
        return f"Context manager: <code>{html.escape(m.group(1)[:120])}</code>. Automatically manages resource cleanup."

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
        return "Skips to the next loop iteration."
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
  <p>This is an ASCII art diagram illustrating a concept in the chapter. Refer to the surrounding text for explanation.</p>
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
    files = sorted(glob.glob(os.path.join(CHAPTERS_DIR, 'ch0[1-9]_*.html')) +
                   glob.glob(os.path.join(CHAPTERS_DIR, 'ch1[0-9]_*.html')) +
                   glob.glob(os.path.join(CHAPTERS_DIR, 'ch2[0-9]_*.html')) +
                   glob.glob(os.path.join(CHAPTERS_DIR, 'ch3[0-6]_*.html')))
    modified = 0
    for fp in files:
        if process_file(fp):
            print(f'  [OK] {os.path.basename(fp)}')
            modified += 1
        else:
            print(f'  [--] {os.path.basename(fp)} (skipped)')
    print(f'\nDone. Enhanced {modified}/{len(files)} Part I-IV files.')


if __name__ == '__main__':
    main()
