#!/usr/bin/env python3
"""
Enhanced post-processor: generates meaningful line-by-line descriptions
for every <pre><code> block in the book chapters.
"""

import re, html, os, glob

CHAPTERS_DIR = os.path.join(os.path.dirname(__file__), "chapters")

# ── Pattern-matching engine for generating descriptions ──────────────────

def describe_line(line: str, context_lines: list[str] = None, line_idx: int = 0) -> str:
    """Generate a human-readable description for one line of Python code."""
    stripped = line.strip()

    if not stripped:
        return ""

    # Docstrings (triple-quoted)
    if stripped.startswith('"\"\"') or stripped.startswith("'''"):
        inner = stripped.strip('"\"\"').strip("'''").strip()
        if inner:
            return f"A docstring documenting the code that follows: <em>{html.escape(inner)}</em>. Docstrings are special strings that describe what a function, class, or module does."
        return "Opens a docstring block. Docstrings document the purpose of functions, classes, and modules."
    if stripped.endswith('"\"\"') or stripped.endswith("'''"):
        return "Ends the docstring block."

    # Comments
    if stripped.startswith('#'):
        inner = stripped.lstrip('#').strip()
        if inner:
            return f"A comment explaining: <em>{html.escape(inner)}</em>. Comments are ignored by the Python interpreter and are used for documentation."
        return "An empty comment line used for visual separation."

    # Import statements
    m = re.match(r'^from\s+(\S+)\s+import\s+(.+)', stripped)
    if m:
        module, names = m.group(1), m.group(2)
        return f"Imports specific names (<code>{html.escape(names)}</code>) from the <code>{html.escape(module)}</code> module, making them available without the module prefix."

    m = re.match(r'^import\s+(.+)', stripped)
    if m:
        modules = m.group(1)
        return f"Imports the <code>{html.escape(modules)}</code> module, making its functions and classes available in this scope."

    # Class definitions
    m = re.match(r'^class\s+(\w+)(?:\(([^)]*)\))?:', stripped)
    if m:
        name, bases = m.group(1), m.group(2) or ""
        if bases:
            return f"Defines a new class <code>{html.escape(name)}</code> that inherits from <code>{html.escape(bases)}</code>. The class will have access to all methods and attributes of its parent class(es)."
        return f"Defines a new class <code>{html.escape(name)}</code>. Classes are blueprints for creating objects that bundle data and behavior together."

    # Function definitions
    m = re.match(r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:', stripped)
    if m:
        name, params, ret = m.group(2), m.group(3), m.group(4)
        param_list = [p.strip().split(':')[0].split('=')[0].strip() for p in params.split(',') if p.strip()]
        param_str = ', '.join(param_list)
        desc = f"Defines a function <code>{html.escape(name)}</code>"
        if param_list:
            desc += f" that accepts parameters: <code>{html.escape(param_str)}</code>"
        if ret:
            desc += f" and returns a value of type <code>{html.escape(ret.strip())}</code>"
        desc += ". Functions encapsulate reusable blocks of code."
        return desc

    # Lambda
    m = re.match(r'^(\w+)\s*=\s*lambda\s+(.+?):', stripped)
    if m:
        name, args = m.group(1), m.group(2)
        return f"Creates an anonymous function (lambda) and assigns it to <code>{html.escape(name)}</code>. Lambda functions are small, single-expression functions useful for short operations."

    # Return statements
    m = re.match(r'^\s*return\s+(.+)', stripped)
    if m:
        val = m.group(1)
        return f"Returns the value <code>{html.escape(val)}</code> to the caller. This exits the current function and passes the result back."

    if stripped == 'return':
        return "Returns <code>None</code> implicitly (no value specified). This exits the current function."

    # if/elif/else
    m = re.match(r'^\s*if\s+(.+):', stripped)
    if m:
        cond = m.group(1)
        return f"Conditional check: <code>{html.escape(cond)}</code>. The indented block below executes only when this condition evaluates to <code>True</code>."

    m = re.match(r'^\s*elif\s+(.+):', stripped)
    if m:
        cond = m.group(1)
        return f"Alternative condition: <code>{html.escape(cond)}</code>. This branch is checked only if all previous <code>if</code>/<code>elif</code> conditions were <code>False</code>."

    if re.match(r'^\s*else:', stripped):
        return "Fallback branch: executes when none of the above <code>if</code>/<code>elif</code> conditions matched."

    # for loops
    m = re.match(r'^\s*for\s+(\w+)\s+in\s+(.+):', stripped)
    if m:
        var, iterable = m.group(1), m.group(2)
        return f"Loop that iterates over <code>{html.escape(iterable)}</code>, assigning each element to <code>{html.escape(var)}</code> in turn. The indented body runs once per element."

    # while loops
    m = re.match(r'^\s*while\s+(.+):', stripped)
    if m:
        cond = m.group(1)
        return f"While loop that repeats the indented body as long as <code>{html.escape(cond)}</code> is <code>True</code>."

    # Try/except/finally
    if re.match(r'^\s*try:', stripped):
        return "Begins a <code>try</code> block. Code inside this block is monitored for exceptions. If an error occurs, execution jumps to the matching <code>except</code> clause."
    m = re.match(r'^\s*except\s+(\w+)', stripped)
    if m:
        exc = m.group(1)
        return f"Catches exceptions of type <code>{html.escape(exc)}</code> (or its subclasses). The handler runs when this type of error occurs in the <code>try</code> block."
    if re.match(r'^\s*except\s*:', stripped):
        return "Catches any exception type. This is a broad catch-all; prefer catching specific exceptions when possible."
    if re.match(r'^\s*finally:', stripped):
        return "The <code>finally</code> block always executes, whether or not an exception occurred. Used for cleanup code (closing files, releasing resources)."
    if re.match(r'^\s*raise\s+', stripped):
        return f"Raises an exception: <code>{html.escape(stripped)}</code>. This stops normal execution and propagates the error up the call stack."

    # with statement
    m = re.match(r'^\s*with\s+(.+):', stripped)
    if m:
        expr = m.group(1)
        return f"Context manager statement: <code>{html.escape(expr)}</code>. Automatically calls <code>__enter__</code> at the start and <code>__exit__</code> at the end, ensuring proper resource cleanup."

    # yield
    if re.match(r'^\s*yield\s', stripped) or stripped == 'yield':
        return "Yields a value to the caller, pausing the function. This makes the function a generator — it can be resumed later to continue execution."

    # Assignment patterns
    m = re.match(r'^(\w+(?:\.\w+)*)\s*=\s*(.+)', stripped)
    if m:
        var, val = m.group(1), m.group(2)
        if val.startswith('{'):
            return f"Assigns a dictionary literal to <code>{html.escape(var)}</code>. Dictionaries map keys to values and are the primary data structure for structured data."
        if val.startswith('['):
            return f"Assigns a list literal to <code>{html.escape(var)}</code>. Lists are ordered, mutable sequences."
        if val.startswith('(') or val.startswith('tuple'):
            return f"Assigns a tuple to <code>{html.escape(var)}</code>. Tuples are ordered, immutable sequences."
        if 'lambda' in val:
            return f"Assigns a lambda (anonymous) function to <code>{html.escape(var)}</code>."
        if val.startswith('"') or val.startswith("'"):
            return f"Assigns a string value to <code>{html.escape(var)}</code>. Strings in Python are immutable sequences of Unicode characters."
        try:
            float(val.replace('**', '').replace('-', '').replace('+', '').replace('(', '').replace(')', '').strip())
            return f"Assigns a numeric value to <code>{html.escape(var)}</code>."
        except:
            pass
        if 'None' in val:
            return f"Assigns <code>None</code> to <code>{html.escape(var)}</code>, indicating no value or an uninitialized state."
        if 'True' in val or 'False' in val:
            return f"Assigns a boolean value to <code>{html.escape(var)}</code>."
        if '(' in val and ')' in val and '.' not in val.split('(')[0]:
            return f"Calls a function and assigns the return value to <code>{html.escape(var)}</code>."
        return f"Assigns the expression <code>{html.escape(val)}</code> to the variable <code>{html.escape(var)}</code>."

    # Augmented assignment
    m = re.match(r'^\s*(\w+(?:\.\w+)*)\s*(\+=|-=|\*=|/=|//=|%=|&=|\|=|\^=|>>=|<<=|\*\*=)\s*(.+)', stripped)
    if m:
        var, op, val = m.group(1), m.group(2), m.group(3)
        op_name = {'+=': 'addition', '-=': 'subtraction', '*=': 'multiplication',
                   '/=': 'division', '//=': 'floor division', '%=': 'modulo'}.get(op, op)
        return f"Augmented assignment: <code>{html.escape(var)}</code> {op_name} <code>{html.escape(val)}</code>. Equivalent to <code>{html.escape(var)} = {html.escape(var)} {op} {html.escape(val)}</code>."

    # Decorator
    if stripped.startswith('@'):
        return f"Decorator: <code>{html.escape(stripped)}</code>. Decorators modify or wrap a function/class, adding behavior without changing the original code."

    # Print statement
    m = re.match(r'^\s*print\s*\((.+)\)', stripped)
    if m:
        return f"Prints output to the console: <code>{html.escape(m.group(1))}</code>. Used for debugging and user-facing messages."

    # self.xxx attribute access
    m = re.match(r'^\s*self\.(\w+)\s*=\s*(.+)', stripped)
    if m:
        attr, val = m.group(1), m.group(2)
        return f"Sets instance attribute <code>{html.escape(attr)}</code> on <code>self</code>. This stores data specific to this particular object instance."

    m = re.match(r'^\s*return\s+self\.(\w+)', stripped)
    if m:
        return f"Returns the instance attribute <code>{html.escape(m.group(1))}</code>. Common in getter methods and for method chaining."

    # Generic return
    if stripped.startswith('return'):
        return f"<code>{html.escape(stripped)}</code> — exits the function and passes a value back to the caller."

    # assert
    m = re.match(r'^\s*assert\s+(.+)', stripped)
    if m:
        return f"Assertion: checks that <code>{html.escape(m.group(1))}</code>. If false, raises <code>AssertionError</code>. Used for debugging and testing."

    # pass
    if stripped == 'pass':
        return "No-op statement. <code>pass</code> does nothing — used as a placeholder where Python requires a statement syntactically."

    # break/continue
    if stripped == 'break':
        return "Exits the innermost <code>for</code> or <code>while</code> loop immediately."
    if stripped == 'continue':
        return "Skips the rest of the current loop iteration and jumps to the next one."

    # del
    m = re.match(r'^\s*del\s+(.+)', stripped)
    if m:
        return f"Deletes the variable or element <code>{html.escape(m.group(1))}</code>, freeing the reference."

    # f-strings
    if 'f"' in stripped or "f'" in stripped:
        return f"F-string: <code>{html.escape(stripped)}</code>. F-strings (Python 3.6+) embed expressions inside string literals using curly braces."

    # Shell/command-line patterns
    m = re.match(r'^(brew|pip|pip3|npm|yarn|sudo|apt|apt-get|conda|python|python3|node|cargo|git|cmake|make)\s+(.+)', stripped)
    if m:
        cmd, args = m.group(1), m.group(2)
        cmd_desc = {
            'brew': 'Homebrew package manager (macOS)',
            'pip': 'Python package installer',
            'pip3': 'Python 3 package installer',
            'sudo': 'Runs a command with administrator (root) privileges',
            'apt': 'Advanced Package Tool (Debian/Ubuntu package manager)',
            'apt-get': 'APT package manager utility',
            'conda': 'Conda package manager (data science)',
            'python': 'Python interpreter command',
            'python3': 'Python 3 interpreter command',
            'node': 'Node.js runtime command',
            'git': 'Git version control command',
        }.get(cmd, f'{cmd} command')
        return f"Shell command: runs <code>{html.escape(cmd)}</code> ({cmd_desc}) with arguments <code>{html.escape(args)}</code>."

    # Generic code line
    return f"<code>{html.escape(stripped)}</code>"


def build_explanation_block(code_text: str, block_index: int) -> str:
    """Build a full explanation <div> for one code block."""
    raw_lines = code_text.split('\n')
    # Filter out truly empty lines but keep comments-only lines
    code_lines = [(i, l) for i, l in enumerate(raw_lines) if l.strip()]

    if not code_lines:
        return ''

    items = []
    for display_num, (orig_idx, line) in enumerate(code_lines, 1):
        desc = describe_line(line, context_lines=code_lines, line_idx=orig_idx)
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
    """Remove HTML tags but keep the text content."""
    text = re.sub(r'<span[^>]*>', '', text)
    text = re.sub(r'</span>', '', text)
    text = html.unescape(text)
    return text


def process_file(filepath: str) -> bool:
    """Process a single HTML file. Returns True if modified."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove any existing basic line-explanation blocks (from previous run)
    content = re.sub(
        r'\n<div class="line-explanation">.*?</div>\n',
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
        explanation = build_explanation_block(code_text, block_index)
        return f'{opening}{code_html}{closing}\n{explanation}'

    new_content = pattern.sub(replacer, content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False


def main():
    files = sorted(glob.glob(os.path.join(CHAPTERS_DIR, '*.html')))
    modified = 0
    for fp in files:
        if process_file(fp):
            print(f'  [OK] {os.path.basename(fp)}')
            modified += 1
        else:
            print(f'  [--] {os.path.basename(fp)} (skipped)')
    print(f'\nDone. Modified {modified}/{len(files)} files.')


if __name__ == '__main__':
    main()
