#!/usr/bin/env python3
"""Batch-add sidebar navigation to all chapter HTML files."""

import os
import re
from pathlib import Path

BOOK_DIR = Path(__file__).parent
CHAPTERS_DIR = BOOK_DIR / "chapters"

# All chapters for the sidebar navigation
SIDEBAR_NAV = """\
<nav class="sidebar">
  <h2>📖 Table of Contents</h2>

  <div class="part-header">Part I — Python Fundamentals</div>
  <ul class="chapters">
    <li><a href="ch01_introduction.html">Ch 1: Introduction &amp; Setup</a></li>
    <li><a href="ch02_variables.html">Ch 2: Variables &amp; Data Types</a></li>
    <li><a href="ch03_control_flow.html">Ch 3: Control Flow</a></li>
    <li><a href="ch04_functions.html">Ch 4: Functions</a></li>
    <li><a href="ch05_strings.html">Ch 5: Strings &amp; Formatting</a></li>
    <li><a href="ch06_collections.html">Ch 6: Lists, Tuples, Sets &amp; Dicts</a></li>
    <li><a href="ch07_file_io.html">Ch 7: File I/O</a></li>
    <li><a href="ch08_error_handling.html">Ch 8: Error Handling</a></li>
    <li><a href="ch09_modules.html">Ch 9: Modules &amp; Packages</a></li>
    <li><a href="ch10_oop_basics.html">Ch 10: OOP Basics</a></li>
  </ul>

  <div class="part-header">Part II — Intermediate Python</div>
  <ul class="chapters">
    <li><a href="ch11_oop_advanced.html">Ch 11: OOP Advanced</a></li>
    <li><a href="ch12_iterators.html">Ch 12: Iterators &amp; Generators</a></li>
    <li><a href="ch13_decorators.html">Ch 13: Decorators</a></li>
    <li><a href="ch14_context.html">Ch 14: Context Managers</a></li>
    <li><a href="ch15_regex.html">Ch 15: Regular Expressions</a></li>
    <li><a href="ch16_datetime.html">Ch 16: Date &amp; Time</a></li>
    <li><a href="ch17_testing.html">Ch 17: Testing with unittest &amp; pytest</a></li>
    <li><a href="ch18_json_xml.html">Ch 18: JSON, XML &amp; Structured Data</a></li>
  </ul>

  <div class="part-header">Part III — Advanced Python</div>
  <ul class="chapters">
    <li><a href="ch19_concurrency.html">Ch 19: Concurrency &amp; Threading</a></li>
    <li><a href="ch20_multiprocessing.html">Ch 20: Multiprocessing</a></li>
    <li><a href="ch21_async.html">Ch 21: Async I/O</a></li>
    <li><a href="ch22_type_hints.html">Ch 22: Type Hints &amp; Static Typing</a></li>
    <li><a href="ch23_metaclasses.html">Ch 23: Metaclasses &amp; Descriptors</a></li>
    <li><a href="ch24_design_patterns.html">Ch 24: Design Patterns in Python</a></li>
    <li><a href="ch25_introspection.html">Ch 25: Introspection &amp; Reflection</a></li>
    <li><a href="ch26_c_extensions.html">Ch 26: C Extensions &amp; ctypes</a></li>
  </ul>

  <div class="part-header">Part IV — wxPython GUI Development</div>
  <ul class="chapters">
    <li><a href="ch27_wxpython_intro.html">Ch 27: wxPython Introduction</a></li>
    <li><a href="ch28_sizers_layout.html">Ch 28: Sizers &amp; Layout</a></li>
    <li><a href="ch29_controls.html">Ch 29: Controls &amp; Widgets</a></li>
    <li><a href="ch30_events.html">Ch 30: Event Handling</a></li>
    <li><a href="ch31_dialogs.html">Ch 31: Dialogs &amp; Wizards</a></li>
    <li><a href="ch32_custom_drawing.html">Ch 32: Custom Drawing &amp; DC</a></li>
    <li><a href="ch33_threading_gui.html">Ch 33: Threading in GUI Apps</a></li>
    <li><a href="ch34_accessibility.html">Ch 34: Accessibility (a11y)</a></li>
    <li><a href="ch35_themes.html">Ch 35: Theming &amp; Styling</a></li>
    <li><a href="ch36_wx_adv.html">Ch 36: wx.adv &amp; Advanced Widgets</a></li>
  </ul>

  <div class="part-header">Part V — Dependencies Deep Dive</div>
  <ul class="chapters">
    <li><a href="ch37_numpy.html">Ch 37: NumPy for Audio Processing</a></li>
    <li><a href="ch38_requests.html">Ch 38: Requests &amp; HTTP</a></li>
    <li><a href="ch39_sherpa_onnx.html">Ch 39: sherpa-onnx TTS Engine</a></li>
    <li><a href="ch40_pypdf_docx.html">Ch 40: PDF &amp; DOCX Parsing</a></li>
    <li><a href="ch41_ffmpeg.html">Ch 41: FFmpeg &amp; Audio Formats</a></li>
    <li><a href="ch42_virtualenvs.html">Ch 42: Virtual Environments &amp; Packaging</a></li>
  </ul>

  <div class="part-header">Part VI — Building AI Voice Studio</div>
  <ul class="chapters">
    <li><a href="ch43_architecture.html">Ch 43: Project Architecture</a></li>
    <li><a href="ch44_paths_settings.html">Ch 44: Paths, Settings &amp; Constants</a></li>
    <li><a href="ch45_project_model.html">Ch 45: Project Model &amp; Persistence</a></li>
    <li><a href="ch46_document_parsing.html">Ch 46: Document Parsing &amp; Splitting</a></li>
    <li><a href="ch47_tts_catalog.html">Ch 47: TTS Catalog &amp; Model Store</a></li>
    <li><a href="ch48_tts_engine.html">Ch 48: TTS Engine Integration</a></li>
    <li><a href="ch49_downloader.html">Ch 49: Model Downloader &amp; Artifacts</a></li>
    <li><a href="ch50_compute_detection.html">Ch 50: Compute Back-end Detection</a></li>
    <li><a href="ch51_main_frame.html">Ch 51: Main Frame &amp; Menu System</a></li>
    <li><a href="ch52_settings_dialog.html">Ch 52: Settings Dialog</a></li>
    <li><a href="ch53_model_panels.html">Ch 53: Model Management Panels</a></li>
    <li><a href="ch54_new_project_wizard.html">Ch 54: New Project Wizard</a></li>
    <li><a href="ch55_recording.html">Ch 55: Recording Dialog</a></li>
    <li><a href="ch56_synthesis_worker.html">Ch 56: Background Synthesis Worker</a></li>
    <li><a href="ch57_clone_engine.html">Ch 57: Voice Cloning (XTTS v2)</a></li>
    <li><a href="ch58_omnivoice.html">Ch 58: OmniVoice TTS Engine</a></li>
    <li><a href="ch59_daisy_builder.html">Ch 59: DAISY Book Builder</a></li>
    <li><a href="ch60_addons.html">Ch 60: Addon System</a></li>
  </ul>

  <div class="part-header">Part VII — Testing &amp; Deployment</div>
  <ul class="chapters">
    <li><a href="ch61_unit_testing.html">Ch 61: Unit Testing Strategies</a></li>
    <li><a href="ch62_mocking.html">Ch 62: Mocking &amp; Test Doubles</a></li>
    <li><a href="ch63_integration.html">Ch 63: Integration Testing</a></li>
    <li><a href="ch64_pyinstaller.html">Ch 64: PyInstaller Packaging</a></li>
    <li><a href="ch65_inno_setup.html">Ch 65: Inno Setup Installers</a></li>
    <li><a href="ch66_ci_cd.html">Ch 66: CI/CD &amp; Release</a></li>
  </ul>

  <div class="part-header">Part VIII — Advanced Topics</div>
  <ul class="chapters">
    <li><a href="ch67_python_runtime.html">Ch 67: Managed Python Runtime</a></li>
    <li><a href="ch68_subprocess.html">Ch 68: Subprocess Architecture</a></li>
    <li><a href="ch69_logging.html">Ch 69: Logging &amp; Diagnostics</a></li>
    <li><a href="ch70_performance.html">Ch 70: Performance Optimization</a></li>
    <li><a href="ch71_security.html">Ch 71: Security Considerations</a></li>
    <li><a href="ch72_roadmap.html">Ch 72: Future Directions &amp; Roadmap</a></li>
  </ul>

  <div class="part-header">Appendices</div>
  <ul class="chapters">
    <li><a href="appendix_a_reference.html">A: Python Quick Reference</a></li>
    <li><a href="appendix_b_api.html">B: AIVS API Reference</a></li>
    <li><a href="appendix_c_glossary.html">C: Glossary</a></li>
  </ul>
</nav>
"""


def get_chapter_number(filename: str) -> str:
    """Extract chapter number from filename like 'ch01_introduction'."""
    match = re.match(r"ch(\d+)_", filename)
    if match:
        return f"Ch {int(match.group(1))}"
    if filename.startswith("appendix_"):
        return filename.replace("appendix_", "").replace("_", " ").title()
    return ""


def get_nav_class(filename: str, current_file: str) -> str:
    """Return 'active' if this nav item matches the current file."""
    if filename == current_file:
        return " active"
    return ""


def add_sidebar(html_content: str, filename: str) -> str:
    """Add sidebar navigation to a chapter HTML file."""
    # Skip if already has sidebar
    if 'class="sidebar"' in html_content:
        print(f"  SKIP (already has sidebar): {filename}")
        return html_content

    # Update the sidebar links to use the correct active class
    sidebar_lines = SIDEBAR_NAV.split("\n")
    updated_sidebar = []
    for line in sidebar_lines:
        # Check if this line contains a link to the current file
        if f'href="{filename}"' in line:
            line = line.replace('class="', 'class="active ')
        updated_sidebar.append(line)
    sidebar = "\n".join(updated_sidebar)

    # Find the insertion point - right after <body> tag
    body_match = re.search(r"(<body[^>]*>)", html_content)
    if not body_match:
        print(f"  SKIP (no <body> tag): {filename}")
        return html_content

    insert_pos = body_match.end()

    # Find the existing content wrapper
    # Look for <div class="book"> or similar
    book_match = re.search(r'(<div\s+class="book"[^>]*>)', html_content)
    if not book_match:
        # Some chapters might use different wrapper
        print(f"  SKIP (no <div class='book'>): {filename}")
        return html_content

    # Build the new HTML structure
    # Replace <div class="book"> with the sidebar + main-content wrapper
    old_book_start = book_match.group(0)
    new_structure = f"""<div class="book-container">
{sidebar}
<div class="main-content">
{old_book_start}"""

    html_content = html_content.replace(old_book_start, new_structure, 1)

    # Add closing tags for main-content and book-container before </body>
    # Find the last </div> before </body>
    body_end_match = re.search(r"</body>", html_content)
    if body_end_match:
        # Insert closing divs before </body>
        insert_pos = body_end_match.start()
        html_content = (
            html_content[:insert_pos]
            + "</div><!-- /.main-content -->\n</div><!-- /.book-container -->\n"
            + html_content[insert_pos:]
        )

    return html_content


def main():
    """Process all chapter files."""
    print("Adding sidebar navigation to all chapter files...")
    print(f"Chapters directory: {CHAPTERS_DIR}")

    # Get all HTML files in chapters directory
    html_files = sorted(CHAPTERS_DIR.glob("*.html"))
    print(f"Found {len(html_files)} HTML files\n")

    updated = 0
    skipped = 0

    for filepath in html_files:
        filename = filepath.name
        print(f"Processing: {filename}")

        try:
            content = filepath.read_text(encoding="utf-8")
            new_content = add_sidebar(content, filename)

            if new_content != content:
                filepath.write_text(new_content, encoding="utf-8")
                updated += 1
                print(f"  [OK] Updated")
            else:
                skipped += 1
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\n{'='*50}")
    print(f"Done! Updated: {updated}, Skipped: {skipped}")
    print(f"Total files: {len(html_files)}")


if __name__ == "__main__":
    main()
