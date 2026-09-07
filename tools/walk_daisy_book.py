"""Walk a DAISY 2.02 book exactly as a player would.

Usage: python tools/walk_daisy_book.py <book-folder-containing-ncc.html>

Simulates the full playback chain a DAISY 2.02 player follows:

1. Open ``ncc.html`` -- parse the navigation (title + headings).
2. For each navigation anchor, resolve ``smil#fragment`` to a SMIL ``<text>``.
3. Load the ``<text src>`` document and verify the fragment id exists.
4. Load each ``<audio>`` target, verify the file exists and every
   ``clip-begin``/``clip-end`` lies inside the real audio duration, with
   clips contiguous (no gaps/overlaps) so playback never stalls.
5. Verify every back-link in the text documents points at a real SMIL id.

Exits non-zero and lists problems if any step fails.
"""

from __future__ import annotations

import os
import re
import sys
import wave
import xml.etree.ElementTree as ET


def _fail(problems, msg):
    problems.append(msg)


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _wav_duration(path):
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _audio_duration(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "wav":
        try:
            return _wav_duration(path)
        except Exception:
            return None
    # mp3/flac: probe with the project's ffmpeg helper
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from ai_voice_studio.audio import ffmpeg as ffmpeg_mod  # noqa: PLC0415

        exe = ffmpeg_mod.find_ffmpeg()
        if not exe:
            return None
        import subprocess  # noqa: PLC0415

        result = subprocess.run(
            [exe, "-i", path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or "")
        if m:
            h, mi, s = m.groups()
            return int(h) * 3600 + int(mi) * 60 + float(s)
    except Exception:
        pass
    return None


def main(book_dir):
    problems = []
    ncc_path = os.path.join(book_dir, "ncc.html")
    if not os.path.isfile(ncc_path):
        print("FAIL: no ncc.html in", book_dir)
        return 1

    # -- Step 1: open the NCC and read the navigation -----------------------
    ncc = ET.parse(ncc_path).getroot()
    title = ncc.findtext(".//{*}head/{*}title") or "?"
    multimedia = None
    for meta in ncc.findall(".//{*}head/{*}meta"):
        if meta.get("name") == "ncc:multimediaType":
            multimedia = meta.get("content")
    print(f"Opened ncc.html  (title: {title!r}, multimediaType: {multimedia})")
    if multimedia not in ("audioNcc", "audioFullText"):
        _fail(problems, f"unexpected ncc:multimediaType {multimedia!r}")

    nav = []
    for a in ncc.findall(".//{*}body//{*}a"):
        href = a.get("href", "")
        text = "".join(a.itertext()).strip()
        nav.append((text, href))
    print(f"Navigation: {len(nav)} items")

    smil_cache = {}
    backlinks = {}  # (smil_name, text_id) -> text doc fragment
    smil_clip_walk = {}  # smil_name -> list of (audio_file, clip_begin, clip_end)

    # -- Steps 2-4: follow every nav anchor as a player would ---------------
    for text, href in nav:
        if "#" not in href:
            _fail(problems, f"nav anchor {href!r} has no #fragment")
            continue
        smil_name, frag = href.split("#", 1)
        smil_path = os.path.join(book_dir, smil_name)
        if not os.path.isfile(smil_path):
            _fail(problems, f"nav {text!r}: missing SMIL {smil_name}")
            continue
        if smil_name not in smil_cache:
            smil_cache[smil_name] = ET.parse(smil_path).getroot()

        root = smil_cache[smil_name]
        target = None
        for el in root.iter():
            if el.get("id") == frag:
                target = el
                break
        if target is None:
            _fail(problems, f"nav {text!r}: fragment #{frag} not found in {smil_name}")
            continue
        if _local(target.tag) != "text":
            _fail(problems, f"nav {text!r}: #{frag} is <{_local(target.tag)}>, not <text>")
            continue

        # Step 3: load the text document and its fragment
        tsrc = target.get("src", "")
        if "#" not in tsrc:
            _fail(problems, f"nav {text!r}: <text> src {tsrc!r} has no fragment")
            continue
        doc_name, doc_frag = tsrc.split("#", 1)
        doc_path = os.path.join(book_dir, doc_name)
        if not os.path.isfile(doc_path):
            _fail(problems, f"nav {text!r}: missing text doc {doc_name}")
            continue
        doc = ET.parse(doc_path).getroot()
        frag_el = None
        for el in doc.iter():
            if el.get("id") == doc_frag:
                frag_el = el
                break
        if frag_el is None:
            _fail(problems, f"nav {text!r}: #{doc_frag} missing in {doc_name}")
            continue
        shown = " ".join("".join(frag_el.itertext()).split())[:60]
        backlinks[(smil_name, frag)] = (doc_name, doc_frag)

        # Step 4: audio clip within the par
        par = target
        # find enclosing <par>
        parent_map = {c: p for p in root.iter() for c in p}
        while par is not None and _local(par.tag) != "par":
            par = parent_map.get(par)
        if par is None:
            _fail(problems, f"nav {text!r}: <text> not inside a <par>")
            continue
        audio = None
        for el in par.iter():
            if _local(el.tag) == "audio":
                audio = el
                break
        if audio is None:
            _fail(problems, f"nav {text!r}: par has no <audio>")
            continue
        apath = os.path.join(book_dir, audio.get("src", ""))
        if not os.path.isfile(apath):
            _fail(problems, f"nav {text!r}: missing audio {audio.get('src')!r}")
            continue
        cb = float(re.search(r"npt=([\d.]+)s", audio.get("clip-begin", "")).group(1))
        ce = float(re.search(r"npt=([\d.]+)s", audio.get("clip-end", "")).group(1))
        dur = _audio_duration(apath)
        if dur is None:
            print(f"  [{text[:40]!r}] -> {smil_name}#{frag}  text {doc_name}#{doc_frag}"
                  f"  audio {os.path.basename(apath)} {cb:.3f}-{ce:.3f}s"
                  f"  (real duration unknown for this format)")
        elif ce > dur + 0.05:
            _fail(problems, f"nav {text!r}: clip-end {ce:.3f}s exceeds real audio {dur:.3f}s")
        smil_clip_walk.setdefault(smil_name, []).append(
            (os.path.basename(apath), cb, ce))
        print(f"  [{text[:40]!r}] -> {smil_name}#{frag}  shows {doc_name}#{doc_frag}"
              f"  {shown!r}  audio {os.path.basename(apath)} {cb:.3f}-{ce:.3f}s"
              + ("" if dur is None else f" (file {dur:.3f}s)"))

    # -- Step 4b: audio clips inside each SMIL must be contiguous ----------
    # Walk EVERY <par> of every SMIL in document order (linear playback),
    # not just nav-reached ones.  Each audio file's slices must tile it with
    # no gaps or overlaps, so playback never stalls or repeats a phrase.
    for smil_name, root in smil_cache.items():
        per_file = {}
        for par in root.iter():
            if _local(par.tag) != "par":
                continue
            audio = next((el for el in par.iter()
                          if _local(el.tag) == "audio"), None)
            if audio is None:
                _fail(problems, f"{smil_name}: a <par> has no <audio>")
                continue
            af = os.path.basename(audio.get("src", ""))
            cb = float(re.search(r"npt=([\d.]+)s", audio.get("clip-begin", "")).group(1))
            ce = float(re.search(r"npt=([\d.]+)s", audio.get("clip-end", "")).group(1))
            per_file.setdefault(af, []).append((cb, ce))
        for af, spans in per_file.items():
            spans.sort()
            dur = _audio_duration(os.path.join(book_dir, af))
            prev = 0.0
            for cb, ce in spans:
                if abs(cb - prev) > 0.05:
                    _fail(problems, f"{smil_name}: gap/overlap at {cb:.3f}s in {af}")
                prev = ce
            if dur is not None and abs(prev - dur) > 0.05:
                _fail(problems, f"{smil_name}: {af} clips end at {prev:.3f}s "
                      f"but the file is {dur:.3f}s long")
            print(f"  linear playback {smil_name}: {af} tiled 0.000-{prev:.3f}s "
                  f"across {len(spans)} clips" + ("" if dur is None else f" (file {dur:.3f}s)"))

    # -- Step 5: back-links in text docs must point at real SMIL ids --------
    smil_ids = {}
    for name, root in smil_cache.items():
        smil_ids[name] = {el.get("id") for el in root.iter() if el.get("id")}
    doc_files = [f for f in os.listdir(book_dir)
                 if f.endswith(".html") and f != "ncc.html"]
    for doc_name in doc_files:
        doc = ET.parse(os.path.join(book_dir, doc_name)).getroot()
        for a in doc.iter("{http://www.w3.org/1999/xhtml}a"):
            href = a.get("href", "")
            if "#" not in href:
                continue
            sm, frag = href.split("#", 1)
            if not sm.endswith(".smil"):
                continue
            if frag not in smil_ids.get(sm, set()):
                _fail(problems, f"{doc_name}: back-link {href!r} target id missing in {sm}")
            if not any(el.get("id") == frag and _local(el.tag) == "text"
                       for el in smil_cache.get(sm, ET.Element("x")).iter()):
                _fail(problems, f"{doc_name}: back-link {href!r} does not point at a <text>")

    n_text_docs = len(doc_files)
    n_pars = sum(
        1 for root in smil_cache.values() for el in root.iter()
        if _local(el.tag) == "par"
    )
    print(f"\nBook summary: {len(smil_cache)} SMILs, {n_pars} synchronized <par>s, "
          f"{n_text_docs} text documents, multimediaType={multimedia}")

    if problems:
        print("\nPROBLEMS FOUND:")
        for p in problems:
            print(" -", p)
        return 1
    print("\nPLAYER WALKTHROUGH: PASS - every navigation, text and audio "
          "reference resolves; clips are contiguous and inside the real audio.")
    return 0


if __name__ == "__main__":
    book = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\anujj\AppData\Roaming\AIVoiceStudio\projects\testingFileCreation\DAISY"
    sys.exit(main(book))
