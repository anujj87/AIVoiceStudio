"""Download minimal Qwen3-TTS ONNX subset (fp16 voice_clone + shared + scripts)."""
import json, os, sys, urllib.request, time

BASE = "https://huggingface.co/xkos/Qwen3-TTS-12Hz-1.7B-ONNX"
API = "https://huggingface.co/api/models/xkos/Qwen3-TTS-12Hz-1.7B-ONNX"
OUT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/anujj/AppData/Local/Temp/qwen_model"
os.makedirs(OUT, exist_ok=True)

req = urllib.request.Request(API + "?blobs=true", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60) as r:
    meta = json.load(r)
names = [s["rfilename"] for s in meta.get("siblings", [])]

keep = []
for n in names:
    if n.startswith("fp16/voice_clone/") or n.startswith("fp16/shared/"):
        keep.append(n)
    elif n.count("/") == 0 and not n.endswith(".py") or n in (
        "tokenizer/merges.txt", "tokenizer/vocab.json", "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json", "config.json", "voice_clone_config.json",
        "voice_design_config.json", "tts_engine.py", "synthesize.py",
        "create_speaker.py", "generate_cache.py", "requirements.txt",
    ):
        keep.append(n)
keep = sorted(set(keep))
print("files to download:", len(keep))

def size_of(n):
    for s in meta["siblings"]:
        if s["rfilename"] == n:
            return s.get("size") or 0
    return 0

total = sum(size_of(n) for n in keep)
print("total size: %.2f GB" % (total / 1e9))

done_bytes = 0
for i, n in enumerate(keep):
    dest = os.path.join(OUT, n)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    sz = size_of(n)
    if os.path.exists(dest) and os.path.getsize(dest) == sz and sz > 0:
        done_bytes += sz
        continue
    url = f"{BASE}/resolve/main/{n}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                with open(dest, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            done_bytes += sz
            print(f"[{i+1}/{len(keep)}] {n}  ({sz/1e6:.0f} MB, total {done_bytes/1e9:.2f} GB)")
            break
        except Exception as e:
            print(f"  retry {attempt+1} for {n}: {e}")
            time.sleep(2)
    else:
        print(f"FAILED: {n}")
        sys.exit(1)

print("done")
