"""Download minimal Qwen3-TTS ONNX subset using huggingface_hub."""
import os, sys
sys.path.insert(0, r"C:/Users/anujj/AppData/Local/Temp/qwen_rt")
from huggingface_hub import snapshot_download

OUT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/anujj/AppData/Local/Temp/qwen_model"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

allow = [
    "fp16/voice_clone/*",
    "fp16/shared/*",
    "tokenizer/*",
    "config.json",
    "voice_clone_config.json",
    "tts_engine.py",
    "requirements.txt",
]
path = snapshot_download(
    repo_id="xkos/Qwen3-TTS-12Hz-1.7B-ONNX",
    local_dir=OUT,
    allow_patterns=allow,
)
print("downloaded to", path)
