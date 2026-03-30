"""
download_model.py — Downloads the dots.mocr model weights from Hugging Face.
Called by setup.bat during first-time setup.
"""
import os
import sys

model_id = "rednote-hilab/dots.mocr"
cache_dir = os.path.join("models", "dots_mocr")
os.makedirs(cache_dir, exist_ok=True)

print("  Downloading tokenizer...")
try:
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    print("  Tokenizer ready.")
except Exception as e:
    print(f"  ERROR downloading tokenizer: {e}")
    sys.exit(1)

print("  Downloading model weights (this may take a while)...")
try:
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        device_map="cpu",       # download to CPU first, ocr.py moves to GPU at runtime
        torch_dtype="auto",
    )
    print("  Model ready.")
except Exception as e:
    print(f"  ERROR downloading model: {e}")
    print()
    print("  Trying fallback download method (snapshot)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        print("  Model files downloaded via snapshot. Ready.")
    except Exception as e2:
        print(f"  ERROR in fallback: {e2}")
        print()
        print("  Please install huggingface_hub and try again:")
        print("    pip install huggingface_hub")
        sys.exit(1)

print("  All done!")
