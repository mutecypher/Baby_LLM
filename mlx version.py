import os
import mlx.core as mx
from mlx_lm import load, generate

HF_TOKEN = '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'

# Use internal SSD or USB drive
base_dir = "/Volumes/Phials4Miles/GitHub/Baby_LLM/Mistral_weights"
cache_dir = os.path.expanduser(f"{base_dir}/.cache/huggingface/hub")
model_dir = os.path.expanduser(f"{base_dir}/Mistral_weights_mlx")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Check disk space
required_space = 20 * 1024 * 1024 * 1024
available_space = os.statvfs(cache_dir).f_frsize * os.statvfs(cache_dir).f_bavail
if available_space < required_space:
    raise OSError(f"Not enough disk space! Need ~20GB, but only {available_space / (1024**3):.2f}GB available")

# Load MLX-converted Mistral model (convert if not already done)
model_path = "mistralai/Mistral-7B-v0.3"
model, tokenizer = load(model_path, {"hf_token": HF_TOKEN})

# Save model (optional, MLX downloads to cache by default)
# Note: MLX uses its own format; conversion happens automatically on first load

# Process text
with open("corpus_subset.txt", "r") as f:
    text = f.read()
filtered_text = [line for line in text.split("\n") if len(line.split()) < 20]
print("Filtered text (first 5 lines):", filtered_text[:5])

# Test inference
input_text = "Hello, how are you?"
output = generate(model, tokenizer, prompt=input_text, max_tokens=50)
print("Generated text:", output)

# Diagnostics
print("MLX array:", mx.array([1, 2, 3]))