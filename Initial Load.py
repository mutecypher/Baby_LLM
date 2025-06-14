import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlx.core as mx
from mistral_inference.transformer import Transformer
import xformers  # Keep for mistral_inference compatibility

HF_TOKEN = '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'

# Use internal SSD or USB drive
base_dir = "/Volumes/Phials4Miles/GitHub/Baby_LLM/Mistral_weights"
cache_dir = os.path.expanduser(f"{base_dir}/.cache/huggingface/hub")
model_dir = os.path.expanduser(f"{base_dir}/Mistral_weights_hf")
weights_dir = os.path.expanduser(f"{base_dir}/Mistral_weights")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

# Check disk space
required_space = 20 * 1024 * 1024 * 1024
available_space = os.statvfs(cache_dir).f_frsize * os.statvfs(cache_dir).f_bavail
if available_space < required_space:
    raise OSError(f"Not enough disk space! Need ~20GB, but only {available_space / (1024**3):.2f}GB available")

# Set device to MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model in FP16 without quantization
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    token=HF_TOKEN,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,  # FP16 for MPS
    device_map={"": device},
    low_cpu_mem_usage=True
)
model.save_pretrained(model_dir)
print("Model saved successfully")

# Convert weights
state_dict = model.state_dict()
torch.save(state_dict, f"{weights_dir}/consolidated.00.pth")
print("Model weights converted")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    token=HF_TOKEN,
    cache_dir=cache_dir
)
print("Tokenizer OK")

# Load mistral_inference model
model = Transformer.from_folder(weights_dir)
print("Mistral inference model loaded")

# Diagnostics
print("PyTorch:", torch.__version__, "MPS:", torch.backends.mps.is_available())
print("xFormers:", xformers.__version__)
print("MLX array:", mx.array([1, 2, 3]))

# Process text
with open("corpus_subset.txt", "r") as f:
    text = f.read()
filtered_text = [line for line in text.split("\n") if len(line.split()) < 20]
print("Filtered text (first 5 lines):", filtered_text[:5])

# Test inference
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=50)
print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))