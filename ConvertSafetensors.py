import torch
from safetensors.torch import load_file

# Load .safetensors (or .bin)
weights = load_file("/Volumes/Phials4Miles/GitHub/Baby_LLM/Mistral_weights/safetensors.json")
# Or: weights = torch.load("pytorch_model.bin", map_location="cpu")

# Save as .pth
torch.save(weights, "/Volumes/Phials4Miles/GitHub/Baby_LLM/Mistral_weights/consolidated.00.pth")