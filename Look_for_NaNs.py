import numpy as np
import os

base_dir = "~/Baby_LLM"
cache_dir = os.path.expanduser(os.path.join(base_dir, "cache"))
model_dir = os.path.expanduser(os.path.join(base_dir, "model"))

# Inspect problematic embedding
problematic_embedding = np.load(os.path.join(model_dir, "pretraining_batch_478_pre_embedding.npy"))
print("Embedding shape:", problematic_embedding.shape)
print("NaN count:", np.isnan(problematic_embedding).sum())
print("Inf count:", np.isinf(problematic_embedding).sum())
print("Sample values:", problematic_embedding.flatten()[:10])

# Load checkpoint with allow_pickle=True
checkpoint = np.load(os.path.join(model_dir, "pretrain_checkpoint.npz"), allow_pickle=True)
for key, value in checkpoint.items():
    if isinstance(value, np.ndarray):
        try:
            # Try to convert to float to check for NaN/Inf
            value_float = value.astype(np.float64)
            if np.any(np.isnan(value_float)) or np.any(np.isinf(value_float)):
                print(f"NaN/Inf in {key}")
        except (TypeError, ValueError) as e:
            print(f"Cannot check NaN/Inf for {key}: type={value.dtype}, error={str(e)}")
    else:
        print(f"Non-array value in {key}: type={type(value)}, value={value}")