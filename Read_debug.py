import numpy as np
import glob
import os

# Define the model directory
model_dir = os.path.expanduser("~/Baby_LLM/model")

# Find the latest debug files (replace <timestamp> with actual files)
output_files = glob.glob(os.path.join(model_dir, "debug_linear1_output_*.npy"))
weights_files = glob.glob(os.path.join(model_dir, "debug_linear1_weights_*.npy"))
input_files = glob.glob(os.path.join(model_dir, "debug_linear1_input_*.npy"))

if not (output_files and weights_files and input_files):
    print("Error: Debug files not found in", model_dir)
    exit(1)

# Use the latest files (sorted by timestamp in filename)
latest_output = max(output_files, key=os.path.getmtime)
latest_weights = max(weights_files, key=os.path.getmtime)
latest_input = max(input_files, key=os.path.getmtime)

# Load the files
output = np.load(latest_output)
weights = np.load(latest_weights)
input_x = np.load(latest_input)

# Analyze for NaN/Inf and value ranges
print(f"Analyzing: {latest_output}")
print(f"Output NaN/Inf: {np.any(np.isnan(output)) or np.any(np.isinf(output))}")
print(f"Output max/min: {output.max()}, {output.min()}")
print(f"Input max/min: {input_x.max()}, {input_x.min()}")
print(f"Input mean/std: {np.mean(input_x)}, {np.std(input_x)}")
print(f"Weights max/min: {weights.max()}, {weights.min()}")
print(f"Weights mean/std: {np.mean(weights)}, {np.std(weights)}")