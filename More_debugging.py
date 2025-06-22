import numpy as np
import glob
import os

# Define the model directory
model_dir = os.path.expanduser("~/Baby_LLM/model")

# Find the latest debug files
output_files = glob.glob(os.path.join(model_dir, "debug_linear1_output_*.npy"))
input_files = glob.glob(os.path.join(model_dir, "debug_linear1_input_*.npy"))

# Sort by timestamp to get the most recent files
if not output_files or not input_files:
    print("No debug files found in", model_dir)
else:
    latest_output_file = max(output_files, key=os.path.getmtime)
    latest_input_file = max(input_files, key=os.path.getmtime)
    
    # Load the files
    output = np.load(latest_output_file)
    input_weights = np.load(latest_input_file)
    
    # Analyze the contents
    print(f"Analyzing {latest_output_file}")
    print("Output stats: max=", np.max(output), "min=", np.min(output), 
          "nan=", np.any(np.isnan(output)), "inf=", np.any(np.isinf(output)))
    print(f"Analyzing {latest_input_file}")
    print("Input weights stats: max=", np.max(input_weights), "min=", np.min(input_weights))
    
    # Additional analysis: Check for extreme values
    if np.any(np.abs(output) > 1e3):
        print("Extreme values in output (|x| > 1e3):", output[np.abs(output) > 1e3])
    if np.any(np.abs(input_weights) > 1e2):
        print("Extreme values in input weights (|x| > 1e2):", input_weights[np.abs(input_weights) > 1e2])