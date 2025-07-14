import numpy as np
data = np.load("/Users/mutecypher/Baby_LLM/model/failed_attention_output_1751609224.288178.npy")
print("Shape:", data.shape)
print("NaN present:", np.any(np.isnan(data)))
print("Inf present:", np.any(np.isinf(data)))