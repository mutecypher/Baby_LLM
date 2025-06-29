import numpy as np
import glob
import os
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", use_fast=False)
file_path = max(glob.glob(os.path.expanduser("/Users/mutecypher/Baby_LLM/model/failed_qa_input_1751143468.6906161.npy")), key=os.path.getctime)
batch = np.load(file_path)
print("Shape:", batch.shape)
print("Min/Max:", np.min(batch), np.max(batch))
for seq in batch:
    print("Decoded:", tokenizer.decode(seq, skip_special_tokens=False))