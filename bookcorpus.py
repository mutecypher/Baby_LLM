from datasets import load_dataset
dataset = load_dataset("tiny_shakespeare", split="train", streaming=True)
print(next(iter(dataset)))