import os
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

# Paths and constants (adjust these based on your setup)
HF_TOKEN = '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'  # Your Hugging Face token
base_dir = "~/Baby_LLM"
model_dir = os.path.expanduser(f"{base_dir}/model")
cache_dir = os.path.expanduser(f"{base_dir}/cache")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", token=HF_TOKEN, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'sep_token': '<SEP>'})
vocab_size = tokenizer.vocab_size + 1

# Define the BabyLLM model (must match your original architecture)
class BabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        class FeedForward(nn.Module):
            def __init__(self, d_in, d_hidden, d_out):
                super().__init__()
                self.linear1 = nn.Linear(d_in, d_hidden)
                self.linear2 = nn.Linear(d_hidden, d_out)
            def __call__(self, x):
                x = self.linear1(x)
                x = nn.gelu(x)
                x = self.linear2(x)
                return x
        self.layers = [
            {
                "attention": nn.MultiHeadAttention(d_model, n_heads, bias=False),
                "norm1": nn.LayerNorm(d_model),
                "ff": FeedForward(d_model, d_ff, d_model),
                "norm2": nn.LayerNorm(d_model)
            } for _ in range(n_layers)
        ]
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def __call__(self, x):
        x = self.embedding(x) * mx.sqrt(self.d_model)
        for layer in self.layers:
            attn_output = layer["attention"](x, x, x)
            x = layer["norm1"](x + attn_output)
            ff_output = layer["ff"](x)
            x = layer["norm2"](x + ff_output)
        x = self.final_norm(x)
        return self.output(x)

# Initialize the model
model = BabyLLM(vocab_size)

# Load the saved weights
weights_path = f"{model_dir}/baby_llm_qa.npz"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found at {weights_path}. Please train the model first.")
model.load_weights(weights_path)
mx.eval(model.parameters())  # Ensure parameters are evaluated

# Text generation function
def generate_text(model, tokenizer, prompt, max_tokens=128, top_k=10, temperature=0.7):
    input_ids = mx.array(tokenizer(f"{prompt} <SEP>", return_tensors="np")["input_ids"])
    output_ids = input_ids  # Shape: [1, seq_len]
    
    for _ in range(max_tokens):
        logits = model(output_ids)  # Shape: [1, seq_len, vocab_size]
        logits = logits[:, -1, :]  # Shape: [1, vocab_size]
        top_k_indices = mx.topk(logits, k=top_k, axis=-1)  # Shape: [1, top_k]
        top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)  # Shape: [1, top_k]
        probs = mx.softmax(top_k_logits / temperature, axis=-1)  # Apply temperature
        next_token_idx = mx.random.categorical(probs.log())
        next_token = top_k_indices[0, next_token_idx].astype(mx.int32)
        next_token_2d = mx.array([next_token]).reshape(1, 1)
        output_ids = mx.concatenate([output_ids, next_token_2d], axis=1)
        if next_token.item() in [tokenizer.eos_token_id, tokenizer.encode('.')[0]]:
            break
    
    output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    if '<SEP>' in output_text:
        answer = output_text.split('<SEP>')[-1].strip()
    else:
        answer = output_text[len(prompt):].strip()  # Fallback if <SEP> isn’t generated
    return answer

# Interactive testing loop
def test_model():
    print("Model loaded successfully! Enter prompts to test the model (type 'exit' to quit).")
    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() == 'exit':
            print("Exiting...")
            break
        try:
            answer = generate_text(model, tokenizer, prompt)
            print(f"Prompt: {prompt}")
            print(f"Answer: {answer}\n")
        except Exception as e:
            print(f"Error generating response: {e}\n")

# Run the testing loop
if __name__ == "__main__":
    test_model()