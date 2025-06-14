import os
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

HF_TOKEN = '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'
base_dir = "~/Baby_LLM"
model_dir = os.path.expanduser(f"{base_dir}/model")

class BabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=1024):
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

def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.7, top_k=40):
    input_ids = mx.array(tokenizer(prompt, return_tensors="np")["input_ids"])  # Shape: (1, seq_len)
    output_ids = input_ids
    for _ in range(max_tokens):
        logits = model(output_ids)  # Shape: (1, seq_len, vocab_size)
        logits = logits[:, -1, :] / temperature  # Shape: (1, vocab_size)
        # Get top-k indices
        top_k_indices = mx.topk(logits, k=top_k, axis=-1)  # Shape: (1, top_k)
        # Extract top-k logits
        top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)  # Shape: (1, top_k)
        # Compute probabilities
        probs = mx.softmax(top_k_logits, axis=-1)  # Shape: (1, top_k)
        # Sample from top-k
        next_token_idx = mx.random.categorical(probs.log())  # Shape: (1,)
        next_token = top_k_indices[0, next_token_idx]  # Scalar
        # Reshape next_token to (1, 1) explicitly
        next_token = mx.expand_dims(mx.expand_dims(next_token, axis=0), axis=1)  # Shape: (1, 1)
        output_ids = mx.concatenate([output_ids, next_token], axis=1)
    return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model = BabyLLM(tokenizer.vocab_size)
model.load_weights(f"{model_dir}/baby_llm.npz")

prompt = "Once upon a time, in a small village,"
for temp in [0.5, 0.7, 1.0]:
    for k in [20, 40, 60]:
        generated = generate_text(model, tokenizer, prompt, temperature=temp, top_k=k)
        print(f"Temperature {temp}, Top-k {k}: {generated}")