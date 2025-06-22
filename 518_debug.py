import re
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import BertTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Preprocessing function for 581.txt
def clean_text(text):
    """Clean Ginx's Baby text by removing metadata and normalizing."""
    text = re.sub(r'\*\*\*.*?(\n\n|$)', '', text, flags=re.DOTALL)
    text = ''.join(c for c in text if c.isprintable())
    text = text.replace('—', '-').replace('“', '"').replace('”', '"')
    normalization_dict = {"Wauxhall": "Vauxhall", "childer'": "children", "Parleyment": "Parliament"}
    for old, new in normalization_dict.items():
        text = text.replace(old, new)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load 581.txt
def load_single_file(file_path):
    """Load and clean 581.txt."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = clean_text(file.read())
        return text
    except Exception as e:
        logger.error(f"Failed to load 581.txt: {str(e)}")
        raise

# Dataset class for MLX
class TextDataset:
    def __init__(self, text, tokenizer, max_length=512, batch_size=8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        self.encodings = []
        for chunk in self.chunks:
            encoding = tokenizer(
                chunk,
                return_tensors="np",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_attention_mask=True,
                clean_up_tokenization_spaces=True
            )
            input_ids = encoding['input_ids'].squeeze()
            if (input_ids < 0).any() or (input_ids >= tokenizer.vocab_size).any():
                logger.warning(f"Invalid token IDs in chunk: {chunk[:100]}...")
                input_ids = np.clip(input_ids, 0, tokenizer.vocab_size - 1)
            self.encodings.append({
                'input_ids': input_ids,
                'attention_mask': encoding['attention_mask'].squeeze()
            })

    def __len__(self):
        return len(self.encodings)

    def get_batch(self, idx):
        """Return a batch of data as MLX arrays."""
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.encodings))
        batch = self.encodings[start:end]
        input_ids = np.stack([item['input_ids'] for item in batch])
        attention_mask = np.stack([item['attention_mask'] for item in batch])
        return {
            'input_ids': mx.array(input_ids, dtype=mx.int32),
            'attention_mask': mx.array(attention_mask, dtype=mx.bool_)
        }

# Simple MLX-based BabyLLM model (decoder-only)
class BabyLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=4, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Manual Xavier uniform initialization
        fan_in, fan_out = embedding_dim, vocab_size
        limit = mx.sqrt(mx.array(6.0 / (fan_in + fan_out)))
        self.embedding.weight = mx.random.uniform(-limit, limit, self.embedding.weight.shape)
        self.decoder = nn.TransformerDecoder(
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def __call__(self, input_ids, attention_mask=None):
        # Create causal mask for decoder
        seq_len = input_ids.shape[1]
        causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        embeddings = self.embedding(input_ids)
        if mx.any(mx.isnan(embeddings)) or mx.any(mx.isinf(embeddings)):
            raise ValueError("NaN/Inf in embedding")
        # Apply attention mask to causal mask
        if attention_mask is not None:
            causal_mask = causal_mask * attention_mask[:, None, :]
        output = self.decoder(embeddings, mask=causal_mask)
        logits = self.fc(output)
        return logits

# Compute cross-entropy loss
def compute_loss(logits, labels):
    """Compute cross-entropy loss with MLX."""
    if mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
        raise ValueError("NaN/Inf in logits")
    logits = logits.reshape((-1, logits.shape[-1]))
    labels = labels.reshape(-1)
    return nn.losses.cross_entropy(logits, labels, ignore_index=0)

# Training loop
def train(model, dataset, optimizer, epochs=1):
    model.train()
    num_batches = (len(dataset) + dataset.batch_size - 1) // dataset.batch_size
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx in range(num_batches):
            try:
                batch = dataset.get_batch(batch_idx)
                input_ids = batch['input_ids']
                attention_mask = batch['input_ids'] != 0  # Mask non-padding tokens
                
                logits = model(input_ids[:, :-1], attention_mask[:, :-1])
                loss = compute_loss(logits, input_ids[:, 1:])
                
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.update()
                
                loss_val = float(loss.item())
                total_loss += loss_val
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss_val:.4f}, Grad Norm: {grad_norm:.4f}")
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {str(e)}")
                input_text = dataset.tokenizer.batch_decode(
                    batch['input_ids'].numpy()[:1],
                    clean_up_tokenization_spaces=True
                )[0][:100]
                print(f"Input text sample: {input_text}...")
                raise
            
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

# Compute perplexity
def compute_perplexity(model, dataset):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = (len(dataset) + dataset.batch_size - 1) // dataset.batch_size
    for batch_idx in range(num_batches):
        batch = dataset.get_batch(batch_idx)
        input_ids = batch['input_ids']
        attention_mask = batch['input_ids'] != 0
        logits = model(input_ids[:, :-1], attention_mask[:, :-1])
        loss = compute_loss(logits, input_ids[:, 1:])
        total_loss += float(loss.item()) * input_ids.shape[1]
        total_tokens += input_ids.shape[1]
    return float(mx.exp(mx.array(total_loss / total_tokens)).item())

# Cosine warmup scheduler
class CosineWarmup:
    def __init__(self, learning_rate, warmup_steps, total_steps):
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))

# Main execution
if __name__ == "__main__":
    file_path = "/Users/mutecypher/Baby_LLM/data/gutenberg/581.txt"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = load_single_file(file_path)
    dataset = TextDataset(text, tokenizer, max_length=512, batch_size=8)
    model = BabyLLM(vocab_size=tokenizer.vocab_size, embedding_dim=256)
    total_steps_lm = len(dataset) * 1
    optimizer = optim.AdamW(learning_rate=CosineWarmup(5e-7, 1000, total_steps_lm), weight_decay=0.05)
    
    try:
        train(model, dataset, optimizer, epochs=1)
        perplexity = compute_perplexity(model, dataset)
        print(f"Perplexity: {perplexity:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise