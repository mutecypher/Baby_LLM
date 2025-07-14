import mlx.core as mx
import mlx.nn as nn

# Create a small test input
vocab_size = 15000
batch_size = 1
seq_len = 64
d_model = 32
n_heads = 1
d_ff = 128

def xavier_uniform(shape, dtype=mx.float32, scale=1.0):
    fan_in, fan_out = shape[-2], shape[-1]
    limit = mx.sqrt(6.0 / (fan_in + fan_out)) * scale  # Add scale factor
    return mx.random.uniform(-limit, limit, shape=shape, dtype=dtype)


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)
        # Xavier initialization
        self.linear1.weight = xavier_uniform((d_in, d_hidden), dtype=mx.float32)  # Changed to float32
        self.linear1.bias = mx.zeros((d_hidden,), dtype=mx.float32)
        self.linear2.weight = xavier_uniform((d_hidden, d_out), dtype=mx.float32)
        self.linear2.bias = mx.zeros((d_out,), dtype=mx.float32)
        # Validate initialization
        for name, param in self.parameters().items():
            if isinstance(param, mx.array):
                if mx.any(mx.isnan(param)) or mx.any(mx.isinf(param)):
                    
                    raise ValueError(f"Invalid initialization for FeedForward {name}")

                
    def __call__(self, x):
        if not isinstance(x, mx.array):
            
            raise ValueError("FeedForward input must be mx.array")
 
        x = x.astype(mx.float32)
        x = self.linear1(x)
        
        x = nn.gelu(x)
        x = mx.clip(x, -1e2, 1e2)
       
        x = self.linear2(x)
        
        x = mx.clip(x, -1e2, 1e2)
        if x.ndim < 2:

            raise ValueError("FeedForward output must be at least 2D")
        return x.astype(mx.float32)
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, n_heads, bias=True)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff = FeedForward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(0.2)
        # Initialize attention parameters with Xavier uniform
        for key, param in self.attention.parameters().items():
            if isinstance(param, mx.array) and 'weight' in key:
                self.attention.parameters()[key] = xavier_uniform(param.shape, dtype=mx.float32)
            elif isinstance(param, mx.array) and 'bias' in key:
                self.attention.parameters()[key] = mx.zeros(param.shape, dtype=mx.float32)

    def __call__(self, x, mask=None):
        if not isinstance(x, mx.array):
             .error(f"TransformerLayer input is not mx.array: type={type(x)}")
            raise ValueError("TransformerLayer input must be mx.array")
         
        if mask is not None and mask.shape[2] != x.shape[1]:
            
            raise ValueError("Mask shape mismatch")

        x = x.astype(mx.float32)
        try:
            # Pass x as queries, keys, and values for self-attention
            attn_output = self.attention(queries=x, keys=x, values=x, mask=mask)
          
            attn_output = mx.clip(attn_output, -1e2, 1e2)  # Tighten clipping for stability
            attn_output = self.dropout(attn_output)
        except Exception as e:
       
            ##np.save(os.path.join(model_dir, f"failed_attention_output_{time.time()}.npy"), np.array(x))
            raise

        x = self.norm1(x + attn_output)
        x = mx.clip(x, -1e2, 1e2)
        ff_output = self.dropout(self.ff(x))
        x = self.norm2(x + ff_output)
        x = mx.clip(x, -1e2, 1e2)
        return x.astype(mx.float32)
  
class BabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = [TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.final_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.output = nn.Linear(d_model, vocab_size)
        self._debug_counter = 0
        # Initialize weights
        self.embedding.weight = xavier_uniform((vocab_size, d_model), dtype=mx.float32)
        self.pos_embedding.weight = xavier_uniform((max_len, d_model), dtype=mx.float32)
        self.output.weight = xavier_uniform((d_model, vocab_size), dtype=mx.float32)
        # Zero bias for output layer
        self.output.bias = mx.zeros((vocab_size,), dtype=mx.float32)
        # Validate initialization
        for name, param in self.parameters().items():
            if isinstance(param, mx.array):
                if mx.any(mx.isnan(param)) or mx.any(mx.isinf(param)):

                    raise ValueError(f"Invalid initialization for {name}")

            else:


    def to_numpy_for_decode(self, array):
        """Convert MLX array to NumPy array for tokenizer decoding."""
        return np.array(array) if isinstance(array, mx.array) else array


    def __call__(self, x):
        self._debug_counter += 1
 
        
        if not isinstance(x, mx.array):

            return None
        if x.ndim != 2:
           
            return None
        
        vocab_size_with_special = self.embedding.weight.shape[0]
        if mx.any(x < 0) or mx.any(x >= vocab_size_with_special):
            
            return None
        
        # Mask creation
        seq_len = x.shape[1]
        causal_mask = mx.triu(mx.ones((1, seq_len, seq_len), dtype=mx.bool_), k=1)
        padding_mask = (x == tokenizer.pad_token_id).astype(mx.bool_)[:, None, :]
        combined_mask = mx.logical_or(causal_mask, padding_mask)
 
        if combined_mask.shape != (1, seq_len, seq_len):
           
            return None
        
        # Embedding
        x = self.embedding(x).astype(mx.float32)
     
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):

            return None
        
        # Positional embedding
        if seq_len > self.pos_embedding.weight.shape[0]:

            return None
        positions = mx.arange(seq_len)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        x = mx.clip(x, -1e2, 1e2)
 
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=combined_mask)
            x = mx.clip(x, -1e2, 1e2)

            if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
              
                return None
        
        # Final norm and output
        x = self.final_norm(x)
     
        x = self.output(x)
         .info(f"Output logits: shape={x.shape}, min={mx.min(x).item()}, max={mx.max(x).item()}")
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
  
            return None
        return x.astype(mx.float32)
    
# Initialize model
model = BabyLLM(vocab_size=vocab_size, d_model=d_model, n_layers=1, n_heads=n_heads, d_ff=d_ff, max_len=128)
test_input = mx.random.randint(0, vocab_size, (batch_size, seq_len), dtype=mx.int32)
mask = mx.triu(mx.ones((1, seq_len, seq_len), dtype=mx.bool_), k=1)

# Run forward pass
try:
    logits = model(test_input)
    print(f"Logits shape: {logits.shape}")
except Exception as e:
    print(f"Error in forward pass: {str(e)}")