import numpy as np

def estimate_model_memory(vocab_size, d_model, n_layers, n_heads, d_ff, max_len, batch_size=8):
    """
    Estimate memory usage for a transformer model in GB
    """
    
    # Model parameters (in millions)
    # Embedding layers
    token_embedding = vocab_size * d_model
    pos_embedding = max_len * d_model
    
    # Per transformer layer
    # Multi-head attention: Q, K, V projections + output projection
    attention_params = 4 * (d_model * d_model)  # 4 linear layers
    # Feed-forward network
    ff_params = d_model * d_ff + d_ff * d_model  # 2 linear layers
    # Layer norms (2 per layer)
    ln_params = 2 * d_model
    
    layer_params = attention_params + ff_params + ln_params
    total_layer_params = n_layers * layer_params
    
    # Output projection
    output_params = d_model * vocab_size
    
    # Final layer norm
    final_ln_params = d_model
    
    total_params = (token_embedding + pos_embedding + total_layer_params + 
                   output_params + final_ln_params)
    
    # Memory calculations (assuming float32 = 4 bytes)
    # Model parameters
    model_memory_gb = (total_params * 4) / (1024**3)
    
    # Gradients (same size as parameters)
    grad_memory_gb = model_memory_gb
    
    # Optimizer state (Adam has momentum and variance, ~2x parameters)
    optimizer_memory_gb = model_memory_gb * 2
    
    # Activations during forward pass (rough estimate)
    # Input embeddings: batch_size * max_len * d_model
    # Attention matrices: batch_size * n_heads * max_len * max_len per layer
    # Feed-forward activations: batch_size * max_len * d_ff per layer
    
    activation_per_layer = (batch_size * max_len * d_model +  # residual connections
                           batch_size * n_heads * max_len * max_len +  # attention weights
                           batch_size * max_len * d_ff)  # FF intermediate
    
    total_activations = activation_per_layer * n_layers
    activation_memory_gb = (total_activations * 4) / (1024**3)
    
    # Total memory needed
    total_memory_gb = model_memory_gb + grad_memory_gb + optimizer_memory_gb + activation_memory_gb
    
    return {
        'total_params_millions': total_params / 1e6,
        'model_memory_gb': model_memory_gb,
        'grad_memory_gb': grad_memory_gb,
        'optimizer_memory_gb': optimizer_memory_gb,
        'activation_memory_gb': activation_memory_gb,
        'total_memory_gb': total_memory_gb,
        'memory_breakdown': {
            'model': model_memory_gb,
            'gradients': grad_memory_gb,
            'optimizer': optimizer_memory_gb,
            'activations': activation_memory_gb
        }
    }

# Your original ambitious parameters
print("=== ORIGINAL AMBITIOUS PARAMETERS ===")
ambitious_config = {
    'vocab_size': 15000,
    'd_model': 768,
    'n_layers': 12,
    'n_heads': 12,
    'd_ff': 3076,
    'max_len': 128,
    'batch_size': 8
}

ambitious_memory = estimate_model_memory(**ambitious_config)
print(f"Parameters: {ambitious_memory['total_params_millions']:.1f}M")
print(f"Total Memory: {ambitious_memory['total_memory_gb']:.2f} GB")
print("Memory breakdown:")
for component, memory in ambitious_memory['memory_breakdown'].items():
    print(f"  {component}: {memory:.2f} GB")

print("\n" + "="*50)

# Recommended configurations for 32GB RAM
configs = [
    {
        'name': 'Conservative (Safe)',
        'vocab_size': 15000,
        'd_model': 256,
        'n_layers': 6,
        'n_heads': 8,
        'd_ff': 1024,
        'max_len': 128,
        'batch_size': 16
    },
    {
        'name': 'Moderate (Balanced)',
        'vocab_size': 15000,
        'd_model': 384,
        'n_layers': 8,
        'n_heads': 8,
        'd_ff': 1536,
        'max_len': 128,
        'batch_size': 12
    },
    {
        'name': 'Aggressive (Push limits)',
        'vocab_size': 15000,
        'd_model': 512,
        'n_layers': 8,
        'n_heads': 8,
        'd_ff': 2048,
        'max_len': 128,
        'batch_size': 8
    },
    {
        'name': 'Maximum (Risky)',
        'vocab_size': 15000,
        'd_model': 512,
        'n_layers': 10,
        'n_heads': 8,
        'd_ff': 2048,
        'max_len': 128,
        'batch_size': 6
    }
]

for config in configs:
    print(f"\n=== {config['name'].upper()} ===")
    
    # Extract only the parameters needed for the function (exclude 'name')
    config_params = {k: v for k, v in config.items() if k != 'name'}
    
    memory_est = estimate_model_memory(**config_params)
    print(f"d_model={config['d_model']}, n_layers={config['n_layers']}, "
          f"n_heads={config['n_heads']}, d_ff={config['d_ff']}")
    print(f"Parameters: {memory_est['total_params_millions']:.1f}M")
    print(f"Total Memory: {memory_est['total_memory_gb']:.2f} GB")
    
    # Safety assessment
    if memory_est['total_memory_gb'] < 20:
        safety = "✅ SAFE"
    elif memory_est['total_memory_gb'] < 25:
        safety = "⚠️  MODERATE RISK"
    elif memory_est['total_memory_gb'] < 30:
        safety = "⚠️  HIGH RISK"
    else:
        safety = "❌ LIKELY TO FAIL"
    
    print(f"Safety for 32GB RAM: {safety}")

print("\n" + "="*50)
print("RECOMMENDATIONS:")
print("1. For 13,000 texts, a 30-50M parameter model should be sufficient")
print("2. Consider the 'Moderate' or 'Aggressive' configurations")
print("3. You can increase max_len to 256 if needed (will increase memory)")
print("4. Monitor memory usage during training and adjust batch_size down if needed")
print("5. Use gradient accumulation if you need effective larger batch sizes")

# Calculate corpus utilization
print(f"\nCORPUS ANALYSIS:")
print(f"- 13,000 texts is a good corpus size for these model sizes")
print(f"- Your current tiny model (32 dim, 1 layer) is severely under-parameterized")
print(f"- Moving to 256-512 d_model will dramatically improve capacity")

print(f"\nSURPRISE FINDING:")
print(f"- Your original ambitious config (768/12/12/3076) only uses ~1.86GB!")
print(f"- This is TOTALLY FEASIBLE with 32GB RAM")
print(f"- You could probably run an even BIGGER model if you wanted")

# Let's test an even bigger model
print(f"\n=== SUPER AMBITIOUS (EXPERIMENTAL) ===")
super_config = {
    'vocab_size': 15000,
    'd_model': 1024,
    'n_layers': 16,
    'n_heads': 16,
    'd_ff': 4096,
    'max_len': 256,
    'batch_size': 4
}

super_memory = estimate_model_memory(**super_config)
print(f"d_model={super_config['d_model']}, n_layers={super_config['n_layers']}, "
      f"n_heads={super_config['n_heads']}, d_ff={super_config['d_ff']}")
print(f"Parameters: {super_memory['total_params_millions']:.1f}M")
print(f"Total Memory: {super_memory['total_memory_gb']:.2f} GB")

if super_memory['total_memory_gb'] < 15:
    print("✅ Even this HUGE model might work!")
else:
    print("❌ This one might be too big")