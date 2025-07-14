# STREAMING ARCHITECTURE FOR 13,000+ TEXTS

import os
import random
import gc
from pathlib import Path
import numpy as np
import mlx.core as mx

class StreamingDataLoader:
    """Efficiently stream large corpus without loading everything into memory"""
    
    def __init__(self, file_paths, tokenizer, batch_size=64, max_length=256, shuffle=True):
        self.file_paths = list(file_paths)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.current_epoch = 0
        
        print(f"StreamingDataLoader initialized:")
        print(f"  - {len(self.file_paths)} files")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Max length: {max_length}")
    
    def process_file_to_sequences(self, file_path, target_sequences=20):
        """Process a single file into multiple training sequences"""
        try:
            # Your existing file processing pipeline
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            # Apply your cleaning pipeline
            cleaned_text = self.clean_text_pipeline(raw_text, file_path)
            if not cleaned_text or len(cleaned_text) < 100:
                return []
            
            # Split into training sequences
            sequences = self.split_into_sequences(cleaned_text, target_sequences)
            return sequences
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
    
    def clean_text_pipeline(self, raw_text, filename):
        """Your existing cleaning pipeline"""
        # Use your existing functions
        stripped_text = simple_strip_headers(raw_text, filename=filename)
        if not stripped_text:
            return ""
        preprocessed_text = preprocess_text(stripped_text)
        cleaned_text = enhanced_clean_text(preprocessed_text)
        return cleaned_text
    
    def split_into_sequences(self, text, target_sequences):
        """Split text into multiple training sequences"""
        from nltk import sent_tokenize
        
        sentences = sent_tokenize(text)
        sequences = []
        current_seq = ""
        target_length = len(text) // target_sequences
        
        for sentence in sentences:
            if len(current_seq) + len(sentence) < target_length:
                current_seq += " " + sentence
            else:
                if current_seq.strip():
                    sequences.append(current_seq.strip())
                current_seq = sentence
        
        # Add final sequence
        if current_seq.strip():
            sequences.append(current_seq.strip())
        
        return sequences[:target_sequences]  # Limit number of sequences per file
    
    def get_batch_iterator(self, steps_per_epoch=1000):
        """Generate batches for training"""
        
        for step in range(steps_per_epoch):
            # Sample random files for this batch
            batch_files = random.sample(self.file_paths, min(self.batch_size // 5, len(self.file_paths)))
            
            batch_texts = []
            for file_path in batch_files:
                sequences = self.process_file_to_sequences(file_path, target_sequences=5)
                batch_texts.extend(sequences)
            
            # If we don't have enough texts, pad with more files
            while len(batch_texts) < self.batch_size and len(self.file_paths) > len(batch_files):
                extra_file = random.choice(self.file_paths)
                if extra_file not in batch_files:
                    extra_sequences = self.process_file_to_sequences(extra_file, target_sequences=3)
                    batch_texts.extend(extra_sequences)
            
            # Tokenize batch
            if batch_texts:
                try:
                    batch_tokens = self.tokenize_batch(batch_texts[:self.batch_size])
                    if batch_tokens is not None:
                        yield batch_tokens, step
                except Exception as e:
                    print(f"Batch tokenization failed at step {step}: {e}")
                    continue
            
            # Cleanup
            del batch_texts
            if step % 10 == 0:
                gc.collect()
    
    def tokenize_batch(self, texts):
        """Tokenize a batch of texts"""
        try:
            batch_inputs = self.tokenizer(
                texts,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )["input_ids"]
            
            # Filter out heavily padded sequences
            non_pad_counts = np.sum(batch_inputs != self.tokenizer.pad_token_id, axis=1)
            valid_mask = non_pad_counts > self.max_length // 8  # Keep sequences with >32 real tokens
            
            if not np.any(valid_mask):
                return None
            
            batch_inputs = batch_inputs[valid_mask]
            return mx.array(batch_inputs, dtype=mx.int32)
            
        except Exception as e:
            print(f"Tokenization error: {e}")
            return None

def train_streaming_model(model, tokenizer, file_paths, optimizer, model_dir, config):
    """Training loop for streaming data"""
    
    # Create streaming data loader
    data_loader = StreamingDataLoader(
        file_paths=file_paths,
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length']
    )
    
    # Training configuration
    steps_per_epoch = config.get('steps_per_epoch', 1000)
    num_epochs = config.get('num_epochs', 3)
    save_every = config.get('save_every', 200)
    
    print(f"Starting streaming training:")
    print(f"  - {len(file_paths)} total files")
    print(f"  - {steps_per_epoch} steps per epoch")
    print(f"  - {num_epochs} epochs")
    print(f"  - Batch size: {config['batch_size']}")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch + 1}/{num_epochs} ===")
        
        epoch_losses = []
        batch_iterator = data_loader.get_batch_iterator(steps_per_epoch)
        
        for batch_tokens, step in batch_iterator:
            try:
                # Training step
                loss, grads, scale = dynamic_loss_scale(model, batch_tokens, loss_fn_lm)
                
                if loss is not None:
                    grads = clip_gradients(grads, max_norm=1.0)
                    optimizer.update(model, grads)
                    mx.eval(model.parameters())
                    
                    epoch_losses.append(loss.item())
                    
                    # Progress reporting
                    if step % 100 == 0:
                        avg_loss = np.mean(epoch_losses[-100:]) if epoch_losses else 0
                        print(f"Epoch {epoch + 1}, Step {step}/{steps_per_epoch}, "
                              f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
                        
                        # Memory monitoring
                        if step % 200 == 0:
                            process = psutil.Process(os.getpid())
                            memory_gb = process.memory_info().rss / (1024**3)
                            print(f"Memory usage: {memory_gb:.1f}GB")
                
                # Save checkpoint
                if global_step % save_every == 0 and global_step > 0:
                    save_checkpoint(model, optimizer, epoch, model_dir)
                    print(f"💾 Saved checkpoint at step {global_step}")
                
                global_step += 1
                
            except Exception as e:
                print(f"Training step failed: {e}")
                continue
        
        # End of epoch summary
        if epoch_losses:
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # Save epoch checkpoint
        save_checkpoint(model, optimizer, epoch, model_dir)
    
    print("🎉 Streaming training complete!")
    return model

# USAGE EXAMPLE FOR 13K FILES

def setup_13k_training():
    """Setup for training on 13,000 scraped texts"""
    
    # Configuration for large corpus
    config = {
        'batch_size': 64,           # Optimize based on your GPU memory
        'max_length': 256,          # Sequence length
        'steps_per_epoch': 2000,    # More steps for larger corpus  
        'num_epochs': 3,            # Fewer epochs since more data
        'save_every': 500,          # Save more frequently
        'd_model': 768,             # Larger model for more data
        'n_layers': 12,             # More layers
        'n_heads': 12,              # More attention heads
        'd_ff': 3072,               # Larger feed-forward
    }
    
    # File paths to your 13K scraped texts
    scraped_files_dir = "/path/to/your/13k/scraped/texts"
    file_paths = list(Path(scraped_files_dir).glob("*.txt"))  # Adjust pattern as needed
    
    print(f"Found {len(file_paths)} scraped text files")
    
    if len(file_paths) < 1000:
        print("⚠️  Warning: Expected ~13K files, found fewer. Check file path and pattern.")
    
    return config, file_paths

# INTEGRATION WITH YOUR EXISTING CODE

def main_13k_training():
    """Main training function for 13K corpus"""
    
    # Setup
    config, file_paths = setup_13k_training()
    
    # Initialize tokenizer (your existing code)
    tokenizer = setup_tokenizer()
    
    # Initialize larger model for bigger corpus
    vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
    model = BabyLLM(
        vocab_size=vocab_size_with_special,
        d_model=config['d_model'],      # 768 instead of 512
        n_layers=config['n_layers'],    # 12 instead of 8  
        n_heads=config['n_heads'],      # 12 instead of 8
        d_ff=config['d_ff'],            # 3072 instead of 2048
        max_len=config['max_length'],
        pad_token_id=tokenizer.pad_token_id
    )
    
    optimizer = optim.Adam(learning_rate=2e-4)  # Slightly higher LR for larger model
    
    # Load checkpoint if exists
    model, optimizer_state, start_epoch = load_checkpoint(model, optimizer, pretrain_checkpoint_path, model_dir)
    optimizer.state = optimizer_state
    
    print(f"🚀 Starting training on {len(file_paths)} files")
    print(f"Model size: {config['d_model']}d, {config['n_layers']} layers")
    
    # Train with streaming
    trained_model = train_streaming_model(
        model=model,
        tokenizer=tokenizer, 
        file_paths=file_paths,
        optimizer=optimizer,
        model_dir=model_dir,
        config=config
    )
    
    return trained_model

# MEMORY ESTIMATES FOR 13K FILES
"""
Memory usage estimates:

Current (50 files):
- 0.3GB peak, 10K sequences

Projected (13K files with streaming):
- ~2-4GB peak (only one batch in memory at a time)
- ~2.6M total sequences (processed over time, not stored)

Model size increase:
- Current: 512d, 8 layers ≈ 50M parameters ≈ 200MB  
- Proposed: 768d, 12 layers ≈ 150M parameters ≈ 600MB

Total projected: ~5GB peak (very manageable!)
"""