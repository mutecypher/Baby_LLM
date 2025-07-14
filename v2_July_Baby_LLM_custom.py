## run with python 3.12
## worked with numpy 2.3.0
## worked with PyTorch 2.6.0
## worked with Transformers 4.44.0

import os
import re
import logging
import gc
from itertools import islice
import shutil
from pathlib import Path
from hashlib import md5
import psutil
import random
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import torch
import transformers
from transformers import BertForMaskedLM, AutoTokenizer, MarianMTModel, MarianTokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer
from nltk.corpus import wordnet
import nltk

import matplotlib.pyplot as plt
from pathlib import Path
import ssl
import string
from collections import Counter
import time
from concurrent.futures import ProcessPoolExecutor
from nltk import sent_tokenize, word_tokenize, pos_tag

from transformers import BertForMaskedLM
import glob
import string   
from functools import partial 
from secrets import HF_TOKEN, BERT_Token
# Fix NLTK SSL issue and download required data

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at bert-base-uncased were not used")

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)



try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    logging.info("Downloading NLTK resources...")
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


import logging.handlers

# Setup logging
##logging.basicConfig(
##    level=logging.DEBUG,
##    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s'
##)


# Compiled regex patterns for cleaning
# Compiled regex patterns for cleaning
patterns = [
    (re.compile(r'\n{3,}', re.MULTILINE), '\n\n'),  # For enhanced_clean_text
]

fallback_metadata_patterns = [
    re.compile(r'^\s*\*{3}\s*START OF THIS PROJECT GUTENBERG EBOOK.*$', re.IGNORECASE),
    re.compile(r'^\s*\*{3}\s*END OF THIS PROJECT GUTENBERG EBOOK.*$', re.IGNORECASE),
    re.compile(r'^\s*\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK.*$', re.IGNORECASE),
    re.compile(r'^\s*\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK.*$', re.IGNORECASE),
    re.compile(r'Produced by.*?\n', re.IGNORECASE),
    re.compile(r'This eBook is for the use of.*?\n', re.IGNORECASE),
]

# Compiled patterns for preprocess_text
preprocess_patterns = [
    re.compile(r'[^\x00-\x7F]+'),  # Non-ASCII characters
    re.compile(r'\s+'),  # Multiple whitespace
]


base_dir = "~/Baby_LLM"
alt_dir = "~/Baby_LLM" ##"/Volumes/Phials4Miles/GitHub/Baby_dat"
cache_dir = os.path.expanduser(os.path.join(alt_dir, "cache"))
model_dir = os.path.expanduser(os.path.join(alt_dir, "model"))
data_dir = os.path.expanduser(os.path.join(base_dir, "data"))
gutenberg_dir = os.path.join(data_dir, "gutenberg")
cleaned_dir = os.path.join(alt_dir, "cleaned")
tokenized_dir = os.path.join(alt_dir, "tokenized")
os.makedirs(cleaned_dir, exist_ok=True)
os.makedirs(tokenized_dir, exist_ok=True)
pretrain_checkpoint_path = os.path.join(model_dir, "pretrain_checkpoint.npz")

# Define log_listener at module level with error handling
def log_listener(queue, log_file):
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s - %(message)s'))
    while True:
        try:
            record = queue.get(timeout=10)
            if record is None:  # Sentinel to stop listener
                break
            file_handler.handle(record)
        except multiprocessing.queues.Empty:
            logging.debug("Log queue timeout, continuing...")
            continue
        except queue.full:
            print("Warning: Log queue full, some logs may be dropped", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Log listener error: {e}", file=sys.stderr)
            continue
    file_handler.close()

def preprocess_text(text):
    ##logging.debug(f"Preprocessing text: length={len(text)}, sample={text[:200]}")
    # Apply compiled regex patterns
    for pattern in preprocess_patterns:
        text = pattern.sub(' ', text)
    # Additional cleaning for Gutenberg artifacts
    text = re.sub(r'\[\d+\]', '', text)  # Remove footnote markers
    text = re.sub(r'CHAPTER [IVXLC]+', 'CHAPTER', text, flags=re.IGNORECASE)  # Normalize chapter headings
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    # Filter allowed characters
    allowed_chars = set(string.ascii_letters + string.digits + '.,?!\'"-;:() ')
    trans_table = str.maketrans('', '', ''.join(chr(i) for i in range(128) if chr(i) not in allowed_chars))
    filtered_text = text.translate(trans_table)
    cleaned_text = filtered_text.strip()
    ##logging.debug(f"Preprocessed text: length={len(cleaned_text)}, sample={cleaned_text[:200]}")
    return cleaned_text

def enhanced_clean_text(raw_text):
    ##logging.debug(f"Raw text length: {len(raw_text)}, sample: {raw_text[:200]}")
    text = raw_text
    for pattern, repl in patterns:
        matches = pattern.findall(text)
        if matches:
            logging.debug(f"Pattern {pattern.pattern} matched {len(matches)} times, sample: {matches[:5]}")
        text = pattern.sub(repl, text)
    ##logging.debug(f"Final cleaned text: length={len(text)}, sample={text[:200]}")
    return text.strip()

def simple_strip_headers(text, include_dedication=True, filename="unknown"):
    lines = text.splitlines()
    total_lines = len(lines)
    logging.debug(f"Total lines in text: {total_lines} for file {filename}")

    first_gutenberg_idx = -1
    for i, line in enumerate(lines):
        if "gutenberg" in line.lower() or any(p.search(line) for p in fallback_metadata_patterns):
            first_gutenberg_idx = i
            logging.debug(f"Found start marker at line {i}: {line[:100]}")
            break
    if first_gutenberg_idx == -1:
        logging.warning(f"No 'Gutenberg' or metadata markers found in {filename}; using fallback processing")
        logging.debug(f"Sample lines from {filename}: {lines[:10]}")
        start_idx = 0
        for i, line in enumerate(lines[:100]):
            line_lower = line.lower()
            if any(term in line_lower for term in ["produced by", "ebook", "copyright", "transcriber", "prefatory note", "printed by"]):
                start_idx = i + 1
            elif len(line.strip()) > 20 and not re.match(r'^\s*(title|author|by|edited by|translated by)', line_lower):
                break
        end_idx = total_lines
        for i in range(len(lines) - 1, -1, -1):
            if re.search(r'\*{3}\s*END OF.*GUTENBERG EBOOK', lines[i], re.IGNORECASE):
                end_idx = i
                logging.debug(f"Found end marker at line {i}: {lines[i][:100]}")
                break
    else:
        start_idx = min(first_gutenberg_idx + 10 + 1, total_lines)
        last_gutenberg_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if "gutenberg" in lines[i].lower() or any(p.search(line) for p in fallback_metadata_patterns):
                last_gutenberg_idx = i
                logging.debug(f"Found end marker at line {i}: {lines[i][:100]}")
                break
        if last_gutenberg_idx == -1 or last_gutenberg_idx <= start_idx + 10:
            end_idx = total_lines
        else:
            end_idx = max(last_gutenberg_idx - 10, start_idx)

    stripped_lines = lines[start_idx:end_idx]
    logging.debug(f"Stripped lines count for {filename}: {len(stripped_lines)}")
    cleaned_lines = [line for line in stripped_lines if len(line.strip()) > 10]
    stripped_text = "\n".join(cleaned_lines).strip()
    if not stripped_text:
        logging.warning(f"Empty output from simple_strip_headers for {filename}; using text after start_idx")
        stripped_lines = lines[start_idx:total_lines]
        cleaned_lines = [line for line in stripped_lines if len(line.strip()) > 5]
        stripped_text = "\n".join(cleaned_lines).strip()
        if not stripped_text:
            logging.error(f"Still no content for {filename}; skipping")
            return ""
    return stripped_text
    
def process_file(filename, tokenizer):
    file_path = os.path.join(gutenberg_dir, filename)
    # REMOVED: cleaned_file_path and caching to disk
    logging.debug(f"Processing file: {filename}")
    
    if not os.path.isfile(file_path):
        logging.warning(f"File not found: {filename}")
        return ""
    if os.path.getsize(file_path) > 10 * 1024 * 1024:
        logging.info(f"Skipping {filename}: File too large (>10MB)")
        return ""
    
    # REMOVED: Check for existing cleaned file - process in memory only
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        logging.info(f"Raw text length for {filename}: {len(raw_text)}")
        stripped_text = simple_strip_headers(raw_text, include_dedication=True, filename=filename)
        logging.info(f"Stripped text length: {len(stripped_text)}")
        preprocessed_text = preprocess_text(stripped_text)
        logging.info(f"Preprocessed text length: {len(preprocessed_text)}")
        cleaned_text = enhanced_clean_text(preprocessed_text)
        logging.info(f"Cleaned text length: {len(cleaned_text)}")
        
        if not cleaned_text.strip():
            logging.warning(f"Empty cleaned text for {filename}")
            return ""
        
        if len(cleaned_text) > 10:
            # REMOVED: Save to disk - return directly
            logging.info(f"Processed file: {filename}")
            return cleaned_text + "\n\n"
        else:
            logging.info(f"Skipping {filename}: Too short (length={len(cleaned_text)})")
            return ""
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return ""
    

def is_narrative(text):
    if not isinstance(text, str) or not text.strip():
        logging.warning("Invalid input to is_narrative: empty or non-string")
        return False
    return len(text) > 10

def process_file_batch(filenames, tokenizer, batch_size=32):
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(filenames), batch_size):
            batch = filenames[i:i + batch_size]
            future_to_file = {executor.submit(process_file, fname, tokenizer): fname for fname in batch}
            for future in future_to_file:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logging.error(f"Error processing {future_to_file[future]}: {str(e)}")
            gc.collect()
            log_memory_usage()
    return results

def safe_remove(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.debug(f"Removed file: {file_path}")
        else:
            logging.debug(f"File not found for removal: {file_path}")
    except Exception as e:
        logging.warning(f"Failed to remove file {file_path}: {str(e)}")
        
# REPLACE your existing load_or_tokenize_texts function with this:

def load_or_tokenize_texts(texts, tokenizer, output_dir, prefix, batch_size=100, max_length=256):
    """
    Memory-only version - ignores output_dir and prefix parameters for compatibility
    but processes everything in memory to save disk space
    """
    # Note: output_dir and prefix are ignored but kept for compatibility
    inputs = []
    vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
    
    # Validate input texts
    logging.info(f"Input texts: total={len(texts)}, sample={[t[:50] + '...' for t in texts[:5]]}")
    valid_texts = [t for t in texts if t.strip() and len(t.strip()) >= 10]
    logging.info(f"Valid texts after initial filtering: total={len(valid_texts)}, sample={[t[:50] + '...' for t in valid_texts[:5]]}")
    
    if not valid_texts:
        logging.error("No valid texts provided for tokenization after initial filtering")
        raise ValueError("No valid texts provided for tokenization")
    
    # Sort texts by token length
    text_lengths = []
    for t in valid_texts:
        try:
            token_ids = tokenizer(t, truncation=True, add_special_tokens=True)["input_ids"]
            length = len(token_ids)
            if length < 5 or length > max_length * 10:  # Cap at 10x max_length (2560)
                logging.warning(f"Skipping text with invalid token length: length={length}, text='{t[:50]}...'")
                continue
            text_lengths.append((t, length))
        except Exception as e:
            logging.warning(f"Failed to tokenize text for length check: {str(e)}, text='{t[:50]}...'")
            continue
    
    text_lengths.sort(key=lambda x: x[1])
    valid_texts = [t for t, length in text_lengths]
    
    if not valid_texts:
        logging.error("No texts with valid token length after filtering")
        raise ValueError("No texts with valid token length")
    
    # Process in batches (in memory only)
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i+batch_size]
        if not batch:
            continue
        
        # Tokenize batch in memory (no file saving/loading)
        try:
            batch_inputs = tokenizer(
                batch,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )["input_ids"]
            
            non_pad_counts = np.sum(batch_inputs != tokenizer.pad_token_id, axis=1)
            valid_mask = non_pad_counts > max_length // 4   # Relaxed threshold
            if not np.any(valid_mask):
                logging.warning(f"Batch {i//batch_size} has excessive padding, non_pad_counts={non_pad_counts.tolist()}")
                continue
            batch_inputs = batch_inputs[valid_mask]
            if np.any(batch_inputs < 0) or np.any(batch_inputs >= vocab_size_with_special):
                logging.error(f"Invalid tokens in batch at index {i}: min={np.min(batch_inputs)}, max={np.max(batch_inputs)}")
                continue
            
            inputs.append(batch_inputs)
            
        except Exception as e:
            logging.error(f"Tokenization failed for batch at index {i}: {str(e)}")
            continue
    
    if not inputs:
        logging.error("No valid batches tokenized")
        raise ValueError("No valid batches tokenized")
    
    input_ids = mx.array(np.concatenate(inputs, axis=0), dtype=mx.int32)
    logging.info(f"Tokenized input_ids: shape={input_ids.shape}, min={mx.min(input_ids).item()}, max={mx.max(input_ids).item()}")
    return input_ids


# Custom learning rate scheduler
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


def flatten_parameters(params, prefix=""):
    flat_params = {}
    for key, value in params.items():
        new_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, mx.array):
            try:
                # Ensure float16 and check for NaN/Inf
                mx_value = value.astype(mx.float16)
                if mx.any(mx.isnan(mx_value)) or mx.any(mx.isinf(mx_value)):
                    logging.error(f"NaN/Inf detected in parameter {new_key}")
                    raise ValueError(f"NaN/Inf in parameter {new_key}")
                flat_params[new_key] = mx_value
            except Exception as e:
                logging.error(f"Failed to process parameter {new_key}: {str(e)}")
                raise
        elif isinstance(value, dict):
            flat_params.update(flatten_parameters(value, f"{new_key}."))
        elif isinstance(value, list) and key == "layers":
            for i, layer in enumerate(value):
                if isinstance(layer, dict):
                    layer_params = flatten_parameters(layer, f"{new_key}.{i}.")
                    flat_params.update(layer_params)
                else:
                    logging.warning(f"Skipping unexpected layer {new_key}.{i}: type={type(layer)}")
        else:
            logging.warning(f"Skipping non-array parameter {new_key}: type={type(value)}")
    return flat_params


# REPLACE THE unflatten_parameters FUNCTION (around line 510) WITH THIS FIXED VERSION:

def unflatten_parameters(flat_params, model):
    """More robust parameter unflattening with better error handling"""
    params = model.parameters()
    
    # Debug: Print some info about what we're trying to load
    print(f"📊 Loading {len(flat_params)} parameters")
    sample_keys = list(flat_params.keys())[:5]
    print(f"📊 Sample parameter keys: {sample_keys}")
    
    successful_loads = 0
    failed_loads = 0
    
    for key, value in flat_params.items():
        try:
            keys = key.split('.')
            current = params
            
            # Navigate through the parameter structure
            for i, k in enumerate(keys[:-1]):
                if k.isdigit():
                    # This is a layer index
                    layer_idx = int(k)
                    if 'layers' in current and isinstance(current['layers'], list):
                        if layer_idx < len(current['layers']):
                            current = current['layers'][layer_idx]
                        else:
                            print(f"⚠️  Layer index {layer_idx} out of range for {key}")
                            break
                    else:
                        print(f"⚠️  Expected layers list but found {type(current.get('layers', 'missing'))} for {key}")
                        break
                else:
                    # Regular parameter name
                    if isinstance(current, dict) and k in current:
                        current = current[k]
                    else:
                        print(f"⚠️  Key '{k}' not found in current parameters for {key}")
                        break
            else:
                # Successfully navigated to the parent, now set the final parameter
                final_key = keys[-1]
                if isinstance(current, dict) and final_key in current:
                    # Check shape compatibility
                    if hasattr(current[final_key], 'shape') and hasattr(value, 'shape'):
                        if current[final_key].shape == value.shape:
                            current[final_key] = value.astype(mx.float16)
                            successful_loads += 1
                        else:
                            print(f"⚠️  Shape mismatch for {key}: expected {current[final_key].shape}, got {value.shape}")
                            failed_loads += 1
                    else:
                        current[final_key] = value.astype(mx.float16)
                        successful_loads += 1
                else:
                    print(f"⚠️  Final key '{final_key}' not found for {key}")
                    failed_loads += 1
                    
        except Exception as e:
            print(f"⚠️  Error loading parameter {key}: {str(e)}")
            failed_loads += 1
            continue
    
    print(f"📊 Parameter loading: {successful_loads} successful, {failed_loads} failed")
    
    if successful_loads == 0:
        raise ValueError("No parameters were successfully loaded! Model architecture may have changed.")
    elif failed_loads > successful_loads:
        print("⚠️  Many parameters failed to load - model may not work correctly")
    
    return params

# ALTERNATIVE SIMPLER FIX - Replace the clip_parameters function with this safer version:

def clip_parameters_safe(model, max_value=10.0):
    """Safer version that doesn't rely on unflatten_parameters"""
    try:
        # Get current parameters
        params = model.parameters()
        
        # Clip parameters in-place
        for param_dict in [params]:
            for key, value in param_dict.items():
                if isinstance(value, mx.array):
                    param_dict[key] = mx.clip(value, -max_value, max_value)
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    clip_nested_params(value, max_value)
                elif isinstance(value, list) and key == "layers":
                    # Handle layers list
                    for layer in value:
                        if isinstance(layer, dict):
                            clip_nested_params(layer, max_value)
        
        # Update model with clipped parameters
        model.update(params)
        return model
        
    except Exception as e:
        logging.warning(f"Parameter clipping failed: {str(e)}, continuing without clipping")
        return model

def clip_nested_params(param_dict, max_value):
    """Helper function to clip nested parameter dictionaries"""
    for key, value in param_dict.items():
        if isinstance(value, mx.array):
            param_dict[key] = mx.clip(value, -max_value, max_value)
        elif isinstance(value, dict):
            clip_nested_params(value, max_value)

# UPDATE THE TRAINING LOOP TO USE THE SAFER VERSION:
# Change this line in your training loop:
# generative_model = clip_parameters(generative_model, max_value=10.0)
# To this:
# generative_model = clip_parameters_safe(generative_model, max_value=10.0)

# REPLACE THE save_checkpoint FUNCTION (around line 575) WITH THIS FIXED VERSION:

def save_checkpoint(model, optimizer, epoch, model_dir, keep_checkpoints=1):
    """Save only the most recent checkpoint(s) to save disk space"""
    checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch}.npz")
    
    try:
        params_to_save = flatten_parameters(model.parameters())
        mx.savez(checkpoint_path, **params_to_save)
        logging.info(f"Saved model checkpoint: {checkpoint_path}")
        print(f"✅ Saved checkpoint for epoch {epoch}")
        
        # Clean up old checkpoints - keep only the most recent ones
        checkpoint_files = sorted(glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.npz")))
        if len(checkpoint_files) > keep_checkpoints:
            for old_file in checkpoint_files[:-keep_checkpoints]:
                safe_remove(old_file)
                logging.info(f"Removed old checkpoint: {old_file}")
                
    except Exception as e:
        logging.error(f"Failed to save checkpoint for epoch {epoch}: {str(e)}")
        print(f"⚠️  Checkpoint save failed for epoch {epoch}, continuing training...")




def xavier_uniform(shape, dtype=mx.float32, scale=1.0):
    fan_in, fan_out = shape[-2], shape[-1]
    limit = mx.sqrt(6.0 / (fan_in + fan_out)) * scale  # Add scale factor
    return mx.random.uniform(-limit, limit, shape=shape, dtype=dtype)

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)
        
        # MLX nn.Linear expects weights in shape (input_features, output_features)
        # The nn.Linear layer will handle the transpose internally
        self.linear1.weight = xavier_uniform((d_hidden, d_in), dtype=mx.float32)  # Changed order
        self.linear1.bias = mx.zeros((d_hidden,), dtype=mx.float32)
        self.linear2.weight = xavier_uniform((d_out, d_hidden), dtype=mx.float32)  # Changed order
        self.linear2.bias = mx.zeros((d_out,), dtype=mx.float32)
        
        # Validate initialization
        for name, param in self.parameters().items():
            if isinstance(param, mx.array):
                if mx.any(mx.isnan(param)) or mx.any(mx.isinf(param)):
                    logging.error(f"NaN/Inf in initialized FeedForward parameter {name}")
                    raise ValueError(f"Invalid initialization for FeedForward {name}")
            else:
                logging.debug(f"Skipping non-array parameter {name}: type={type(param)}")

    def __call__(self, x):
        if not isinstance(x, mx.array):
            logging.error(f"FeedForward input is not mx.array: type={type(x)}")
            raise ValueError("FeedForward input must be mx.array")
        logging.debug(f"FeedForward input shape: {x.shape}, min={mx.min(x).item()}, max={mx.max(x).item()}")
        
        x = x.astype(mx.float32)
        x = self.linear1(x)
        logging.debug(f"After linear1 shape: {x.shape}, min={mx.min(x).item()}, max={mx.max(x).item()}")
        x = nn.gelu(x)
        x = mx.clip(x, -1e2, 1e2)
        logging.debug(f"After GELU shape: {x.shape}, min={mx.min(x).item()}, max={mx.max(x).item()}")
        x = self.linear2(x)
        logging.debug(f"After linear2 shape: {x.shape}, min={mx.min(x).item()}, max={mx.max(x).item()}")
        x = mx.clip(x, -1e2, 1e2)
        
        if x.ndim < 2:
            logging.error(f"FeedForward output is scalar or 1D: shape={x.shape}")
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
        attention_params = self.attention.parameters()
        for key, param in attention_params.items():
            if isinstance(param, mx.array) and 'weight' in key:
                if param.ndim == 2:
                    out_features, in_features = param.shape
                    self.attention.parameters()[key] = xavier_uniform((out_features, in_features), dtype=mx.float32)
            elif isinstance(param, mx.array) and 'bias' in key:
                self.attention.parameters()[key] = mx.zeros(param.shape, dtype=mx.float32)

    def __call__(self, x, mask=None):
        if not isinstance(x, mx.array):
            logging.error(f"TransformerLayer input is not mx.array: type={type(x)}")
            raise ValueError("TransformerLayer input must be mx.array")
        
        # FIXED: Check mask shape properly
        if mask is not None:
            expected_shape = (x.shape[0], self.attention.num_heads, x.shape[1], x.shape[1])
            if mask.shape != expected_shape:
                logging.error(f"Mask shape {mask.shape} does not match expected {expected_shape}")
                logging.error(f"Input shape: {x.shape}, num_heads: {self.attention.num_heads}")
                raise ValueError("Mask shape mismatch")

        x = x.astype(mx.float32)
        try:
            # Pass x as queries, keys, and values for self-attention
            attn_output = self.attention(queries=x, keys=x, values=x, mask=mask)
            attn_output = mx.clip(attn_output, -1e2, 1e2)
            attn_output = self.dropout(attn_output)
        except Exception as e:
            logging.error(f"Attention computation failed: {str(e)}")
            if mask is not None:
                logging.error(f"Input shape: {x.shape}, Mask shape: {mask.shape}")
            raise

        x = self.norm1(x + attn_output)
        x = mx.clip(x, -1e2, 1e2)
        ff_output = self.dropout(self.ff(x))
        x = self.norm2(x + ff_output)
        x = mx.clip(x, -1e2, 1e2)
        return x.astype(mx.float32)
    

class BabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id 
        self.d_model = d_model
        self.n_heads = n_heads  # Store n_heads for mask reshaping
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = [TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.final_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.output = nn.Linear(d_model, vocab_size)
        self._debug_counter = 0
        
        # Initialize weights (same as before)
        self.embedding.weight = xavier_uniform((vocab_size, d_model), dtype=mx.float32)
        self.pos_embedding.weight = xavier_uniform((max_len, d_model), dtype=mx.float32)
        self.output.weight = xavier_uniform((vocab_size, d_model), dtype=mx.float32)
        self.output.bias = mx.zeros((vocab_size,), dtype=mx.float32)
        
        # Validate initialization
        for name, param in self.parameters().items():
            if isinstance(param, mx.array):
                if mx.any(mx.isnan(param)) or mx.any(mx.isinf(param)):
                    logging.error(f"NaN/Inf in initialized parameter {name}")
                    raise ValueError(f"Invalid initialization for {name}")

    def to_numpy_for_decode(self, array):
        """Convert MLX array to NumPy array for tokenizer decoding."""
        return np.array(array) if isinstance(array, mx.array) else array

    def __call__(self, x):
        self._debug_counter += 1
        
        if not isinstance(x, mx.array):
            logging.error(f"Input is not mx.array: type={type(x)}")
            return None
        if x.ndim != 2:
            logging.error(f"Input is not 2D: shape={x.shape}")
            return None
        
        vocab_size_with_special = self.embedding.weight.shape[0]
        if mx.any(x < 0) or mx.any(x >= vocab_size_with_special):
            logging.error(f"Invalid tokens: min={mx.min(x).item()}, max={mx.max(x).item()}, vocab_size={vocab_size_with_special}")
            return None
        
        # Get dimensions
        batch_size, seq_len = x.shape
        
        # FIXED MASK CREATION - Create 4D mask for multi-head attention
        # MLX MultiHeadAttention expects mask shape: (batch_size, n_heads, seq_len, seq_len)
        
        # 1. Create causal mask (lower triangular)
        causal_mask = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=1)
        # Expand to (batch_size, n_heads, seq_len, seq_len)
        causal_mask = mx.broadcast_to(
            causal_mask[None, None, :, :], 
            (batch_size, self.n_heads, seq_len, seq_len)
        )
        
        # 2. Create padding mask
        padding_mask = (x == self.pad_token_id)  # Shape: (batch_size, seq_len)
        # Expand to (batch_size, n_heads, seq_len, seq_len)
        # Each padded position should mask all positions in both dimensions
        padding_mask = mx.broadcast_to(
            padding_mask[:, None, None, :], 
            (batch_size, self.n_heads, seq_len, seq_len)
        )
        
        # 3. Combine masks
        combined_mask = mx.logical_or(causal_mask, padding_mask)
        
        # Optional: Log mask shapes only occasionally
        if self._debug_counter % 1000 == 0:
            logging.info(f"Mask shapes - causal: {causal_mask.shape}, padding: {padding_mask.shape}, combined: {combined_mask.shape}")
        
        # Embedding
        x = self.embedding(x).astype(mx.float32)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf in embedding output")
            return None
        
        # Positional embedding
        if seq_len > self.pos_embedding.weight.shape[0]:
            logging.error(f"Sequence length {seq_len} exceeds max_len {self.pos_embedding.weight.shape[0]}")
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
                logging.error(f"NaN/Inf in layer {i} output")
                return None
        
        # Final norm and output
        x = self.final_norm(x)
        x = self.output(x)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf in final output")
            return None
        return x.astype(mx.float32)
    

def validate_tokens(input_ids, vocab_size, model):
    if not isinstance(input_ids, mx.array):
        logging.error(f"validate_tokens input is not mx.array: type={type(input_ids)}")
        raise ValueError("Input to validate_tokens must be mx.array")
    invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
    if mx.any(invalid_mask):
        invalid_indices = mx.where(invalid_mask)[0]
        invalid_samples = [model.to_numpy_for_decode(input_ids[i]) for i in invalid_indices[:5]]
        decoded_samples = [tokenizer.decode(sample) for sample in invalid_samples]
        logging.error(f"Invalid tokens at indices: {invalid_indices.tolist()[:10]}, samples: {decoded_samples}")
        ##np.save(os.path.join(model_dir, f"invalid_tokens_{time.time()}.npy"), np.array(input_ids))
        raise ValueError("Invalid tokens detected")
    logging.debug(f"Validated tokens: shape={input_ids.shape}, min={mx.min(input_ids).item()}, max={mx.max(input_ids).item()}")
    return input_ids

def scale_gradients(grads, scale):
    if isinstance(grads, mx.array):
        return grads.astype(mx.float16) * scale
    elif isinstance(grads, dict):
        return {k: scale_gradients(v, scale) for k, v in grads.items() if isinstance(v, (mx.array, dict))}
    else:
        logging.error(f"Invalid gradient type in scale_gradients: {type(grads)}")
        raise TypeError(f"Invalid gradient type: {type(grads)}")

def add_grads(acc, new):
    if acc is None:
        return {k: v.copy() for k, v in new.items()} if isinstance(new, dict) else new.copy()
    if isinstance(acc, mx.array) and isinstance(new, mx.array):
        return acc + new
    elif isinstance(acc, dict) and isinstance(new, dict):
        result = {}
        for k in acc:
            if k not in new:
                logging.warning(f"Key {k} missing in new gradients, skipping")
                continue
            if isinstance(acc[k], mx.array) and isinstance(new[k], mx.array):
                result[k] = acc[k] + new[k]
            elif isinstance(acc[k], dict) and isinstance(new[k], dict):
                result[k] = add_grads(acc[k], new[k])
            else:
                logging.error(f"Invalid gradient types for key {k}: acc={type(acc[k])}, new={type(new[k])}")
                raise TypeError(f"Invalid gradient types for key {k}")
        return result
    else:
        logging.error(f"Invalid input types: acc={type(acc)}, new={type(new)}")
        raise TypeError("Invalid input types for gradient accumulation")

def clean_answer(text):
    text = re.sub(r'[.!?\n]+$', '', text)
    text = re.sub(r'^\s*Answer:\s*', '', text, flags=re.IGNORECASE)
    return text.strip() or "Unknown"

def cleanup_intermediate_files():
    """Clean up all intermediate files at start"""
    # Clear cleaned files
    if os.path.exists(cleaned_dir):
        shutil.rmtree(cleaned_dir)
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Clear tokenized files  
    if os.path.exists(tokenized_dir):
        shutil.rmtree(tokenized_dir)
    os.makedirs(tokenized_dir, exist_ok=True)
    
    # Clear old debug files
    debug_files = glob.glob(os.path.join(model_dir, "failed_*.npy"))
    debug_files.extend(glob.glob(os.path.join(model_dir, "invalid_*.npy")))
    debug_files.extend(glob.glob(os.path.join(model_dir, "nan_*.npy")))
    for debug_file in debug_files:
        safe_remove(debug_file)
    
    logging.info("Cleaned up intermediate files")
def nucleus_sampling(logits, p=0.7):
    sorted_logits, sorted_indices = mx.sort(logits, axis=-1, descending=True)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
    mask = cumsum_probs <= p
    top_p_indices = sorted_indices[mask]
    top_p_logits = mx.take_along_axis(logits, top_p_indices, axis=-1)
    probs = mx.softmax(top_p_logits, axis=-1)
    next_token_idx = mx.random.categorical(probs.log())
    return top_p_indices[0, next_token_idx].reshape(1, 1)

# Add this after nucleus_sampling
def initialize_bert_pipeline(bert_token, cache_dir, max_retries=3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Selected device for BERT pipeline: {device}")
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Attempt {attempt}/{max_retries} to initialize BERT fill-mask pipeline...")
            bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=bert_token, cache_dir=cache_dir)
            bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased", token=bert_token, cache_dir=cache_dir, ignore_mismatched_sizes=True)
            bert_model.to(device)
            return pipeline("fill-mask", model=bert_model, tokenizer=bert_tokenizer, device=device)
        except Exception as e:
            logging.error(f"Attempt {attempt} failed: {str(e)}")
            if attempt == max_retries:
                logging.warning("All attempts failed, returning None")
                return None
            time.sleep(2)

# Data augmentation functions
def get_synonyms(word, pos_tag):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    try:
        for syn in wordnet.synsets(word, pos=pos_tag):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
    except Exception as e:
        logging.warning(f"Failed to get synonyms for '{word}': {e}")
    return list(synonyms)

def synonym_replacement(question, prob=0.3):
    try:
        words = nltk.word_tokenize(question)
        pos_tags = nltk.pos_tag(words)
        new_words = words.copy()
        for i, (word, pos) in enumerate(pos_tags):
            if random.random() < prob and pos.startswith(('NN', 'VB', 'JJ', 'RB')):
                pos_map = {'NN': 'n', 'VB': 'v', 'JJ': 'a', 'RB': 'r'}
                if pos[:2] in pos_map:
                    synonyms = get_synonyms(word, pos_map[pos[:2]])
                    if synonyms:
                        new_words[i] = random.choice(synonyms)
        return ' '.join(new_words)
    except Exception as e:
        logging.warning(f"Synonym replacement failed: {e}")
        return question


# REPLACE your existing paraphrase_question function with this:

def paraphrase_question(question, max_retries=3):
    """
    Memory-only version - no file caching to save disk space
    """
    # Note: All caching logic removed but function signature stays the same
    
    for attempt in range(1, max_retries + 1):
        try:
            t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=cache_dir, token=HF_TOKEN)
            t5_model = T5ForConditionalGeneration.from_pretrained("t5-small", cache_dir=cache_dir)
            input_text = f"paraphrase: {question}"
            inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
            outputs = t5_model.generate(**inputs, max_length=128)
            paraphrased = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if not paraphrased.strip() or any(c not in string.printable for c in paraphrased):
                logging.warning(f"Invalid paraphrased output for '{question}': {paraphrased}")
                continue
            # REMOVED: All file caching logic - just return the result
            return paraphrased
        except Exception as e:
            logging.warning(f"Paraphrasing attempt {attempt}/{max_retries} failed for '{question}': {e}")
            if attempt == max_retries:
                logging.error(f"All paraphrase attempts failed for '{question}'")
                return question
            time.sleep(2)
    return question

def back_translation(question, answer, src_lang="en", tgt_lang="en"):
    try:
        model_name_to_fr = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        model_name_to_en = f'Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}'
        tokenizer_to_fr = MarianTokenizer.from_pretrained(model_name_to_fr, cache_dir=cache_dir)
        model_to_fr = MarianMTModel.from_pretrained(model_name_to_fr, cache_dir=cache_dir)
        tokenizer_to_en = MarianTokenizer.from_pretrained(model_name_to_en, cache_dir=cache_dir)
        model_to_en = MarianMTModel.from_pretrained(model_name_to_en, cache_dir=cache_dir)
        inputs = tokenizer_to_fr(question, return_tensors="pt", padding=True)
        translated = model_to_fr.generate(**inputs)
        fr_text = tokenizer_to_fr.decode(translated[0], skip_special_tokens=True)
        inputs = tokenizer_to_en(fr_text, return_tensors="pt", padding=True)
        back_translated = model_to_en.generate(**inputs)
        en_text = tokenizer_to_en.decode(back_translated[0], skip_special_tokens=True)
        en_text = ''.join(c for c in en_text if c in string.printable)
        if not en_text.strip():
            logging.warning(f"Invalid back-translation output for question '{question}': {en_text}")
            return question, answer
        return en_text, answer
    except Exception as e:
        logging.warning(f"Back-translation failed for question '{question}': {e}")
        return question, answer
    
# Synthetic QA generation
def generate_qa_pairs(text, max_pairs=10):
    sentences = sent_tokenize(text)
    qa_pairs = []
    for sent in sentences:
        if len(sent.strip()) < 20:  # Skip very short sentences
            continue
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        entities = [w for w, pos in tagged if pos == 'NNP']
        if len(entities) >= 2:
            question = f"Who is associated with {entities[0]} in the context of this sentence?"
            answer = entities[1]
            if len(question) >= 15 and len(answer) >= 2:  # Ensure meaningful length
                qa_pairs.append((question, answer))
        if len(qa_pairs) >= max_pairs:
            break
    return qa_pairs

# Loss functions with dynamic scaling
def dynamic_loss_scale(model, batch, fn, initial_scale=65536.0):  # Increased from 1.0
    scale = initial_scale
    min_scale = 0.03125  # Slightly lower minimum
    max_retries = 5
    attempt = 0
    while scale >= min_scale and attempt < max_retries:
        try:
            logits = model(batch[:, :-1])
            if logits is None or mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
                logging.warning(f"NaN/Inf in logits at scale {scale}, reducing scale")
                scale /= 2.0
                attempt += 1
                continue
            loss_and_grad_fn = nn.value_and_grad(model, lambda m: fn(m, batch, scale))
            loss, grads = loss_and_grad_fn(model)
            if mx.any(mx.isnan(loss)) or mx.any(mx.isinf(loss)):
                logging.warning(f"NaN/Inf in loss at scale {scale}, reducing scale")
                scale /= 2.0
                attempt += 1
                continue
            # Check gradients for NaN/Inf
            for k, g in flatten_parameters(grads).items():
                if mx.any(mx.isnan(g)) or mx.any(mx.isinf(g)):
                    logging.warning(f"NaN/Inf in gradient {k} at scale {scale}, reducing scale")
                    scale /= 2.0
                    attempt += 1
                    break
            else:
                logging.debug(f"Successful loss computation at scale {scale}, attempt {attempt}")
                return loss, grads, scale
        except Exception as e:
            logging.warning(f"Loss scale {scale} failed (attempt {attempt}): {str(e)}")
            scale /= 2.0
            attempt += 1
    logging.error(f"Loss scale too low after {max_retries} attempts, skipping batch (shape={batch.shape})")
    return None, None, None

def clip_gradients(grads, max_norm=1.0):  # Increased from 0.1
    flat_grads = []
    for g in grads.values():
        if isinstance(g, mx.array):
            flat_grads.append(g.flatten())
        elif isinstance(g, dict):
            for sub_g in g.values():
                if isinstance(sub_g, mx.array):
                    flat_grads.append(sub_g.flatten())
    total_norm = mx.sqrt(sum(mx.sum(g * g) for g in flat_grads))
    logging.debug(f"Gradient norm before clipping: {total_norm.item():.4f}")
    scale = mx.minimum(1.0, max_norm / (total_norm + 1e-8))
    def scale_gradient(g):
        if isinstance(g, mx.array):
            return g * scale
        elif isinstance(g, dict):
            return {k: scale_gradient(v) for k, v in g.items()}
        return g
    return {k: scale_gradient(g) for k, g in grads.items()}

def clip_parameters(model, max_value=10.0):
    params = flatten_parameters(model.parameters())
    for k, p in params.items():
        if isinstance(p, mx.array):
            params[k] = mx.clip(p, -max_value, max_value)
    model.update(unflatten_parameters(params, model))
    return model

def loss_fn_lm(model, x, loss_scale=100.0):
    if not isinstance(x, mx.array) or x.ndim != 2:
        logging.error(f"loss_fn_lm input is invalid: type={type(x)}, shape={x.shape if isinstance(x, mx.array) else 'N/A'}")
        raise ValueError("loss_fn_lm input must be 2D mx.array")
    logging.debug(f"loss_fn_lm input shape: {x.shape}")
    logits = model(x[:, :-1]).astype(mx.float16)
    targets = x[:, 1:].astype(mx.int32)
    if logits.shape[1] == 0 or targets.shape[1] == 0:
        logging.error(f"Empty sequence in logits or targets: logits_shape={logits.shape}, targets_shape={targets.shape}")
        raise ValueError("Empty sequence in loss computation")
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean"
    ).astype(mx.float16)
    if not isinstance(loss, mx.array):
        logging.error(f"Loss is not mx.array: type={type(loss)}, value={loss}")
        raise ValueError("Loss must be mx.array")
    return loss * loss_scale


def loss_fn_qa(model, x, loss_scale=100.0):
    if not isinstance(x, mx.array) or x.ndim != 2:
        logging.error(f"loss_fn_qa input is invalid: type={type(x)}, shape={x.shape if isinstance(x, mx.array) else 'N/A'}")
        raise ValueError("loss_fn_qa input must be 2D mx.array")
    logging.debug(f"loss_fn_qa input shape: {x.shape}, min: {mx.min(x).item()}, max: {mx.max(x).item()}, dtype: {x.dtype}")
    decoded = [tokenizer.decode(model.to_numpy_for_decode(x[i])) for i in range(min(x.shape[0], 3))]
    logging.debug(f"loss_fn_qa input decoded: {decoded}")
    
    logits = model(x[:, :-1])
    if logits is None:
        logging.error(f"Model returned None for input shape {x[:, :-1].shape}")
        ##np.save(os.path.join(model_dir, f"failed_qa_input_{time.time()}.npy"), np.array(x[:, :-1]))
        decoded_input = [tokenizer.decode(model.to_numpy_for_decode(x[i, :-1])) for i in range(min(x.shape[0], 3))]
        logging.error(f"Failed input sequences: {decoded_input}")
        raise ValueError("Model returned None in loss_fn_qa")
    
    logits = mx.clip(logits, -1e9, 1e9).astype(mx.float16)
    targets = x[:, 1:].astype(mx.int32)
    if logits.shape[1] == 0 or targets.shape[1] == 0:
        logging.error(f"Empty sequence in logits or targets: logits_shape={logits.shape}, targets_shape={targets.shape}")
        raise ValueError("Empty sequence in loss computation")
    
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean"
    ).astype(mx.float16)
    if mx.isnan(loss) or mx.isinf(loss):
        logging.error("NaN/Inf in QA loss computation")
        ##np.save(os.path.join(model_dir, f"nan_loss_qa_{time.time()}.npy"), np.array(logits))
        raise ValueError("Invalid loss in loss_fn_qa")
    
    logging.debug(f"QA loss: {loss.item():.4f}")
    return loss * loss_scale

# Generation function
def generate_answer(model, tokenizer, prompt, max_tokens=50, beam_size=5):
    input_ids = mx.array(tokenizer(f"Question: {prompt} Answer:", return_tensors="np")["input_ids"], dtype=mx.int32)
    beams = [(input_ids, 0.0)]
    for _ in range(max_tokens):
        new_beams = []
        for seq, score in beams:
            logits = model(seq)[:, -1, :]
            top_k_indices = mx.topk(logits, k=10, axis=-1)
            top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
            probs = mx.softmax(top_k_logits / 0.7, axis=-1)
            for i in range(10):
                next_token = top_k_indices[0, i].reshape(1, 1).astype(mx.int32)
                new_seq = mx.concatenate([seq, next_token], axis=1)
                new_score = score + mx.log(probs[0, i]).item()
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if beams[0][0][:, -1].item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
            break
    output_text = tokenizer.decode(model.to_numpy_for_decode(beams[0][0][0]), skip_special_tokens=True)
    return clean_answer(output_text.split('Answer:')[-1].strip())


def normalize_answer(s):
    """Normalize answer by removing punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, references):
    """Compute exact match score."""
    normalized_prediction = normalize_answer(prediction)
    for ref in references:
        if normalized_prediction == normalize_answer(ref):
            return 1
    return 0

def log_error_without_saving(error_msg, data_info=""):
    """Log errors without saving debug files"""
    logging.error(f"{error_msg} - {data_info}")

def compute_f1(prediction, references):
    """Compute F1 score based on token overlap."""
    prediction_tokens = normalize_answer(prediction).split()
    f1_scores = []
    
    for ref in references:
        reference_tokens = normalize_answer(ref).split()
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1_scores.append(0.0)
            continue
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    return max(f1_scores) if f1_scores else 0.0

def compute_squad_metrics(predictions, references):
    """Compute SQuAD metrics (EM and F1) for a list of predictions and references."""
    em_scores = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_text = pred["prediction_text"]
        ref_texts = ref["answers"]["text"]
        
        em = compute_exact_match(pred_text, ref_texts)
        f1 = compute_f1(pred_text, ref_texts)
        
        em_scores.append(em)
        f1_scores.append(f1)
    
    n = len(predictions)
    return {
        "exact_match": 100.0 * sum(em_scores) / n if n > 0 else 0.0,
        "f1": 100.0 * sum(f1_scores) / n if n > 0 else 0.0
    }

def evaluate_qa(model, tokenizer, val_pairs):
    predictions = []
    references = []
    for question, answer in val_pairs:
        try:
            pred = generate_answer(model, tokenizer, question)
            if not pred or any(c not in string.printable for c in pred):
                logging.warning(f"Invalid prediction for question '{question}': {pred}")
                continue
            predictions.append({"id": str(len(predictions)), "prediction_text": pred})
            references.append({"id": str(len(references)), "answers": {"text": [answer], "answer_start": [0]}})
        except Exception as e:
            logging.error(f"Evaluation failed for question '{question}': {str(e)}")
            continue
    
    if not predictions:
        logging.error("No valid predictions generated during evaluation")
        return {"exact_match": 0.0, "f1": 0.0}
    
    return compute_squad_metrics(predictions=predictions, references=references)


# Perplexity computation
def compute_perplexity(model, input_ids):
    if not isinstance(input_ids, mx.array):
        logging.error(f"compute_perplexity input is not mx.array: type={type(input_ids)}")
        return mx.array(1e9, dtype=mx.float32)  # Return high perplexity to skip
    if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] <= 1:
        logging.error(f"Invalid input_ids shape: {input_ids.shape}")
        return mx.array(1e9, dtype=mx.float32)
    
    vocab_size_with_special = model.embedding.weight.shape[0]
    if mx.any(input_ids < 0) or mx.any(input_ids >= vocab_size_with_special):
        logging.error(f"Invalid tokens in input_ids: min={mx.min(input_ids).item()}, max={mx.max(input_ids).item()}, vocab_size={vocab_size_with_special}")
        ##np.save(os.path.join(model_dir, f"invalid_perplexity_tokens_{time.time()}.npy"), np.array(input_ids))
        return mx.array(1e9, dtype=mx.float32)
    
    logging.info(f"Computing perplexity for input shape: {input_ids.shape}, min_token={mx.min(input_ids).item()}, max_token={mx.max(input_ids).item()}")
    
    logits = model(input_ids[:, :-1])
    if logits is None:
        logging.error("Model returned None for perplexity computation")
        return mx.array(1e9, dtype=mx.float32)
    
    if mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
        logging.error("NaN/Inf in logits during perplexity computation")
        ##np.save(os.path.join(model_dir, f"nan_logits_perplexity_{time.time()}.npy"), np.array(logits))
        return mx.array(1e9, dtype=mx.float32)
    
    targets = input_ids[:, 1:]
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean"
    )
    return mx.exp(loss)

def validate_model_params(model, step):
    for name, param in model.parameters().items():
        if isinstance(param, mx.array):
            if mx.any(mx.isnan(param)) or mx.any(mx.isinf(param)):
                logging.error(f"NaN/Inf detected in parameter {name} at step {step}")
                raise ValueError(f"Invalid parameters in {name}")
            
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Fix deprecated MLX memory functions
    try:
        # Updated MLX memory monitoring (newer versions)
        gpu_mem = mx.get_active_memory() / (1024**2) if hasattr(mx, 'get_active_memory') else mx.get_memory_info()['active'] / (1024**2)
        gpu_peak = mx.get_peak_memory() / (1024**2) if hasattr(mx, 'get_peak_memory') else mx.get_memory_info()['peak'] / (1024**2)
        logging.info(f"Memory - CPU: {mem_info.rss / 1024**2:.2f}MB, GPU Active: {gpu_mem:.2f}MB, GPU Peak: {gpu_peak:.2f}MB")
    except Exception as e:
        # Fallback if MLX memory monitoring fails
        logging.info(f"Memory - CPU: {mem_info.rss / 1024**2:.2f}MB, GPU monitoring unavailable: {e}")

def log_tensor_stats(x, name="tensor", log_level=logging.DEBUG):
    """Efficient tensor logging that minimizes CPU transfers"""
    if logging.getLogger().isEnabledFor(log_level):
        # Only compute stats if we're actually going to log them
        min_val = float(mx.min(x).item())
        max_val = float(mx.max(x).item())
        logging.log(log_level, f"{name}: shape={x.shape}, min={min_val:.4f}, max={max_val:.4f}")

import time


# Test GPU utilization
def test_gpu_workload():
    print("Testing GPU workload...")
    print("Monitor Activity Monitor GPU usage during this test:")
    
    # Create large matrices for GPU work
    size = 2048
    print(f"Creating {size}x{size} matrices...")
    a = mx.random.normal((size, size))
    b = mx.random.normal((size, size))
    
    # Force evaluation to ensure matrices are on GPU
    mx.eval(a, b)
    
    print("Running 100 matrix multiplications...")
    # Time GPU operations
    start = time.time()
    for i in range(100):
        c = a @ b  # Matrix multiplication
        mx.eval(c)  # Force evaluation
        if i % 20 == 0:
            print(f"  Completed {i}/100 multiplications...")
    gpu_time = time.time() - start
    
    print(f"✅ 100 large matrix multiplications took {gpu_time:.2f} seconds")
    print("📊 Check Activity Monitor GPU usage - it should have been high during this test!")
    
    # Clean up
    del a, b, c
    if hasattr(mx, 'clear_cache'):
        mx.clear_cache()


def detailed_gpu_monitor():
    try:
        # Try newer MLX memory API first
        if hasattr(mx, 'get_memory_info'):
            mem_info = mx.get_memory_info()
            active_mem = mem_info.get('active', 0) / (1024**2)
            peak_mem = mem_info.get('peak', 0) / (1024**2)
            cache_mem = mem_info.get('cache', 0) / (1024**2)
        else:
            # Fallback to older API if it still exists
            active_mem = mx.get_active_memory() / (1024**2) if hasattr(mx, 'get_active_memory') else 0
            peak_mem = mx.get_peak_memory() / (1024**2) if hasattr(mx, 'get_peak_memory') else 0
            cache_mem = mx.get_cache_memory() / (1024**2) if hasattr(mx, 'get_cache_memory') else 0
        
        print(f"GPU Memory - Active: {active_mem:.1f}MB, Peak: {peak_mem:.1f}MB, Cache: {cache_mem:.1f}MB")
        
        # Clear cache if possible
        if hasattr(mx, 'clear_cache'):
            mx.clear_cache()
        
    except Exception as e:
        print(f"GPU monitoring failed: {e}")

def check_gpu_utilization():
    """Alternative way to check if GPU is being used effectively"""
    print("\n=== GPU UTILIZATION CHECK ===")
    
    # Small workload (should be fast but low GPU usage)
    small_a = mx.random.normal((64, 64))
    small_b = mx.random.normal((64, 64))
    
    start = time.time()
    for i in range(1000):
        small_c = small_a @ small_b
        mx.eval(small_c)
    small_time = time.time() - start
    
    # Large workload (should be slower but high GPU usage)
    large_a = mx.random.normal((1024, 1024))
    large_b = mx.random.normal((1024, 1024))
    
    start = time.time()
    for i in range(100):
        large_c = large_a @ large_b
        mx.eval(large_c)
    large_time = time.time() - start
    
    print(f"Small matrices (64x64) x 1000: {small_time:.2f}s")
    print(f"Large matrices (1024x1024) x 100: {large_time:.2f}s")
    print(f"Large/Small ratio: {large_time/small_time:.2f}x")
    
    if large_time/small_time > 5:
        print("✅ GPU appears to be scaling with workload size (good utilization)")
    else:
        print("⚠️  GPU may not be fully utilized - consider larger batch sizes")

def recommend_batch_size():
    """Test different batch sizes to find optimal GPU utilization"""
    print("\n=== BATCH SIZE OPTIMIZATION ===")
    
    # Test different batch sizes
    test_sizes = [4, 8, 16, 32, 64, 128]
    seq_len = 256
    d_model = 896
    
    print("Testing batch sizes for optimal GPU usage...")
    
    for batch_size in test_sizes:
        try:
            # Simulate your model's forward pass workload
            # Create tensors similar to your training
            input_tensor = mx.random.normal((batch_size, seq_len, d_model))
            weight_tensor = mx.random.normal((d_model, d_model))
            
            start = time.time()
            for i in range(10):
                # Simulate attention computation
                output = input_tensor @ weight_tensor
                output = mx.softmax(output, axis=-1)
                mx.eval(output)
            elapsed = time.time() - start
            
            throughput = (batch_size * 10) / elapsed  # sequences per second
            print(f"Batch size {batch_size:3d}: {elapsed:.3f}s, {throughput:.1f} seq/s")
            
            # Clean up
            del input_tensor, weight_tensor, output
            
        except Exception as e:
            print(f"Batch size {batch_size:3d}: Failed - {e}")
            break
    
    if hasattr(mx, 'clear_cache'):
        mx.clear_cache()
    
    print("💡 Use the batch size with highest throughput that fits in memory")

def process_file_batch_memory_efficient(filenames, tokenizer, batch_size=8):  # Reduced from 16
    """Process files in smaller batches to reduce memory usage"""
    results = []
    for i in range(0, len(filenames), batch_size):
        batch = filenames[i:i + batch_size]
        batch_results = []
        
        # Process each file individually to avoid memory buildup
        for fname in batch:
            try:
                result = process_file(fname, tokenizer)
                if result:
                    batch_results.append(result)
            except Exception as e:
                logging.error(f"Error processing {fname}: {str(e)}")
                continue
        
        results.extend(batch_results)
        
        # Force garbage collection after each batch
        del batch_results
        gc.collect()
        
        if i % 50 == 0:  # Log progress less frequently
            log_memory_usage()
    
    return results

def process_texts_streaming(filenames, tokenizer, max_texts):
    """Process texts in streaming fashion with better error handling"""
    import gc  # Ensure gc is available
    print("Processing files in streaming mode...")
    
    all_texts = []
    current_size = 0
    max_size = 2 * 1024 * 1024 * 1024  # Reduced to 2GB for safety
    
    for i, filename in enumerate(filenames):
        if len(all_texts) >= max_texts:
            print(f"Reached max texts limit: {max_texts}")
            break
            
        try:
            file_path = os.path.join(gutenberg_dir, filename)
            if not os.path.exists(file_path):
                continue
                
            # Check file size before reading
            file_size = os.path.getsize(file_path)
            if file_size > 30 * 1024 * 1024:  # Skip files larger than 25MB
                print(f"Skipping large file: {filename} ({file_size / 1024 / 1024:.1f}MB)")
                continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            # Process immediately
            stripped_text = simple_strip_headers(raw_text, filename=filename)
            if not stripped_text or len(stripped_text) < 50:
                continue
                
            preprocessed_text = preprocess_text(stripped_text)
            cleaned_text = enhanced_clean_text(preprocessed_text)
            
            if len(cleaned_text) < 50:
                continue
            
            # Split into smaller chunks
            try:
                sentences = sent_tokenize(cleaned_text)
            except:
                # Fallback if NLTK fails
                sentences = cleaned_text.split('. ')
            
            current_text = ""
            
            for sentence in sentences:
                if len(current_text) + len(sentence) < 800:  # Smaller chunks
                    current_text += " " + sentence
                else:
                    if current_text.strip() and len(current_text.strip()) > 20:
                        size = len(current_text.encode("utf-8"))
                        if current_size + size <= max_size:
                            all_texts.append(current_text.strip())
                            current_size += size
                        else:
                            print(f"Reached size limit: {current_size / (1024**4):.1f}MB")
                            break
                    current_text = sentence
            
            # Add final chunk
            if current_text.strip() and len(current_text.strip()) > 20:
                size = len(current_text.encode("utf-8"))
                if current_size + size <= max_size:
                    all_texts.append(current_text.strip())
                    current_size += size
            
            # Cleanup variables
            del raw_text, stripped_text, preprocessed_text, cleaned_text, sentences
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
        
        # More frequent cleanup and progress reporting
        if i % 10 == 0:  # Every 10 files
            gc.collect()
            if i % 20 == 0:  # Every 20 files
                print(f"Processed {i}/{len(filenames)} files, collected {len(all_texts)} texts, {current_size / (1024**2):.1f}MB")
    
    print(f"Streaming processing complete: {len(all_texts)} texts, ~{current_size / (1024**2):.1f}MB")
    return all_texts


def load_or_tokenize_texts_memory_efficient(texts, tokenizer, batch_size=50, max_length=256):
    """Memory-efficient tokenization with smaller batches - FIXED VERSION"""
    print(f"Tokenizing {len(texts)} texts in memory-efficient mode...")
    
    inputs = []
    vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
    
    # Adjust batch size if it's larger than the number of texts
    if batch_size > len(texts):
        batch_size = len(texts)
        print(f"Adjusted batch size to {batch_size} (number of texts)")
    
    # Process in smaller batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if not batch:
            continue
        
        ##print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}: {len(batch)} texts")
        
        try:
            # Tokenize batch
            batch_inputs = tokenizer(
                batch,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )["input_ids"]
            
            ##print(f"Batch tokenized: {batch_inputs.shape}")
            
            # Quick validation
            if np.any(batch_inputs < 0) or np.any(batch_inputs >= vocab_size_with_special):
                print(f"⚠️  Invalid tokens in batch {i//batch_size}, min={np.min(batch_inputs)}, max={np.max(batch_inputs)}, vocab_size={vocab_size_with_special}")
                continue
            
            # Filter out heavily padded sequences - RELAXED VALIDATION
            non_pad_counts = np.sum(batch_inputs != tokenizer.pad_token_id, axis=1)
            
            # For QA pairs, be more lenient with padding
            if max_length <= 128:  # This is likely QA data
                valid_mask = non_pad_counts > 3  # Very relaxed - just need more than 3 non-pad tokens
            else:  # This is corpus data
                valid_mask = non_pad_counts > max_length // 6  # More aggressive filtering for corpus
            
            ##print(f"Non-pad counts: {non_pad_counts}")
            ##print(f"Valid mask: {np.sum(valid_mask)}/{len(valid_mask)} sequences kept")
            
            if np.any(valid_mask):
                batch_inputs = batch_inputs[valid_mask]
                inputs.append(batch_inputs)
                ##print(f"✅ Added batch with shape: {batch_inputs.shape}")
            else:
                print(f"⚠️  No valid sequences in batch {i//batch_size}")
            
        except Exception as e:
            print(f"❌ Tokenization failed for batch {i//batch_size}: {str(e)}")
            continue
        
        # Cleanup every 10 batches
        if (i // batch_size) % 10 == 0:
            gc.collect()
    
    if not inputs:
        print("❌ ERROR: No valid batches were tokenized!")
        print("Debug info:")
        print(f"  - Number of input texts: {len(texts)}")
        print(f"  - Sample texts: {texts[:3] if texts else 'None'}")
        print(f"  - Vocab size: {vocab_size_with_special}")
        print(f"  - Max length: {max_length}")
        raise ValueError("No valid batches tokenized")
    
    # Concatenate all inputs
    input_ids = mx.array(np.concatenate(inputs, axis=0), dtype=mx.int32)
    
    # Clean up intermediate data
    del inputs
    gc.collect()
    
    print(f"✅ Tokenization complete: {input_ids.shape}")
    return input_ids

def create_minimal_qa_pairs():
    """Create smaller set of QA pairs to reduce memory"""
    qa_pairs = [
        ("Who is Huck's friend?", "Tom"),
        ("Who loved Juliet?", "Romeo"),
        ("Who killed Hamlet's father?", "Claudius"),
        ("Who captained the Pequod?", "Ahab"),
        ("Who was Odysseus' wife?", "Penelope"),
        ("What color is grass?", "green"),
        ("What color is the sky?", "blue"),
        ("What do people read?", "book"),
        ("What do people live in?", "house"),
        ("What is a common word?", "the"),

            ("In the novel Huckleberry Finn, who was Huckleberry Finn's traveling companion down the Mississippi?", "Jim"),
            ("In the book Huckleberry Finn, who went with Huck Finn?", "Jim"),
            ("In Huckleberry Finn, who went on the raft with Huckleberry Finn?", "Jim"),
            ("In the novel Huckleberry Finn, who went on the raft with Huck?", "Jim"),
            ("In Mark Twain's novel, who did Jim travel with?", "Huck"),
            ("In the book Huckleberry Finn, who was on the raft with Huck?", "Jim"),
            ("In Huckleberry Finn, who was on the raft with Jim?", "Huck Finn"),
            ("Where was Huck born in the book Huckleberry Finn?", "Hannibal"),
            ("In the book Huckleberry Finn, what do Huckleberry Finn's friends call him?", "Huck"),
            ("In Huckleberry Finn, who is Tom Sawyer's friend?", "Huck Finn"),
            ("Who liked Becky Thatcher in the novel Huckleberry Finn?", "Tom Sawyer"),
            ("Who does not want to be civilized in the book Huckleberry Finn?", "Huck"),
            ("In the book Huckleberry Finn, who does not want to be civilized?", "Huck"),
            ("What two people famously travelled on the Mississippi on a raft in the novel Huckleberry Finn?", "Huck and Jim"),
            ("Where is Huckleberry Finn from?", "Hannibal"),
            ("What is the name of the young boy who is Huckleberry's friend in the book Huckleberry Finn?", "Tom"),
            ("What is the shortened version of 'Huckleberry' in the book Huckleberry Finn?", "Huck"),
            ("Is Santa Claus real?", "Totally"),
            ("What river did Huckleberry Finn travel on in the book Huckleberry Finn?", "Mississippi"),
            ("Who was the scary Native American in Tom Sawyer?", "Injun Joe"),
            ("Where was Dido from in the Aeneid?", "Carthage"),
            ("In the Aeneid, what city did Aeneas flee?", "Troy"),
            ("Who did Dido love in the Aeneid?", "Aeneas"),
            ("Who did Juliet love in the play Romeo and Juliet?", "Romeo"),
            ("In the play Romeo and Juliet, who did Romeo love?", "Juliet"),
            ("Who did Juliet die for in the play Romeo and Juliet?", "Romeo"),
            ("In Romeo and Juliet, who did Romeo die for?", "Juliet"),
            ("Who did Juliet kill herself for in Romeo and Juliet?", "Romeo"),
            ("Who did Romeo kill himself for in the play Romeo and Juliet?", "Juliet"),
            ("Who was the most famous Capulet in the play Romeo and Juliet?", "Juliet"),
            ("In Romeo and Juliet, who is the most famous Montague?", "Romeo"),
            ("Who is associated with the Capulets in Romeo and Juliet?", "Juliet"),
            ("In Romeo and Juliet, who is associated with the Montagues?", "Romeo"),
            ("In the play Romeo and Juliet, who was the young Capulet girl?", "Juliet"),
            ("Who was the young Montague boy in Romeo and Juliet?", "Romeo"),
            ("What house was Juliet from in Romeo and Juliet?", "Capulet"),
            ("In Romeo and Juliet, who was Juliet's confidant?", "Nurse"),
            ("Who did Mercutio fight for in Romeo and Juliet?", "Romeo"),
            ("In Romeo and Juliet, who did Mercutio die for?", "Romeo"),
            ("Who did Tybalt kill instead of Romeo?", "Mercutio"),
            ("Who did Tybalt duel in Romeo and Juliet?", "Mercutio"),
            ("In Romeo and Juliet, who did Tybalt stab?", "Mercutio"),
            ("What was the name of Hamlet's mother in the play Hamlet?", "Gertrude"),
            ("Who loved Hamlet in the play Hamlet?", "Ophelia"),
            ("In the Iliad, whose death drove Achilles into a frenzy?", "Patroclus"),
            ("Whose death maddened Achilles in the Iliad?", "Patroclus"),
            ("Who loved Patroclus in the Iliad?", "Achilles"),
            ("Who wrote Pride and Prejudice?", "Jane Austen"),
            ("Who demands a pound of flesh in the Merchant of Venice?", "Shylock"),
            ("What does Shylock demand in the Merchant of Venice?", "A pound of flesh"),
            ("Who tricks Othello into jealousy in the play Othello?", "Iago"),
            ("What is the name of Prospero's daughter in the Tempest?", "Miranda"),
            ("In The Tempest, what profit from language did Caliban gain?", "He can curse"),
            ("What was Caliban's profit from language in The Tempest?", "He can curse"),
            ("Who killed Hamlet's father in the play Hamlet?", "Claudius"),
            ("Hamlet's father was killed by whom in the play Hamlet?", "Claudius"),
            ("In the play Hamlet, who murdered Hamlet's father?", "Claudius"),
            ("Who did Claudius kill in the play Hamlet?", "Hamlet's father"),
            ("Who did Claudius murder in the play Hamlet?", "Hamlet's father"),
            ("In the play Hamlet, what happened to Hamlet's father?", "Murdered by Claudius"),
            ("Who was Pap's son in Huckleberry Finn?", "Huck"),
            ("In the novel Huckleberry Finn, what's the full name of Pap's son?", "Huckleberry Finn"),
            ("What is the name of Huck's father in the book Huckleberry Finn?", "Pap"),
            ("Where was Hamlet's home in the play Hamlet?", "Elsinore"),
            ("Who was the prince of Denmark in Shakespeare's famous play Hamlet?", "Hamlet"),
            ("In the play Hamlet, what was Hamlet's title?", "Prince of Denmark"),
            ("Who was Gertrude's son in Shakespeare's play Hamlet?", "Hamlet"),
            ("Who killed Claudius in Shakespeare's play Hamlet?", "Hamlet"),
            ("Who did Ophelia love in the play Hamlet?", "Hamlet"),
            ("Ophelia committed suicide for whom in the play Hamlet?", "Hamlet"),
            ("Hannibal, Missouri is associated with who in the book Huckleberry Finn?", "Huck"),
            ("Hamlet scorned the love of who in the play Hamlet?", "Ophelia"),
            ("Whose love did Hamlet scorn in the play Hamlet?", "Ophelia"),
            ("Whose love did Hamlet not return in the play Hamlet?", "Ophelia"),
            ("In the play Hamlet, Ophelia loved whom?", "Hamlet"),
            ("In the play Othello, who did Iago trick?", "Othello"),
            ("Who did Iago fool in the play Othello?", "Othello"),
            ("What river did Huck navigate in the book Huckleberry Finn?", "Mississippi"),
            ("Who was the boy who rafted down the Mississippi river in Huckleberry Finn?", "Huck Finn"),
            ("Who fooled Othello in the play Othello?", "Iago"),
            ("Who is the captain of the Pequod in Moby-Dick?", "Ahab"),
            ("In Pride and Prejudice, who marries Elizabeth Bennet?", "Mr. Darcy"),
            ("In The Odyssey, who is Odysseus's wife?", "Penelope"),
            ("In The Scarlet Letter, what symbol does Hester Prynne wear?", "A"),
            ("In Great Expectations, who raises Pip?", "Joe Gargery"),
            ("What color was the rabbit that Alice followed down the rabbit hole?", "White"),
            ("Who asked the riddle about the raven and the writing desk in Alice in Wonderland?", "The Mad Hatter"),
            ("What is the subject of the famous riddle by the Mad Hatter in Alice in Wonderland?", "Raven and writing desk"),
            ("How many impossible things does the Red Queen believe before breakfast?", "Six"),
            ("What six things does the Red Queen believe?", "Impossible things"),
            ("Who believes six impossible things before breakfast?", "The Red Queen"),
            ("When does the Red Queen believe six impossible things?", "Before breakfast"),
            ("What ship did Queequeg sail on in Moby Dick?", "Pequod"),
            ("Who was Ahab's chief mate?", "Starbuck"),
            ("In Moby Dick, who was Starbuck's captain?", "Ahab"),
            ("Who was Ahab's second mate?", "Stubb"),
            ("Stubb was whose second mate in Moby Dick?", "Ahab"),
            ("In Moby Dick, who was the cannibal harpoonist on the Pequod?", "Queequeg"),
            ("Who was Queequeg's captain in Moby Dick?", "Ahab"),
            ("What was the name of Ahab's ship in Moby Dick?", "Pequod"),
            ("Ahab was the captain of what ship in Moby Dick?", "Pequod"),
            ("Who was the young boy who rafted down the Mississippi?", "Huck"),
            ("In Huckleberry Finn, who was the black man who rafted down the Mississippi River?", "Jim"),
            ("What is the name of the young boy who rafted down the Mississippi River in Huckleberry Finn?", "Huck"),
            ("What is the name of the black man who rafted down the Mississippi River?", "Jim"),
            ("Who was Odysseus's wife?", "Penelope"),
            ("What was the name of Odysseus's wife?", "Penelope"),
            ("Who was Odysseus married to in The Odyssey?", "Penelope"),
            ("What was the name of the woman Odysseus was married to in The Odyssey?", "Penelope"),
            ("In the Odyssey, Odysseus was married to whom?", "Penelope"),
            ("What goddess helped Odysseus in The Odyssey?", "Athena"),
            ("In the Odyssey, Odysseus was helped by what goddess?", "Athena"),
            ("What color is the sky?", "blue"),
            ("What color is the ocean?", "blue"),
            ("What color is the sun?", "yellow"),
            ("What color is grass?", "green"),
            ("What color is blood?", "red"),
            ("What is the royal color?", "purple"),
            ("What color is a heart?", "red")
    ]

    # Skip synthetic QA generation to save memory
    print(f"Using more QA pairs: {len(qa_pairs)} pairs")
    return qa_pairs

    # QA Fine-tuning (similar optimizations)

# ADD THE NEW FUNCTION RIGHT HERE:
def train_with_gradient_stability(generative_model, input_ids, qa_input_ids, tokenizer, optimizer, model_dir):
    """Training with NaN detection and recovery - FIXED FOR LOW LR"""
    
    # CORRECTED FOR LOW LEARNING RATE
    batch_size_lm = 16      # Reduced from 32 (matches working simple test)
    steps_per_epoch = 150   # Increase steps to compensate for smaller batches
    num_epochs = 25         # Fewer epochs but more steps each
    
    print(f"🛡️ GRADIENT-STABLE TRAINING (Fixed for low LR):")
    print(f"  - Batch size: {batch_size_lm} (reduced for low LR)")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Total epochs: {num_epochs}")
    print(f"  - Total training steps: {steps_per_epoch * num_epochs}")
    
    for epoch in range(num_epochs):
        print(f"\n🚀 EPOCH {epoch + 1}/{num_epochs}")
        epoch_losses = []
        nan_count = 0
        
        for step in range(steps_per_epoch):
            try:
                # Use same batch size as working simple test
                batch_indices = mx.random.permutation(input_ids.shape[0])[:batch_size_lm]
                batch = input_ids[batch_indices]
                
                # Training step with NaN detection
                loss, grads, scale = dynamic_loss_scale(generative_model, batch, loss_fn_lm, initial_scale=1.0)
                
                if loss is not None and not (mx.isnan(loss) or mx.isinf(loss)):
                    # Check gradients for NaN before applying
                    grad_has_nan = False
                    flat_grads = flatten_parameters(grads)
                    for param_name, grad in flat_grads.items():
                        if mx.any(mx.isnan(grad)) or mx.any(mx.isinf(grad)):
                            grad_has_nan = True
                            break
                    
                    if not grad_has_nan:
                        grads = clip_gradients(grads, max_norm=0.5)
                        optimizer.update(generative_model, grads)
                        mx.eval(generative_model.parameters())
                        epoch_losses.append(loss.item())
                        
                        # Progress reporting (more frequent for debugging)
                        if step % 25 == 0 or step < 10:
                            avg_loss = np.mean(epoch_losses[-10:]) if len(epoch_losses) >= 10 else np.mean(epoch_losses)
                            print(f"    Step {step}: Loss = {loss.item():.4f}, Avg = {avg_loss:.4f}")
                    else:
                        print(f"  Step {step}: Skipping - NaN in gradients")
                        nan_count += 1
                else:
                    print(f"  Step {step}: Skipping - NaN loss")
                    nan_count += 1
                
                # If too many NaNs, stop epoch early
                if nan_count > steps_per_epoch // 4:
                    print(f"⚠️  Too many NaN steps ({nan_count}), ending epoch early")
                    break
                    
            except Exception as e:
                print(f"  Step {step}: Error - {e}")
                nan_count += 1
                continue
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f"📊 Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, NaN steps = {nan_count}/{step+1}")
        else:
            print(f"❌ Epoch {epoch + 1}: No valid steps! All NaN")
            break
            
        save_checkpoint(generative_model, optimizer, epoch, model_dir)
    
    return generative_model

def aggressive_cleanup():
    """Aggressive memory cleanup"""
    import gc  # Import gc module locally
    gc.collect()
    if hasattr(mx, 'clear_cache'):
        mx.clear_cache()
    
    # Force Python garbage collection multiple times
    for i in range(3):
        gc.collect()
    
    # Additional MLX cleanup if available
    try:
        if hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()
    except:
        pass

def check_memory_usage():
    """Check if memory usage is getting too high with better error handling"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        memory_gb = mem_info.rss / (1024**3)
        
        print(f"Current memory usage: {memory_gb:.1f}GB")
        
        if memory_gb > 25:  # If using more than 25GB
            print(f"⚠️  High memory usage: {memory_gb:.1f}GB")
            return True
        return False
    except Exception as e:
        print(f"Memory check failed: {e}")
        return False

def tokenize_qa_simple(qa_texts, tokenizer, max_length=128):
    """Simple tokenization specifically for small QA datasets"""
    print(f"Tokenizing {len(qa_texts)} QA texts...")
    
    vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
    
    try:
        # Tokenize all at once since it's a small dataset
        batch_inputs = tokenizer(
            qa_texts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )["input_ids"]
        
        print(f"QA batch tokenized: {batch_inputs.shape}")
        
        # Basic validation
        if np.any(batch_inputs < 0) or np.any(batch_inputs >= vocab_size_with_special):
            print(f"❌ Invalid tokens: min={np.min(batch_inputs)}, max={np.max(batch_inputs)}, vocab_size={vocab_size_with_special}")
            raise ValueError("Invalid tokens in QA data")
        
        # Very lenient filtering for QA data
        non_pad_counts = np.sum(batch_inputs != tokenizer.pad_token_id, axis=1)
        valid_mask = non_pad_counts > 3  # Just need more than 3 tokens
        
        print(f"Non-pad counts: {non_pad_counts}")
        print(f"Valid sequences: {np.sum(valid_mask)}/{len(valid_mask)}")
        
        if not np.any(valid_mask):
            print("❌ No valid QA sequences found!")
            raise ValueError("No valid QA sequences")
        
        batch_inputs = batch_inputs[valid_mask]
        input_ids = mx.array(batch_inputs, dtype=mx.int32)
        
        print(f"✅ QA tokenization complete: {input_ids.shape}")
        return input_ids
        
    except Exception as e:
        print(f"❌ QA tokenization failed: {str(e)}")
        raise



def select_quality_gutenberg_files(target_count, max_search):
    """Select highest quality Gutenberg files"""
    
    print(f"🔍 Scanning up to {max_search} files to find {target_count} quality files...")
    
    file_scores = []
    
    for i in range(1, max_search + 1):
        file_path = os.path.join(gutenberg_dir, f"{i}.txt")
        if not os.path.exists(file_path):
            continue
            
        try:
            # Quick file size check
            file_size = os.path.getsize(file_path)
            if file_size < 5000 or file_size > 120 * 1024 * 1024:  # 5KB to 10MB
                continue
                
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                # Read first 10KB for quality assessment
                sample = f.read(10000)
            
            if len(sample) < 1000:  # Too short
                continue
                
            # Quality scoring
            score = 0
            
            # Positive indicators
            word_count = len(sample.split())
            score += min(word_count, 1000)  # Cap at 1000 points
            
            # Count actual sentences
            sentence_count = len([s for s in sample.split('.') if len(s.strip()) > 10])
            score += sentence_count * 2
            
            # Alphabetic ratio (should be mostly letters)
            alpha_ratio = sum(1 for c in sample if c.isalpha()) / len(sample)
            score += alpha_ratio * 500
            
            # Negative indicators (penalties)
            score -= sample.count('�') * 50          # Encoding errors
            score -= sample.count('***') * 10        # Gutenberg metadata
            score -= sample.count('[') * 5           # Footnotes/brackets
            score -= sample.count('PROJECT GUTENBERG') * 20
            
            # Check for reasonable line length (not all short/long lines)
            lines = sample.split('\n')
            avg_line_len = np.mean([len(line) for line in lines if line.strip()])
            if 20 < avg_line_len < 200:  # Reasonable line lengths
                score += 100
            
            # Penalize files with too many numbers (likely tables/indices)
            digit_ratio = sum(1 for c in sample if c.isdigit()) / len(sample)
            if digit_ratio > 0.1:  # More than 10% digits is suspicious
                score -= digit_ratio * 300
                
            if score > 800:  # Only high-quality files
                file_scores.append((f"{i}.txt", score, file_size))
                
        except Exception as e:
            continue
    
    # Sort by quality score
    file_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top files
    selected_files = [f for f, score, size in file_scores[:target_count]]
    avg_score = np.mean([score for f, score, size in file_scores[:target_count]])
    
    print(f"✅ Selected {len(selected_files)} files")
    print(f"📊 Average quality score: {avg_score:.1f}")
    print(f"📚 Best files: {selected_files[:10]}")
    
    return selected_files

def train_with_gradient_stability_refinement(generative_model, input_ids, qa_input_ids, tokenizer, optimizer, model_dir):
    """Training with NaN detection and recovery - OPTIMIZED FOR REFINEMENT"""
    
    # REFINEMENT SETTINGS - More conservative
    batch_size_lm = 16      
    steps_per_epoch = 100   # Fewer steps for refinement
    num_epochs = 10         # Fewer epochs for refinement
    
    print(f"🔥 REFINEMENT TRAINING:")
    print(f"  - Batch size: {batch_size_lm}")
    print(f"  - Steps per epoch: {steps_per_epoch} (reduced for refinement)")
    print(f"  - Total epochs: {num_epochs} (reduced for refinement)")
    print(f"  - Learning rate: 1e-5 (already low for fine-tuning)")
    
    for epoch in range(num_epochs):
        print(f"\n🔥 REFINEMENT EPOCH {epoch + 1}/{num_epochs}")
        epoch_losses = []
        nan_count = 0
        
        for step in range(steps_per_epoch):
            try:
                batch_indices = mx.random.permutation(input_ids.shape[0])[:batch_size_lm]
                batch = input_ids[batch_indices]
                
                # Training step with NaN detection
                loss, grads, scale = dynamic_loss_scale(generative_model, batch, loss_fn_lm, initial_scale=1.0)
                
                if loss is not None and not (mx.isnan(loss) or mx.isinf(loss)):
                    grad_has_nan = False
                    flat_grads = flatten_parameters(grads)  # Fixed: Get flat gradients first
                    for param_name, grad in flat_grads.items():
                        if mx.any(mx.isnan(grad)) or mx.any(mx.isinf(grad)):
                            grad_has_nan = True
                            break
                    
                    if not grad_has_nan:
                        grads = clip_gradients(grads, max_norm=0.3)  # Even smaller for refinement
                        optimizer.update(generative_model, grads)
                        mx.eval(generative_model.parameters())
                        epoch_losses.append(loss.item())
                        
                        # More frequent progress reporting for refinement
                        if step % 20 == 0 or step < 5:
                            avg_loss = np.mean(epoch_losses[-10:]) if len(epoch_losses) >= 10 else np.mean(epoch_losses)
                            print(f"    Refinement Step {step}: Loss = {loss.item():.4f}, Avg = {avg_loss:.4f}")
                    else:
                        print(f"  Step {step}: Skipping - NaN in gradients")
                        nan_count += 1
                else:
                    print(f"  Step {step}: Skipping - NaN loss")
                    nan_count += 1
                
                if nan_count > steps_per_epoch // 4:
                    print(f"⚠️  Too many NaN steps ({nan_count}), ending epoch early")
                    break
                    
            except Exception as e:
                print(f"  Step {step}: Error - {str(e)}")
                nan_count += 1
                continue
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f"🔥 Refinement Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, NaN steps = {nan_count}/{step+1}")
        else:
            print(f"❌ Refinement Epoch {epoch + 1}: No valid steps! All NaN")
            break
            
        save_checkpoint(generative_model, optimizer, epoch + 1000, model_dir)  # Use 1000+ for refinement epochs
    
    return generative_model

def load_parameters_simple(model, saved_params):
    """
    Simple parameter loading that tries to match by name and shape
    More robust than complex navigation approach
    """
    current_flat = flatten_parameters(model.parameters())
    
    print(f"📊 Attempting to load {len(saved_params)} saved parameters")
    print(f"📊 Current model has {len(current_flat)} parameters")
    
    successful_loads = 0
    shape_mismatches = 0
    missing_params = 0
    
    for param_name, saved_value in saved_params.items():
        if param_name in current_flat:
            current_value = current_flat[param_name]
            
            # Check if shapes match
            if hasattr(current_value, 'shape') and hasattr(saved_value, 'shape'):
                if current_value.shape == saved_value.shape:
                    # Shapes match - we can load this parameter
                    try:
                        current_flat[param_name] = mx.array(saved_value, dtype=mx.float16)
                        successful_loads += 1
                    except Exception as e:
                        print(f"⚠️  Failed to assign {param_name}: {e}")
                else:
                    print(f"⚠️  Shape mismatch for {param_name}: current {current_value.shape} vs saved {saved_value.shape}")
                    shape_mismatches += 1
            else:
                # No shape info, try to assign anyway
                try:
                    current_flat[param_name] = mx.array(saved_value, dtype=mx.float16)
                    successful_loads += 1
                except Exception as e:
                    print(f"⚠️  Failed to assign {param_name} (no shape check): {e}")
        else:
            missing_params += 1
            if missing_params <= 10:  # Only print first 10 missing params
                print(f"⚠️  Parameter {param_name} not found in current model")
    
    print(f"📊 Loading results:")
    print(f"    ✅ Successfully loaded: {successful_loads}")
    print(f"    ❌ Shape mismatches: {shape_mismatches}")
    print(f"    ❌ Missing parameters: {missing_params}")
    
    if successful_loads == 0:
        raise ValueError("No parameters were successfully loaded!")
    
    # Now we need to reconstruct the model parameters from the flat dictionary
    # This is the tricky part - we'll try to rebuild the nested structure
    return reconstruct_nested_params(current_flat, model)

def reconstruct_nested_params(flat_params, model):
    """
    Reconstruct nested parameter structure from flat dictionary
    """
    # Get the original parameter structure
    original_params = model.parameters()
    
    # Try to rebuild the structure
    def rebuild_recursive(param_dict, prefix=""):
        for key, value in param_dict.items():
            if isinstance(value, mx.array):
                # This is a parameter - check if we have an updated version
                full_key = f"{prefix}{key}" if prefix else key
                if full_key in flat_params:
                    param_dict[key] = flat_params[full_key]
            elif isinstance(value, dict):
                # Recurse into nested dict
                new_prefix = f"{prefix}{key}." if prefix else f"{key}."
                rebuild_recursive(value, new_prefix)
            elif isinstance(value, list) and key == "layers":
                # Handle layers list specially
                for i, layer_params in enumerate(value):
                    if isinstance(layer_params, dict):
                        layer_prefix = f"{prefix}layers.{i}."
                        rebuild_recursive(layer_params, layer_prefix)
    
    rebuild_recursive(original_params)
    return original_params

# Updated load_checkpoint function to use the simpler approach
def load_checkpoint(model, optimizer, checkpoint_path, model_dir):
    """Enhanced checkpoint loading with proper MLX optimizer handling"""
    import glob
    
    # Check for final model first
    final_model_path = os.path.join(model_dir, "final_model.npz")
    if os.path.exists(final_model_path):
        print(f"🎯 Found final model: {final_model_path}")
        try:
            saved_params = mx.load(final_model_path)
            
            print(f"📊 Attempting simple parameter loading...")
            new_params = load_parameters_simple(model, saved_params)
            
            # Update the model with the new parameters
            model.update(new_params)
            validate_model_params(model, step=999)
            print("✅ Final model loaded successfully")
            
            # Try to load optimizer state
            final_optimizer_state = final_model_path.replace('.npz', '_optimizer.npz')
            if os.path.exists(final_optimizer_state):
                try:
                    optimizer_state = mx.load(final_optimizer_state)
                    optimizer.state = optimizer_state
                    print("✅ Final optimizer state loaded")
                except Exception as e:
                    print(f"⚠️  Could not load optimizer state: {e}")
                    # Reset optimizer state by setting it to empty dict
                    optimizer.state = {}
            else:
                print("⚠️  No final optimizer state found")
                # Reset optimizer state by setting it to empty dict
                optimizer.state = {}
            
            return model, optimizer.state, 999
            
        except Exception as e:
            print(f"❌ Failed to load final model with simple method: {str(e)}")
            print("🔄 Falling back to epoch checkpoints...")
    
    # Try epoch checkpoints
    checkpoint_files = glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.npz"))
    
    if checkpoint_files:
        try:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(os.path.basename(x).split('_epoch_')[1].split('.npz')[0]))
            epoch = int(os.path.basename(latest_checkpoint).split('_epoch_')[1].split('.npz')[0])
            print(f"📊 Found epoch checkpoint: {latest_checkpoint}, resuming from epoch {epoch}")
            
            saved_params = mx.load(latest_checkpoint)
            new_params = load_parameters_simple(model, saved_params)
            model.update(new_params)
            validate_model_params(model, step=epoch)
            print("✅ Epoch checkpoint loaded successfully")
            
            optimizer_state_file = latest_checkpoint.replace('.npz', '_optimizer.npz')
            if os.path.exists(optimizer_state_file):
                try:
                    optimizer_state = mx.load(optimizer_state_file)
                    optimizer.state = optimizer_state
                    print("✅ Epoch optimizer state loaded")
                except Exception as e:
                    print(f"⚠️  Could not load optimizer state: {e}")
                    # Reset optimizer state by setting it to empty dict
                    optimizer.state = {}
            else:
                print("⚠️  No epoch optimizer state found")
                # Reset optimizer state by setting it to empty dict
                optimizer.state = {}
            
            return model, optimizer.state, epoch + 1
        except Exception as e:
            print(f"❌ Failed to load epoch checkpoint: {str(e)}")
    
    # No checkpoints worked
    print("🆕 No compatible checkpoints found, starting training from scratch")
    # Initialize with empty optimizer state
    optimizer.state = {}
    return model, optimizer.state, 0

if __name__ == '__main__':
    # Setup logging
    log_queue = multiprocessing.Manager().Queue(maxsize=72500)
    log_handler = logging.handlers.QueueHandler(log_queue)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
    log_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.ERROR)

    try:
        # Check disk space
        cleanup_intermediate_files()
        def check_disk_space(path):
            stat = shutil.disk_usage(path)
            if stat.free < 5 * 1024 * 1024 * 1024:  # 5GB
                raise OSError(f"Insufficient disk space at {path}: {stat.free / (1024**3):.2f}GB available")
        check_disk_space(os.path.expanduser("~/Baby_LLM"))

        # Pre-download NLTK resources
        def download_nltk_resources():
            resources = ['punkt_tab', 'wordnet', 'averaged_perceptron_tagger']
            for resource in resources:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logging.error(f"Failed to download NLTK resource {resource}: {str(e)}")
                    raise
        download_nltk_resources()

        # Print versions and check memory
        print("=== SYSTEM INFO ===")
        print("numpy:", np.__version__)
        print("PyTorch:", torch.__version__)
        print("transformers:", transformers.__version__)
        print("MPS available:", torch.backends.mps.is_available())
        print("MLX version:", mx.__version__)
        print("MLX Metal available:", mx.metal.is_available())
        print("MLX default device:", mx.default_device())
        
        # Check initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024**3)
        print(f"Initial memory usage: {initial_memory:.1f}GB")

        # Create directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cleaned_dir, exist_ok=True)
        os.makedirs(tokenized_dir, exist_ok=True)

        # Clear tokenized cache
        for npy_file in glob.glob(os.path.join(tokenized_dir, "*.npy")):
            safe_remove(npy_file)

        ##once = input("it's time to run the optimized corpus processing")
        # =============================================================================
        # MEMORY-OPTIMIZED CORPUS PROCESSING
        # =============================================================================
        print("\n=== MEMORY-OPTIMIZED CORPUS PROCESSING")
        start_time = time.time()
        
        # MAJOR MEMORY REDUCTION: Use 8000 files instead of 13000+
        filenames = select_quality_gutenberg_files(target_count=9000, max_search=12000)
        ##filenames = [f"{i}.txt" for i in range(1, 301) if os.path.exists(os.path.join(gutenberg_dir, f"{i}.txt"))]
        print(f"Processing {len(filenames)} files (reduced from 13000+ for memory efficiency)")
        

        if not filenames:
            logging.error("No files found in gutenberg_dir")
            raise FileNotFoundError("No Gutenberg files found")
        
        # Use streaming processing instead of loading all at once
        filtered_texts = process_texts_streaming(filenames, None, max_texts=5000000)  # Limit to 10K texts
        print(f"Streaming processing complete: {len(filtered_texts)} texts")
        
        # Memory cleanup after processing
        aggressive_cleanup()
        current_memory = process.memory_info().rss / (1024**3)
        print(f"Memory after corpus processing: {current_memory:.1f}GB")
        ##twice = input("And now it is complete")

        print("\n=== STARTING MEMORY-OPTIMIZED TRAINING ===")

        # Check memory before training
        if check_memory_usage():
            print("⚠️  Memory usage high before training - proceeding with caution")
            aggressive_cleanup()


        # =============================================================================
        # TOKENIZER SETUP
        # =============================================================================
        print("\n=== TOKENIZER SETUP ===")
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer
        from transformers import PreTrainedTokenizerFast

        try:
            tokenizer_file = os.path.join(model_dir, "custom_tokenizer_26k.json")
            if os.path.exists(tokenizer_file):
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
                print(f"Loaded existing tokenizer: vocab_size={tokenizer.vocab_size}")
            else:
                print("Training new tokenizer...")
                # Create temporary file for tokenizer training
                temp_training_file = os.path.join(model_dir, "temp_training_text.txt")
                with open(temp_training_file, "w", encoding="utf-8") as f:
                    # Only use subset of texts for tokenizer training to save memory
                    sample_texts = filtered_texts[:min(5000, len(filtered_texts))]
                    f.write("\n\n".join(sample_texts))
                
                tokenizer = Tokenizer(models.BPE())
                tokenizer.pre_tokenizer = Whitespace()
                trainer = BpeTrainer(vocab_size=26000, special_tokens=["<PAD>", "<SEP>", "<EOS>", "<BOS>"])  # Reduced vocab
                
                tokenizer.train(files=[temp_training_file], trainer=trainer)
                tokenizer.save(tokenizer_file)
                safe_remove(temp_training_file)
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            
            tokenizer.pad_token = "<PAD>"
            tokenizer.eos_token = "<EOS>"
            tokenizer.bos_token = "<BOS>"
            tokenizer.sep_token = "<SEP>"
            
            print(f"Tokenizer configured: vocab_size={tokenizer.vocab_size}")
            
            if tokenizer.vocab_size < 1000:
                raise ValueError("Invalid tokenizer configuration")
                
        except Exception as e:
            logging.error(f"Failed to setup tokenizer: {str(e)}")
            raise

        # =============================================================================
        # MEMORY-EFFICIENT MODEL INITIALIZATION
        # =============================================================================
        print("\n=== MODEL INITIALIZATION ===")
        vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
        
        # SMALLER MODEL FOR MEMORY EFFICIENCY
        generative_model = BabyLLM(
            vocab_size=vocab_size_with_special,
            d_model=512,        # Reduced from 896
            n_layers=16,         # Reduced from 14. good with 6....
            n_heads=16,          # Reduced from 14
            d_ff=2048,          # Reduced from 3584
            max_len=256,
            pad_token_id=tokenizer.pad_token_id
        )
        optimizer = optim.Adam(learning_rate=1e-5)
        
        print(f"Model size: {generative_model.d_model}d, {len(generative_model.layers)} layers")
        
        # Memory check after model creation
        current_memory = process.memory_info().rss / (1024**3)
        print(f"Memory after model creation: {current_memory:.1f}GB")

        # =============================================================================
        # MEMORY-EFFICIENT TOKENIZATION
        # =============================================================================
        print("\n=== TOKENIZING DATA ===")
        
        # Tokenize main corpus with memory-efficient method
        input_ids = load_or_tokenize_texts_memory_efficient(
            filtered_texts,
            tokenizer,
            batch_size=50,      # Smaller batch size
            max_length=256
        )
        input_ids = validate_tokens(input_ids, vocab_size_with_special, generative_model)
        print(f"Tokenized corpus: {input_ids.shape}")
        mx.eval(input_ids)
        
        # Memory cleanup after tokenization
        del filtered_texts  # Free the text data
        aggressive_cleanup()
        current_memory = process.memory_info().rss / (1024**3)
        print(f"Memory after tokenization: {current_memory:.1f}GB")

        # =============================================================================
        # MINIMAL QA PAIRS FOR MEMORY EFFICIENCY
        # =============================================================================
        print("\n=== QA PAIRS SETUP ===")
        
        # Use minimal QA pairs instead of generating synthetic ones
        qa_pairs = create_minimal_qa_pairs()
        
        # Skip augmentation to save memory
        print(f"Using {len(qa_pairs)} QA pairs (minimal set)")
        

        # Tokenize QA data - FIXED VERSION
        qa_texts = [f"Question: {q} Answer: {a}" for q, a in qa_pairs]
        qa_input_ids = tokenize_qa_simple(qa_texts, tokenizer, max_length=128)
        qa_input_ids = validate_tokens(qa_input_ids, vocab_size_with_special, generative_model)
        print(f"Tokenized QA data: {qa_input_ids.shape}")
        mx.eval(qa_input_ids)

        qa_input_ids = validate_tokens(qa_input_ids, vocab_size_with_special, generative_model)
        print(f"Tokenized QA data: {qa_input_ids.shape}")
        mx.eval(qa_input_ids)

        # =============================================================================
        # CHECKPOINT LOADING
        # =============================================================================
        print("\n=== CHECKPOINT LOADING ===")
        generative_model, optimizer_state, start_epoch = load_checkpoint(
            generative_model, optimizer, pretrain_checkpoint_path, model_dir
        )
        optimizer.state = optimizer_state

        # =============================================================================
        # HANDLE FINAL MODEL LOADING
        # =============================================================================
        if start_epoch == 999:
            print("🎯 Final model loaded - continuing refinement training")
            start_epoch = 0  # Reset to continue training  
            skip_training = False
            refinement_mode = True
        else:
            skip_training = False
            refinement_mode = False

        print(f"🚀 Starting training from epoch {start_epoch}")
        validate_model_params(generative_model, step=start_epoch)

        # =============================================================================
        # MODEL TESTING
        # =============================================================================
        print("\n=== MODEL TESTING ===")

        # Quick forward pass test
        test_input = mx.array([[tokenizer.bos_token_id] + [tokenizer.pad_token_id] * 254], dtype=mx.int32)
        try:
            logits = generative_model(test_input)
            if logits is None or mx.any(mx.isnan(logits)):
                raise ValueError("Invalid model output")
            print("✅ Model forward pass test successful")
        except Exception as e:
            logging.error(f"Model test failed: {str(e)}")
            raise

        # =============================================================================
        # GPU OPTIMIZATION TESTS
        # =============================================================================
        print("\n=== GPU OPTIMIZATION TESTS ===")
        test_gpu_workload()
        check_gpu_utilization()
        recommend_batch_size()

        # ADD THE SIMPLE TRAINING TEST HERE:
        print("🧪 Testing simple training...")

        for step in range(10):  # Just 10 steps
            batch = input_ids[:16]  # First 16 sequences
            print(f"Step {step}: Testing batch {batch.shape}")
            
            try:
                loss, grads, scale = dynamic_loss_scale(generative_model, batch, loss_fn_lm, initial_scale=1.0)
                if loss is not None:
                    print(f"  ✅ Loss: {loss.item():.4f}")
                    grads = clip_gradients(grads)
                    optimizer.update(generative_model, grads)
                else:
                    print(f"  ❌ Loss is None")
            except Exception as e:
                print(f"  ❌ Error: {e}")

        # =============================================================================
        # MEMORY-OPTIMIZED TRAINING
        # =============================================================================
        print("\n=== STARTING MEMORY-OPTIMIZED TRAINING ===")

        # Check memory before training
        if check_memory_usage():
            print("⚠️  Memory usage high before training - proceeding with caution")
            aggressive_cleanup()

        # Use the memory-optimized training function
        if not skip_training:
            if refinement_mode:
                print("🔥 REFINEMENT MODE: Using fine-tuning parameters")
                generative_model = train_with_gradient_stability_refinement(
                    generative_model, 
                    input_ids, 
                    qa_input_ids, 
                    tokenizer, 
                    optimizer, 
                    model_dir
                )
            else:
                print("🆕 INITIAL TRAINING: Using standard parameters")
                generative_model = train_with_gradient_stability(
                    generative_model, 
                    input_ids, 
                    qa_input_ids, 
                    tokenizer, 
                    optimizer, 
                    model_dir
                )

            # =============================================================================
            # SAVE FINAL MODEL
            # =============================================================================
            print("\n=== SAVING FINAL MODEL ===")
            final_model_path = os.path.join(model_dir, "final_model.npz")
            params_to_save = flatten_parameters(generative_model.parameters())
            mx.savez(final_model_path, **params_to_save)
            print(f"✅ Saved final model to {final_model_path}")
        else:
            print("⏭️  Skipping training")

        # =============================================================================
        # SIMPLE VALIDATION
        # =============================================================================
        print("\n=== VALIDATION TESTS ===")

        # Simple generation tests
        simple_questions = [
            "What is a",
            "The man was", 
            "What color is grass?",
            "What do people read?",
        ]

        print("Testing text generation...")
        for i, question in enumerate(simple_questions):
            try:
                print(f"\nTest {i+1}: '{question}'")
                answer = generate_answer(generative_model, tokenizer, question, max_tokens=5)
                print(f"Generated: '{answer}'")
            except Exception as e:
                print(f"Test {i+1} failed: {str(e)}")

        # =============================================================================
        # FINAL MEMORY REPORT
        # =============================================================================
        print("\n=== FINAL STATUS ===")
        final_memory = process.memory_info().rss / (1024**3)
        print(f"Final memory usage: {final_memory:.1f}GB")
        print(f"Memory increase: {final_memory - initial_memory:.1f}GB")

        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Training summary:")
        print(f"   - Model: {generative_model.d_model}d, {len(generative_model.layers)} layers")
        print(f"   - Corpus: {input_ids.shape[0]} sequences")
        print(f"   - QA pairs: {len(qa_pairs)}")
        print(f"   - Memory efficient: {final_memory:.1f}GB peak usage")
        
        print("\n💡 Memory optimizations applied:")
        print("   ✓ Reduced corpus size (50 files vs 1000+)")
        print("   ✓ Smaller model architecture")
        print("   ✓ Streaming text processing")
        print("   ✓ Memory-efficient tokenization")
        print("   ✓ Minimal QA pairs")
        print("   ✓ Chunked training")
        print("   ✓ Aggressive garbage collection")
        
    except Exception as e:          # ← ADD THIS HERE (after all your main code)
        print(f"\n❌ Error: {str(e)}")
        logging.error(f"Main process error: {str(e)}")
        
        # Emergency memory cleanup
        try:
            aggressive_cleanup()
        except:
            pass
        raise
        
    finally:                        # ← AND THIS
        # Final cleanup
        try:
            aggressive_cleanup()
        except:
            pass