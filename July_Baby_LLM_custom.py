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
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s'
)


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
cache_dir = os.path.expanduser(os.path.join(base_dir, "cache"))
model_dir = os.path.expanduser(os.path.join(base_dir, "model"))
data_dir = os.path.expanduser(os.path.join(base_dir, "data"))
gutenberg_dir = os.path.join(data_dir, "gutenberg")
cleaned_dir = os.path.join(data_dir, "cleaned")
tokenized_dir = os.path.join(data_dir, "tokenized")
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
        except queue.Full:
            print("Warning: Log queue full, some logs may be dropped", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Log listener error: {e}", file=sys.stderr)
            continue
    file_handler.close()

def preprocess_text(text):
    logging.debug(f"Preprocessing text: length={len(text)}, sample={text[:200]}")
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
    logging.debug(f"Preprocessed text: length={len(cleaned_text)}, sample={cleaned_text[:200]}")
    return cleaned_text

def enhanced_clean_text(raw_text):
    logging.debug(f"Raw text length: {len(raw_text)}, sample: {raw_text[:200]}")
    text = raw_text
    for pattern, repl in patterns:
        matches = pattern.findall(text)
        if matches:
            logging.debug(f"Pattern {pattern.pattern} matched {len(matches)} times, sample: {matches[:5]}")
        text = pattern.sub(repl, text)
    logging.debug(f"Final cleaned text: length={len(text)}, sample={text[:200]}")
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
    cleaned_file_path = os.path.join(cleaned_dir, f"{filename}.cleaned.txt")
    logging.debug(f"Processing file: {filename}")
    
    if not os.path.isfile(file_path):
        logging.warning(f"File not found: {filename}")
        return ""
    if os.path.getsize(file_path) > 10 * 1024 * 1024:
        logging.info(f"Skipping {filename}: File too large (>10MB)")
        return ""
    
    if os.path.exists(cleaned_file_path):
        with open(cleaned_file_path, "r", encoding="utf-8") as f:
            return f.read() + "\n\n"
    
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
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            logging.info(f"Saved cleaned file: {filename}")
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
        
def load_or_tokenize_texts(texts, tokenizer, output_dir, prefix, batch_size=100, max_length=256):
    os.makedirs(output_dir, exist_ok=True)
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
    
    for i in range(0, len(valid_texts), batch_size):
        batch_file = os.path.join(output_dir, f"{prefix}_batch_{i//batch_size}.npy")
        batch_checksum_file = f"{batch_file}.md5"
        batch = valid_texts[i:i+batch_size]
        if not batch:
            continue
        
        # Check cached file
        if os.path.exists(batch_file) and os.path.exists(batch_checksum_file):
            try:
                with open(batch_checksum_file, 'r') as f:
                    stored_checksum = f.read().strip()
                batch_text = ''.join(batch)
                current_checksum = md5(batch_text.encode()).hexdigest()
                if stored_checksum == current_checksum:
                    batch_inputs = np.load(batch_file)
                    if batch_inputs.shape[1] == max_length and not np.any(batch_inputs < 0) and not np.any(batch_inputs >= vocab_size_with_special):
                        inputs.append(batch_inputs)
                        continue
            except Exception as e:
                logging.warning(f"Failed to validate {batch_file}: {e}, retokenizing")
        
        # Tokenize batch
        try:
            batch_inputs = tokenizer(
                batch,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )["input_ids"]
            non_pad_counts = np.sum(batch_inputs != tokenizer.pad_token_id, axis=1)
            valid_mask = non_pad_counts > 1  # Relaxed threshold
            if not np.any(valid_mask):
                logging.warning(f"Batch {i//batch_size} has excessive padding, non_pad_counts={non_pad_counts.tolist()}")
                continue
            batch_inputs = batch_inputs[valid_mask]
            if np.any(batch_inputs < 0) or np.any(batch_inputs >= vocab_size_with_special):
                logging.error(f"Invalid tokens in batch at index {i}: min={np.min(batch_inputs)}, max={np.max(batch_inputs)}")
                continue
            np.save(batch_file, batch_inputs)
            with open(batch_checksum_file, 'w') as f:
                f.write(md5(np.load(batch_file).tobytes()).hexdigest())
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
            flat_params[new_key] = np.array(value, dtype=np.float16)
        elif isinstance(value, dict):
            flat_params.update(flatten_parameters(value, f"{new_key}."))
        elif isinstance(value, list) and key == "layers":
            for i, layer in enumerate(value):
                if isinstance(layer, dict):  # Handle MLX's parameter dictionaries
                    layer_params = flatten_parameters(layer, f"{new_key}.{i}.")
                    flat_params.update(layer_params)
                else:
                    logging.warning(f"Skipping unexpected layer {new_key}.{i}: type={type(layer)}")
        else:
            logging.warning(f"Skipping non-array parameter {new_key}: type={type(value)}")
    return flat_params


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)
        fan_in = d_in
        fan_out = d_hidden
        scale = float(np.sqrt(1.0 / fan_in))
        self.linear1.weight = mx.random.normal(shape=(d_hidden, d_in), loc=0.0, scale=scale, dtype=mx.float16)
        self.linear1.bias = mx.zeros(d_hidden, dtype=mx.float16)
        fan_in = d_hidden
        fan_out = d_out
        scale = float(np.sqrt(1.0 / fan_in))
        self.linear2.weight = mx.random.normal(shape=(d_out, d_hidden), loc=0.0, scale=scale, dtype=mx.float16)
        self.linear2.bias = mx.zeros(d_out, dtype=mx.float16)
    def __call__(self, x):
        if not isinstance(x, mx.array):
            logging.error(f"FeedForward input is not mx.array: type={type(x)}")
            raise ValueError("FeedForward input must be mx.array")
        logging.debug(f"FeedForward input shape: {x.shape}")
        x = x.astype(mx.float16)
        x = self.linear1(x)
        logging.debug(f"After linear1 shape: {x.shape}")
        x = nn.gelu(x)
        x = mx.clip(x, -1e4, 1e4)
        x = self.linear2(x)
        logging.debug(f"After linear2 shape: {x.shape}")
        x = mx.clip(x, -1e4, 1e4)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf in feed-forward output")
            raise ValueError("NaN/Inf in feed-forward")
        if x.ndim < 2:
            logging.error(f"FeedForward output is scalar or 1D: shape={x.shape}")
            raise ValueError("FeedForward output must be at least 2D")
        return x.astype(mx.float16)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, n_heads, bias=True)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)  # Increase epsilon for stability
        self.ff = FeedForward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(0.2)
        # Ensure attention parameters are float16
        for key, param in self.attention.parameters().items():
            if isinstance(param, mx.array) and param.dtype != mx.float16:
                self.attention.parameters()[key] = param.astype(mx.float16)

    def __call__(self, x, mask=None, padding_mask=None):
        if not isinstance(x, mx.array):
            logging.error(f"TransformerLayer input is not mx.array: type={type(x)}")
            raise ValueError("TransformerLayer input must be mx.array")
        logging.debug(f"TransformerLayer input: shape={x.shape}, max={mx.max(x).item()}, min={mx.min(x).item()}")
        if mask is not None and mask.shape[2] != x.shape[1]:
            logging.error(f"Mask shape {mask.shape} does not match input sequence length {x.shape[1]}")
            raise ValueError("Mask shape mismatch")
        if padding_mask is not None:
            logging.debug(f"Padding mask: shape={padding_mask.shape}, dtype={padding_mask.dtype}")
            if padding_mask.shape != (x.shape[0], x.shape[1]):
                logging.error(f"Padding mask shape {padding_mask.shape} does not match input shape ({x.shape[0]}, {x.shape[1]})")
                raise ValueError("Padding mask shape mismatch")
        
        x = x.astype(mx.float16)
        try:
            attn_output = self.attention(x, x, x, mask=mask, key_padding_mask=padding_mask)
            logging.debug(f"Attention output: shape={attn_output.shape}, max={mx.max(attn_output).item()}, min={mx.min(attn_output).item()}")
            if mx.any(mx.isnan(attn_output)) or mx.any(mx.isinf(attn_output)):
                logging.error(f"NaN/Inf in attention output: max={mx.max(attn_output).item()}, min={mx.min(attn_output).item()}")
                raise ValueError("NaN/Inf in attention")
            attn_output = mx.clip(attn_output, -1e3, 1e3)
            attn_output = self.dropout(attn_output)
        except Exception as e:
            logging.error(f"Attention computation failed: {str(e)}")
            raise
        
        x = self.norm1(x + attn_output)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after first normalization")
            raise ValueError("NaN/Inf after norm1")
        
        ff_output = self.dropout(self.ff(x))
        if mx.any(mx.isnan(ff_output)) or mx.any(mx.isinf(ff_output)):
            logging.error("NaN/Inf in feed-forward output")
            raise ValueError("NaN/Inf in feed-forward")
        x = self.norm2(x + ff_output)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after second normalization")
            raise ValueError("NaN/Inf after norm2")
        
        return x.astype(mx.float16)

class BabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=8, d_ff=512, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = [TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.final_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.output = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        std = 0.02
        self.embedding.weight = mx.random.normal(
            shape=self.embedding.weight.shape, loc=0.0, scale=std, dtype=mx.float16
        )
        self.pos_embedding.weight = mx.random.normal(
            shape=self.pos_embedding.weight.shape, loc=0.0, scale=std, dtype=mx.float16
        )
        self.output.weight = mx.random.normal(
            shape=self.output.weight.shape, loc=0.0, scale=std, dtype=mx.float16
        )
        self.output.bias = mx.zeros(self.output.bias.shape, dtype=mx.float16)
        
        # Validate initialization
        for name, param in self.parameters().items():
            if isinstance(param, mx.array) and (mx.any(mx.isnan(param)) or mx.any(mx.isinf(param))):
                logging.error(f"NaN/Inf in initialized parameter {name}")
                raise ValueError(f"Invalid initialization for {name}")
        
        logging.info(f"Initialized BabyLLM: vocab_size={vocab_size}, d_model={d_model}, "
                     f"n_layers={n_layers}, n_heads={n_heads}, d_ff={d_ff}, max_len={max_len}")

    def __call__(self, x):
        logging.debug(f"BabyLLM.__call__ input: shape={x.shape}, dtype={x.dtype}, min={mx.min(x).item()}, max={mx.max(x).item()}")
        if not isinstance(x, mx.array):
            logging.error(f"BabyLLM input is not mx.array: type={type(x)}")
            return None
        if x.ndim != 2:
            logging.error(f"BabyLLM input is not 2D: shape={x.shape}")
            return None
        
        vocab_size_with_special = self.embedding.weight.shape[0]
        if mx.any(x < 0) or mx.any(x >= vocab_size_with_special):
            logging.error(f"Invalid tokens in input: min={mx.min(x).item()}, max={mx.max(x).item()}, vocab_size={vocab_size_with_special}")
            np.save(os.path.join(model_dir, f"invalid_input_tokens_{time.time()}.npy"), np.array(x))
            decoded = [tokenizer.decode(to_numpy_for_decode(x[i])) for i in range(x.shape[0])]
            logging.error(f"Invalid input sequences: {decoded}")
            return None
        
        # Create padding mask for input sequence
        try:
            seq_len = x.shape[1]
            padding_mask = (x != tokenizer.pad_token_id).astype(mx.bool_)  # Shape: (batch_size, seq_len)
            non_pad_counts = mx.sum(padding_mask, axis=1)
            padding_ratio = 1 - mx.mean(padding_mask).item()
            logging.debug(f"Padding mask: shape={padding_mask.shape}, dtype={padding_mask.dtype}, "
                        f"non_pad_counts={non_pad_counts.tolist()}, padding_ratio={padding_ratio:.2f}")
        except Exception as e:
            logging.error(f"Padding mask creation failed: {str(e)}")
            return None
        
        # Embedding lookup
        try:
            x = self.embedding(x).astype(mx.float32)
            x = mx.clip(x, -1e2, 1e2).astype(mx.float16)
            logging.debug(f"Embedding output: shape={x.shape}, min={mx.min(x).item()}, max={mx.max(x).item()}")
            if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
                logging.error("NaN/Inf in embedding output")
                np.save(os.path.join(model_dir, f"nan_embedding_output_{time.time()}.npy"), np.array(x))
                return None
        except Exception as e:
            logging.error(f"Embedding computation failed: {str(e)}")
            np.save(os.path.join(model_dir, f"failed_embedding_input_{time.time()}.npy"), np.array(x))
            return None
        
        # Positional embedding
        try:
            if seq_len > self.pos_embedding.weight.shape[0]:
                logging.error(f"Sequence length {seq_len} exceeds max_len {self.pos_embedding.weight.shape[0]}")
                return None
            positions = mx.arange(seq_len)
            pos_emb = self.pos_embedding(positions)
            logging.debug(f"Positional embedding: shape={pos_emb.shape}, min={mx.min(pos_emb).item()}, max={mx.max(pos_emb).item()}")
            x = x + pos_emb
            x = mx.clip(x, -1e2, 1e2)
            logging.debug(f"After positional embedding: shape={x.shape}, min={mx.min(x).item()}, max={mx.max(x).item()}")
            if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
                logging.error("NaN/Inf after positional embedding")
                np.save(os.path.join(model_dir, f"nan_pos_embedding_{time.time()}.npy"), np.array(x))
                return None
        except Exception as e:
            logging.error(f"Positional embedding failed: {str(e)}")
            return None
        
        # Causal mask
        try:
            mask = mx.triu(mx.ones((1, 1, seq_len, seq_len), dtype=mx.bool_), k=1)
            logging.debug(f"Causal mask: shape={mask.shape}, dtype={mask.dtype}")
            if mask.shape[2] != seq_len or mask.shape[3] != seq_len:
                logging.error(f"Mask shape mismatch: mask_shape={mask.shape}, expected=(1, 1, {seq_len}, {seq_len})")
                return None
        except Exception as e:
            logging.error(f"Mask creation failed: {str(e)}")
            return None
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            try:
                # Adjust padding mask for sliced input in loss computation
                layer_padding_mask = padding_mask if layer == self.layers[0] else padding_mask[:, :-1]  # Shape: (batch_size, seq_len-1) for subsequent layers
                x = layer(x, mask=mask, padding_mask=layer_padding_mask)
                x = mx.clip(x, -1e2, 1e2)
                logging.debug(f"Layer {i} output: shape={x.shape}, min={mx.min(x).item()}, max={mx.max(x).item()}")
                if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
                    logging.error(f"NaN/Inf in layer {i} output")
                    np.save(os.path.join(model_dir, f"nan_layer_{i}_output_{time.time()}.npy"), np.array(x))
                    return None
            except Exception as e:
                logging.error(f"Layer {i} failed: {str(e)}")
                np.save(os.path.join(model_dir, f"failed_layer_{i}_input_{time.time()}.npy"), np.array(x))
                return None
        
        # Final normalization and output
        try:
            x = mx.clip(x, -1e2, 1e2)
            x = self.final_norm(x)
            logging.debug(f"Final norm output: shape={x.shape}, min={mx.min(x).item()}, max={mx.max(x).item()}")
            if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
                logging.error("NaN/Inf after final normalization")
                np.save(os.path.join(model_dir, f"nan_final_norm_{time.time()}.npy"), np.array(x))
                return None
            x = self.output(x)
            logging.debug(f"Output logits: shape={x.shape}, min={mx.min(x).item()}, max={mx.max(x).item()}")
            if x.ndim < 2:
                logging.error(f"BabyLLM output is scalar or 1D: shape={x.shape}")
                return None
            if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
                logging.error("NaN/Inf in final output")
                np.save(os.path.join(model_dir, f"nan_output_{time.time()}.npy"), np.array(x))
                return None
            return x.astype(mx.float16)
        except Exception as e:
            logging.error(f"Final output computation failed: {str(e)}")
            return None
    
# Utility functions
def to_numpy_for_decode(array):
    return np.array(array) if isinstance(array, mx.array) else array

def validate_tokens(input_ids, vocab_size):
    invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
    if mx.any(invalid_mask):
        invalid_indices = mx.where(invalid_mask)[0]
        logging.error(f"Invalid tokens at indices: {invalid_indices.tolist()}")
        raise ValueError("Invalid tokens detected")
    return input_ids


def clip_gradients(grads, max_norm=0.5):  # Reduced max_norm for sdef clip_gradients(grads, max_norm=0.5):
    flat_grads = []
    for g in grads.values():
        if isinstance(g, mx.array):
            flat_grads.append(g.flatten())
        elif isinstance(g, dict):
            for sub_g in g.values():
                if isinstance(sub_g, mx.array):
                    flat_grads.append(sub_g.flatten())
    total_norm = mx.sqrt(sum(mx.sum(g * g) for g in flat_grads))
    logging.info(f"Gradient norm before clipping: {total_norm.item():.4f}")
    scale = mx.minimum(1.0, max_norm / (total_norm + 1e-8))
    def scale_gradient(g):
        if isinstance(g, mx.array):
            return g * scale
        elif isinstance(g, dict):
            return {k: scale_gradient(v) for k, v in g.items()}
        return g
    return {k: scale_gradient(g) for k, g in grads.items()}

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
    synonyms = set()
    for syn in wordnet.synsets(word):
        if syn.pos() in pos_tag:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
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


def paraphrase_question(question, max_retries=3):
    cache_file = os.path.join(cache_dir, f"paraphrase_{md5(question.encode()).hexdigest()}.txt")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.warning(f"Failed to load cached paraphrase for '{question}': {e}")
    
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
            # Cache valid result
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(paraphrased)
            return paraphrased
        except Exception as e:
            logging.warning(f"Paraphrasing attempt {attempt}/{max_retries} failed for '{question}': {e}")
            if attempt == max_retries:
                logging.error(f"All paraphrase attempts failed for '{question}'")
                return question
            time.sleep(2)
    return question

def back_translation(question, answer, src_lang="en", tgt_lang="fr"):
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
    sentences = sent_tokenize(text)[:100]
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
def dynamic_loss_scale(model, batch, fn, initial_scale=2.0):
    scale = initial_scale
    min_scale = 0.1
    while scale >= min_scale:
        try:
            logits = model(batch[:, :-1])
            if logits is None or mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
                scale /= 2.0
                continue
            loss_and_grad_fn = nn.value_and_grad(model, lambda m: fn(m, batch, scale))
            loss, grads = loss_and_grad_fn(model)
            if not (mx.isnan(loss) or mx.isinf(loss)):
                return loss, grads, scale
            scale /= 2.0
        except Exception as e:
            scale /= 2.0
    logging.error(f"Loss scale too low for batch, skipping (shape={batch.shape})")
    return None, None, None

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
    decoded = [tokenizer.decode(to_numpy_for_decode(x[i])) for i in range(min(x.shape[0], 3))]
    logging.debug(f"loss_fn_qa input decoded: {decoded}")
    
    logits = model(x[:, :-1])
    if logits is None:
        logging.error(f"Model returned None for input shape {x[:, :-1].shape}")
        np.save(os.path.join(model_dir, f"failed_qa_input_{time.time()}.npy"), np.array(x[:, :-1]))
        decoded_input = [tokenizer.decode(to_numpy_for_decode(x[i, :-1])) for i in range(min(x.shape[0], 3))]
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
        np.save(os.path.join(model_dir, f"nan_loss_qa_{time.time()}.npy"), np.array(logits))
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
    output_text = tokenizer.decode(to_numpy_for_decode(beams[0][0][0]), skip_special_tokens=True)
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
        np.save(os.path.join(model_dir, f"invalid_perplexity_tokens_{time.time()}.npy"), np.array(input_ids))
        return mx.array(1e9, dtype=mx.float32)
    
    logging.info(f"Computing perplexity for input shape: {input_ids.shape}, min_token={mx.min(input_ids).item()}, max_token={mx.max(input_ids).item()}")
    
    logits = model(input_ids[:, :-1])
    if logits is None:
        logging.error("Model returned None for perplexity computation")
        return mx.array(1e9, dtype=mx.float32)
    
    if mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
        logging.error("NaN/Inf in logits during perplexity computation")
        np.save(os.path.join(model_dir, f"nan_logits_perplexity_{time.time()}.npy"), np.array(logits))
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
        logging.info(f"Memory usage: RSS={mem_info.rss / 1024**2:.2f}MB, VMS={mem_info.vms / 1024**2:.2f}MB")
            

#

# Main block (replace the existing `if __name__ == '__main__':` block)
if __name__ == '__main__':
    # Setup logging
    log_queue = multiprocessing.Manager().Queue(maxsize=20000)
    log_handler = logging.handlers.QueueHandler(log_queue)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
    log_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)

    log_process = multiprocessing.Process(
        target=log_listener,
        args=(log_queue, os.path.expanduser(f'~/Baby_LLM/qa_training_{os.getpid()}.log'))
    )
    log_process.start()

    try:
        # Check disk space
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

        # Print versions
        print("numpy:", np.__version__)
        print("PyTorch:", torch.__version__)
        print("transformers:", transformers.__version__)
        print("MPS available:", torch.backends.mps.is_available())
        print("MLX version:", mx.__version__)

        # Create directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cleaned_dir, exist_ok=True)
        os.makedirs(tokenized_dir, exist_ok=True)

        # Process Gutenberg corpus to generate cleaned files
        print("Processing Gutenberg corpus...")
        start_time = time.time()
        text = ""
        try:
            filenames = [f"{i}.txt" for i in range(500, 540) if os.path.exists(os.path.join(gutenberg_dir, f"{i}.txt"))]
            logging.info(f"Found {len(filenames)} files in {gutenberg_dir}")
            if not filenames:
                logging.error("No files found in gutenberg_dir")
                raise FileNotFoundError("No Gutenberg files found")
            results = process_file_batch(filenames, tokenizer=None, batch_size=16)
            text = "".join(results)
            logging.info(f"Processed {len(filenames)} files, total text length: {len(text)}")
        except Exception as e:
            logging.error(f"Error during corpus processing: {str(e)}")
            raise

        # Load and train custom tokenizer
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer
        from transformers import PreTrainedTokenizerFast

        try:
            tokenizer_file = os.path.join(model_dir, "custom_tokenizer.json")
            if os.path.exists(tokenizer_file):
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
                logging.info(f"Loaded existing custom tokenizer from {tokenizer_file}")
            else:
                tokenizer = Tokenizer(models.BPE())
                tokenizer.pre_tokenizer = Whitespace()
                trainer = BpeTrainer(vocab_size=15000, special_tokens=["<PAD>", "<SEP>", "<EOS>", "<BOS>"])
                cleaned_files = [os.path.join(cleaned_dir, f) for f in os.listdir(cleaned_dir) if f.endswith('.cleaned.txt')]
                if not cleaned_files:
                    logging.error("No cleaned files found in cleaned_dir for tokenizer training")
                    raise FileNotFoundError("No cleaned files found for tokenizer training")
                logging.info(f"Training custom tokenizer on {len(cleaned_files)} files")
                tokenizer.train(files=cleaned_files, trainer=trainer)
                tokenizer.save(tokenizer_file)
                logging.info(f"Saved custom tokenizer to {tokenizer_file}")
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            
            tokenizer.pad_token = "<PAD>"
            tokenizer.eos_token = "<EOS>"
            tokenizer.bos_token = "<BOS>"
            tokenizer.sep_token = "<SEP>"
            
            logging.info(f"Tokenizer configured: vocab_size={tokenizer.vocab_size}, "
                         f"eos_token={tokenizer.eos_token} (ID={tokenizer.eos_token_id}), "
                         f"pad_token={tokenizer.pad_token} (ID={tokenizer.pad_token_id}), "
                         f"bos_token={tokenizer.bos_token} (ID={tokenizer.bos_token_id}), "
                         f"sep_token={tokenizer.sep_token} (ID={tokenizer.sep_token_id})")
            
            if tokenizer.vocab_size < 1000 or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
                logging.error("Invalid tokenizer configuration")
                raise ValueError("Invalid tokenizer configuration")
        except Exception as e:
            logging.error(f"Failed to load or train tokenizer: {str(e)}")
            raise

        # Log a sample tokenized sequence
        sample_text = "Question: Who is Huck's friend? Answer: Tom"
        sample_tokens = tokenizer(sample_text, padding=True, truncation=True, max_length=64, return_tensors="np")["input_ids"]
        logging.info(f"Sample tokenized sequence: {sample_tokens.tolist()}")
        logging.info(f"Sample decoded: {tokenizer.decode(sample_tokens[0], skip_special_tokens=False)}")

        # Split and concatenate texts
        texts = []
        for t in text.split("\n\n"):
            if t.strip():
                sentences = sent_tokenize(t)
                current_text = ""
                max_chunk_tokens = 120
                current_tokens = []
                for s in sentences:
                    sentence_tokens = tokenizer(s, add_special_tokens=False)["input_ids"]
                    if len(current_tokens) + len(sentence_tokens) < max_chunk_tokens:
                        current_text += " " + s
                        current_tokens.extend(sentence_tokens)
                    else:
                        if current_text.strip() and len(current_text) > 10:
                            texts.append(current_text.strip())
                        current_text = s
                        current_tokens = sentence_tokens
                if current_text.strip() and len(current_text) > 10:
                    texts.append(current_text.strip())
        logging.info(f"Collected {len(texts)} texts after concatenation, sample={[t[:50] + '...' for t in texts[:5]]}")

        # Filter texts
        max_size = 20 * 1024 * 1024 * 1024  # 20GB
        current_size = 0
        filtered_texts = []
        for i, t in enumerate(texts):
            size = len(t.encode("utf-8"))
            if current_size + size <= max_size and len(t.strip()) >= 10:
                try:
                    token_ids = tokenizer(t, truncation=True, add_special_tokens=True)["input_ids"]
                    if 5 <= len(token_ids) <= 2560:  # Cap at 10x max_length (256)
                        filtered_texts.append(t)
                        current_size += size
                    else:
                        logging.warning(f"Skipping text {i}: token_length={len(token_ids)}, text='{t[:50]}...'")
                except Exception as e:
                    logging.warning(f"Failed to tokenize text {i}: {str(e)}, text='{t[:50]}...'")
                    continue
            else:
                logging.warning(f"Skipping text {i}: size={size}, total_size={current_size / (1024**2):.2f}MB, text='{t[:50]}...'")
            if i % 1000 == 0:
                logging.info(f"Processed {i} texts, current_size={current_size / (1024**2):.2f}MB")
        logging.info(f"Collected {len(filtered_texts)} texts, ~{current_size / (1024**2):.2f}MB")
        print(f"Collected {len(filtered_texts)} texts, ~{current_size / (1024**2):.2f}MB")

        # Validate filtered_texts
        if not filtered_texts:
            logging.error("No valid texts after filtering")
            raise ValueError("No valid texts after filtering")

        # Tokenize corpus
        print("Tokenizing corpus...")
        try:
            input_ids = load_or_tokenize_texts(
                filtered_texts,
                tokenizer,
                tokenized_dir,
                "corpus",
                batch_size=100,
                max_length=128  # Match model’s max_len
            )
            input_ids = validate_tokens(input_ids, tokenizer.vocab_size + len(tokenizer.all_special_tokens))
            logging.info(f"Validated input_ids: shape={input_ids.shape}, min={mx.min(input_ids).item()}, max={mx.max(input_ids).item()}")
            print(f"Tokenized corpus shape: {input_ids.shape}")
            mx.eval(input_ids)
        except Exception as e:
            logging.error(f"Tokenization failed: {str(e)}")
            raise

        # ... rest of your code (validation corpus, QA pairs, pretraining, fine-tuning) ...
    except Exception as e:
        logging.error(f"Main process error: {str(e)}")
        raise
    finally:
        log_queue.put(None)
        log_process.join(timeout=30)
        if log_process.is_alive():
            log_process.terminate()

    # ... Previous code from updated main block (tokenizer, corpus processing, sentence concatenation, corpus tokenization) ...



        # Tokenize validation corpus
        print("Tokenizing validation corpus...")
        val_corpus = filtered_texts[-300:]
        val_corpus_ids = load_or_tokenize_texts(
            val_corpus,
            tokenizer,
            tokenized_dir,
            "val_corpus",
            batch_size=100,
            max_length=128  # Match model’s max_len
        )
        val_corpus_ids = validate_tokens(val_corpus_ids, tokenizer.vocab_size + len(tokenizer.all_special_tokens))
        logging.info(f"Validated val_corpus_ids: shape={val_corpus_ids.shape}, min={mx.min(val_corpus_ids).item()}, max={mx.max(val_corpus_ids).item()}")
        print(f"Tokenized validation corpus shape: {val_corpus_ids.shape}")
        mx.eval(val_corpus_ids)



    # Define QA pairs
    qa_pairs = [
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
        ("In the Odyssey, Odysseus was helped by what goddess?", "Athena")
    ]
    qa_pairs = [(re.sub(r'\s+', ' ', q.strip()), a.strip()) for q, a in qa_pairs if q and a]

    # Generate synthetic QA pairs
    print("Generating synthetic QA pairs...")
    for i, text in enumerate(filtered_texts[:100]):
        qa_pairs.extend(generate_qa_pairs(text, max_pairs=10))
    print(f"Added {len(qa_pairs)} synthetic QA pairs")

    # Validate and augment QA pairs
    def is_valid_qa_pair(question, answer):
        if not question.strip() or not answer.strip():
            logging.warning(f"Invalid QA pair: empty question or answer (question='{question[:50]}...', answer='{answer[:50]}...')")
            return False
        try:
            text = f"Question: {question} Answer: {answer}"
            tokens = tokenizer(
                text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=64  # Reduced for QA pairs
            )["input_ids"]
            non_pad_count = np.sum(tokens != tokenizer.pad_token_id)
            if tokens.shape[1] > 64 or non_pad_count < 4:  # Relaxed threshold
                logging.warning(f"Invalid QA pair: tokens_shape={tokens.shape}, non_pad_count={non_pad_count}, text='{text[:50]}...'")
                return False
            if np.any(tokens < 0) or np.any(tokens >= tokenizer.vocab_size + len(tokenizer.all_special_tokens)):
                logging.warning(f"Invalid tokens in QA pair: min={np.min(tokens)}, max={np.max(tokens)}, text='{text[:50]}...'")
                return False
            if any(c not in string.printable for c in text):
                logging.warning(f"Non-printable characters in QA pair: text='{text[:50]}...'")
                return False
            return True
        except Exception as e:
            logging.warning(f"QA pair validation failed: {str(e)}, text='{text[:50]}...'")
            return False

    augmented_pairs = []
    for question, answer in qa_pairs:
        augmented_pairs.append((question, answer))
        syn_question = synonym_replacement(question)
        if is_valid_qa_pair(syn_question, answer):
            augmented_pairs.append((syn_question, answer))
        # Skip context_aware_synonym if using custom tokenizer
        para_question = paraphrase_question(question)
        if is_valid_qa_pair(para_question, answer):
            augmented_pairs.append((para_question, answer))
        bt_question, bt_answer = back_translation(question, answer)
        if is_valid_qa_pair(bt_question, bt_answer):
            augmented_pairs.append((bt_question, bt_answer))

    qa_pairs = augmented_pairs
    logging.info(f"Total QA pairs after augmentation: {len(qa_pairs)}, Sample: {qa_pairs[:5]}")

    # Preprocess QA data
    qa_texts = [f"Question: {q} Answer: {a}" for q, a in qa_pairs]
    logging.info(f"Total QA texts: {len(qa_texts)}, Sample: {qa_texts[:5]}")

    # Clear tokenized QA cache
    for npy_file in glob.glob(os.path.join(tokenized_dir, "qa_*.npy")):
        safe_remove(npy_file)

    # Tokenize QA data
    print("Tokenizing QA data...")
    qa_input_ids = load_or_tokenize_texts(
        qa_texts,
        tokenizer,
        tokenized_dir,
        "qa",
        batch_size=100,
        max_length=64
    )
    qa_input_ids = validate_tokens(qa_input_ids, tokenizer.vocab_size + len(tokenizer.all_special_tokens))
    print(f"Tokenized QA data shape: {qa_input_ids.shape}")
    mx.eval(qa_input_ids)

        # Initialize model
    vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
    generative_model = BabyLLM(
        vocab_size=vocab_size_with_special,
        d_model=128,
        n_layers=4,
        n_heads=8,
        d_ff=512,
        max_len=128
    )

    # Pretraining
    print("Starting pretraining...")
    optimizer = optim.Adam(learning_rate=1e-4)
    scheduler = CosineWarmup(learning_rate=1e-4, warmup_steps=100, total_steps=1000)
    state = [generative_model, optimizer.state]
    batch_size_lm = 2
    for epoch in range(3):
        indices = mx.random.permutation(input_ids.shape[0])  # Keep as MLX array
        for i in range(0, len(indices), batch_size_lm):
            batch_indices = indices[i:i + batch_size_lm]
            batch = input_ids[batch_indices]
            loss, grads, scale = dynamic_loss_scale(generative_model, batch, loss_fn_lm)
            if loss is None:
                continue
            grads = clip_gradients(grads)
            optimizer.update(generative_model, grads)
            mx.eval(generative_model.parameters(), optimizer.state)
            logging.info(f"Epoch {epoch}, Step {i//batch_size_lm}, Loss: {loss.item():.4f}")
        mx.savez(pretrain_checkpoint_path, **flatten_parameters(generative_model.parameters()))

    # Fine-tuning
    print("Starting fine-tuning...")
    batch_size_qa = 16
    for epoch in range(3):
        indices = mx.random.permutation(qa_input_ids.shape[0])
        for i in range(0, len(indices), batch_size_qa):
            batch_indices = indices[i:i + batch_size_qa]
            batch = qa_input_ids[batch_indices]
            loss, grads, scale = dynamic_loss_scale(generative_model, batch, loss_fn_qa)
            if loss is None:
                continue
            grads = clip_gradients(grads)
            optimizer.update(generative_model, grads)
            mx.eval(generative_model.parameters(), optimizer.state)
            logging.info(f"Fine-tuning Epoch {epoch}, Step {i//batch_size_qa}, Loss: {loss.item():.4f}")

            # Validation prompts
    validation_prompts = [
            ("Who killed Hamlet's dad?", "Claudius"),
            ("Who is Huck's friend?", "Tom"),
            ("Who loved Juliet?", "Romeo"),
            ("Who ignored Ophelia?", "Hamlet"),
            ("Who tricked Othello?", "Iago"),
            ("Who killed Mercutio?", "Tybalt"),
            ("In The Odyssey, who is the cyclops encountered by Odysseus?", "Polyphemus"),
            ("In Jane Eyre, who is the governess at Thornfield Hall?", "Jane Eyre"),
            ("Who was Odysseus' wife?", "Penelope"),
            ("What did the Red Queen believe before breakfast?", "Six impossible things"),
            ("Who captained the Pequod?", "Ahab"),
            ("What is the name of Ahab's ship?", "Pequod")
        ]
    val_texts = [f"Question: {q} Answer: {a}" for q, a in validation_prompts]
    val_input_ids = load_or_tokenize_texts(
        val_texts,
        tokenizer,
        tokenized_dir,
        "val",
        batch_size=50,
        max_length=64
    )
    val_input_ids = validate_tokens(val_input_ids, tokenizer.vocab_size + len(tokenizer.all_special_tokens))
    print(f"Tokenized validation data shape: {val_input_ids.shape}")
    mx.eval(val_input_ids)

    # Evaluate
    metrics = evaluate_qa(generative_model, tokenizer, validation_prompts)
    logging.info(f"Validation metrics: {metrics}")