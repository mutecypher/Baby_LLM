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
from transformers import AutoTokenizer, MarianMTModel, MarianTokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer
from nltk.corpus import wordnet
import nltk

import matplotlib.pyplot as plt
from pathlib import Path
import ssl
##import evaluate
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

# Setup logging
logging.basicConfig(
    filename=f'qa_training_{multiprocessing.current_process().name}.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s'
)





# Compiled regex patterns for cleaning


patterns = [
    (re.compile(r'\n{3,}', re.MULTILINE), '\n\n'),  # Only keep this for enhanced_clean_text
]

fallback_metadata_patterns = [
    re.compile(r'^\s*\*{3}\s*START OF THIS PROJECT GUTENBERG EBOOK.*$', re.IGNORECASE),
    re.compile(r'^\s*\*{3}\s*END OF THIS PROJECT GUTENBERG EBOOK.*$', re.IGNORECASE),
    re.compile(r'^\s*\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK.*$', re.IGNORECASE),
    re.compile(r'^\s*\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK.*$', re.IGNORECASE),
    re.compile(r'Produced by.*?\n', re.IGNORECASE),
    re.compile(r'This eBook is for the use of.*?\n', re.IGNORECASE),
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

def preprocess_text(text):
    logging.debug(f"Preprocessing text: length={len(text)}, sample={text[:200]}")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    cleaned_text = text.strip()
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
    



def is_narrative(text):
    if not isinstance(text, str) or not text.strip():
        logging.warning("Invalid input to is_narrative: empty or non-string")
        return False
    try:
        return len(text) > 20  # Only check for minimum length
    except Exception as e:
        logging.error(f"is_narrative failed for text (length={len(text)}): {str(e)}")
        return False

def process_file(filename, tokenizer):
    file_path = os.path.join(gutenberg_dir, filename)
    cleaned_file_path = os.path.join(cleaned_dir, f"{filename}.cleaned.txt")
    logging.debug(f"Processing file: {filename}, path: {file_path}")
    
    if not os.path.isfile(file_path):
        logging.warning(f"File not found: {filename}")
        return ""
    if os.path.getsize(file_path) > 10 * 1024 * 1024:
        logging.info(f"Skipping {filename}: File too large (>100MB)")
        return ""
    
    if os.path.exists(cleaned_file_path):
        logging.info(f"Using cached cleaned file: {filename}")
        with open(cleaned_file_path, "r", encoding="utf-8") as f:
            return f.read() + "\n\n"
    
    try:
        logging.info(f"Reading raw text for {filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        logging.debug(f"Raw text length for {filename}: {len(raw_text)}")
        
        ##with open(os.path.join(cleaned_dir, f"{filename}.raw.txt"), "w", encoding="utf-8") as f:
          ##  f.write(raw_text)
        
        stripped_text = simple_strip_headers(raw_text, include_dedication=True, filename=filename)
        ##with open(os.path.join(cleaned_dir, f"{filename}.stripped.txt"), "w", encoding="utf-8") as f:
     ##       f.write(stripped_text)
        logging.debug(f"Saved stripped file: {filename}.stripped.txt")
        
        preprocessed_text = preprocess_text(stripped_text)
       ## with open(os.path.join(cleaned_dir, f"{filename}.preprocessed.txt"), "w", encoding="utf-8") as f:
         ##   f.write(preprocessed_text)
        logging.debug(f"Saved preprocessed file: {filename}.preprocessed.txt")
        
        cleaned_text = enhanced_clean_text(preprocessed_text)
        ##with open(os.path.join(cleaned_dir, f"{filename}.cleaned.debug.txt"), "w", encoding="utf-8") as f:
        ##    f.write(cleaned_text)
        logging.debug(f"Saved cleaned debug file: {filename}.cleaned.debug.txt")
        
        if not cleaned_text.strip():
            logging.warning(f"Empty cleaned text for {filename}")
            return ""
        
        if (is_narrative(cleaned_text) and len(cleaned_text) > 50 and
            len(tokenizer(cleaned_text, max_length=128, truncation=True)["input_ids"]) > 20):
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            logging.info(f"Saved cleaned file: {filename}")
            return cleaned_text + "\n\n"
        else:
            logging.info(f"Skipping {filename}: Not narrative, too short, or untokenizable (length={len(cleaned_text)})")
            return ""
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return ""

# Updated process_file_batch
def process_file_batch(filenames, tokenizer, batch_size=64):
    for i in range(0, len(filenames), batch_size):
        batch = filenames[i:i + batch_size]
        process_file_with_tokenizer = partial(process_file, tokenizer=tokenizer)
        try:
            with ProcessPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(process_file_with_tokenizer, batch))
            yield from results
        except Exception as e:
            logging.error(f"Multiprocessing batch {i//batch_size} failed: {str(e)}")
            results = [process_file_with_tokenizer(f) for f in batch]
            yield from results
        gc.collect()
        log_memory_usage()

def safe_remove(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.debug(f"Removed file: {file_path}")
        else:
            logging.debug(f"File not found for removal: {file_path}")
    except Exception as e:
        logging.warning(f"Failed to remove file {file_path}: {str(e)}")
        
def load_or_tokenize_texts(texts, tokenizer, output_dir, prefix, batch_size=1000, max_length=128):
    os.makedirs(output_dir, exist_ok=True)
    inputs = []
    vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
    
    valid_texts = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            logging.warning(f"Skipping invalid text at index {i}: '{t[:50] if isinstance(t, str) else t}'")
            continue
        if any(ord(c) > 0x10FFFF for c in t):
            logging.warning(f"Skipping text with invalid characters at index {i}: {t[:50]}")
            continue
        try:
            tokens = tokenizer(t, max_length=max_length, truncation=True)["input_ids"]
            if any(token < 0 or token >= vocab_size_with_special for token in tokens):
                logging.warning(f"Skipping text with invalid tokens at index {i}: {t[:50]}")
                continue
            valid_texts.append(t)
        except Exception as e:
            logging.warning(f"Tokenization failed for text at index {i}: {str(e)}")
            continue
    
    if not valid_texts:
        logging.error("No valid texts provided for tokenization")
        raise ValueError("No valid texts provided for tokenization")
    
    for i in range(0, len(valid_texts), batch_size):
        batch_file = os.path.join(output_dir, f"{prefix}_batch_{i//batch_size}.npy")
        batch_checksum_file = f"{batch_file}.md5"
        batch = valid_texts[i:i+batch_size]
        if not batch:
            logging.warning(f"Empty batch at index {i}")
            continue
        
        if os.path.exists(batch_file) and os.path.exists(batch_checksum_file):
            try:
                with open(batch_checksum_file, 'r') as f:
                    stored_checksum = f.read().strip()
                with open(batch_file, 'rb') as f:
                    current_checksum = md5(f.read()).hexdigest()
                if stored_checksum == current_checksum:
                    batch_inputs = np.load(batch_file)
                    if batch_inputs.shape[1] != max_length:
                        batch_inputs = np.pad(
                            batch_inputs,
                            ((0, 0), (0, max_length - batch_inputs.shape[1])),
                            mode="constant",
                            constant_values=tokenizer.pad_token_id
                        ) if batch_inputs.shape[1] < max_length else batch_inputs[:, :max_length]
                    if np.any(batch_inputs < 0) or np.any(batch_inputs >= vocab_size_with_special):
                        logging.warning(f"Corrupted tokens in {batch_file}, retokenizing")
                    else:
                        inputs.append(batch_inputs)
                        continue
            except Exception as e:
                logging.warning(f"Failed to validate {batch_file}: {e}, retokenizing")
        
        try:
            batch_inputs = tokenizer(
                batch,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )["input_ids"]
            if np.any(batch_inputs < 0) or np.any(batch_inputs >= vocab_size_with_special):
                logging.error(f"Invalid tokens in batch at index {i}: {batch_inputs}")
                np.save(os.path.join(output_dir, f"invalid_batch_{i}.npy"), batch_inputs)
                continue
            if batch_inputs.shape[1] != max_length:
                batch_inputs = np.pad(
                    batch_inputs,
                    ((0, 0), (0, max_length - batch_inputs.shape[1])),
                    mode="constant",
                    constant_values=tokenizer.pad_token_id
                ) if batch_inputs.shape[1] < max_length else batch_inputs[:, :max_length]
            np.save(batch_file, batch_inputs)
            with open(batch_file, 'rb') as f:
                checksum = md5(f.read()).hexdigest()
            with open(batch_checksum_file, 'w') as f:
                f.write(checksum)
            logging.info(f"Saved tokenized batch to {batch_file} with checksum")
            inputs.append(batch_inputs)
        except Exception as e:
            logging.error(f"Tokenization failed for batch at index {i}: {str(e)}")
            continue
    
    if not inputs:
        logging.error("No valid batches tokenized after processing all texts")
        raise ValueError("No valid batches tokenized")
    
    input_ids = mx.array(np.concatenate(inputs, axis=0), dtype=mx.int32)
    if mx.any(input_ids < 0) or mx.any(input_ids >= vocab_size_with_special):
        logging.error(f"Invalid token IDs in {prefix}: {input_ids}")
        np.save(os.path.join(output_dir, f"invalid_final_{prefix}.npy"), np.array(input_ids))
        raise ValueError(f"Invalid token IDs in {prefix}")
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
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        # Initialize attention weights (optional, as nn.MultiHeadAttention handles this internally)
        # If you need custom initialization, access internal weights like self.attention.qkv_weight
        for key, param in self.attention.parameters().items():
            if isinstance(param, mx.array) and param.dtype != mx.float16:
                self.attention.parameters()[key] = param.astype(mx.float16)

    def __call__(self, x, mask=None):
        if not isinstance(x, mx.array):
            logging.error(f"TransformerLayer input is not mx.array: type={type(x)}")
            raise ValueError("TransformerLayer input must be mx.array")
        logging.debug(f"TransformerLayer input shape: {x.shape}, max: {mx.max(x).item()}, min: {mx.min(x).item()}")
        x = x.astype(mx.float16)
        
        # Compute attention output using nn.MultiHeadAttention
        attn_output = self.attention(x, x, x, mask=mask)  # Pass x as query, key, and value
        if mx.any(mx.isnan(attn_output)) or mx.any(mx.isinf(attn_output)):
            logging.error(f"NaN/Inf in attention output: max={mx.max(attn_output).item()}, min={mx.min(attn_output).item()}")
            raise ValueError("NaN/Inf in attention")
        attn_output = mx.clip(attn_output, -1e4, 1e4)
        attn_output = self.dropout(attn_output)
        
        # Residual connection and normalization
        x = self.norm1(x + attn_output)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after first normalization")
            raise ValueError("NaN/Inf after norm1")
        
        # Feed-forward and second residual connection
        ff_output = self.dropout(self.ff(x))
        if mx.any(mx.isnan(ff_output)) or mx.any(mx.isinf(ff_output)):
            logging.error("NaN/Inf in feed-forward output")
            raise ValueError("NaN/Inf in feed-forward")
        x = self.norm2(x + ff_output)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after second normalization")
            raise ValueError("NaN/Inf after norm2")
        
        if x.ndim < 2:
            logging.error(f"TransformerLayer output is scalar or 1D: shape={x.shape}")
            raise ValueError("TransformerLayer output must be at least 2D")
        return x.astype(mx.float16)
    


class BabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = [TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        # Initialize weights with smaller scale for stability
        std = 0.02  # Smaller standard deviation to prevent extreme values
        self.embedding.weight = mx.random.normal(
            self.embedding.weight.shape, loc=0.0, scale=std, dtype=mx.float16
        )
        self.pos_embedding.weight = mx.random.normal(
            self.pos_embedding.weight.shape, loc=0.0, scale=std, dtype=mx.float16
        )
        self.output.weight = mx.random.normal(
            self.output.weight.shape, loc=0.0, scale=std, dtype=mx.float16
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
        if not isinstance(x, mx.array):
            logging.error(f"BabyLLM input is not mx.array: type={type(x)}")
            raise ValueError("BabyLLM input must be mx.array")
        if x.ndim != 2:
            logging.error(f"BabyLLM input is not 2D: shape={x.shape}")
            raise ValueError("BabyLLM input must be 2D (batch, seq_len)")
        logging.debug(f"BabyLLM input shape: {x.shape}, max: {mx.max(x).item()}, min: {mx.min(x).item()}")
        
        # Validate input tokens before embedding
        vocab_size_with_special = self.embedding.weight.shape[0]
        if mx.any(x < 0) or mx.any(x >= vocab_size_with_special):
            logging.error(f"Invalid tokens in input: min={mx.min(x).item()}, max={mx.max(x).item()}, "
                          f"vocab_size={vocab_size_with_special}")
            np.save(os.path.join(model_dir, f"invalid_input_tokens_{time.time()}.npy"), np.array(x))
            raise ValueError("Invalid tokens in input")
        
        # Embedding lookup and scaling in float32 for stability
        x = self.embedding(x).astype(mx.float32) * mx.sqrt(self.d_model / 10.0)  # Reduced scaling
        x = x.astype(mx.float16)  # Convert back to float16
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf in embedding output")
            raise ValueError("NaN/Inf in embedding")
        
        # Positional embedding
        seq_len = x.shape[1]
        positions = mx.arange(seq_len)
        x = x + self.pos_embedding(positions)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after positional embedding")
            raise ValueError("NaN/Inf after pos_embedding")
        
        # Causal mask for attention
        mask = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=1)
        mask = mask[None, None, :, :]
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask)
            if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
                logging.error(f"NaN/Inf in layer {i} output")
                raise ValueError(f"NaN/Inf in layer {i}")
        
        # Final normalization and output
        x = self.final_norm(x)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after final normalization")
            raise ValueError("NaN/Inf after final_norm")
        
        x = self.output(x)
        if x.ndim < 2:
            logging.error(f"BabyLLM output is scalar or 1D: shape={x.shape}")
            raise ValueError("BabyLLM output must be at least 2D")
        
        return x.astype(mx.float16)
    
# Utility functions
def to_numpy_for_decode(array):
    return np.array(array) if isinstance(array, mx.array) else array

def clip_gradients(grads, max_norm=0.5):  # Reduced max_norm for stricter clipping
    flat_grads = []
    for g in grads.values():
        if g is not None and isinstance(g, mx.array):
            if mx.any(mx.isnan(g)) or mx.any(mx.isinf(g)):
                logging.warning(f"NaN/Inf in gradients before clipping")
                g = mx.where(mx.isnan(g) | mx.isinf(g), mx.zeros_like(g), g)
            flat_grads.append(g.flatten())
        elif isinstance(g, dict):
            for sub_g in g.values():
                if sub_g is not None and isinstance(sub_g, mx.array):
                    if mx.any(mx.isnan(sub_g)) or mx.any(mx.isinf(sub_g)):
                        logging.warning(f"NaN/Inf in sub-gradients before clipping")
                        sub_g = mx.where(mx.isnan(sub_g) | mx.isinf(sub_g), mx.zeros_like(sub_g), sub_g)
                    flat_grads.append(sub_g.flatten())
    if not flat_grads:
        logging.warning("No valid gradients to clip")
        return grads
    total_norm = mx.sqrt(sum(mx.sum(g * g) for g in flat_grads))
    logging.info(f"Gradient norm: {total_norm.item()}")
    scale = mx.minimum(1.0, max_norm / (total_norm + 1e-8))
    def scale_gradient(g):
        if isinstance(g, mx.array):
            return mx.where(mx.isnan(g) | mx.isinf(g), mx.zeros_like(g), g * scale)
        elif isinstance(g, dict):
            return {k: scale_gradient(v) for k, v in g.items()}
        return g
    clipped_grads = {k: scale_gradient(g) for k, g in grads.items()}
    # Validate clipped gradients
    for k, g in clipped_grads.items():
        if isinstance(g, mx.array) and (mx.any(mx.isnan(g)) or mx.any(mx.isinf(g))):
            logging.error(f"NaN/Inf in clipped gradient {k}")
            raise ValueError(f"Invalid clipped gradient {k}")
    return clipped_grads

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
    global fill_mask
    fill_mask = None  # Initialize to avoid undefined variable
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Selected device for BERT pipeline: {device}")
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Attempt {attempt}/{max_retries} to initialize BERT fill-mask pipeline...")
            bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=bert_token, cache_dir=cache_dir)
            bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased", token=bert_token, cache_dir=cache_dir)
            bert_model.to(device)
            fill_mask = pipeline("fill-mask", model=bert_model, tokenizer=bert_tokenizer, device=device)
            logging.info("Initialized BERT fill-mask pipeline successfully")
            return
        except Exception as e:
            logging.error(f"Attempt {attempt} failed: {str(e)}")
            if attempt == max_retries:
                logging.warning("All attempts failed, setting fill_mask to None")
                fill_mask = None
            time.sleep(2)
            

# Replace existing context_aware_synonym
def context_aware_synonym(question, prob=0.3):
    try:
        words = nltk.word_tokenize(question)
        pos_tags = nltk.pos_tag(words)
        new_words = words.copy()
        for i, (word, pos) in enumerate(pos_tags):
            if random.random() < prob and pos.startswith(('NN', 'VB', 'JJ', 'RB')):
                masked = words[:i] + ['[MASK]'] + words[i+1:]
                masked_sent = ' '.join(masked)
                try:
                    predictions = fill_mask_pipeline(masked_sent, top_k=1)
                    new_words[i] = predictions[0]['token_str']
                except Exception as e:
                    logging.debug(f"Failed to predict synonym for word '{word}': {str(e)}")
                    continue
        return ' '.join(new_words)
    except Exception as e:
        logging.warning(f"Context-aware synonym replacement failed for question '{question}': {str(e)}")
        return question
    
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
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        entities = [w for w, pos in tagged if pos == 'NNP']
        if len(entities) >= 2:
            question = f"Who is associated with {entities[0]} in this context?"
            answer = entities[1]
            qa_pairs.append((question, answer))
        if len(qa_pairs) >= max_pairs:
            break
    return qa_pairs

# Loss functions with dynamic scaling
def dynamic_loss_scale(model, batch, fn, initial_scale=2.0):  # Reduced initial scale
    scale = initial_scale
    min_scale = 0.1  # Minimum scale to prevent excessive reduction
    while scale >= min_scale:
        loss_and_grad_fn = nn.value_and_grad(model, lambda m: fn(m, batch, scale))
        loss, grads = loss_and_grad_fn(model)
        if not (mx.isnan(loss) or mx.isinf(loss)):
            return loss, grads, scale
        scale /= 2.0
        logging.warning(f"Reducing loss scale to {scale} due to NaN/Inf")
    logging.error("Loss scale too low, training unstable")
    raise ValueError("Loss scale too low, training unstable")

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
    logits = model(x[:, :-1]).astype(mx.float16)
    logits = mx.clip(logits, -1e9, 1e9)
    targets = x[:, 1:].astype(mx.int32)
    mask = mx.zeros(targets.shape, dtype=mx.bool_)
    for i in range(targets.shape[0]):
        decoded = tokenizer.decode(to_numpy_for_decode(x[i]))
        answer_idx = decoded.find("Answer:") + len("Answer:")
        if answer_idx < len("Answer:"):
            continue
        tokenized = tokenizer(decoded, return_tensors="np")["input_ids"][0]
        answer_token_idx = len(tokenizer(decoded[:answer_idx], return_tensors="np")["input_ids"][0])
        mask[i, answer_token_idx:] = True
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)
    loss = nn.losses.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="none"
    ).astype(mx.float16)
    masked_loss = mx.where(mask_flat, loss, 0.0)
    return mx.mean(masked_loss) * loss_scale

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

# QA evaluation
##squad_metric = evaluate.load("squad")

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
    return squad_metric.compute(predictions=predictions, references=references)

# Perplexity computation
def compute_perplexity(model, input_ids):
    logits = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean")
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

# Main block
if __name__ == '__main__':
    # Suppress warnings

    # Pre-download NLTK resources
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)

    # Print versions once
    print("The version of numpy is ", np.__version__)
    print("PyTorch version ", torch.__version__)
    print("transfomers version ", transformers.__version__)
    print("MPS available:", torch.backends.mps.is_available())
    print("the random version is ", random.__file__)
    print("About to start the directory setup")

    # Logging setup per process
    logging.basicConfig(
        filename=f'qa_training_{multiprocessing.current_process().name}.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s'
    )

    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(tokenized_dir, exist_ok=True)

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.3",
            token=HF_TOKEN,
            cache_dir=cache_dir,
            use_fast=False,
            clean_up_tokenization_spaces=False
        )
        logging.info("Tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {str(e)}")
        raise
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'sep_token': '<SEP>'})

    # Initialize BERT pipeline
    initialize_bert_pipeline(BERT_Token, cache_dir)

    # Load and process Gutenberg corpus
    print("Loading and processing Gutenberg corpus...")
    start_time = time.time()
    text = ""
    try:
        filenames = os.listdir(gutenberg_dir)
        logging.info(f"Found {len(filenames)} files in {gutenberg_dir}")
        for result in process_file_batch(filenames, tokenizer):
            if result:
                text += result
    except Exception as e:
        logging.error(f"Error during corpus processing: {str(e)}")
        raise
    print(f"Finished loading and cleaning, Total Time: {time.time() - start_time:.2f}s")
    print(f"Total processed text length: {len(text)}")
    logging.info(f"Total processed text length: {len(text)}")
    gc.collect()
    log_memory_usage()
    pause = input("okay, proceed")

    # Split texts
    texts = [t for t in text.split("\n\n") if t.strip()]
    max_size = 20 * 1024 * 1024 * 1024  # 20GB
    current_size = 0
    filtered_texts = []
    for i, t in enumerate(texts):
        size = len(t.encode("utf-8"))
        if current_size + size <= max_size:
            filtered_texts.append(t)
            current_size += size
        else:
            break
        if i % 1000 == 0:
            logging.info(f"Processed {i} texts, current_size={current_size / (1024**2):.2f}MB")
    logging.info(f"Collected {len(filtered_texts)} texts, ~{current_size / (1024**2):.2f}MB")
    print(f"Collected {len(filtered_texts)} texts, ~{current_size / (1024**2):.2f}MB")
    pause = input("Okay, now proceed")

    # Tokenize corpus
    print("Tokenizing corpus...")
    input_ids = load_or_tokenize_texts(
        filtered_texts,
        tokenizer,
        tokenized_dir,
        "corpus",
        batch_size=1000,
        max_length=128
    )
    print(f"Tokenized corpus shape: {input_ids.shape}")
    pause = input("Wait some more")
    mx.eval(input_ids)

    val_corpus = filtered_texts[-300:]
    val_corpus_ids = load_or_tokenize_texts(
        val_corpus,
        tokenizer,
        tokenized_dir,
        "val_corpus",
        batch_size=100,
        max_length=128
    )
    print(f"Tokenized validation corpus shape: {val_corpus_ids.shape}")
    mx.eval(val_corpus_ids)

    # Load and augment QA data
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
            return False
        try:
            text = f"Question: {question} Answer: {answer}"
            tokens = tokenizer(text, return_tensors="np")["input_ids"]
            if tokens.shape[1] > 128 or tokens.shape[1] < 5:
                return False
            if np.any(tokens < 0) or np.any(tokens >= tokenizer.vocab_size + len(tokenizer.all_special_tokens)):
                return False
            if any(c not in string.printable for c in text):
                return False
            return True
        except Exception:
            return False

    augmented_pairs = []
    for question, answer in qa_pairs:
        augmented_pairs.append((question, answer))
        syn_question = synonym_replacement(question)
        if is_valid_qa_pair(syn_question, answer):
            augmented_pairs.append((syn_question, answer))
        ctx_question = context_aware_synonym(question)
        if is_valid_qa_pair(ctx_question, answer):
            augmented_pairs.append((ctx_question, answer))
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
        max_length=128
    )
    print(f"Tokenized QA data shape: {qa_input_ids.shape}")
    mx.eval(qa_input_ids)

    # Validate qa_input_ids
    vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
    invalid_mask = (qa_input_ids < 0) | (qa_input_ids >= vocab_size_with_special)
    if mx.any(invalid_mask):
        invalid_indices = mx.where(invalid_mask)[0]
        logging.error(f"Invalid tokens in qa_input_ids at indices: {invalid_indices.tolist()}")
        for idx in invalid_indices:
            logging.error(f"Invalid sequence {idx}: {qa_input_ids[idx].tolist()}")
            decoded = tokenizer.decode(to_numpy_for_decode(qa_input_ids[idx]))
            logging.error(f"Decoded invalid sequence {idx}: {decoded}")
        raise ValueError("Invalid tokens in qa_input_ids")

    # Validation data
        # Validation data
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
        batch_size=100,
        max_length=128
    )
    print(f"Tokenized validation data shape: {val_input_ids.shape}")
    mx.eval(val_input_ids)

    # Pretraining setup
    batch_size_lm = 8
    accumulation_steps_lm = 8
    num_epochs_lm = 2
    total_steps_lm = num_epochs_lm * (len(input_ids) // batch_size_lm)
    optimizer_lm = optim.AdamW(learning_rate=CosineWarmup(5e-7, 1000, total_steps_lm), weight_decay=0.05)
    optimizer_lm.state = {k: v.astype(mx.float16) if isinstance(v, mx.array) else v for k, v in optimizer_lm.state.items()}
    train_losses_gen = []

    # Clear tokenized files
    for npy_file in glob.glob(os.path.join(tokenized_dir, "*.npy")):
        safe_remove(npy_file)

    generative_model = BabyLLM(vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens))
    embedding_weights = generative_model.embedding.weight
    if mx.any(mx.isnan(embedding_weights)) or mx.any(mx.isinf(embedding_weights)):
        logging.error("NaN/Inf in embedding weights after initialization")
        raise ValueError("Invalid embedding weights")
    weights_path = os.path.join(model_dir, "baby_llm_pretrain.npz")
    os.makedirs(model_dir, exist_ok=True)
    try:
        weights = np.load(weights_path)
        weights_dict = dict(weights)
        for k, v in weights_dict.items():
            if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                logging.warning(f"NaN/Inf in weight {k}, replacing with zeros")
                v[np.isnan(v) | np.isinf(v)] = 0
                weights_dict[k] = v
        np.savez(weights_path, **weights_dict)
        generative_model.load_weights(weights_path)
        logging.info("Loaded pretrained weights")
    except FileNotFoundError:
        logging.info("Pretrained weights not found. Using initialized weights.")
    except Exception as e:
        logging.error(f"Failed to load weights: {e}")
        raise
    mx.eval(generative_model.parameters())

    pretrain_checkpoint_path = os.path.join(model_dir, "pretrain_checkpoint.npz")
    start_epoch_lm = 0
    if os.path.exists(pretrain_checkpoint_path):
        try:
            checkpoint = np.load(pretrain_checkpoint_path)
            start_epoch_lm = int(checkpoint["epoch"])
            generative_model.load_weights(pretrain_checkpoint_path)
            optimizer_lm_state = np.load(os.path.join(model_dir, "optimizer_lm_state.npz"))
            optimizer_lm.state = {k: mx.array(v, dtype=mx.float16) if isinstance(v, np.ndarray) else v for k, v in optimizer_lm_state.items()}
            logging.info(f"Resumed pretraining from epoch {start_epoch_lm}")
        except Exception as e:
            logging.error(f"Failed to load pretraining checkpoint: {str(e)}")
            start_epoch_lm = 0


    # Before pretraining loop
    embedding_weights = generative_model.embedding.weight
    if mx.any(mx.isnan(embedding_weights)) or mx.any(mx.isinf(embedding_weights)):
        std = 0.02
        generative_model.embedding.weight = mx.random.normal(
            generative_model.embedding.weight.shape, loc=0.0, scale=std, dtype=mx.float16
        )
        logging.info("Reinitialized embedding weights due to NaN/Inf in checkpoint")
        
        
            
    # Pretraining loop
    for epoch in range(start_epoch_lm, num_epochs_lm):
        logging.info(f"Starting language modeling epoch {epoch + 1}/{num_epochs_lm}")
        indices = mx.random.permutation(len(input_ids))
        accumulated_grads = None
        print(f"Pretraining epoch {epoch + 1}")
        for i in range(0, len(input_ids), batch_size_lm):
            batch_idx = i // batch_size_lm
            batch_indices = indices[i:i + batch_size_lm]
            if batch_indices.shape[0] < batch_size_lm:
                continue
            batch = mx.take(input_ids, batch_indices, axis=0)
            vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
            if mx.any(batch < 0) or mx.any(batch >= vocab_size_with_special):
                logging.warning(f"Invalid tokens in pretraining batch {batch_idx}")
                np.save(os.path.join(model_dir, f"invalid_pretraining_batch_{batch_idx}.npy"), np.array(batch))
                continue
            non_pad_counts = mx.sum(batch != tokenizer.pad_token_id, axis=1)
            valid_mask = non_pad_counts > 0.1 * batch.shape[1]
            valid_mask_np = np.array(valid_mask)
            valid_indices = np.where(valid_mask_np)[0]
            if valid_indices.size == 0 or valid_indices.size < 2:
                logging.warning(f"Pretraining batch {batch_idx} has excessive padding or too few sequences")
                continue
            valid_indices = mx.array(valid_indices, dtype=mx.int32)
            batch = mx.take(batch, valid_indices, axis=0)
            mx.eval(batch)
            if mx.any(mx.isnan(batch.astype(mx.float32))) or mx.any(mx.isinf(batch.astype(mx.float32))):
                logging.warning(f"Pretraining batch {batch_idx} contains NaN/Inf tokens")
                np.save(os.path.join(model_dir, f"nan_pretraining_batch_{batch_idx}.npy"), np.array(batch))
                continue
            logging.info(f"Pretraining batch {batch_idx}: shape={batch.shape}, "
                        f"max_token={int(mx.max(batch))}, "
                        f"min_token={int(mx.min(batch))}, "
                        f"padding_ratio={(1 - mx.mean(batch != tokenizer.pad_token_id).item()):.2f}")
            try:
                np.save(os.path.join(model_dir, f"pretraining_batch_{batch_idx}_pre_embedding.npy"), np.array(batch))
                validate_model_params(generative_model, batch_idx)
                loss, grads, scale = dynamic_loss_scale(generative_model, batch, loss_fn_lm, initial_scale=2.0)
                grads = clip_gradients(grads, max_norm=0.5)
                grads = scale_gradients(grads, 1.0 / scale)
                if any(mx.any(mx.isnan(g) | mx.isinf(g)) for g in grads.values() if isinstance(g, mx.array)):
                    logging.warning(f"Skipping pretraining batch {batch_idx} due to NaN/Inf gradients")
                    np.save(os.path.join(model_dir, f"failed_batch_{batch_idx}.npy"), np.array(batch))
                    continue
                scaled_grads = scale_gradients(grads, 1.0 / accumulation_steps_lm)
                accumulated_grads = add_grads(accumulated_grads, scaled_grads)
                if (batch_idx + 1) % accumulation_steps_lm == 0:
                    optimizer_lm.update(generative_model, accumulated_grads)
                    mx.eval(generative_model.parameters(), optimizer_lm.state)
                    accumulated_grads = None
                train_losses_gen.append(loss.item() / scale)
                logging.info(f"Pretraining batch {batch_idx}, Loss: {loss.item() / scale:.4f}")
                if batch_idx % 100 == 0:
                    print(f"Pretraining batch {batch_idx + 1}")
                    log_memory_usage()
                    gc.collect()
            except Exception as e:
                logging.error(f"Pretraining batch {batch_idx} failed: {str(e)}")
                np.save(os.path.join(model_dir, f"failed_batch_{batch_idx}.npy"), np.array(batch))
                continue
        perplexity = compute_perplexity(generative_model, val_corpus_ids)
        logging.info(f"Epoch {epoch + 1}, Validation Perplexity: {perplexity.item():.4f}")
        generative_model.save_weights(os.path.join(model_dir, f"baby_llm_pretrain_epoch_{epoch}.npz"))
        np.savez(pretrain_checkpoint_path, epoch=epoch, **generative_model.parameters())
    # Fine-tuning setup
    batch_size_qa = 32
    accumulation_steps = 16
    num_epochs_qa = 50
    total_steps_qa = num_epochs_qa * (len(qa_input_ids) // batch_size_qa)
    optimizer_gen = optim.AdamW(learning_rate=CosineWarmup(1e-6, 1000, total_steps_qa), weight_decay=0.05)
    patience = 10
    best_val_em = 0.0
    patience_counter = 0
    val_losses_gen = []

    # Validate QA input
    vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
    invalid_mask = (qa_input_ids < 0) | (qa_input_ids >= vocab_size_with_special)
    if mx.any(invalid_mask):
        invalid_indices = mx.where(invalid_mask)[0]
        logging.error(f"Invalid tokens in qa_input_ids at indices: {invalid_indices.tolist()}")
        raise ValueError("Invalid tokens in qa_input_ids")

    # Validate embedding weights
    embedding_weights = generative_model.embedding.weight
    if mx.any(mx.isnan(embedding_weights)) or mx.any(mx.isinf(embedding_weights)):
        std = float(np.sqrt(2.0 / (vocab_size_with_special + generative_model.d_model)))
        generative_model.embedding.weight = mx.random.normal(
            generative_model.embedding.weight.shape, loc=0.0, scale=std, dtype=mx.float16
        )
        logging.info("Reinitialized embedding weights due to NaN/Inf")

    # Curriculum learning: Sort QA pairs by question length
    qa_pairs.sort(key=lambda x: len(x[0].split()))
    log_memory_usage()

    # Fine-tuning loop
    for epoch in range(num_epochs_qa):
        logging.info(f"Starting generative QA epoch {epoch + 1}/{num_epochs_qa}")
        if epoch < num_epochs_qa // 2:
            indices = mx.array(np.random.permutation(len(qa_input_ids) // 2))
        else:
            indices = mx.array(np.random.permutation(len(qa_input_ids)))
        accumulated_grads = None
        print(f"fine tuning epoch {epoch}")
        for i in range(0, len(indices), batch_size_qa):
            batch_idx = i // batch_size_qa
            batch_indices = indices[i:i + batch_size_qa]
            if batch_indices.shape[0] < batch_size_qa:
                continue
            if mx.any(batch_indices < 0) or mx.any(batch_indices >= len(qa_input_ids)):
                logging.warning(f"Invalid batch indices in QA batch {batch_idx}: {batch_indices.tolist()}")
                continue
            batch = mx.take(qa_input_ids, batch_indices, axis=0)
            if mx.any(batch < 0) or mx.any(batch >= vocab_size_with_special):
                logging.warning(f"Invalid tokens in QA batch {batch_idx}")
                np.save(os.path.join(model_dir, f"invalid_qa_batch_{batch_idx}.npy"), np.array(batch))
                continue
            non_pad_counts = mx.sum(batch != tokenizer.pad_token_id, axis=1)
            valid_mask = non_pad_counts > 0.1 * batch.shape[1]
            valid_mask_np = np.array(valid_mask)
            valid_indices = np.where(valid_mask_np)[0]
            if valid_indices.size == 0 or valid_indices.size < 2:
                logging.warning(f"QA batch {batch_idx} has excessive padding or too few sequences")
                continue
            valid_indices = mx.array(valid_indices, dtype=mx.int32)
            batch = mx.take(batch, valid_indices, axis=0)
            mx.eval(batch)
            if mx.any(mx.isnan(batch.astype(mx.float32))) or mx.any(mx.isinf(batch.astype(mx.float32))):
                logging.warning(f"QA batch {batch_idx} contains NaN/Inf tokens")
                continue
            logging.info(f"QA batch {batch_idx}: shape={batch.shape}, "
                         f"max_token={int(mx.max(batch))}, "
                         f"min_token={int(mx.min(batch))}, "
                         f"padding_ratio={(1 - mx.mean(batch != tokenizer.pad_token_id).item()):.2f}")
            try:
                np.save(os.path.join(model_dir, f"qa_batch_{batch_idx}_pre_embedding.npy"), np.array(batch))
                decoded_batch = [tokenizer.decode(to_numpy_for_decode(batch[j])) for j in range(batch.shape[0])]
                logging.info(f"QA batch {batch_idx} decoded: {decoded_batch}")
                loss, grads, scale = dynamic_loss_scale(generative_model, batch, loss_fn_qa, initial_scale=1.0)
                grads = scale_gradients(grads, 1.0 / scale)
                if any(mx.any(mx.isnan(g) | mx.isinf(g)) for g in grads.values() if isinstance(g, mx.array)):
                    logging.warning(f"Skipping QA batch {batch_idx} due to NaN/Inf gradients")
                    continue
                scaled_grads = scale_gradients(grads, 1.0 / accumulation_steps)
                accumulated_grads = add_grads(accumulated_grads, scaled_grads)
                if (batch_idx + 1) % accumulation_steps == 0:
                    accumulated_grads = clip_gradients(accumulated_grads, max_norm=1.0)
                    optimizer_gen.update(generative_model, accumulated_grads)
                    mx.eval(generative_model.parameters(), optimizer_gen.state)
                    accumulated_grads = None
                train_losses_gen.append(loss.item() / scale)
                logging.info(f"QA batch {batch_idx}, Loss: {loss.item() / scale:.4f}")
                if batch_idx % 100 == 0:
                    print(f"fine-tuning batch {batch_idx + 1}")
                    log_memory_usage()
            except Exception as e:
                logging.error(f"QA batch {batch_idx} failed: {str(e)}")
                continue
        val_loss_gen = loss_fn_qa(generative_model, val_input_ids, 1.0)
        val_metrics = evaluate_qa(generative_model, tokenizer, validation_prompts)
        val_em = val_metrics["exact_match"]
        val_f1 = val_metrics["f1"]
        val_losses_gen.append(val_loss_gen.item())
        logging.info(f"Epoch {epoch + 1}, Val Loss: {val_loss_gen.item():.4f}, EM: {val_em:.4f}, F1: {val_f1:.4f}")
        if val_em > best_val_em:
            best_val_em = val_em
            generative_model.save_weights(os.path.join(model_dir, f"baby_llm_qa_best_epoch_{epoch}.npz"))
            np.savez(os.path.join(model_dir, f"optimizer_gen_state_epoch_{epoch}.npz"), **optimizer_gen.state)
            tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

    # Save final model
    generative_model.save_weights(os.path.join(model_dir, "baby_llm_qa_final.npz"))
    np.savez(os.path.join(model_dir, "optimizer_gen_state_final.npz"), **optimizer_gen.state)

    # Plot losses
    plt.plot(np.convolve(train_losses_gen, np.ones(10)/10, mode='valid'), label='Train Loss')
    plt.plot(np.convolve(val_losses_gen, np.ones(10)/10, mode='valid'), label='Val Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'loss_plot.png'))
    plt.close()

    # Interactive testing
    while True:
        prompt = input("Enter a question (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        answer = generate_answer(generative_model, tokenizer, prompt)
        print(f"Generative Answer: {answer}")