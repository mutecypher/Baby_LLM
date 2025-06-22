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

from collections import Counter

from transformers import BertForMaskedLM
import glob
import string   
from functools import partial 
from secrets import HF_TOKEN, BERT_Token
# Fix NLTK SSL issue and download required data

import warnings
# Near the top, after warnings import
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*", message="A parameter name that contains.*")

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




def normalize_answer(s):
    """Normalize answer by removing punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text, flags=re.IGNORECASE)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Compute F1 score between prediction and ground_truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def exact_match_score(prediction, ground_truth):
    """Compute exact match score between prediction and ground_truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def compute_squad_metrics(predictions, references):
    """Compute SQuAD-like metrics (exact match and F1) for predictions and references."""
    if len(predictions) != len(references):
        logging.error(f"Mismatch in predictions ({len(predictions)}) and references ({len(references)})")
        return {"exact_match": 0.0, "f1": 0.0}
    
    exact_scores = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_text = pred["prediction_text"]
        # Handle multiple possible ground truths (take the best score)
        ground_truths = ref["answers"]["text"]
        best_em = 0
        best_f1 = 0.0
        
        for ground_truth in ground_truths:
            em = exact_match_score(pred_text, ground_truth)
            f1 = f1_score(pred_text, ground_truth)
            best_em = max(best_em, em)
            best_f1 = max(best_f1, f1)
        
        exact_scores.append(best_em)
        f1_scores.append(best_f1)
    
    exact_match = 100.0 * sum(exact_scores) / len(exact_scores) if exact_scores else 0.0
    f1 = 100.0 * sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return {"exact_match": exact_match, "f1": f1}


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

# Directory definitions (place near existing directory setup, around line 215)
base_dir = "~/Baby_LLM"
cache_dir = os.path.expanduser(os.path.join(base_dir, "cache"))
model_dir = os.path.expanduser(os.path.join(base_dir, "model"))
data_dir = os.path.expanduser(os.path.join(base_dir, "data"))
gutenberg_dir = os.path.join(data_dir, "gutenberg")
cleaned_dir = os.path.join(data_dir, "cleaned")
tokenized_dir = os.path.join(data_dir, "tokenized")
stripped_dir = os.path.join(data_dir, "stripped")  # New directory
os.makedirs(cleaned_dir, exist_ok=True)
os.makedirs(tokenized_dir, exist_ok=True)
os.makedirs(stripped_dir, exist_ok=True)


def preprocess_text(text):
    logging.debug(f"Preprocessing text: length={len(text)}, sample={text[:200]}")
    # Only remove control characters, preserve valid UTF-8
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    cleaned_text = text.strip()
    logging.debug(f"Preprocessed text: length={len(cleaned_text)}, sample={cleaned_text[:200]}")
    return cleaned_text

def enhanced_clean_text(raw_text, tokenizer):
    logging.debug(f"Raw text length: {len(raw_text)}, sample: {raw_text[:200]}")
    text = raw_text
    for pattern, repl in patterns:
        matches = pattern.findall(text)
        if matches:
            logging.debug(f"Pattern {pattern.pattern} matched {len(matches)} times, sample: {matches[:5]}")
        text = pattern.sub(repl, text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    text = text.replace('—', '-').replace('“', '"').replace('”', '"')
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    normalization_dict = {"Wauxhall": "Vauxhall", "childer'": "children", "Parleyment": "Parliament"}
    for old, new in normalization_dict.items():
        text = text.replace(old, new)
    text = re.sub(r'\s+', ' ', text).strip()
    logging.debug(f"Cleaned text: length={len(text)}, sample: {text[:200]}")
    tokens = tokenizer(text[:500], return_tensors="np")["input_ids"]
    unk_count = np.sum(tokens == tokenizer.unk_token_id)
    if unk_count > 0:
        logging.debug(f"Found {unk_count} <unk> tokens in cleaned text: {text[:200]}")
    return text

def simple_strip_headers(text, include_dedication=True, filename="unknown"):
    # Normalize quotes and dashes before splitting
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'").replace('—', '-')
    lines = text.splitlines()
    total_lines = len(lines)
    logging.debug(f"Total lines in text: {total_lines} for file {filename}")

    start_idx = 0
    end_idx = total_lines
    metadata_terms = [
        "gutenberg", "produced by", "ebook", "posting date", "release date",
        "character set encoding", "start of this project", "end of this project",
        "language:", "[ebook #", "title:", "author:", "by ", "release date:", "posting date:",
        "*** start", "*** end", "this ebook is for the use of", "project gutenberg license"
    ]
    metadata_count = 0
    content_started = False

    # Scan for start of content
    for i, line in enumerate(lines[:2000]):
        line_lower = line.lower().strip()
        if any(term in line_lower for term in metadata_terms):
            start_idx = i + 1
            metadata_count += 1
            continue
        elif re.match(r'^\s*(chapter|section|part)\s+[ivxlc0-9]', line_lower, re.IGNORECASE):
            start_idx = i
            content_started = True
            break
        elif (len(line.strip()) > 50 and 
              not re.match(r'^\s*[-*=\s]+$', line_lower) and 
              not re.match(r'^\s*(title|author|by|produced by|release date|language|character set encoding)', line_lower, re.IGNORECASE) and 
              not content_started):
            if len(line.strip()) < 100 and i > 0 and any(term in lines[i-1].lower() for term in metadata_terms):
                metadata_count += 1
                continue
            start_idx = i
            content_started = True
            break

    # Scan for end marker
    for i in range(total_lines - 1, max(total_lines - 2000, -1), -1):
        if re.search(r'\*{3}\s*end of.*gutenberg ebook', lines[i], re.IGNORECASE):
            end_idx = i
            logging.debug(f"Found end marker at line {i}: {lines[i][:100]}")
            break

    stripped_lines = lines[start_idx:end_idx]
    cleaned_lines = []
    for line in stripped_lines:
        line_lower = line.lower().strip()
        if (len(line.strip()) >= 10 and 
            not any(term in line_lower for term in metadata_terms) and 
            not re.match(r'^\s*[-*=\s]+$', line_lower) and 
            not re.match(r'^\s*(title|author|by|produced by|release date|language|character set encoding)', line_lower, re.IGNORECASE)):
            cleaned_lines.append(line)
        else:
            metadata_count += 1

    stripped_text = "\n".join(cleaned_lines).strip()
    if not stripped_text:
        logging.warning(f"Empty output from simple_strip_headers for {filename}")
        return ""

    # Log sample of stripped text for debugging
    logging.debug(f"Stripped text sample for {filename}: {stripped_text[:200]}")
    remaining_metadata = [term for term in metadata_terms if term in stripped_text.lower()]
    if remaining_metadata or metadata_count > 5000:
        logging.warning(f"Persistent metadata in {filename}: {remaining_metadata}, count={metadata_count}")
        with open(os.path.join(cleaned_dir, f"{filename}.metadata_issue.txt"), "w", encoding="utf-8") as f:
            f.write(f"Raw text:\n{text[:1000]}\n\nStripped text:\n{stripped_text[:1000]}")
        return ""
    return stripped_text

def is_narrative(text):
    if not isinstance(text, str) or not text.strip():
        logging.warning("Invalid input to is_narrative: empty or non-string")
        return False
    try:
        if len(text) <= 15:
            logging.warning(f"Text too short: {text[:100]}")
            return False
        if re.search(r'<(?!\bunk\b)\w+.*?>', text):
            logging.warning(f"Text contains HTML-like pattern: {text[:100]}")
            return False
        return True
    except Exception as e:
        logging.error(f"is_narrative failed: {str(e)}")
        return False

def identify_high_tokens(tokenizer, vocab_size_threshold=0.9, sample_text=None):
    vocab_size = tokenizer.vocab_size
    threshold = int(vocab_size * vocab_size_threshold)  # e.g., 29494.8
    high_tokens = {}
    
    # Optionally analyze sample text
    if sample_text:
        tokens = tokenizer(sample_text, max_length=128, truncation=True, return_tensors="np")["input_ids"]
        unique_tokens = np.unique(tokens)
        for token_id in unique_tokens:
            if token_id >= threshold:
                decoded = tokenizer.decode([token_id], skip_special_tokens=False)
                high_tokens[token_id] = decoded
    
    # Analyze entire vocabulary
    for token_id in range(threshold, vocab_size):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        high_tokens[token_id] = decoded
    
    logging.info(f"Identified {len(high_tokens)} high tokens (>= {threshold})")
    with open(os.path.join(cleaned_dir, "high_tokens_mapping.txt"), "w", encoding="utf-8") as f:
        for token_id, decoded in high_tokens.items():
            f.write(f"Token ID: {token_id}, Decoded: {decoded}\n")
    
    return high_tokens
    
def remove_high_token_content(text, high_tokens, replace_with="<unk>"):
    # Escape special regex characters and create a pattern
    ##patterns = [re.escape(decoded.strip()) for decoded in high_tokens.values() if decoded.strip()]
    ##if not patterns:
    logging.debug("No high token patterns to remove")
    return text
    
    pattern = "|".join(patterns)
    try:
        if replace_with:
            cleaned_text = re.sub(pattern, replace_with, text, flags=re.UNICODE)
            logging.debug(f"Replaced {len(re.findall(pattern, text))} high token instances with '{replace_with}'")
        else:
            cleaned_text = re.sub(pattern, "", text, flags=re.UNICODE)
            logging.debug(f"Removed {len(re.findall(pattern, text))} high token instances")
        return cleaned_text
    except Exception as e:
        logging.warning(f"Failed to remove high tokens: {str(e)}")
        return text

def process_file(filename, tokenizer, high_tokens=None):
    file_path = os.path.join(gutenberg_dir, filename)
    cleaned_file_path = os.path.join(cleaned_dir, f"{filename}.cleaned.txt")
    debug_file_path = os.path.join(cleaned_dir, f"{filename}.debug.txt")
    stripped_file_path = os.path.join(stripped_dir, f"{filename}.stripped.txt")
    tokenized_file = os.path.join(tokenized_dir, f"{filename}.tokens.npy")
    logging.info(f"Processing file: {filename}")

    if not os.path.isfile(file_path):
        logging.warning(f"File not found: {file_path}")
        return ""
    if os.path.getsize(file_path) > 10 * 1024 * 1024:
        logging.info(f"Skipping {filename}: File too large (>10MB)")
        return ""

    if os.path.exists(cleaned_file_path):
        try:
            with open(cleaned_file_path, "r", encoding="utf-8") as f:
                cleaned_text = f.read()
            if cleaned_text.strip() and is_narrative(cleaned_text):
                logging.info(f"Using cached cleaned file: {cleaned_file_path}")
                return cleaned_text + "\n\n"
        except Exception as e:
            logging.warning(f"Invalid cached file {cleaned_file_path}: {str(e)}")
            safe_remove(cleaned_file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        logging.debug(f"Raw text length for {filename}: {len(raw_text)}")

        # Save raw text for debugging
        with open(os.path.join(cleaned_dir, f"{filename}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)

        # Process and save stripped text
        stripped_text = simple_strip_headers(raw_text, include_dedication=True, filename=filename)
        with open(stripped_file_path, "w", encoding="utf-8") as f:
            f.write(stripped_text)
        logging.debug(f"Saved stripped text: {stripped_file_path}")

        preprocessed_text = preprocess_text(stripped_text)
        with open(os.path.join(cleaned_dir, f"{filename}.preprocessed.txt"), "w", encoding="utf-8") as f:
            f.write(preprocessed_text)
        logging.debug(f"Saved preprocessed text: {filename}.preprocessed.txt")

        # Remove high token content
        if high_tokens:
            preprocessed_text = remove_high_token_content(preprocessed_text, high_tokens, replace_with="<unk>")
            with open(os.path.join(cleaned_dir, f"{filename}.high_tokens_removed.txt"), "w", encoding="utf-8") as f:
                f.write(preprocessed_text)
            logging.debug(f"Saved text after high token removal: {filename}.high_tokens_removed.txt")

        cleaned_text = enhanced_clean_text(preprocessed_text)
        with open(debug_file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logging.debug(f"Saved debug text: {filename}.debug.txt")

        # Check for residual metadata
        metadata_terms = ["gutenberg", "ebook", "posting date", "release date", "[ebook #"]
        if any(term in cleaned_text.lower() for term in metadata_terms):
            logging.warning(f"Residual metadata in {filename} after cleaning: {cleaned_text[:200]}")
            with open(os.path.join(cleaned_dir, f"{filename}.residual_metadata.txt"), "w", encoding="utf-8") as f:
                f.write(cleaned_text[:1000])
            return ""

        if not cleaned_text.strip():
            logging.warning(f"Empty cleaned text for {filename}. Raw text sample: {raw_text[:200]}")
            return ""

        # Tokenize and debug
        tokens = tokenizer(cleaned_text, max_length=128, truncation=True, return_tensors="np")["input_ids"]
        # In process_file, around line 400
        vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
        if (tokens < 0).any() or (tokens >= vocab_size_with_special).any():
            invalid_tokens = tokens[(tokens < 0) | (tokens >= vocab_size_with_special)]
            logging.error(f"Invalid tokens in {filename}: {invalid_tokens}")
            with open(os.path.join(cleaned_dir, f"{filename}.invalid_tokens.txt"), "w") as f:
                f.write(f"Invalid tokens: {invalid_tokens}\nText sample: {cleaned_text[:500]}")
            return ""
        if tokens.max() >= vocab_size_with_special * 0.9:
            logging.warning(f"High token values in {filename}: max={tokens.max()}")
            decoded = tokenizer.decode(tokens[0])
            with open(os.path.join(cleaned_dir, f"{filename}.high_tokens.txt"), "w") as f:
                f.write(f"Max token: {tokens.max()}\nSequence: {decoded[:1000]}")
            # Save tokenized output without replacement (for debugging)
            np.save(tokenized_file, tokens)
            logging.info(f"Saved tokenized file: {tokenized_file}")

        if is_narrative(cleaned_text) and len(tokens[0]) > 20:
            with open(cleaned_file_path, "w") as f:
                f.write(cleaned_text)
            logging.info(f"Saved cleaned file: {cleaned_file_path}")
            logging.info(f"Tokenized {filename}: tokens={len(tokens[0])}, max={tokens.max()}, min={tokens.min()}")
            return cleaned_text + "\n\n"
        else:
            logging.info(f"Skipping {filename}: Not narrative or too short (length={len(cleaned_text)})")
            return ""
        if not cleaned_text.strip():
            logging.warning(f"Empty cleaned text for {filename}. Saving intermediates for debugging.")
            with open(os.path.join(cleaned_dir, f"{filename}.failed.txt"), "w", encoding="utf-8") as f:
                f.write(f"Raw: {raw_text[:500]}\nStripped: {stripped_text[:500]}\nPreprocessed: {preprocessed_text[:500]}\n")
            return ""
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return ""
    
# Updated process_file_batch
def process_file_batch(filenames, tokenizer, high_tokens=None, batch_size=64):
    for i in range(0, len(filenames), batch_size):
        batch = filenames[i:i + batch_size]
        process_file_with_tokenizer = partial(process_file, tokenizer=tokenizer, high_tokens=high_tokens)
        try:
            with ProcessPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(process_file_with_tokenizer, batch))
            logging.info(f"Processed batch {i//batch_size} with multiprocessing")
            yield from results
        except Exception as e:
            logging.error(f"Multiprocessing batch {i//batch_size} failed: {str(e)}, falling back to sequential")
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
        
def load_or_tokenize_texts(texts, tokenizer, output_dir, prefix, batch_size=100, max_length=128):
    os.makedirs(output_dir, exist_ok=True)
    inputs = []
    vocab_size_with_special = tokenizer.vocab_size + len(tokenizer.all_special_tokens)

    valid_texts = []
    tokenized_files = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            logging.warning(f"Skipping invalid text at index {i}: {t[:50] if isinstance(t, str) else t}")
            continue
        if any(ord(c) > 0x10FFFF for c in t):
            logging.warning(f"Skipping text with invalid characters at index {i}: {t[:50]}")
            continue
        # Check for pre-tokenized file
        filename = f"{i}.txt"  # Match Gutenberg filename
        tokenized_file = os.path.join(output_dir, f"{filename}.tokens.npy")
        if os.path.exists(tokenized_file):
            try:
                tokens = np.load(tokenized_file)
                if tokens.shape[1] != max_length:
                    tokens = np.pad(
                        tokens,
                        ((0, 0), (0, max_length - tokens.shape[1])),
                        mode="constant",
                        constant_values=tokenizer.pad_token_id
                    ) if tokens.shape[1] < max_length else tokens[:, :max_length]
                if np.any(tokens < 0) or np.any(tokens >= vocab_size_with_special):
                    logging.warning(f"Corrupted tokens in {tokenized_file}, retokenizing")
                else:
                    valid_texts.append(t)
                    tokenized_files.append(tokens)
                    continue
            except Exception as e:
                logging.warning(f"Failed to load {tokenized_file}: {e}, retokenizing")
        try:
            tokens = tokenizer(t, max_length=max_length, truncation=True)["input_ids"]
            if any(token < 0 or token >= vocab_size_with_special for token in tokens):
                logging.error(f"Invalid tokens in text {i}: {tokens}")
                with open(os.path.join(output_dir, f"invalid_text_{i}.txt"), "w") as f:
                    f.write(f"Text: {t[:500]}\nTokens: {tokens}")
                continue
            valid_texts.append(t)
            tokenized_files.append(np.array(tokens).reshape(1, -1))
        except Exception as e:
            logging.warning(f"Tokenization failed for text {i}: {str(e)}")
            continue

    if not valid_texts:
        logging.error("No valid texts for tokenization")
        raise ValueError("No valid texts for tokenization")

    for i in range(0, len(valid_texts), batch_size):
        batch_file = os.path.join(output_dir, f"{prefix}_batch_{i//batch_size}.npy")
        batch_checksum_file = f"{batch_file}.md5"
        batch_texts = valid_texts[i:i+batch_size]
        batch_tokens = tokenized_files[i:i+batch_size]
        if not batch_texts:
            logging.warning(f"Empty batch at index {i}")
            continue

        logging.info(f"Processing batch {i//batch_size} with {len(batch_texts)} texts")

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
                        logging.info(f"Loaded cached batch: {batch_file}")
                        continue
            except Exception as e:
                logging.warning(f"Failed to validate {batch_file}: {e}, retokenizing")

        try:
            batch_inputs = np.concatenate(batch_tokens, axis=0)
            if np.any(batch_inputs < 0) or np.any(batch_inputs >= vocab_size_with_special):
                logging.error(f"Invalid tokens in batch {i//batch_size}: {batch_inputs}")
                np.save(os.path.join(output_dir, f"invalid_batch_{i//batch_size}.npy"), batch_inputs)
                with open(os.path.join(output_dir, f"invalid_batch_{i//batch_size}.txt"), "w") as f:
                    for j, text in enumerate(batch_texts):
                        f.write(f"Text {j}: {text[:500]}\n")
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
            logging.info(f"Saved tokenized batch: {batch_file}")
            inputs.append(batch_inputs)
        except Exception as e:
            logging.error(f"Tokenization failed for batch {i//batch_size}: {str(e)}")
            with open(os.path.join(output_dir, f"failed_batch_{i//batch_size}.txt"), "w") as f:
                for j, text in enumerate(batch_texts):
                    f.write(f"Text {j}: {text[:500]}\n")
            continue

    if not inputs:
        logging.error("No valid batches tokenized after processing all batches")
        raise ValueError("No valid batches tokenized after processing all batches")

    input_ids = mx.array(np.concatenate(inputs, axis=0), dtype=mx.int32)
    if mx.any(input_ids < 0) or mx.any(input_ids >= vocab_size_with_special):
        logging.error(f"Invalid token IDs in {prefix}: {np.asarray(input_ids)}")
        try:
            np.save(os.path.join(output_dir, f"invalid_final_{prefix}.npy"), np.asarray(input_ids))
            with open(os.path.join(output_dir, f"invalid_final_{prefix}_texts.txt"), "w") as f:
                for idx in np.where((np.asarray(input_ids) < 0) | (np.asarray(input_ids) >= vocab_size_with_special))[0][:10]:
                    text_idx = idx // max_length
                    if text_idx < len(valid_texts):
                        f.write(f"Text {text_idx}: {valid_texts[text_idx][:500]}\n")
        except Exception as e:
            logging.error(f"Failed to save invalid_final_{prefix}.npy: {str(e)}")
        raise ValueError(f"Invalid token IDs in {prefix}")

    # Save token stats for debugging
    debug_tokens_path = os.path.join(output_dir, f"debug_tokens_{prefix}.npy")
    try:
        np.save(debug_tokens_path, np.asarray(input_ids))
        logging.info(f"Saved debug tokens to: {debug_tokens_path}")
    except Exception as e:
        logging.error(f"Failed to save debug_tokens_{prefix}.npy: {str(e)}")
        raise

    debug_stats_path = os.path.join(output_dir, f"debug_tokens_{prefix}_stats.txt")
    try:
        with open(debug_stats_path, "w") as f:
            f.write(f"Token count: {input_ids.size}\n")
            f.write(f"Max token: {int(mx.max(input_ids))}\n")
            f.write(f"Min token: {int(mx.min(input_ids))}\n")
            invalid_mask = (input_ids < 0) | (input_ids >= vocab_size_with_special)
            if mx.any(invalid_mask):
                invalid_indices = mx.where(invalid_mask)[0]
                invalid_indices_np = np.asarray(invalid_indices)
                f.write(f"Invalid tokens at indices: {invalid_indices_np.tolist()}\n")
                for idx in invalid_indices_np[:10]:
                    decoded = tokenizer.decode(np.asarray(input_ids[idx]))
                    f.write(f"Invalid sequence {idx}: {decoded}\n")
        logging.info(f"Saved debug stats to: {debug_stats_path}")
    except Exception as e:
        logging.error(f"Failed to save debug_tokens_{prefix}_stats.txt: {str(e)}")
        raise

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
        scale = float(np.sqrt(1e-5 / fan_in))  # Reduced scale, e.g., ≈0.0014 for fan_in=256
        self.linear1.weight = mx.random.normal(shape=(d_hidden, d_in), loc=0.0, scale=scale, dtype=mx.float16)
        self.linear1.bias = mx.zeros(d_hidden, dtype=mx.float16)
        fan_in = d_hidden
        scale = float(np.sqrt(1e-5 / fan_in))  # Reduced scale
        self.linear2.weight = mx.random.normal(shape=(d_out, d_hidden), loc=0.0, scale=scale, dtype=mx.float16)
        self.linear2.bias = mx.zeros(d_out, dtype=mx.float16)
        for name, param in [('linear1.weight', self.linear1.weight), ('linear1.bias', self.linear1.bias),
                            ('linear2.weight', self.linear2.weight), ('linear2.bias', self.linear2.bias)]:
            if mx.any(mx.isnan(param)) or mx.any(mx.isinf(param)):
                logging.error(f"NaN/Inf in initialized {name}")
                raise ValueError(f"Invalid initialization for {name}")

    def __call__(self, x):
        if not isinstance(x, mx.array):
            logging.error(f"FeedForward input is not mx.array: type={type(x)}")
            raise ValueError("FeedForward input must be mx.array")
        logging.debug(f"FeedForward input shape: {x.shape}, max: {mx.max(x).item()}, min: {mx.min(x).item()}")
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        std = mx.sqrt(mx.mean((x - mean)**2, axis=-1, keepdims=True) + 1e-3)
        logging.debug(f"Normalization std: min={mx.min(std).item()}, max={mx.max(std).item()}")
        if mx.any(std < 1e-5):
            logging.warning("Near-zero std in normalization, potential instability")
            print("std is low : ", std)
        x = (x - mean) / mx.maximum(std, 1e-3)
        x = mx.clip(x, -10.0, 10.0)
        logging.debug(f"Linear1 input: shape={x.shape}, max={mx.max(x).item()}, min={mx.min(x).item()}, mean={mx.mean(x).item()}, std={mx.std(x).item()}")
        np.save(os.path.join(model_dir, f"debug_feedforward_input_{time.time()}.npy"), np.asarray(x))
        x_input = x
        x = self.linear1(x.astype(mx.float32))
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after linear1")
            np.save(os.path.join(model_dir, f"debug_linear1_output_{time.time()}.npy"), np.asarray(x))
            np.save(os.path.join(model_dir, f"debug_linear1_weights_{time.time()}.npy"), np.asarray(self.linear1.weight))
            np.save(os.path.join(model_dir, f"debug_linear1_input_{time.time()}.npy"), np.asarray(x_input))
            raise ValueError("NaN/Inf after linear1")
        x = x * mx.sigmoid(x)  # SiLU (comment out for ReLU: x = mx.maximum(x, 0.0))
        x = mx.clip(x, -10.0, 10.0)
        x_input_linear2 = x
        x = self.linear2(x)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after linear2")
            np.save(os.path.join(model_dir, f"debug_linear2_output_{time.time()}.npy"), np.asarray(x))
            np.save(os.path.join(model_dir, f"debug_linear2_weights_{time.time()}.npy"), np.asarray(self.linear2.weight))
            np.save(os.path.join(model_dir, f"debug_linear2_input_{time.time()}.npy"), np.asarray(x_input_linear2))
            raise ValueError("NaN/Inf after linear2")
        x = x.astype(mx.float16)
        x = mx.clip(x, -10.0, 10.0)
        return x
    
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, n_heads, bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        for key, param in self.attention.parameters().items():
            if isinstance(param, mx.array) and 'weight' in key:
                fan_in = param.shape[-1]
                scale = float(np.sqrt(5e-5 / fan_in))  # Match FeedForward
                self.attention.parameters()[key] = mx.random.normal(param.shape, loc=0.0, scale=scale, dtype=mx.float16)
            elif isinstance(param, mx.array):
                self.attention.parameters()[key] = mx.zeros(param.shape, dtype=mx.float16)

    def __call__(self, x, mask=None):
        if not isinstance(x, mx.array):
            logging.error(f"TransformerLayer input is not mx.array: type={type(x)}")
            raise ValueError("TransformerLayer input must be mx.array")
        logging.debug(f"TransformerLayer input shape: {x.shape}, max: {mx.max(x).item()}, min: {mx.min(x).item()}")
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        std = mx.sqrt(mx.mean((x - mean)**2, axis=-1, keepdims=True) + 1e-4)
        logging.debug(f"TransformerLayer norm: std min={mx.min(std).item()}, max={mx.max(std).item()}")
        if mx.any(std < 1e-5):
            logging.warning("Near-zero std in TransformerLayer norm")
        x = (x - mean) / mx.maximum(std, 1e-4)
        x = mx.clip(x, -20.0, 20.0)
        attn_output = self.attention(x, x, x, mask=mask).astype(mx.float32)
        logging.debug(f"Attention output: max={mx.max(attn_output).item()}, min={mx.min(attn_output).item()}")
        if mx.any(mx.isnan(attn_output)) or mx.any(mx.isinf(attn_output)):
            logging.error("NaN/Inf in attention output")
            np.save(os.path.join(model_dir, f"debug_attn_output_{time.time()}.npy"), np.asarray(attn_output))
            raise ValueError("NaN/Inf in attention")
        attn_output = mx.clip(attn_output, -20.0, 20.0).astype(mx.float16)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output).astype(mx.float16)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after norm1")
            raise ValueError("NaN/Inf after norm1")
        ff_output = self.dropout(self.ff(x))
        if mx.any(mx.isnan(ff_output)) or mx.any(mx.isinf(ff_output)):
            logging.error("NaN/Inf in feed-forward output")
            raise ValueError("NaN/Inf in feed-forward")
        x = self.norm2(x + ff_output)
        return x.astype(mx.float16)
    
class BabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=8, d_ff=1024, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = [TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        std = 0.02  # Increased from 0.01
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
        
        vocab_size_with_special = self.embedding.weight.shape[0]
        if mx.any(x < 0) or mx.any(x >= vocab_size_with_special):
            logging.error(f"Invalid tokens in input: min={mx.min(x).item()}, max={mx.max(x).item()}, vocab_size={vocab_size_with_special}")
            np.save(os.path.join(model_dir, f"invalid_input_tokens_{time.time()}.npy"), np.asarray(x))
            raise ValueError("Invalid tokens in input")
        
        x_embed = self.embedding(x).astype(mx.float32)
        embed_norm = mx.sqrt(mx.mean(x_embed**2, axis=-1, keepdims=True) + 1e-4)
        x_embed = x_embed / mx.maximum(embed_norm, 1e-4)
        logging.debug(f"Embedding norm: min={mx.min(embed_norm).item()}, max={mx.max(embed_norm).item()}")
        if mx.any(embed_norm < 1e-5):
            logging.warning("Near-zero norm in embedding")
        x = (x_embed * mx.sqrt(self.d_model / 1000.0)).astype(mx.float16)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf in embedding output")
            np.save(os.path.join(model_dir, f"debug_embedding_output_{time.time()}.npy"), np.asarray(x))
            raise ValueError("NaN/Inf in embedding")
        
        seq_len = x.shape[1]
        positions = mx.arange(seq_len)
        x = x + self.pos_embedding(positions)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after positional embedding")
            raise ValueError("NaN/Inf after pos_embedding")
        
        mask = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=1)
        mask = mask[None, None, :, :]
        
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask)
            if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
                logging.error(f"NaN/Inf in layer {i} output")
                raise ValueError(f"NaN/Inf in layer {i}")
        
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

def clip_gradients(grads, max_norm=0.05):  # Reduced from 0.1
    flat_grads = []
    for g in grads.values():
        if g is not None and isinstance(g, mx.array):
            g = mx.where(mx.isnan(g) | mx.isinf(g), mx.zeros_like(g), g)  # Replace NaN/Inf
            flat_grads.append(g.flatten())
        elif isinstance(g, dict):
            for sub_g in g.values():
                if sub_g is not None and isinstance(sub_g, mx.array):
                    sub_g = mx.where(mx.isnan(sub_g) | mx.isinf(sub_g), mx.zeros_like(sub_g), sub_g)
                    flat_grads.append(sub_g.flatten())
    if not flat_grads:
        logging.warning("No valid gradients to clip")
        return grads
    total_norm = mx.sqrt(sum(mx.sum(g * g) for g in flat_grads))
    logging.info(f"Gradient norm: {total_norm.item()}")
    if mx.isnan(total_norm) or mx.isinf(total_norm):
        logging.error("NaN/Inf in gradient norm")
        return {k: mx.zeros_like(g) if isinstance(g, mx.array) else g for k, g in grads.items()}
    scale = mx.minimum(1.0, max_norm / (total_norm + 1e-8))
    def scale_gradient(g):
        if isinstance(g, mx.array):
            return mx.where(mx.isnan(g) | mx.isinf(g), mx.zeros_like(g), g * scale)
        elif isinstance(g, dict):
            return {k: scale_gradient(v) for k, v in g.items()}
        return g
    clipped_grads = {k: scale_gradient(g) for k, g in grads.items()}
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

fill_mask = None

def initialize_bert_pipeline(bert_token, cache_dir, max_retries=3):
    global fill_mask
    if fill_mask is not None:
        logging.info("BERT pipeline already initialized, skipping")
        return
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
                    predictions = fill_mask(masked_sent, top_k=1)  # Correct
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
def dynamic_loss_scale(model, batch, fn, initial_scale=1.0):  # Reduced initial scale
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

def loss_fn_lm(model, x, loss_scale=10.0):
    if not isinstance(x, mx.array) or x.ndim != 2:
        logging.error(f"loss_fn_lm input is invalid: type={type(x)}, shape={x.shape if isinstance(x, mx.array) else 'N/A'}")
        raise ValueError("loss_fn_lm input must be 2D mx.array")
    logging.debug(f"loss_fn_lm input shape: {x.shape}, max: {mx.max(x).item()}, min: {mx.min(x).item()}")
    logging.debug(f"Input to model: shape={x[:, :-1].shape}")
    logits = model(x[:, :-1]).astype(mx.float16)
    targets = x[:, 1:].astype(mx.int32)
    logging.debug(f"logits shape: {logits.shape}, max: {mx.max(logits).item()}, min: {mx.min(logits).item()}")
    logging.debug(f"targets shape: {targets.shape}, max: {mx.max(targets).item()}, min: {mx.min(targets).item()}")
    if logits.shape[1] == 0 or targets.shape[1] == 0:
        logging.error(f"Empty sequence in logits or targets: logits_shape={logits.shape}, targets_shape={targets.shape}")
        raise ValueError("Empty sequence in loss computation")
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean"
    ).astype(mx.float16)
    logging.debug(f"Loss value: {loss.item()}")
    if mx.isnan(loss) or mx.isinf(loss):
        logging.error(f"NaN/Inf in loss: {loss.item()}")
        raise ValueError("NaN/Inf in loss")
    return loss * loss_scale

def loss_fn_qa(model, x, loss_scale=10.0):  # Reduced from 100.0
    if not isinstance(x, mx.array) or x.ndim != 2:
        logging.error(f"loss_fn_qa input is invalid: type={type(x)}, shape={x.shape if isinstance(x, mx.array) else 'N/A'}")
        raise ValueError("loss_fn_qa input must be 2D mx.array")
    logging.debug(f"loss_fn_qa input shape: {x.shape}, max: {mx.max(x).item()}, min: {mx.min(x).item()}")
    logits = model(x[:, :-1]).astype(mx.float16)
    logits = mx.clip(logits, -1e9, 1e9)
    targets = x[:, 1:].astype(mx.int32)
    logging.debug(f"logits shape: {logits.shape}, max: {mx.max(logits).item()}, min: {mx.min(logits).item()}")
    logging.debug(f"targets shape: {targets.shape}, max: {mx.max(targets).item()}, min: {mx.min(targets).item()}")
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
    mean_loss = mx.mean(masked_loss)
    logging.debug(f"Loss value: {mean_loss.item()}")
    if mx.isnan(mean_loss) or mx.isinf(mean_loss):
        logging.error(f"NaN/Inf in loss: {mean_loss.item()}")
        raise ValueError("NaN/Inf in loss")
    return mean_loss * loss_scale

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
    return compute_squad_metrics(predictions, references)  # Use custom function

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
            

if __name__ == '__main__':
    # Pre-download NLTK resources
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)

    # Print versions
    print("NumPy version:", np.__version__)
    print("PyTorch version:", torch.__version__)
    print("Transformers version:", transformers.__version__)
    print("MPS available:", torch.backends.mps.is_available())
    print("Random module:", random.__file__)
    print("Starting directory setup")

    # Logging setup
    logging.basicConfig(
        filename='debug_tokenization.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )

    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(tokenized_dir, exist_ok=True)
    stripped_dir = os.path.expanduser(os.path.join(base_dir, "data", "stripped"))
    os.makedirs(stripped_dir, exist_ok=True)  # Create the stripped directory if it doesn't exist

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            token=HF_TOKEN,
            cache_dir=cache_dir,
            use_fast=False,
            clean_up_tokenization_spaces=False
        )
        logging.info("BERT tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load BERT tokenizer: {str(e)}")
        raise


    tokenizer.pad_token = tokenizer.sep_token  # Use [SEP] as pad token
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})  # Ensure [SEP] is available


    # Initialize BERT pipeline
    initialize_bert_pipeline(BERT_Token, cache_dir)
    if fill_mask is None:
        logging.error("BERT pipeline initialization failed, exiting")
        raise RuntimeError("BERT pipeline initialization failed")
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Vocab size with special tokens: {tokenizer.vocab_size + len(tokenizer.all_special_tokens)}")

    # Identify high tokens
    sample_text = """For a long time after the course of the steamer _Sofala_ had been altered for the land, the low swampy coast had retained its appearance of a mere smudge of darkness beyond a belt of glitter."""
    high_tokens = identify_high_tokens(tokenizer, vocab_size_threshold=0.9, sample_text=sample_text)
    logging.info(f"High tokens mapping saved to {cleaned_dir}/high_tokens_mapping.txt")

    # Clear existing tokenized files
    for npy_file in glob.glob(os.path.join(tokenized_dir, "*.npy")):
        safe_remove(npy_file)

    # Load and process Gutenberg corpus
    print("Loading and processing Gutenberg corpus...")
    start_time = time.time()
    text = ""
    try:
        filenames = [f"{i}.txt" for i in range(527, 529) if os.path.exists(os.path.join(gutenberg_dir, f"{i}.txt"))]
        logging.info(f"Found {len(filenames)} files: {filenames}")
        for result in process_file_batch(filenames, tokenizer, high_tokens=high_tokens, batch_size=4):
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

    # Debug tokenizer
    sample_tokens = tokenizer(sample_text, max_length=128, truncation=True, return_tensors="np")["input_ids"]
    high_token_threshold = tokenizer.vocab_size * 0.9
    high_tokens_sample = sample_tokens[sample_tokens >= high_token_threshold]
    if high_tokens_sample.size > 0:
        logging.info(f"High tokens in sample text: {high_tokens_sample}")
        decoded_high = tokenizer.decode(high_tokens_sample)
        logging.info(f"Decoded high tokens: {decoded_high}")

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

    # Save sample cleaned text
    if filtered_texts:
        with open(os.path.join(cleaned_dir, "sample_cleaned.txt"), "w", encoding="utf-8") as f:
            f.write(filtered_texts[0][:1000])
        logging.info(f"Saved sample cleaned text to {cleaned_dir}/sample_cleaned.txt")
    
    # Debug sample text
    sample_text = filtered_texts[0][:500] if filtered_texts else "This is a test sentence with <SEP> separator."
    tokens = tokenizer(sample_text, max_length=128, truncation=True, return_tensors="np")["input_ids"]
    print(f"Sample tokens: shape={tokens.shape}, max={tokens.max()}, min={tokens.min()}")
    decoded = tokenizer.decode(tokens[0])
    print(f"Decoded sample: {decoded}")

    # Tokenize corpus
    print("Tokenizing corpus...")
    try:
        input_ids = load_or_tokenize_texts(
            filtered_texts,
            tokenizer,
            tokenized_dir,
            "corpus",
            batch_size=16,  # Reduced for stability
            max_length=128
        )
        print(f"Tokenized corpus shape: {input_ids.shape}")
        mx.eval(input_ids)
        tokens = np.load(os.path.expanduser("~/Baby_LLM/data/tokenized/debug_tokens_corpus.npy"))
        print(f"Tokens: shape={tokens.shape}, max={tokens.max()}, min={tokens.min()}")
        vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
        if tokens.max() > vocab_size * 0.9:
            logging.warning(f"High max token value: {tokens.max()} (vocab_size={vocab_size})")
            high_token_indices = np.where(tokens >= vocab_size * 0.9)
            if high_token_indices[0].size > 0:
                for idx in range(min(5, high_token_indices[0].size)):
                    row, col = high_token_indices[0][idx], high_token_indices[1][idx]
                    sequence = tokens[row]
                    token_id = tokens[row, col]
                    decoded = tokenizer.decode(sequence)
                    logging.info(f"High token at index ({row}, {col}): token={token_id}, decoded={decoded[:200]}")
                    with open(os.path.join(tokenized_dir, f"high_token_seq_{row}_{col}.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Token ID: {token_id}\nSequence: {decoded}")
        tokens = np.asarray(input_ids)
        print(f"Token shape: {tokens.shape}, Max: {tokens.max()}, Min: {tokens.min()}")
        
        invalid_tokens = tokens[(tokens < 0) | (tokens >= vocab_size)]
        if invalid_tokens.size > 0:
            print(f"Invalid tokens: {invalid_tokens}")
            invalid_indices = np.where((tokens < 0) | (tokens >= vocab_size))[0]
            for idx in invalid_indices[:10]:
                decoded = tokenizer.decode(tokens[idx])
                print(f"Invalid sequence {idx}: {decoded}")

        if mx.any(input_ids < 0) or mx.any(input_ids >= vocab_size):
            logging.error(f"Invalid tokens in input_ids: min={mx.min(input_ids)}, max={mx.max(input_ids)}")
            raise ValueError("Invalid tokens in input_ids")

        # Simple training loop
        model = BabyLLM(vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens))
        optimizer = optim.Adam(learning_rate=1e-6)
        scheduler = CosineWarmup(learning_rate=1e-6, warmup_steps=100, total_steps=1000)
        batch_size = 1
        for epoch in range(2):
            for i in range(0, input_ids.shape[0], batch_size):
                batch = input_ids[i:i+batch_size]
                logging.debug(f"Batch shape: {batch.shape}, type: {type(batch)}")
                loss, grads, scale = dynamic_loss_scale(model, batch, loss_fn_lm, initial_scale=2.0)
                grads = clip_gradients(grads, max_norm=0.05)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                logging.info(f"Epoch {epoch}, Batch {i//batch_size}, Loss: {loss.item()}, Scale: {scale}")
            validate_model_params(model, epoch)
        print("Training complete")
        # Evaluate QA
        val_pairs = [("Who is the author?", "Jane Austen"), ("What is the capital?", "Paris")]
        metrics = evaluate_qa(model, tokenizer, val_pairs)
        print(f"QA Metrics: {metrics}")
    except Exception as e:
        logging.error(f"Training or QA evaluation failed: {str(e)}")
        raise

    # Verify debug tokens
    debug_tokens_path = os.path.join(tokenized_dir, "debug_tokens_corpus.npy")
    if os.path.exists(debug_tokens_path):
        try:
            tokens = np.load(debug_tokens_path)
            print(f"Loaded debug tokens: shape={tokens.shape}, Max={tokens.max()}, Min={tokens.min()}")
            vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
            invalid_tokens = tokens[(tokens < 0) | (tokens >= vocab_size)]
            if invalid_tokens.size > 0:
                print(f"Invalid tokens: {invalid_tokens}")
                invalid_indices = np.where((tokens < 0) | (tokens >= vocab_size))[0]
                for idx in invalid_indices[:10]:
                    decoded = tokenizer.decode(tokens[idx])
                    print(f"Invalid sequence {idx}: {decoded}")
        except Exception as e:
            logging.error(f"Failed to load debug_tokens_corpus.npy: {str(e)}")
            raise
    else:
        logging.error(f"Debug tokens file not found: {debug_tokens_path}")
        raise FileNotFoundError(f"Debug tokens file not found: {debug_tokens_path}")

    print("Debugging complete. Check 'debug_tokenization.log' and tokenized files.")