import os
import re
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer, MarianMTModel, MarianTokenizer
from nltk.corpus import wordnet
import nltk
import random
import matplotlib.pyplot as plt
from pathlib import Path
import ssl
import evaluate
from gutenberg.cleanup import strip_headers
import time
from concurrent.futures import ProcessPoolExecutor

# Fix NLTK SSL issue and download required data
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
logging.basicConfig(filename='qa_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory setup
base_dir = "~/Baby_LLM"
cache_dir = os.path.expanduser(os.path.join(base_dir, "cache"))
model_dir = os.path.expanduser(os.path.join(base_dir, "model"))
data_dir = os.path.expanduser(os.path.join(base_dir, "data"))
gutenberg_dir = os.path.join(data_dir, "gutenberg")
cleaned_dir = os.path.join(data_dir, "cleaned")

# Compiled regex patterns for cleaning
patterns = [
    (re.compile(r'Scanned and proofed by.*?\n', re.IGNORECASE), ''),
    (re.compile(r'\[NOTE:.*?\]\n', re.DOTALL), ''),
    (re.compile(r'This eBook was produced by.*?\n', re.IGNORECASE), ''),
    (re.compile(r'^\s*[A-Z\s]+\n\s*TALES\s*\n', re.MULTILINE), ''),
    (re.compile(r'CONTENTS.*?(CHAPTER|BOOK|I\.|1\.)', re.DOTALL | re.IGNORECASE), ''),
    (re.compile(r'CHAPTER [IVXLC\d]+\..*?\n', re.IGNORECASE), ''),
    (re.compile(r'^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s.*?\n', re.MULTILINE), ''),
    (re.compile(r'BOOK THE (FIRST|SECOND|THIRD|FOURTH|FIFTH).*?\n', re.IGNORECASE), ''),
    (re.compile(r'\[.*?\]'), ''),
    (re.compile(r'^\s*[A-Z\s]+[A-Z\s]+\.\s*$', re.MULTILINE), ''),
    (re.compile(r'\*{3,}'), ''),
    (re.compile(r'^\s*[A-Z\s&]+\s*$', re.MULTILINE), ''),
    (re.compile(r'\b_+|_+\b'), ''),
    (re.compile(r'\*{2,}|\*\s+\*'), ''),
    (re.compile(r'\s*&\s*'), ' and '),
    (re.compile(r'\n{3,}'), '\n\n')
]

# Data cleaning functions
def enhanced_clean_text(raw_text):
    text = strip_headers(raw_text).strip()
    for pattern, repl in patterns:
        text = pattern.sub(repl, text)
    return text.strip()

def preprocess_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def is_narrative(text):
    text_array = np.array(list(text))
    letters = np.sum(np.char.isalpha(text_array))
    punctuation = np.sum(np.isin(text_array, ['.', ',', '!', '?', ';', ':']))
    lines = text.split('\n')
    avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
    uppercase_lines = sum(1 for line in lines if line.strip().isupper())
    short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 15)
    return (letters > 20 * punctuation and
            len(text) > 2000 and
            avg_line_length > 50 and
            uppercase_lines < len(lines) * 0.05 and
            short_lines < len(lines) * 0.2 and
            not re.search(r'Project Gutenberg|Transcriber|Editor', text, re.IGNORECASE))

        
def process_file(filename):
    file_path = os.path.join(gutenberg_dir, filename)
    cleaned_file_path = os.path.join(cleaned_dir, f"{filename}.cleaned.txt")
    if not os.path.isfile(file_path):
        return ""
    if os.path.getsize(file_path) > 10 * 1024 * 1024:
        print(f"Skipping {filename}: File too large (>10MB)")
        return ""
    try:
        if os.path.exists(cleaned_file_path):
            print(f"Using cached cleaned file: {filename}")
            with open(cleaned_file_path, "r", encoding="utf-8") as f:
                return f.read() + "\n\n"
        print(f"Processing {filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned_text = enhanced_clean_text(preprocess_text(raw_text))
        if is_narrative(cleaned_text) and len(cleaned_text) > 1000:
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"Saved cleaned file: {filename}")
            return cleaned_text + "\n\n"
        else:
            print(f"Deleting {filename}: Not narrative or too short")
            logging.info(f"Deleting {filename}: Not narrative or too short (length={len(cleaned_text)})")
            # Delete original and cached files
            try:
                os.remove(file_path)
                if os.path.exists(cleaned_file_path):
                    os.remove(cleaned_file_path)
            except Exception as e:
                logging.error(f"Failed to delete {filename} or its cached file: {e}")
            return ""
    except Exception as e:
        print(f"Deleting after skipping {filename}: {e}")
        os.remove(file_path)
    return ""

# Custom learning rate scheduler
class LinearWarmupConstant:
    def __init__(self, learning_rate, warmup_steps):
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        return self.learning_rate

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)
        # Initialize with smaller variance
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
        return x.astype(mx.float32)
        
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, n_heads, bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        for key, param in self.attention.parameters().items():
            if isinstance(param, mx.array) and param.dtype != mx.float16:
                self.attention.parameters()[key] = param.astype(mx.float16)
    def __call__(self, x):
        if not isinstance(x, mx.array):
            logging.error(f"TransformerLayer input is not mx.array: type={type(x)}")
            raise ValueError("TransformerLayer input must be mx.array")
        logging.debug(f"TransformerLayer input shape: {x.shape}")
        x = x.astype(mx.float16)
        attn_output = self.dropout(self.attention(x, x, x))
        if mx.any(mx.isnan(attn_output)) or mx.any(mx.isinf(attn_output)):
            logging.error("NaN/Inf in attention output")
            raise ValueError("NaN/Inf in attention")
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
        if x.ndim < 2:
            logging.error(f"TransformerLayer output is scalar or 1D: shape={x.shape}")
            raise ValueError("TransformerLayer output must be at least 2D")
        return x.astype(mx.float32)
    
    
class BabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = [TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.embedding.weight = self.embedding.weight.astype(mx.float16)
        self.pos_embedding.weight = self.pos_embedding.weight.astype(mx.float16)
        self.output.weight = self.output.weight.astype(mx.float16)
        self.output.bias = self.output.bias.astype(mx.float16)
    def __call__(self, x):
        if not isinstance(x, mx.array):
            logging.error(f"BabyLLM input is not mx.array: type={type(x)}")
            raise ValueError("BabyLLM input must be mx.array")
        if x.ndim != 2:
            logging.error(f"BabyLLM input is not 2D: shape={x.shape}")
            raise ValueError("BabyLLM input must be 2D (batch, seq_len)")
        logging.debug(f"BabyLLM input shape: {x.shape}")
        seq_len = x.shape[1]
        x = self.embedding(x) * mx.sqrt(self.d_model)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf in embedding output")
            raise ValueError("NaN/Inf in embedding")
        positions = mx.arange(seq_len)
        x = x + self.pos_embedding(positions)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN/Inf after positional embedding")
            raise ValueError("NaN/Inf after pos_embedding")
        for i, layer in enumerate(self.layers):
            x = layer(x)
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
        return x.astype(mx.float32)
    
# Utility functions
def to_numpy_for_decode(array):
    return np.array(array) if isinstance(array, mx.array) else array

def clip_gradients(grads, max_norm=0.1):  # Tighter clipping
    flat_grads = []
    for g in grads.values():
        if g is not None and isinstance(g, mx.array):
            flat_grads.append(g.flatten())
        elif isinstance(g, dict):
            for sub_g in g.values():
                if sub_g is not None and isinstance(sub_g, mx.array):
                    flat_grads.append(sub_g.flatten())
    if not flat_grads:
        return grads
    total_norm = mx.sqrt(sum(mx.sum(g * g) for g in flat_grads))
    logging.info(f"Gradient norm: {total_norm.item()}")
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
        return grads * scale
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

def back_translation(question, src_lang="en", tgt_lang="fr"):
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
        return en_text
    except Exception as e:
        logging.warning(f"Back-translation failed: {e}")
        return question

# Loss functions with dynamic scaling
def dynamic_loss_scale(model, batch, fn, initial_scale=100.0):
    scale = initial_scale
    while scale > 1.0:
        loss_and_grad_fn = nn.value_and_grad(model, lambda m: fn(m, batch, scale))
        loss, grads = loss_and_grad_fn(model)
        if not (mx.isnan(loss) or mx.isinf(loss)):
            return loss, grads, scale
        scale /= 2.0
        logging.warning(f"Reducing loss scale to {scale} due to NaN/Inf")
    raise ValueError("Loss scale too low, training unstable")

def loss_fn_lm(model, x, loss_scale=100.0):
    if not isinstance(x, mx.array) or x.ndim != 2:
        logging.error(f"loss_fn_lm input is invalid: type={type(x)}, shape={x.shape if isinstance(x, mx.array) else 'N/A'}")
        raise ValueError("loss_fn_lm input must be 2D mx.array")
    logging.debug(f"loss_fn_lm input shape: {x.shape}")
    logits = model(x[:, :-1]).astype(mx.float32)
    logging.debug(f"Logits shape: {logits.shape}")
    targets = x[:, 1:].astype(mx.int32)
    logging.debug(f"Targets shape: {targets.shape}")
    if logits.shape[1] == 0 or targets.shape[1] == 0:
        logging.error(f"Empty sequence in logits or targets: logits_shape={logits.shape}, targets_shape={targets.shape}")
        raise ValueError("Empty sequence in loss computation")
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean"
    )
    if not isinstance(loss, mx.array):
        logging.error(f"Loss is not mx.array: type={type(loss)}, value={loss}")
        raise ValueError("Loss must be mx.array")
    return loss * loss_scale

def loss_fn_qa(model, x, loss_scale=100.0):
    logits = model(x[:, :-1]).astype(mx.float32)
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
    )
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
squad_metric = evaluate.load("squad")
def evaluate_qa(model, tokenizer, val_pairs):
    predictions = []
    references = []
    for question, answer in val_pairs:
        pred = generate_answer(model, tokenizer, question)
        predictions.append({"id": str(len(predictions)), "prediction_text": pred})
        references.append({"id": str(len(references)), "answers": {"text": [answer], "answer_start": [0]}})
    return squad_metric.compute(predictions=predictions, references=references)

if __name__ == '__main__':
    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)

    # Load tokenizer
    HF_TOKEN = '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'
    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", token=HF_TOKEN, cache_dir=cache_dir)
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'sep_token': '<SEP>'})

    # Load and clean Gutenberg corpus
    print("Loading and processing Gutenberg corpus...")
    start_time = time.time()
    text = ""
    filenames = os.listdir(gutenberg_dir)
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, filenames))
        for result, filename in zip(results, filenames):
            if result:
                text += result
    print(f"Finished loading and cleaning, Total Time: {time.time() - start_time:.2f}s")

    # Filter texts by size
    print("Now splitting texts")
    texts = text.split("\n\n")
    max_size = 2 * 1024 * 1024 * 1024  # 2GB limit
    current_size = 0
    filtered_texts = []
    for i, t in enumerate(texts):
        size = len(t.encode("utf-8"))
        if current_size + size <= max_size:
            filtered_texts.append(t)
            current_size += size
        else:
            break
        if i % 100000 == 0:
            print(f"Processed {i} texts")
    print(f"Collected {len(filtered_texts)} texts, ~{current_size / (1024**2):.2f}MB")

    # Tokenization with validation
    print("Tokenizing corpus...")
    inputs = []
    for i in range(0, len(filtered_texts), 1000):
        batch = [text for text in filtered_texts[i:i+1000] if text.strip()]  # Remove empty texts
        if not batch:
            logging.warning(f"Skipping empty batch at index {i}")
            continue
        try:
            batch_inputs = tokenizer(
                batch,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            tokenized_ids = batch_inputs["input_ids"]
            if tokenized_ids.shape[1] != 128:
                logging.warning(f"Batch at index {i} has incorrect sequence length: {tokenized_ids.shape[1]}")
                if tokenized_ids.shape[1] < 128:
                    pad_width = ((0, 0), (0, 128 - tokenized_ids.shape[1]))
                    tokenized_ids = np.pad(
                        tokenized_ids,
                        pad_width,
                        mode="constant",
                        constant_values=tokenizer.pad_token_id
                    )
                else:
                    tokenized_ids = tokenized_ids[:, :128]
            inputs.append(tokenized_ids)
        except Exception as e:
            logging.error(f"Tokenization failed for batch at index {i}: {e}")
            continue
        if i % 100000 == 0:
            print(f"Tokenized {i} texts")
    if not inputs:
        raise ValueError("No valid batches tokenized. Check input data.")
    input_ids = mx.array(np.concatenate(inputs, axis=0), dtype=mx.int32)
    if mx.any(input_ids < 0) or mx.any(input_ids >= tokenizer.vocab_size):
        invalid_indices = mx.where((input_ids < 0) | (input_ids >= tokenizer.vocab_size))
        logging.error(f"Invalid token IDs at indices: {invalid_indices}")
        raise ValueError("Invalid token IDs in corpus")
    print(f"Tokenized corpus shape: {input_ids.shape}")
    mx.eval(input_ids)

    input_ids = input_ids[16280 * 4:16310 * 4]  # Near batch 16287
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
    # Clean QA pairs
    qa_pairs = [(re.sub(r'\s+', ' ', q.strip()), a.strip()) for q, a in qa_pairs if q and a]
    augmented_pairs = []
    for question, answer in qa_pairs:
        augmented_pairs.append((question, answer))
        augmented_pairs.append((synonym_replacement(question), answer))
        augmented_pairs.append((back_translation(question), answer))
    qa_pairs.extend(augmented_pairs)

    # Preprocess QA data
    qa_texts = [f"Question: {q} Answer: {a}" for q, a in qa_pairs]
    inputs = tokenizer(qa_texts, return_tensors="np", padding=True, truncation=True, max_length=128)
    qa_input_ids = mx.array(inputs["input_ids"], dtype=mx.int32)
    if mx.any(qa_input_ids < 0) or mx.any(qa_input_ids >= tokenizer.vocab_size):
        logging.error(f"Invalid token IDs in qa_input_ids: {qa_input_ids[mx.logical_or(qa_input_ids < 0, qa_input_ids >= tokenizer.vocab_size)]}")
        raise ValueError("Invalid token IDs in QA data")
    mx.eval(qa_input_ids)

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
    val_inputs = tokenizer(val_texts, return_tensors="np", padding=True, truncation=True, max_length=128)
    val_input_ids = mx.array(val_inputs["input_ids"], dtype=mx.int32)
    if mx.any(val_input_ids < 0) or mx.any(val_input_ids >= tokenizer.vocab_size):
        logging.error(f"Invalid token IDs in val_input_ids: {val_input_ids[mx.logical_or(val_input_ids < 0, val_input_ids >= tokenizer.vocab_size)]}")
        raise ValueError("Invalid token IDs in validation data")
    mx.eval(val_input_ids)



    # Pretraining setup
    batch_size_lm = 4  # Reduced for stability
    accumulation_steps_lm = 16
    num_epochs_lm = 2
    optimizer_lm = optim.AdamW(learning_rate=LinearWarmupConstant(5e-6, 1000), weight_decay=0.05)
    train_losses_gen = []

    # Initialize model
    generative_model = BabyLLM(vocab_size=tokenizer.vocab_size + 1)
    weights_path = os.path.join(model_dir, "baby_llm_pretrain.npz")
    os.makedirs(model_dir, exist_ok=True)  # Ensure model directory exists
    try:
        weights = np.load(weights_path)
        weights_dict = dict(weights)
        for k, v in weights_dict.items():
            if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                logging.warning(f"NaN/Inf in weight {k}, replacing with zeros")
                v[np.isnan(v) | np.isinf(v)] = 0
                weights_dict[k] = v
        if 'embedding.weight' in weights_dict and weights_dict['embedding.weight'].shape[1] == 512:
            weights_dict['embedding.weight'] = np.pad(
                weights_dict['embedding.weight'], ((0, 0), (0, 768-512)), mode='constant')
            weights_dict['output.weight'] = np.pad(
                weights_dict['output.weight'], ((0, 0), (0, 768-512)), mode='constant')
        np.savez(weights_path, **weights_dict)
        generative_model.load_weights(weights_path)
        logging.info("Loaded and adjusted pretrained weights")
    except FileNotFoundError:
        logging.info("Pretrained weights not found. Initializing with Xavier/Glorot.")
        def xavier_init(shape, fan_in, fan_out):
            std = float(np.sqrt(2.0 / (fan_in + fan_out)))  # Convert to Python float
            return mx.random.normal(shape=shape, loc=0.0, scale=std, dtype=mx.float16)
        for module in [generative_model.embedding, generative_model.pos_embedding, generative_model.output]:
            if hasattr(module, 'weight'):
                fan_in = module.weight.shape[1]
                fan_out = module.weight.shape[0]
                module.weight = xavier_init(module.weight.shape, fan_in, fan_out)
        for layer in generative_model.layers:
            for module in [layer.attention, layer.ff.linear1, layer.ff.linear2]:
                for param in ['query', 'key', 'value', 'weight']:
                    if hasattr(module, param):
                        w = getattr(module, param)
                        fan_in = w.shape[1]
                        fan_out = w.shape[0]
                        setattr(module, param, xavier_init(w.shape, fan_in, fan_out))
        logging.info("Model initialized with Xavier/Glorot weights")
    except Exception as e:
        logging.error(f"Failed to load or initialize weights: {e}")
        raise
    mx.eval(generative_model.parameters())

    logging.info(f"Embedding weight shape: {generative_model.embedding.weight.shape}")

    # Pretraining loop
    for epoch in range(num_epochs_lm):
        logging.info(f"Starting language modeling epoch {epoch + 1}/{num_epochs_lm}")
        indices = mx.array(np.random.permutation(len(input_ids)))
        accumulated_grads = None
        for i in range(0, len(input_ids), batch_size_lm):
            batch_idx = i // batch_size_lm
            batch_indices = indices[i:i + batch_size_lm]
            if batch_indices.shape[0] < batch_size_lm:
                continue
            batch = mx.take(input_ids, batch_indices, axis=0)
            # Validate batch
            if mx.any(batch < 0) or mx.any(batch >= tokenizer.vocab_size):
                logging.warning(f"Invalid tokens in batch {batch_idx}: {batch.tolist()}")
                continue
            # Filter excessive padding using integer indices
            non_pad_counts = mx.sum(batch != tokenizer.pad_token_id, axis=1)
            valid_mask = non_pad_counts > 0.2 * batch.shape[1]  # Stricter threshold
            valid_mask_np = np.array(valid_mask)
            valid_indices = np.where(valid_mask_np)[0]
            if valid_indices.size == 0 or valid_indices.size < 2:  # Require at least 2 sequences
                logging.warning(f"Batch {batch_idx} has excessive padding or too few sequences: {valid_indices.size}")
                continue
            valid_indices = mx.array(valid_indices, dtype=mx.int32)
            batch = mx.take(batch, valid_indices, axis=0)
            mx.eval(batch)
            # Log batch statistics
            logging.info(f"Batch {batch_idx}: shape={batch.shape}, "
                        f"max_token={int(mx.max(batch))}, "
                        f"min_token={int(mx.min(batch))}, "
                        f"padding_ratio={(1 - mx.mean(batch != tokenizer.pad_token_id).item()):.2f}")
            # Save batch for inspection
            try:
                np.save(os.path.join(model_dir, f"batch_{batch_idx}_failed.npy"), np.array(batch))
            except Exception as e:
                logging.warning(f"Failed to save batch {batch_idx}: {e}")
            # Log decoded batch
            decoded_batch = [tokenizer.decode(to_numpy_for_decode(batch[j])) for j in range(batch.shape[0])]
            logging.info(f"Batch {batch_idx} decoded: {decoded_batch}")
            # Dynamic loss scaling
            try:
                loss, grads, scale = dynamic_loss_scale(generative_model, batch, loss_fn_lm, initial_scale=10.0)
                grads = scale_gradients(grads, 1.0 / scale)
                # Check gradients
                if any(mx.any(mx.isnan(g) | mx.isinf(g)) for g in grads.values() if isinstance(g, mx.array)):
                    logging.warning(f"Skipping batch {batch_idx} due to NaN/Inf gradients")
                    continue
                scaled_grads = scale_gradients(grads, 1.0 / accumulation_steps_lm)
                logging.debug(f"Batch {batch_idx}: scaled_grads type={type(scaled_grads)}, "
                            f"keys={list(scaled_grads.keys()) if isinstance(scaled_grads, dict) else None}")
                accumulated_grads = add_grads(accumulated_grads, scaled_grads)
                if (batch_idx + 1) % accumulation_steps_lm == 0:
                    accumulated_grads = clip_gradients(accumulated_grads, max_norm=0.05)
                    optimizer_lm.update(generative_model, accumulated_grads)
                    mx.eval(generative_model.parameters(), optimizer_lm.state)
                    accumulated_grads = None
                train_losses_gen.append(loss.item() / scale)
                logging.info(f"Batch {batch_idx}, Loss: {loss.item() / scale:.4f}")
            except Exception as e:
                logging.error(f"Batch {batch_idx} failed: {str(e)}")
                if 'grads' in locals():
                    logging.error(f"Gradient types: {[type(g) for g in grads.values() if isinstance(g, (mx.array, dict))]}")
                continue
        generative_model.save_weights(os.path.join(model_dir, f"baby_llm_pretrain_epoch_{epoch}.npz"))
        
                  
    # Fine-tuning setup
    batch_size_qa = 8  # Reduced for stability
    accumulation_steps = 16
    num_epochs_qa = 50
    scheduler_qa = LinearWarmupConstant(learning_rate=5e-6, warmup_steps=1000)
    optimizer_gen = optim.AdamW(learning_rate=scheduler_qa, weight_decay=0.05)
    patience = 10
    best_val_em = 0.0
    patience_counter = 0
    val_losses_gen = []

    # Fine-tuning loop
    for epoch in range(num_epochs_qa):
        logging.info(f"Starting generative QA epoch {epoch + 1}/{num_epochs_qa}")
        indices = mx.array(np.random.permutation(len(qa_input_ids)))
        accumulated_grads = None
        for i in range(0, len(qa_input_ids), batch_size_qa):
            batch_idx = i // batch_size_qa
            batch_indices = indices[i:i + batch_size_qa]
            if batch_indices.shape[0] < batch_size_qa:
                continue
            batch = mx.take(qa_input_ids, batch_indices, axis=0)
            # Validate batch
            if mx.any(batch < 0) or mx.any(batch >= tokenizer.vocab_size):
                logging.warning(f"Invalid tokens in QA batch {batch_idx}: {batch.tolist()}")
                continue
            # Filter excessive padding using integer indices
            non_pad_counts = mx.sum(batch != tokenizer.pad_token_id, axis=1)
            valid_mask = non_pad_counts > 0.5 * batch.shape[1]
            valid_mask_np = np.array(valid_mask)
            valid_indices = np.where(valid_mask_np)[0]
            if valid_indices.size == 0 or valid_indices.size < 2:
                logging.warning(f"QA batch {batch_idx} has excessive padding or too few sequences: {valid_indices.size}")
                continue
            valid_indices = mx.array(valid_indices, dtype=mx.int32)
            batch = mx.take(batch, valid_indices, axis=0)
            mx.eval(batch)
            # Log batch statistics
            logging.info(f"QA batch {batch_idx}: shape={batch.shape}, "
                        f"max_token={int(mx.max(batch))}, "
                        f"min_token={int(mx.min(batch))}, "
                        f"padding_ratio={(1 - mx.mean(batch != tokenizer.pad_token_id).item()):.2f}")
            try:
                np.save(os.path.join(model_dir, f"qa_batch_{batch_idx}_failed.npy"), np.array(batch))
            except Exception as e:
                logging.warning(f"Failed to save QA batch {batch_idx}: {e}")
            decoded_batch = [tokenizer.decode(to_numpy_for_decode(batch[j])) for j in range(batch.shape[0])]
            logging.info(f"QA batch {batch_idx} decoded: {decoded_batch}")
            try:
                loss, grads, scale = dynamic_loss_scale(generative_model, batch, loss_fn_qa, initial_scale=10.0)
                grads = scale_gradients(grads, 1.0 / scale)
                if any(mx.any(mx.isnan(g) | mx.isinf(g)) for g in grads.values() if isinstance(g, mx.array)):
                    logging.warning(f"Skipping QA batch {batch_idx} due to NaN/Inf gradients")
                    continue
                scaled_grads = scale_gradients(grads, 1.0 / accumulation_steps)
                logging.debug(f"QA batch {batch_idx}: scaled_grads type={type(scaled_grads)}, "
                            f"keys={list(scaled_grads.keys()) if isinstance(scaled_grads, dict) else None}")
                accumulated_grads = add_grads(accumulated_grads, scaled_grads)
                if (batch_idx + 1) % accumulation_steps == 0:
                    accumulated_grads = clip_gradients(accumulated_grads, max_norm=0.05)
                    optimizer_gen.update(generative_model, accumulated_grads)
                    mx.eval(generative_model.parameters(), optimizer_gen.state)
                    accumulated_grads = None
                train_losses_gen.append(loss.item() / scale)
                logging.info(f"QA batch {batch_idx}, Loss: {loss.item() / scale:.4f}")
            except Exception as e:
                logging.error(f"QA batch {batch_idx} failed: {str(e)}")
                if 'grads' in locals():
                    logging.error(f"Gradient types: {[type(g) for g in grads.values() if isinstance(g, (mx.array, dict))]}")
                continue
        # Validation
        val_loss_gen = loss_fn_qa(generative_model, val_input_ids, 100.0) / 100.0
        val_metrics = evaluate_qa(generative_model, tokenizer, validation_prompts)
        val_em = val_metrics["exact_match"]
        val_f1 = val_metrics["f1"]
        val_losses_gen.append(val_loss_gen.item())
        logging.info(f"Epoch {epoch + 1}, Val Loss: {val_loss_gen.item():.4f}, EM: {val_em:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
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