import os
import re
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer
from nltk.corpus import wordnet
import nltk
import random
import matplotlib.pyplot as plt
from pathlib import Path
import ssl
import evaluate
from gutenberg.cleanup import strip_headers
import time
import mmap
import gzip
import pickle
from collections import Counter
import psutil
from concurrent.futures import ProcessPoolExecutor
import gc
import threading
from queue import Queue

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
logging.basicConfig(filename='qa_training_debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory setup
base_dir = "~/Baby_LLM"
cache_dir = os.path.expanduser(os.path.join(base_dir, "cache"))
model_dir = os.path.expanduser(os.path.join(base_dir, "model"))
data_dir = os.path.expanduser(os.path.join(base_dir, "data"))
gutenberg_dir = os.path.join(data_dir, "gutenberg")
cleaned_dir = os.path.join(data_dir, "cleaned")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(cleaned_dir, exist_ok=True)

# Memory management utilities
class MemoryManager:
    def __init__(self, max_memory_gb=24):  # Leave 8GB for system
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.current_usage = 0
        
    def check_memory(self):
        mem = psutil.virtual_memory()
        return mem.percent < 85  # Stay under 85% usage
    
    def force_cleanup(self):
        gc.collect()
        mx.metal.clear_cache()  # Clear MLX GPU cache
        
    def memory_available(self):
        mem = psutil.virtual_memory()
        return mem.available

memory_manager = MemoryManager()

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

# Optimized data cleaning functions
def enhanced_clean_text(raw_text):
    text = strip_headers(raw_text).strip()
    for pattern, repl in patterns:
        text = pattern.sub(repl, text)
    return text.strip()

def preprocess_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_narrative(text):
    # Optimized version using numpy operations
    if len(text) < 2000:
        return False
    
    # Sample text for faster processing on large texts
    sample_size = min(5000, len(text))
    sample_text = text[:sample_size]
    
    text_array = np.array(list(sample_text))
    letters = np.sum(np.char.isalpha(text_array))
    punctuation = np.sum(np.isin(text_array, ['.', ',', '!', '?', ';', ':']))
    
    lines = sample_text.split('\n')
    if not lines:
        return False
        
    avg_line_length = sum(len(line) for line in lines) / len(lines)
    uppercase_lines = sum(1 for line in lines if line.strip().isupper())
    short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 15)
    
    return (letters > 20 * punctuation and
            avg_line_length > 50 and
            uppercase_lines < len(lines) * 0.05 and
            short_lines < len(lines) * 0.2 and
            not re.search(r'Project Gutenberg|Transcriber|Editor', sample_text, re.IGNORECASE))

def process_file_optimized(filename):
    """Optimized file processing with better memory management"""
    file_path = os.path.join(gutenberg_dir, filename)
    cleaned_file_path = os.path.join(cleaned_dir, f"{filename}.gz")
    
    if not os.path.isfile(file_path) or not filename.endswith('.txt'):
        return ""
    
    file_size = os.path.getsize(file_path)
    
    # Adjust size limits based on available memory
    max_file_size = min(10 * 1024 * 1024, memory_manager.memory_available() // 100)
    if file_size > max_file_size or file_size < 100:
        return ""
    
    try:
        if os.path.exists(cleaned_file_path):
            with gzip.open(cleaned_file_path, "rt", encoding="utf-8") as f:
                cleaned_text = f.read()
            if len(cleaned_text) < 1000 or not is_narrative(cleaned_text):
                os.remove(cleaned_file_path)
                return ""
            return cleaned_text + "\n\n"
        
        # Use memory mapping for large files
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            if file_size > 1024 * 1024:  # Use mmap for files > 1MB
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    raw_text = mm.read().decode("utf-8", errors='ignore')
            else:
                raw_text = f.read()
        
        cleaned_text = enhanced_clean_text(preprocess_text(raw_text))
        del raw_text  # Explicit cleanup
        
        if is_narrative(cleaned_text) and len(cleaned_text) > 1000:
            with gzip.open(cleaned_file_path, "wt", encoding="utf-8") as f:
                f.write(cleaned_text)
            return cleaned_text + "\n\n"
        else:
            return ""
            
    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        return ""

# Streaming data loader for better memory efficiency
class StreamingDataLoader:
    def __init__(self, data_dir, batch_size=32, max_length=48):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.files = [f for f in os.listdir(data_dir) if f.startswith('tokenized_batch_')]
        self.current_file_idx = 0
        self.current_batch = None
        self.batch_idx = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current_batch is None or self.batch_idx >= len(self.current_batch):
            if self.current_file_idx >= len(self.files):
                raise StopIteration
            
            # Load next file
            file_path = os.path.join(self.data_dir, self.files[self.current_file_idx])
            self.current_batch = np.load(file_path)
            self.current_file_idx += 1
            self.batch_idx = 0
            
        # Get next batch
        start_idx = self.batch_idx
        end_idx = min(start_idx + self.batch_size, len(self.current_batch))
        batch = self.current_batch[start_idx:end_idx]
        self.batch_idx = end_idx
        
        return mx.array(batch, dtype=mx.int16)

# Optimized model with gradient checkpointing
class OptimizedFeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)
        
        # More efficient initialization
        scale1 = np.sqrt(2.0 / d_in)  # He initialization
        scale2 = np.sqrt(2.0 / d_hidden)
        
        self.linear1.weight = mx.random.normal(shape=(d_hidden, d_in), scale=scale1, dtype=mx.float16)
        self.linear1.bias = mx.zeros(d_hidden, dtype=mx.float16)
        self.linear2.weight = mx.random.normal(shape=(d_out, d_hidden), scale=scale2, dtype=mx.float16)
        self.linear2.bias = mx.zeros(d_out, dtype=mx.float16)
        
    def __call__(self, x):
        x = x.astype(mx.float16)
        x = self.linear1(x)
        x = nn.gelu(x)
        x = self.linear2(x)
        return x.astype(mx.float32)

class OptimizedTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, n_heads, bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = OptimizedFeedForward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def __call__(self, x):
        # Pre-norm architecture for better training stability
        normed_x = self.norm1(x)
        attn_output = self.dropout(self.attention(normed_x, normed_x, normed_x))
        x = x + attn_output
        
        normed_x2 = self.norm2(x)
        ff_output = self.dropout(self.ff(normed_x2))
        x = x + ff_output
        
        return x

class OptimizedBabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=3, n_heads=4, d_ff=512, max_len=48):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.layers = []
        for i in range(n_layers):
            self.layers.append(OptimizedTransformerLayer(d_model, n_heads, d_ff))
            
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        # Weight tying for memory efficiency
        self.output.weight = self.embedding.weight
        
    def __call__(self, x):
        seq_len = x.shape[1]
        
        # Token embeddings
        x_emb = self.embedding(x) * mx.sqrt(float(self.d_model))
        
        # Position embeddings
        positions = mx.arange(seq_len).reshape(1, seq_len)
        pos_emb = self.pos_embedding(positions)
        
        x = x_emb + pos_emb
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        return self.output(x)

# Optimized training utilities
def adaptive_batch_size(memory_usage):
    """Dynamically adjust batch size based on memory usage"""
    if memory_usage > 80:
        return 1
    elif memory_usage > 70:
        return 2
    elif memory_usage > 60:
        return 4
    else:
        return 8

def optimized_loss_fn(model, batch, loss_scale=1.0):
    """Optimized loss function with memory management"""
    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    
    # Use label smoothing for better generalization
    smooth_targets = targets * 0.9 + 0.1 / logits.shape[-1]
    
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean"
    )
    
    return loss * loss_scale

def clip_gradients_optimized(grads, max_norm=1.0):
    """More efficient gradient clipping"""
    total_norm = 0.0
    
    def calculate_norm(g):
        nonlocal total_norm
        if isinstance(g, mx.array):
            total_norm += mx.sum(g * g).item()
        elif isinstance(g, dict):
            for v in g.values():
                calculate_norm(v)
    
    calculate_norm(grads)
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        
        def scale_gradient(g):
            if isinstance(g, mx.array):
                return g * scale
            elif isinstance(g, dict):
                return {k: scale_gradient(v) for k, v in g.items()}
            return g
            
        return scale_gradient(grads)
    
    return grads

# Optimized generation function
def generate_answer_optimized(model, tokenizer, prompt, max_tokens=32, temperature=0.7):
    """More memory-efficient generation"""
    input_text = f"Question: {prompt} Answer:"
    input_ids = tokenizer(input_text, return_tensors="np", max_length=32, truncation=True)["input_ids"]
    input_ids = mx.array(input_ids, dtype=mx.int16)
    
    generated_tokens = []
    
    for _ in range(max_tokens):
        with mx.stream(mx.cpu):  # Use CPU stream for memory efficiency
            logits = model(input_ids)[:, -1, :]
            
            # Temperature scaling
            logits = logits / temperature
            
            # Top-k sampling for efficiency
            k = min(50, logits.shape[-1])
            top_k_logits, top_k_indices = mx.topk(logits, k=k, axis=-1)
            
            probs = mx.softmax(top_k_logits, axis=-1)
            next_token_idx = mx.random.categorical(probs.log())
            next_token = top_k_indices[0, next_token_idx].item()
            
            if next_token == tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token)
            
            # Update input_ids efficiently
            next_token_tensor = mx.array([[next_token]], dtype=mx.int16)
            input_ids = mx.concatenate([input_ids, next_token_tensor], axis=1)
            
            # Keep only recent context to save memory
            if input_ids.shape[1] > 40:
                input_ids = input_ids[:, -32:]
    
    # Decode only the generated part
    if generated_tokens:
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_text.strip()
    else:
        return "I don't know"

# Main training function with optimizations
def train_optimized():
    """Main training loop with memory optimizations"""
    
    # Load tokenizer
    HF_TOKEN = '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'
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.3", 
        token=HF_TOKEN, 
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'sep_token': '<SEP>'})
    
    # Process files in smaller batches to manage memory
    print("Processing Gutenberg corpus with memory optimization...")
    filenames = [f for f in os.listdir(gutenberg_dir) if f.endswith('.txt')][:2000]  # Limit for memory
    
    # Process files in smaller chunks
    chunk_size = 100
    all_texts = []
    
    for i in range(0, len(filenames), chunk_size):
        chunk = filenames[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(filenames) + chunk_size - 1)//chunk_size}")
        
        with ProcessPoolExecutor(max_workers=4) as executor:  # Limit workers for memory
            chunk_texts = list(executor.map(process_file_optimized, chunk))
            all_texts.extend([text for text in chunk_texts if text])
        
        memory_manager.force_cleanup()
        
        if not memory_manager.check_memory():
            print("Memory limit reached, stopping text processing")
            break
    
    print(f"Processed {len(all_texts)} texts")
    
    # Tokenize in batches and save immediately
    print("Tokenizing corpus...")
    batch_size = 500  # Smaller batches for memory efficiency
    
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        
        if not batch_texts:
            continue
            
        batch_inputs = tokenizer(
            batch_texts, 
            return_tensors="np", 
            padding=True, 
            truncation=True, 
            max_length=48
        )
        
        tokenized_ids = np.array(batch_inputs["input_ids"], dtype=np.int16)
        np.save(os.path.join(data_dir, f"tokenized_batch_{i//batch_size}.npy"), tokenized_ids)
        
        del batch_inputs, tokenized_ids
        memory_manager.force_cleanup()
    
    # Clear all_texts to free memory
    del all_texts
    memory_manager.force_cleanup()
    
    # Load QA data
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
        
    qa_texts = [f"Question: {q} Answer: {a}" for q, a in qa_pairs]
    qa_inputs = tokenizer(qa_texts, return_tensors="np", padding=True, truncation=True, max_length=48)
    qa_input_ids = mx.array(qa_inputs["input_ids"], dtype=mx.int16)
    
    # Initialize model with smaller dimensions for memory efficiency
    vocab_size = len(tokenizer.vocab)
    model = OptimizedBabyLLM(
        vocab_size=vocab_size, 
        d_model=128,  # Reduced from 192
        n_layers=3, 
        n_heads=4, 
        d_ff=384,  # Reduced from 768
        max_len=48
    )
    
    # Training setup
    learning_rate = 1e-4
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    
    # Training loop with adaptive batch sizing
    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Adaptive batch size based on memory
        memory_usage = psutil.virtual_memory().percent
        current_batch_size = adaptive_batch_size(memory_usage)
        
        data_loader = StreamingDataLoader(data_dir, batch_size=current_batch_size)
        
        epoch_loss = 0
        batch_count = 0
        
        try:
            for batch in data_loader:
                if not memory_manager.check_memory():
                    memory_manager.force_cleanup()
                    continue
                
                # Compute loss and gradients
                loss_and_grad_fn = nn.value_and_grad(model, optimized_loss_fn)
                loss, grads = loss_and_grad_fn(model, batch)
                
                # Clip gradients
                grads = clip_gradients_optimized(grads)
                
                # Update model
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                
                epoch_loss += loss.item()
                batch_count += 1
                
                if batch_count % 100 == 0:
                    print(f"Batch {batch_count}, Loss: {loss.item():.4f}, Memory: {psutil.virtual_memory().percent:.1f}%")
                
                # Periodic cleanup
                del batch, loss, grads
                if batch_count % 50 == 0:
                    memory_manager.force_cleanup()
                    
        except Exception as e:
            print(f"Training interrupted: {e}")
            memory_manager.force_cleanup()
            continue
        
        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        model.save_weights(os.path.join(model_dir, f"model_epoch_{epoch}.npz"))
        
    print("Training completed!")
    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = train_optimized()
    
    # Interactive testing
    print("\nModel ready for testing!")
    while True:
        prompt = input("Enter a question (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        answer = generate_answer_optimized(model, tokenizer, prompt)
        print(f"Answer: {answer}")