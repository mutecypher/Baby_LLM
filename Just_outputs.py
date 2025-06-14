import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer
import numpy as np

HF_TOKEN = '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'
base_dir = "~/Baby_LLM"
cache_dir = os.path.expanduser(f"{base_dir}/cache")
model_dir = os.path.expanduser(f"{base_dir}/model")
data_dir = os.path.expanduser(f"{base_dir}/data")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

required_space = 10 * 1024 * 1024 * 1024
available_space = os.statvfs(data_dir).f_frsize * os.statvfs(data_dir).f_bavail
if available_space < required_space:
    raise OSError(f"Not enough disk space! Need ~10GB, but only {available_space / (1024**3):.2f}GB available")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", token=HF_TOKEN, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'sep_token': '<SEP>'})
vocab_size = tokenizer.vocab_size + 1
print(f"Vocabulary size is: {vocab_size}")

class BabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        class FeedForward(nn.Module):
            def __init__(self, d_in, d_hidden, d_out):
                super().__init__()
                self.linear1 = nn.Linear(d_in, d_hidden)
                self.linear2 = nn.Linear(d_hidden, d_out)
            def __call__(self, x):
                x = self.linear1(x)
                x = nn.gelu(x)
                x = self.linear2(x)
                return x
        
        class TransformerLayer(nn.Module):
            def __init__(self, d_model, n_heads, d_ff):
                super().__init__()
                self.attention = nn.MultiHeadAttention(d_model, n_heads, bias=False)
                self.norm1 = nn.LayerNorm(d_model)
                self.ff = FeedForward(d_model, d_ff, d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(0.2)
            
            def __call__(self, x):
                attn_output = self.dropout(self.attention(x, x, x))
                x = self.norm1(x + attn_output)
                ff_output = self.dropout(self.ff(x))
                x = self.norm2(x + ff_output)
                return x
        
        # Use a list of TransformerLayer modules
        self.layers = [TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def __call__(self, x):
        x = self.embedding(x) * mx.sqrt(self.d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.output(x)
       
# Initialize model
model = BabyLLM(vocab_size)

# Check if pretrained weights exist and load them
pretrain_weights_path = f"{model_dir}/baby_llm_pretrain.npz"
if os.path.exists(pretrain_weights_path):
    print(f"Found existing pretrained weights at {pretrain_weights_path}. Loading weights...")
    model.load_weights(pretrain_weights_path)
    print("Weights loaded successfully. Resuming training from saved state.")
else:
    print("No pretrained weights found. Starting training from scratch.")
mx.eval(model.parameters())

print("Loading Gutenberg corpus...")
text = ""
gutenberg_dir = os.path.join(data_dir, "gutenberg")
for filename in os.listdir(gutenberg_dir):
    file_path = os.path.join(gutenberg_dir, filename)
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                start = content.find("*** START OF ") + 50
                end = content.find("*** END OF ")
                if start > 50 and end > start:
                    content = content[start:end]
                text += content + "\n\n"
        except Exception as e:
            print(f"Skipping {filename}: {e}")

texts = text.split("\n\n")
max_size = 2 * 1024 * 1024 * 1024
current_size = 0
filtered_texts = []
for t in texts:
    size = len(t.encode("utf-8"))
    if current_size + size <= max_size:
        filtered_texts.append(t)
        current_size += size
    else:
        break
print(f"Collected {len(filtered_texts)} texts, ~{current_size / (1024**2):.2f}MB")

inputs = tokenizer(filtered_texts, return_tensors="np", padding=True, truncation=True, max_length=256)
input_ids = mx.array(inputs["input_ids"])
print(f"Tokenized corpus shape: {input_ids.shape}")


qa_pairs = [
    ("Who was Huckleberry Finn's traveling companion down the Mississippi?", "Jim"),
    ("who went with Huck Finn?", "Jim"),
    ("Who went on the raft with Huckleberry Finn?", "Jim"),
    ("Who went on the raft with Huck?", "Jim"),
    ("Who did Jim travel with?", "Huck"),
    ("Who was on the raft with Huck?", "Jim"),
    ("Who was on the raft with Jim?", "Huck Finn"),
    ("Where was Huck born?", "Hannibal"),
    ("What do Huckleberry Finn's friends call him?", "Huck"),
    ("Who is Tom Sawyer's friend?", "Huck Finn"),
    ("Who like Becky Thatcher?", "Tom Sawyer"),
    ("Who does not want to be civilized?", "Huck"),
    ("Does Santa Clause exist?", "Absolutely"),
    ("What two people famously travelled on the Mississippi on a raft?", "Huck and Jim"),
    ("Where is Huckberry Finn from?", "Hannibal"),
    ("What is the name of the young boy who is Huckberry's friend?", "Tom"),
    ("What is the shortened version of 'Huckleberry?'", "Huck"),
    ("Is Santa Clause real?", "Totally."),
    ("What river did Huckleberry Finn travel on?", "Mississippi"),
    ("Who was the scary Hative American in Tom Sawyer?", "Injun Joe"),
    ("Where was Dido from?", "Carthage"),
    ("What city did Aeneas flee?", "Troy",),
    ("Who did Dido love?", "Aeneas"),
    ("Who did Juliet love?", "Romeo"),
    ("Who did Romeo love?", "Juliet"),
    ("Who did Juliet die for?", "Romeo"),
    ("Who did Romeo die for?", "Juliet"),
    ("Who did Juliet kill herself for?", "Romeo"),
    ("Who did Romeo kill himself for?", "Juliet"),
    ("Who was the most famous Capulet?", "Juliet"),
    ("Who is the most famous Montague?", "Romeo"),
    ("Who is associated with the Capulets?", "Juliet"),
    ("Who is associated with the Montagues?", "Romeo"),
    ("Who was the young Capulet girl?", "Juliet"),
    ("Who was the young Montague boy?", "Romeo"),
    ("What house was Juliet from?", "Capulet"),
    ("Who was Juliet's confidant?", "Nurse"),
    ("Who did Mercutio fight for?", "Romeo"),
    ("Who did Mercutio die for?", "Romeo"),
    ("Who did Tybalt kill?", "Mercutio"),
    ("Who did Tybalt duel?", "Mercutio"),
    ("Who did Tybalt stab?", "Mercutio"),
    ("What was the name of Hamlet's mother?", "Gertrude"),
    ("Who loved Hamlet?", "Ophelia"),
    ("Whose death drove Achilles into a frenzy?", "Patroclus"),
    ("Whose death maddened Achilles?", "Patroclus"),
    ("Who loved Patroclus?", "Achilles"),
    ("Who wrote Pride and Prejudice?", "Jane Austin"),
    ("Who demands a pound of flesh in the Merchant of Venice?", "Shylock"),
    ("What does Shylock demand in the Merchang of Venice?", "A pound of flesh"),
    ("Who tricks Othello into jealousy?", "Iago"),
    ("What is the name of Prospero's daughter?", "Miranda"),
    ("What profit from language did Caliban gain?", "He can curse."),
    ("Who killed Hamlet father?", "Claudius"),
    ("Hamlet father was killed by whom?", "Claudius"),
    ("Who murdered Hamlet father?", "Caudius"),
    ("Who did Claudius kill in Hamlet?", "Hamlet's father"),
    ("Who did Claudius murder?", "Hamlet's father"),
    ("What happened to Hamlet's father?", "Murdered by Claudius"),
    ("Who was Pap's son?", "Huck"),
    ("What's the full name of Pap's son?","Huckleberry Finn"),
    ("What is the name of Huck's father?", "Pap"),
    ("Where was Hamlet's home?", "Elsinore"),
    ("Who was the prince of Denmark in Shakespear's famous play?", "Hamlet"),
    ("What was Hamlet's title?", "Prince of Denmark"),
    ("Who was Gertrude's son in Shakespere's famous play?", "Hamlet"),
    ("Who killed Claudius in Shakespeare's famous play?", "Hamlet"),
    ("Who did Ophelia love in Shakespeare's famous play?", "Hamlet"),
    ("Ophelia commited suicide for whom?", "Hamlet"),
    ("Hannibal Missouri is associated with who?", "Huck"),
    ("Hamlet scorned the love of who?", "Ophelia"),
    ("Whose love did Hamlet scorn?", "Ophelia"),
    ("Whose love did Hamlet not return?", "Ophelia"),
    ("Ophelia loved whom?", "Hamlet"),
    ("Who did Iago trick?", "Othello"),
    ("Who did Iago fool?", "Othello")
        
]

prompts = [
    "Who was Huck's traveling companion down the Mississippi?",
    "Where is Huck Finn from?",
    "What is the name of the young boy who is Huck's friend?",
    "Who did Iago trick?",
    "Who did Tybalt kill?",
    "Who did Claudius murder?",
    "Who did Ophelia love?"
]

validation_prompts = [
    ("Who killed Hamlet dad?", "Claudius"),
    ("Who is Huck friend?", "Tom"),
    ("Who loved Juliet?", "Romeo"),
    ("Who ignored Ophelia?", "Hamlet")
]
   
qa_texts = [f"{q} <SEP> {a}" for q, a in qa_pairs]
inputs = tokenizer(qa_texts, return_tensors="np", padding=True, truncation=True, max_length=128)
qa_input_ids = mx.array(inputs["input_ids"])
print(f"Tokenized QA data shape: {qa_input_ids.shape}")
validation_texts = [f"{q} <SEP> {a}" for q, a in validation_prompts]
val_inputs = tokenizer(validation_texts, return_tensors="np", padding=True, truncation=True, max_length=128)
val_input_ids = mx.array(val_inputs["input_ids"])


loss_fn = nn.losses.cross_entropy

def loss_fn_corpus(model, x):
    logits = model(x[:, :-1])
    targets = x[:, 1:]
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    return mx.mean(loss_fn(logits_flat, targets_flat, reduction='none'))

def loss_fn_qa(model, x):
    logits = model(x[:, :-1])
    targets = x[:, 1:]
    sep_token_id = tokenizer.sep_token_id
    mask = mx.zeros(targets.shape, dtype=mx.bool_)
    for i in range(targets.shape[0]):
        # Convert targets[i] to a Python list for manual inspection
        row = targets[i].tolist()
        sep_idx = -1
        for j, val in enumerate(row):
            if val == sep_token_id:
                sep_idx = j
                break
        if sep_idx >= 0:
            mask[i, sep_idx + 1:] = True
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)
    loss = loss_fn(logits_flat, targets_flat, reduction='none')  # Changed None to 'none'
    masked_loss = mx.where(mask_flat, loss, mx.zeros_like(loss))
    return mx.mean(masked_loss)

batch_size_pretrain = 16
num_epochs_pretrain = 5
total_steps = num_epochs_pretrain * (len(input_ids) // batch_size_pretrain)
scheduler = optim.cosine_decay(1e-4, num_epochs_pretrain * (len(input_ids) // batch_size_pretrain))
optimizer = optim.Adam(learning_rate=scheduler, bias_correction=True)
batch_size_qa = 64  # For QA pairs (matches your 4 pairs)
num_epochs_qa = 30
optimizer_qa = optim.Adam(learning_rate=5e-5, bias_correction= True)
print("Fine-tuning on QA pairs...")
for epoch in range(num_epochs_qa):
    print(f"Epoch {epoch + 1}/{num_epochs_qa}")
    for i in range(0, len(qa_input_ids), batch_size_qa):
        batch = qa_input_ids[i:i + batch_size_qa]
        if batch.shape[0] < batch_size_qa:
            continue
        batch_idx = i // batch_size_qa
        loss_and_grad_fn = nn.value_and_grad(model, lambda m: loss_fn_qa(m, batch))
        loss, grads = loss_and_grad_fn(model)
        optimizer_qa.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if batch_idx % 10000 == 0 or i + batch_size_qa >= len(qa_input_ids):  # Every 1000 batches or last batch
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    # Validation step
    val_loss = loss_fn_qa(model, val_input_ids)
    print(f"Validation Loss: {val_loss.item():.4f}")
    
model.save_weights(f"{model_dir}/baby_llm_qa.npz")

model.save_weights(f"{model_dir}/baby_llm_pretrain.npz")
def generate_text(model, tokenizer, prompt, max_tokens=15, top_k=2):
    input_ids = mx.array(tokenizer(f"{prompt} <SEP>", return_tensors="np")["input_ids"])
    output_ids = input_ids
    print(f"Input IDs: {input_ids.tolist()}")
    for _ in range(max_tokens):
        logits = model(output_ids)
        logits = logits[:, -1, :]
        top_k_indices = mx.topk(logits, k=top_k, axis=-1)
        top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
        probs = mx.softmax(top_k_logits, axis=-1)
        next_token_idx = mx.random.categorical(probs.log())
        next_token = top_k_indices[0, next_token_idx].astype(mx.int32)
        ##print(f"Next token ID: {next_token.item()}, Decoded: {tokenizer.decode([next_token.item()])}")
        output_ids = mx.concatenate([output_ids, mx.array([next_token]).reshape(1, 1)], axis=1)
        if next_token.item() in [tokenizer.eos_token_id, tokenizer.sep_token_id]:
            break
    output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)  # Skip special tokens
    return output_text.split('<SEP>')[-1].strip() if '<SEP>' in output_text else output_text.strip()

# Test it
test_prompt = "Who was Huck Finn traveling companion down the Mississippi?"
generated = generate_text(model, tokenizer, test_prompt)
print(f"Generated: {generated}")


def generate_temp(model, tokenizer, prompt, max_tokens=5, temperature=0.7):
    input_ids = mx.array(tokenizer(prompt, return_tensors="np")["input_ids"])
    output_ids = input_ids
    for _ in range(max_tokens):
        logits = model(output_ids)
        logits = logits[:, -1, :] / temperature
        probs = mx.softmax(logits, axis=-1)
        next_token = mx.random.categorical(probs.log())
        output_ids = mx.concatenate([output_ids, next_token[:, None]], axis=1)
    return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

def generate_single(model, tokenizer, prompt, max_tokens, top_k):
    input_ids = mx.array(tokenizer(prompt, return_tensors="np")["input_ids"])   
    output_ids = input_ids
    for _ in range(max_tokens):
        logits = model(output_ids)
        logits = logits[:, -1, :]
        ##print(f"Logits shape: {logits.shape}, Top 5: {mx.topk(logits, k=5).tolist()}")
        top_k_indices = mx.topk(logits, k=top_k, axis=-1)
        top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
        probs = mx.softmax(top_k_logits, axis=-1)
        next_token_idx = mx.random.categorical(probs.log())
        next_token = top_k_indices[0, next_token_idx].astype(mx.int32)
        ##print(f"Next token: {next_token.item()} ({tokenizer.decode([next_token.item()])})")
        output_ids = mx.concatenate([output_ids, mx.array([next_token]).reshape(1, 1)], axis=1)
        if next_token.item() in [tokenizer.eos_token_id, tokenizer.encode('.')[0]]:
            break
    return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True).split('<SEP>')[-1].strip()

test_prompt = "Huck Finn traveled down the "
generated = generate_temp(model, tokenizer, test_prompt)
print(f"Pretraining test: {generated}")

for prompt in prompts:
    generated = generate_text(model, tokenizer, prompt)
    print(f"Prompt for top_k: {prompt}")
    print(f"Answer - generate_text for top_k: {generated}\n")
    generated_temp = generate_temp(model, tokenizer, prompt)
    print(f"Prompt for temp: {prompt}")
    print(f"Answer - generate_temp: {generated}\n")
    
for prompt in validation_prompts:
    question = prompt[0]  # Use only the question part of the tuple
    generated = generate_text(model, tokenizer, question)
    print(f"Validation Prompt for top_k: {question}")
    print(f"Validation Answer - generate_text for top_k: {generated}\n")
    generated_temp = generate_temp(model, tokenizer, question)
    print(f"Prompt for temp: {question}")
    print(f"Validation Answer generate_temp for temp: {generated_temp}\n")
    
for prompt in prompts:
    question = prompt  # Use only the question part of the tuple
    generated = generate_single(model, tokenizer, question, vocab_size // 200, 5)
    print(f"Prompt for single: {question}")
    print(f"Single Answer for : {generated}\n")
for prompt in validation_prompts:
    question = prompt[0]
    generated_single = generate_single(model, tokenizer, question, vocab_size // 2, 5)
    print(f"Prompt for temp: {question}")
    print(f"Single Answer for temp: {generated_single}\n")