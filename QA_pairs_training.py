import os
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer
import matplotlib

# Setup logging
logging.basicConfig(filename='qa_training.log', level=logging.INFO)

# Define and create directories
base_dir = "~/Baby_LLM"
cache_dir = os.path.expanduser(os.path.join(base_dir, "cache"))
model_dir = os.path.expanduser(os.path.join(base_dir, "model"))
data_dir = os.path.expanduser(os.path.join(base_dir, "data"))
gutenberg_dir = os.path.join(data_dir, "gutenberg")
cleaned_dir = os.path.join(data_dir, "cleaned")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(gutenberg_dir, exist_ok=True)
os.makedirs(cleaned_dir, exist_ok=True)

# Model definition
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
        self.attention = nn.MultiHeadAttention(d_model, n_heads, bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.3)
    def __call__(self, x):
        attn_output = self.dropout(self.attention(x, x, x))
        x = self.norm1(x + attn_output)
        ff_output = self.dropout(self.ff(x))
        x = self.norm2(x + ff_output)
        return x

class BabyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = [TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def __call__(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x) * mx.sqrt(self.d_model)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN or Inf after embedding")
        positions = mx.arange(seq_len)
        x = x + self.pos_embedding(positions)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN or Inf after pos_embedding")
        for layer in self.layers:
            x = layer(x)
            if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
                logging.error("NaN or Inf after layer")
        x = self.final_norm(x)
        if mx.any(mx.isnan(x)) or mx.any(mx.isinf(x)):
            logging.error("NaN or Inf after final_norm")
        return self.output(x)

# Utility functions - FIXED VERSION
def to_numpy_for_decode(array):
    """Convert MLX array to NumPy for tokenizer decoding"""
    if isinstance(array, mx.array):
        return np.array(array)  # Convert MLX array to NumPy
    return array

def clip_gradients(grads, max_norm=0.5):
    flat_grads = []
    for g in grads.values():
        if g is not None and isinstance(g, mx.array):
            flat_grads.append(g.flatten())
        elif isinstance(g, dict):
            for sub_g in g.values():
                if sub_g is not None and isinstance(sub_g, mx.array):
                    flat_grads.append(sub_g.flatten())
        elif isinstance(g, list):
            for sub_g in g:
                if isinstance(sub_g, dict):
                    for sub_sub_g in sub_g.values():
                        if sub_sub_g is not None and isinstance(sub_sub_g, mx.array):
                            flat_grads.append(sub_sub_g.flatten())
    if not flat_grads:
        return grads
    total_norm = mx.sqrt(sum(mx.sum(g * g) for g in flat_grads))
    scale = mx.minimum(1.0, max_norm / (total_norm + 1e-8))
    def scale_gradient(g):
        if isinstance(g, mx.array):
            return g * scale
        elif isinstance(g, dict):
            return {k: scale_gradient(v) for k, v in g.items()}
        elif isinstance(g, list):
            return [scale_gradient(v) for v in g]
        return g
    return {k: scale_gradient(g) for k, g in grads.items()}

def scale_gradients(grads, scale):
    if grads is None:
        return None
    
    # Handle different gradient structures
    if isinstance(grads, mx.array):
        return grads * scale
    elif isinstance(grads, dict):
        scaled = {}
        for k, v in grads.items():
            scaled[k] = scale_gradients(v, scale)  # Recursive call
        return scaled
    elif isinstance(grads, list):
        return [scale_gradients(item, scale) for item in grads]  # Recursive call for each item
    else:
        # For any other type, return as-is
        return grads
    
def add_grads(acc, new):
    if acc is None or new is None:
        return acc if new is None else new
    
    # Handle different gradient structures
    if isinstance(acc, mx.array) and isinstance(new, mx.array):
        return acc + new
    elif isinstance(acc, dict) and isinstance(new, dict):
        result = {}
        for k in acc.keys():
            if k in new:
                result[k] = add_grads(acc[k], new[k])  # Recursive call
            else:
                result[k] = acc[k]
        return result
    elif isinstance(acc, list) and isinstance(new, list):
        return [add_grads(acc[i], new[i]) for i in range(len(acc))]  # Recursive call for each item
    else:
        # If types don't match or are unexpected, return the new gradient
        return new
    

def compute_accuracy(model, tokenizer, val_pairs):
    correct = 0
    for question, true_answer in val_pairs:
        generated = generate_answer(model, tokenizer, question, sampling='top_k')
        if clean_answer(generated).lower() == true_answer.lower():
            correct += 1
    return correct / len(val_pairs)

def clean_answer(text):
    text = text.strip('.!?\n')
    return text if text else "Unknown"

def nucleus_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = mx.sort(logits, axis=-1, descending=True)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
    mask = cumsum_probs <= p
    top_p_indices = sorted_indices[mask]
    top_p_logits = mx.take_along_axis(logits, top_p_indices, axis=-1)
    probs = mx.softmax(top_p_logits, axis=-1)
    next_token_idx = mx.random.categorical(probs.log())
    return top_p_indices[0, next_token_idx].reshape(1, 1)

# Load tokenizer and model
HF_TOKEN = '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'

try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", token=HF_TOKEN, cache_dir=cache_dir)
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    raise
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'sep_token': '<SEP>'})

# Sanitize and pad pretrained weights
weights_path = os.path.join(model_dir, "baby_llm_pretrain.npz")
try:
    weights = np.load(weights_path)
    weights_dict = dict(weights)
    expected_vocab_size = tokenizer.vocab_size + 1  # 32770
    for k, v in weights_dict.items():
        if np.any(np.isnan(v)) or np.any(np.isinf(v)):
            logging.warning(f"Invalid values in {k}. Replacing NaN/Inf with 0.")
            v[np.isnan(v) | np.isinf(v)] = 0
            weights_dict[k] = v
    # Pad embedding.weight and output.weight if necessary
    if 'embedding.weight' in weights_dict and weights_dict['embedding.weight'].shape[0] < expected_vocab_size:
        logging.info(f"Padding embedding.weight from {weights_dict['embedding.weight'].shape} to ({expected_vocab_size}, 512)")
        weights_dict['embedding.weight'] = np.pad(
            weights_dict['embedding.weight'],
            ((0, expected_vocab_size - weights_dict['embedding.weight'].shape[0]), (0, 0)),
            mode='constant'
        )
    if 'output.weight' in weights_dict and weights_dict['output.weight'].shape[0] < expected_vocab_size:
        logging.info(f"Padding output.weight from {weights_dict['output.weight'].shape} to ({expected_vocab_size}, 512)")
        weights_dict['output.weight'] = np.pad(
            weights_dict['output.weight'],
            ((0, expected_vocab_size - weights_dict['output.weight'].shape[0]), (0, 0)),
            mode='constant'
        )
    if 'output.bias' in weights_dict and weights_dict['output.bias'].shape[0] < expected_vocab_size:
        logging.info(f"Padding output.bias from {weights_dict['output.bias'].shape} to ({expected_vocab_size},)")
        weights_dict['output.bias'] = np.pad(
            weights_dict['output.bias'],
            (0, expected_vocab_size - weights_dict['output.bias'].shape[0]),
            mode='constant'
        )
    # Save modified weights
    np.savez(weights_path, **weights_dict)
except FileNotFoundError:
    logging.error("Pretrained weights not found. Starting from scratch.")

model = BabyLLM(vocab_size=tokenizer.vocab_size + 1)
try:
    model.load_weights(weights_path)
except FileNotFoundError:
    logging.warning("No pretrained weights found. Initializing from scratch.")
    mx.eval(model.parameters())

# QA data
qa_pairs = [
    ("In the novel Huckleberry Finn, who was Huckleberry Finn's traveling companion down the Mississippi?", "Jim"),
    ("In the book Huckleberry Finn, who went with Huck Finn?", "Jim"),
    ("In Huckleberry Finn, who went on the raft with Huckleberry Finn?", "Jim"),
    ("In the novle Huckleberry Finn, who went on the raft with Huck?", "Jim"),
    ("In Mark Twain's novel, who did Jim travel with?", "Huck"),
    ("In the book Huckleberry Finn, wWho was on the raft with Huck?", "Jim"),
    ("In Huckleberry Finn, who was on the raft with Jim?", "Huck Finn"),
    ("Where was Huck born in the book Huckleberry Finn?", "Hannibal"),
    ("In the book Huckleberry Fin, what do Huckleberry Finn's friends call him?", "Huck"),
    ("In huckleberry Finn, who is Tom Sawyer's friend?", "Huck Finn"),
    ("Who like Becky Thatcher in the novel Huckleberry Finn?", "Tom Sawyer"),
    ("Who does not want to be civilized in the book Huckleberry Finn?", "Huck"),
    ("In the book Huckleberry Finn ,who does not want to be civilized", "Huck"),
    ("What two people famously travelled on the Mississippi on a raftin the novel Huckleberry Finn?", "Huck and Jim"),
    ("Where is Huckberry Finn from?", "Hannibal"),
    ("What is the name of the young boy who is Huckberry's friend in the book Huckleberry Finn?", "Tom"),
    ("What is the shortened version of 'Huckleberry'in the book Huckleberry Finn?", "Huck"),
    ("Is Santa Clause real?", "Totally"),
    ("What river did Huckleberry Finn travel onin the book Huckleberry Finn?", "Mississippi"),
    ("Who was the scary Native American in Tom Sawyer?", "Injun Joe"),
    ("Where was Dido from in the Aeneid?", "Carthage"),
    ("In the Aeneid hat city did Aeneas flee?", "Troy"),
    ("Who did Dido love in the Aeneid?", "Aeneas"),
    ("Who did Juliet love in the play Romeo and Juliet?", "Romeo"),
    ("In the play Romeo and Juliet, who did Romeo love?", "Juliet"),
    ("Who did Juliet die for in the play Romeo and Juliet?", "Romeo"),
    ("In Romeo and Juliet, who did Romeo die for?", "Juliet"),
    ("Who did Juliet kill herself for in fomeo and Juliet?", "Romeo"),
    ("Who did Romeo kill himself for in the play Romeo and Juliet?", "Juliet"),
    ("Who was the most famous Capulet in the play Romeo and Juliet?", "Juliet"),
    ("In Romeo and Juliet, who is the most famous Montague?", "Romeo"),
    ("Who is associated with the Capulets in Romeo an dJuliet?", "Juliet"),
    ("In Romeo and Juliet who is associated with the Montagues?", "Romeo"),
    ("In the play Romeo and Juliet, Who was the young Capulet girl?", "Juliet"),
    ("Who was the young Montague boy in Romeo and Juliet?", "Romeo"),
    ("What house was Juliet from in Romeo and Juliet?", "Capulet"),
    ("In Romeo and Juliet, who was Juliet's confidant?", "Nurse"),
    ("Who did Mercutio fight for in Romeo and Juliet?", "Romeo"),
    ("In Romeo and Juliet, who did Mercutio die for?", "Romeo"),
    ("Who did Tybalt kill instead of Romeo?", "Mercutio"),
    ("Who did Tybalt duel in Romeo and Juliet?", "Mercutio"),
    ("In romeo and Juliet, who did Tybalt stab?", "Mercutio"),
    ("What was the name of Hamlet's mother in the play Hamlet?", "Gertrude"),
    ("Who loved Hamlet in the play Hamlet?", "Ophelia"),
    ("In the Illiad, Whose death drove Achilles into a frenzy?", "Patroclus"),
    ("Whose death maddened Achilles in the Iliad?", "Patroclus"),
    ("Who loved Patroclus in the iliad?", "Achilles"),
    ("Who wrote Pride and Prejudice?", "Jane Austen"),
    ("Who demands a pound of flesh in the Merchant of Venice?", "Shylock"),
    ("What does Shylock demand in the Merchant of Venice?", "A pound of flesh"),
    ("Who tricks Othello into jealousy in the play Othello?", "Iago"),
    ("What is the name of Prospero's daughter in the Tempest?", "Miranda"),
    ("In The Tempest, what profit from language did Caliban gain?", "He can curse"),
    ("What was Caliban's profit from language in The Tempest?", "He can curse"),
    ("Who killed Hamlet's father in the play Hamlet?", "Claudius"),
    ("Hamlet's father was killed by whom in the play Hamlet?", "Claudius"),
    ("In the play Hamlet,Who murdered Hamlet's father?", "Claudius"),
    ("Who did Claudius kill in the play Hamlet?", "Hamlet's father"),
    ("Who did Claudius murder in the play Hamlet?", "Hamlet's father"),
    ("In the play Hamlet,what happened to Hamlet's father?", "Murdered by Claudius"),
    ("Who was Pap's son in Huckleberry Finn?", "Huck"),
    ("In the novel Huckleberry Finn, what's the full name of Pap's son?", "Huckleberry Finn"),
    ("What is the name of Huck's father in the book Huckleberry Rinn?", "Pap"),
    ("Where was Hamlet's home in the play Hamlet?", "Elsinore"),
    ("Who was the prince of Denmark in Shakespeare's famous play Hamlet?", "Hamlet"),
    ("In the play Hamlet, what was Hamlet's title?", "Prince of Denmark"),
    ("Who was Gertrude's son in Shakespeare'splay Hamlet?", "Hamlet"),
    ("Who killed Claudius in Shakespeare's play hamlet?", "Hamlet"),
    ("Who did Ophelia love in the play Hamlet?", "Hamlet"),
    ("Ophelia committed suicide for whom in the play Hamlet?", "Hamlet"),
    ("Hannibal, Missouri is associated with who in the book Huckleberry Finn?", "Huck"),
    ("Hamlet scorned the love of who in the play  Hamlet?", "Ophelia"),
    ("Whose love did Hamlet scorn in the play Hamet?", "Ophelia"),
    ("Whose love did Hamlet not return in the play Hamlet?", "Ophelia"),
    ("in the play Hamlet, Ophelia loved whom?", "Hamlet"),
    ("In the play Othello, who did Iago trick?", "Othello"),
    ("Who did Iago fool in the play Othello?", "Othello"),
    ("What river did Huck navigate in the book Huckleberry Finn?", "Mississippi"),
    ("Who was the boy who rafted down the Mississippi riverin Huckleberry Finn?", "Huck Finn"),
    ("Who fooled Othello in the play Othello?", "Iago"),
    ("Who is the captain of the Pequod in Moby-Dick,?", "Ahab"),
    ("In Pride and Prejudice, who marries Elizabeth Bennet?", "Mr. Darcy"),
    ("In The Odyssey, who is Odysseus's wife?", "Penelope"),
    ("In The Scarlet Letter, what symbol does Hester Prynne wear?", "A"),
    ("In Great Expectations, who raises Pip?", "Joe Gargery"),
    ("What color was the rabbit that Alice followed down the rabbit hole?", "White"),
    ("Who asked the riddle about the raven and the writing desk?", "The Mad Hatter"),
    ("What is the subject of the famous riddle by the Mad Hatter?", "Raven and writing desk"),
    ("How many impossible things does the Red Queen believe before breakfast?", "Six"),
    ("What six things does the Red Queen believe?", "Impossible things"),
    ("Who believes six impossible things before breakfast?", "The Red Queen"),
    ("When does the Red Queen believe six impossible things?", "Before breakfast"),
    ("What ship did Queequeg sail on?", "Pequod"),
    ("Who was Ahab's chief mate?", "Starbuck"),
    ("Who was Starbuck's captain?", "Ahab"),
    ("Who was Ahab's second mate?", "Stubb"),
    ("Stubb was whose second mate?", "Ahab"),
    ("Who was the cannibal harpoonist on the Pequod?", "Queequeg"),
    ("Who was Queequeg's captain?", "Ahab"),
    ("What was the name of Ahab's ship?","Pequod"),
    ("Ahab was the captain of what ship?", "Pequod"),
    ("Who was the young boy who rafted down the Mississippi?", "Huck"),
    ("Who was the black man who rafted down the Mississippi River?", "Nigger Jim"),
    ("What is the name of the young boy who rafted down the Mississippi River?", "Huck"),
    ("What is the name of the black man who rafted down the Missippi River?","Nigger Jim"),
    ("Who was Odysseus's wife?", "Penelope"),
    ("What was the name of Odysseus's wife?", "Penelope"),
    ("Who was Odysseus married to?", "Penelope"),
    ("What was the name of the woman Odysseus was married to?", "Penelope"),
    ("Odysseus was married to whom?", "Penelope"),
    ("What goddess helped Odysseus?", "Athena"),
    ("Odysseus was helped by what goddess?", "Athena"),
    ("Athena helped which character in the Odyssey?", "Odysseus"),
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
    ("Is Santa Clause real?", "Totally"),
    ("What river did Huckleberry Finn travel on?", "Mississippi"),
    ("Who was the scary Native American in Tom Sawyer?", "Injun Joe"),
    ("Where was Dido from?", "Carthage"),
    ("What city did Aeneas flee?", "Troy"),
    ("Who did Dido love?", "Aeneas"),
    ("Who did Juliet love?", "Romeo"),
    ("Who did Romeo love?", "Juliet"),
    ("Who did Juliet die for?", "Romeo"),
    ("Who did Romeo die for?", "Juliet"),
    ("Who did Juliet kill herself for?", "Romeo"),
    ("Who did Romeo kill herself for?", "Juliet"),
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
    ("Who wrote Pride and Prejudice?", "Jane Austen"),
    ("Who demands a pound of flesh in the Merchant of Venice?", "Shylock"),
    ("What does Shylock demand in the Merchant of Venice?", "A pound of flesh"),
    ("Who tricks Othello into jealousy?", "Iago"),
    ("What is the name of Prospero's daughter?", "Miranda"),
    ("What profit from language did Caliban gain?", "He can curse"),
    ("Who killed Hamlet's father?", "Claudius"),
    ("Hamlet's father was killed by whom?", "Claudius"),
    ("Who murdered Hamlet's father?", "Claudius"),
    ("Who did Claudius kill in Hamlet?", "Hamlet's father"),
    ("Who did Claudius murder?", "Hamlet's father"),
    ("What happened to Hamlet's father?", "Murdered by Claudius"),
    ("Who was Pap's son?", "Huck"),
    ("What's the full name of Pap's son?", "Huckleberry Finn"),
    ("What is the name of Huck's father?", "Pap"),
    ("Where was Hamlet's home?", "Elsinore"),
    ("Who was the prince of Denmark in Shakespeare's famous play?", "Hamlet"),
    ("What was Hamlet's title?", "Prince of Denmark"),
    ("Who was Gertrude's son in Shakespeare's famous play?", "Hamlet"),
    ("Who killed Claudius in Shakespeare's famous play?", "Hamlet"),
    ("Who did Ophelia love in Shakespeare's famous play?", "Hamlet"),
    ("Ophelia committed suicide for whom?", "Hamlet"),
    ("Hannibal, Missouri is associated with who?", "Huck"),
    ("Hamlet scorned the love of who?", "Ophelia"),
    ("Whose love did Hamlet scorn?", "Ophelia"),
    ("Whose love did Hamlet not return?", "Ophelia"),
    ("Ophelia loved whom?", "Hamlet"),
    ("Who did Iago trick?", "Othello"),
    ("Who did Iago fool?", "Othello"),
    ("What river did Huck navigate?", "Mississippi"),
    ("Who was the boy who rafted down the Mississippi river?", "Huck Finn"),
    ("Who fooled Othello?", "Iago"),
    ("In Moby-Dick, who is the captain of the Pequod?", "Ahab"),
    ("In Pride and Prejudice, who marries Elizabeth Bennet?", "Mr. Darcy"),
    ("In The Odyssey, who is Odysseus's wife?", "Penelope"),
    ("In The Scarlet Letter, what symbol does Hester Prynne wear?", "A"),
    ("In Great Expectations, who raises Pip?", "Joe Gargery"),
    ("What color was the rabbit that Alice followed down the rabbit hole?", "White"),
    ("Who asked the riddle about the raven and the writing desk?", "The Mad Hatter"),
    ("What is the subject of the famous riddle by the Mad Hatter?", "Raven and writing desk"),
    ("How many impossible things does the Red Queen believe before breakfast?", "Six"),
    ("What six things does the Red Queen believe?", "Impossible things"),
    ("Who believes six impossible things before breakfast?", "The Red Queen"),
    ("When does the Red Queen believe six impossible things?", "Before breakfast"),
    ("What ship did Queequeg sail on?", "Pequod"),
    ("Who was Ahab's chief mate?", "Starbuck"),
    ("Who was Starbuck's captain?", "Ahab"),
    ("Who was Ahab's second mate?", "Stubb"),
    ("Stubb was whose second mate?", "Ahab"),
    ("Who was the cannibal harpoonist on the Pequod?", "Queequeg"),
    ("Who was Queequeg's captain?", "Ahab"),
    ("What was the name of Ahab's ship?","Pequod"),
    ("Ahab was the captain of what ship?", "Pequod"),
    ("Who was the young boy who rafted down the Mississippi?", "Huck"),
    ("Who was the black man who rafted down the Mississippi River?", "Nigger Jim"),
    ("What is the name of the young boy who rafted down the Mississippi River?", "Huck"),
    ("What is the name of the black man who rafted down the Missippi River?","Nigger Jim"),
    ("Who was Odysseus's wife?", "Penelope"),
    ("What was the name of Odysseus's wife?", "Penelope"),
    ("Who was Odysseus married to?", "Penelope"),
    ("What was the name of the woman Odysseus was married to?", "Penelope"),
    ("Odysseus was married to whom?", "Penelope"),
    ("What goddess helped Odysseus?", "Athena"),
    ("Odysseus was helped by what goddess?", "Athena"),
    ("Athena helped which character in the Odyssey?", "Odysseus")

    
]

qa_texts = [f"Question: {q} Answer: {a}" for q, a in qa_pairs]
inputs = tokenizer(qa_texts, return_tensors="np", padding=True, truncation=True, max_length=128)
qa_input_ids = mx.array(inputs["input_ids"], dtype=mx.int32)
if mx.any(qa_input_ids < 0) or mx.any(qa_input_ids >= tokenizer.vocab_size + 1):
    logging.error("Invalid token IDs in qa_input_ids")
    raise ValueError("Invalid token IDs")

# FIXED: Check sequences only once, not in a loop
print(f"QA input shape: {qa_input_ids.shape}")
print(f"First QA sequence: {tokenizer.decode(to_numpy_for_decode(qa_input_ids[0]))}")

# Check for Answer: in sequences
for i in range(len(qa_input_ids)):
    seq = qa_input_ids[i]
    decoded = tokenizer.decode(to_numpy_for_decode(seq))  # FIXED: Use the corrected function
    if 'Answer:' not in decoded:
        logging.warning(f"No 'Answer:' in qa_input_ids[{i}]: {decoded}")

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
if mx.any(val_input_ids < 0) or mx.any(val_input_ids >= tokenizer.vocab_size + 1):
    logging.error("Invalid token IDs in val_input_ids")
    raise ValueError("Invalid token IDs")

# Check validation sequences
for i in range(len(val_input_ids)):
    seq = val_input_ids[i]
    decoded = tokenizer.decode(to_numpy_for_decode(seq))  # FIXED: Use the corrected function
    if 'Answer:' not in decoded:
        logging.warning(f"No 'Answer:' in val_input_ids[{i}]: {decoded}")

prompts = [
    "In Moby-Dick, what motivates Ahab's pursuit of the whale?",
    "In Pride and Prejudice, what is Mr. Darcy's first name?",
    "In The Odyssey, who is the goddess who helps Odysseus?",
    "In The Scarlet Letter, why does Hester Prynne wear the scarlet letter?",
    "In Great Expectations, what is Pip's ultimate ambition?"
]

# Loss function (FIXED to handle MLX arrays properly)
def loss_fn_qa(model, x):
    logits = model(x[:, :-1])
    if mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
        logging.error("NaN or Inf in logits")
    logits = mx.clip(logits, -1e9, 1e9)
    targets = x[:, 1:]
    mask = mx.zeros(targets.shape, dtype=mx.bool_)
    for i in range(targets.shape[0]):
        # Convert MLX array to NumPy for decoding - FIXED
        decoded = tokenizer.decode(to_numpy_for_decode(x[i]))
        answer_idx = decoded.find("Answer:") + len("Answer:")
        if answer_idx < len("Answer:"):
            logging.warning(f"No 'Answer:' in sequence {i}: {decoded}")
            continue
        tokenized = tokenizer(decoded, return_tensors="np")["input_ids"][0]
        answer_token_idx = len(tokenizer(decoded[:answer_idx], return_tensors="np")["input_ids"][0])
        mask[i, answer_token_idx:] = True
    if not mx.any(mask):
        logging.error("Empty mask in loss_fn_qa")
        return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)
    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
    if mx.any(mx.isnan(loss)) or mx.any(mx.isinf(loss)):
        logging.error("NaN or Inf in cross_entropy loss")
    masked_loss = mx.where(mask_flat, loss, mx.zeros_like(loss))
    mask_sum = mx.sum(mask_flat) + 1e-8
    final_loss = mx.sum(masked_loss) / mask_sum
    if mx.isnan(final_loss) or mx.isinf(final_loss):
        logging.error("NaN or Inf in final loss")
    elif final_loss == 0:
        logging.error("Zero loss detected")
    return final_loss
        
# Generation function with beam search
def generate_answer(model, tokenizer, prompt, max_tokens=50, top_k=10, temperature=0.7, p=0.9, sampling='top_k', beam_size=2):
    if sampling == 'beam':
        input_ids = mx.array(tokenizer(f"Question: {prompt} Answer:", return_tensors="np")["input_ids"], dtype=mx.int32)
        beams = [(input_ids, 0.0)]
        for _ in range(max_tokens):
            new_beams = []
            for seq, score in beams:
                logits = model(seq)[:, -1, :]
                if mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
                    logging.error("NaN or Inf in generation logits")
                    return "Generation failed due to invalid logits"
                top_k_indices = mx.topk(logits, k=top_k, axis=-1)
                top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
                probs = mx.softmax(top_k_logits / temperature, axis=-1)
                for i in range(top_k):
                    next_token = top_k_indices[0, i].reshape(1, 1).astype(mx.int32)
                    new_seq = mx.concatenate([seq, next_token], axis=1)
                    new_score = score + mx.log(probs[0, i]).item()
                    new_beams.append((new_seq, new_score))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            if beams[0][0][:, -1].item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break
        output_text = tokenizer.decode(to_numpy_for_decode(beams[0][0][0]), skip_special_tokens=True)
        return clean_answer(output_text.split('Answer:')[-1].strip())
    else:
        input_ids = mx.array(tokenizer(f"Question: {prompt} Answer:", return_tensors="np")["input_ids"], dtype=mx.int32)
        output_ids = input_ids
        for _ in range(max_tokens):
            logits = model(output_ids)[:, -1, :]
            if mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
                logging.error("NaN or Inf in generation logits")
                return "Generation failed due to invalid logits"
            if sampling == 'top_k':
                top_k_indices = mx.topk(logits, k=top_k, axis=-1)
                top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
                probs = mx.softmax(top_k_logits / temperature, axis=-1)
                next_token_idx = mx.random.categorical(probs.log())
                next_token = top_k_indices[0, next_token_idx].reshape(1, 1).astype(mx.int32)
            elif sampling == 'nucleus':
                next_token = nucleus_sampling(logits, p=p).astype(mx.int32)
            else:
                probs = mx.softmax(logits / temperature, axis=-1)
                next_token = mx.random.categorical(probs.log())[:, None].astype(mx.int32)
            output_ids = mx.concatenate([output_ids, next_token], axis=1)
            if next_token.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break
        output_text = tokenizer.decode(to_numpy_for_decode(output_ids[0]), skip_special_tokens=True)
        return clean_answer(output_text.split('Answer:')[-1].strip())
    
## Training setup
batch_size_qa = 128
num_epochs_qa = 50
total_steps_qa = num_epochs_qa * (len(qa_input_ids) // batch_size_qa)
scheduler_qa = optim.cosine_decay(5e-6, decay_steps=total_steps_qa)
optimizer_qa = optim.AdamW(learning_rate=scheduler_qa, weight_decay=0.01)
patience = 10
best_val_loss = float('inf')
patience_counter = 0
accumulation_steps = 8
train_losses, val_losses = [], []

for epoch in range(num_epochs_qa):
    print(f"Epoch {epoch + 1}/{num_epochs_qa}")
    indices = mx.array(np.random.permutation(len(qa_input_ids)))
    accumulated_grads = None
    valid_batches = 0
    for i in range(0, len(qa_input_ids), batch_size_qa):
        batch_indices = indices[i:i + batch_size_qa]
        if batch_indices.shape[0] < batch_size_qa:
            continue
        batch = mx.take(qa_input_ids, batch_indices, axis=0)
        batch_idx = i // batch_size_qa
        loss_and_grad_fn = nn.value_and_grad(model, lambda m: loss_fn_qa(m, batch))
        loss, grads = loss_and_grad_fn(model)
        if mx.isnan(loss) or mx.isinf(loss):
            logging.warning(f"Invalid loss ({loss.item()}) at batch {batch_idx}. Skipping.")
            continue
        valid_batches += 1
        scaled_grads = scale_gradients(grads, 1.0 / accumulation_steps)
        accumulated_grads = add_grads(accumulated_grads, scaled_grads)
        if (batch_idx + 1) % accumulation_steps == 0 or i + batch_size_qa >= len(qa_input_ids):
            if accumulated_grads is not None:
                accumulated_grads = clip_gradients(accumulated_grads, max_norm=1.0)
                optimizer_qa.update(model, accumulated_grads)
                mx.eval(model.parameters(), optimizer_qa.state)
                accumulated_grads = None
        train_losses.append(loss.item())
        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        logging.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    if valid_batches == 0:
        print(f"Epoch {epoch + 1}: No valid batches processed. Stopping.")
        break
    
    # Validation
    val_loss = loss_fn_qa(model, val_input_ids)
    val_accuracy = compute_accuracy(model, tokenizer, validation_prompts)
    val_losses.append(val_loss.item())
    print(f"Validation Loss: {val_loss.item():.4f}, Accuracy: {val_accuracy:.4f}")
    logging.info(f"Epoch {epoch + 1}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_weights(os.path.join(model_dir, "baby_llm_qa_best.npz"))
        tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Save final model
model.save_weights(os.path.join(model_dir, "baby_llm_qa_final.npz"))


# Plot losses
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.savefig(os.path.join(model_dir, 'loss_plot.png'))

# Test generation
for prompt in prompts:
    answer = generate_answer(model, tokenizer, prompt, sampling='beam', beam_size=2)
    print(f"Prompt: {prompt}\nAnswer: {answer}\n")
for prompt, _ in validation_prompts:
    answer = generate_answer(model, tokenizer, prompt, sampling='beam', beam_size=2)
    print(f"Prompt: {prompt}\nAnswer: {answer}\n")