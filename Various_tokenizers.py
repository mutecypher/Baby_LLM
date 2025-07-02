from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'sep_token': '<SEP>'})


## custome bpe tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=15000, special_tokens=["<PAD>", "<SEP>", "<EOS>", "<BOS>"])
tokenizer.train(files=[os.path.join(cleaned_dir, f) for f in os.listdir(cleaned_dir) if f.endswith('.cleaned.txt')], trainer=trainer)
tokenizer.save(os.path.join(model_dir, "custom_tokenizer.json"))

# Load in Hugging Face format
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "custom_tokenizer.json"))
tokenizer.pad_token = "<PAD>"
tokenizer.eos_token = "<EOS>"
tokenizer.bos_token = "<BOS>"
tokenizer.sep_token = "<SEP>"


from collections import Counter
import nltk

# Build vocabulary
all_words = []
for text in filtered_texts:
    words = nltk.word_tokenize(text.lower())
    all_words.extend([w for w in words if w in string.ascii_letters + string.digits + '.,?!\'"-;:() '])
vocab = {word: idx + 4 for idx, (word, _) in enumerate(Counter(all_words).most_common(10000))}
vocab.update({"<PAD>": 0, "<SEP>": 1, "<EOS>": 2, "<BOS>": 3})
reverse_vocab = {idx: word for word, idx in vocab.items()}

def custom_tokenize(text, max_length=128):
    tokens = nltk.word_tokenize(text.lower())[:max_length-2]
    token_ids = [vocab.get("<BOS>", 3)] + [vocab.get(word, 0) for word in tokens] + [vocab.get("<EOS>", 2)]
    if len(token_ids) < max_length:
        token_ids += [vocab["<PAD>"]] * (max_length - len(token_ids))
    return np.array(token_ids, dtype=np.int32)

# Update load_or_tokenize_texts
def load_or_tokenize_texts(texts, tokenizer, output_dir, prefix, batch_size=500, max_length=128):
    inputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_inputs = np.array([custom_tokenize(t, max_length) for t in batch])
        batch_file = os.path.join(output_dir, f"{prefix}_batch_{i//batch_size}.npy")
        np.save(batch_file, batch_inputs)
        inputs.append(batch_inputs)
    return mx.array(np.concatenate(inputs, axis=0), dtype=mx.int32)