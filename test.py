import json
from transformers import GPT2LMHeadModel, BertTokenizer
from tqdm import tqdm

def generate_text(model, tokenizer, prompt_text, max_length=256, top_k=50):
    model.eval()
    inputs = tokenizer.encode(prompt_text, return_tensors='pt')
    prompt_length = len(inputs[0])
    outputs = model.generate(
        inputs,
        max_length=prompt_length + max_length,  # Ensure total length is prompt + 256
        num_return_sequences=1,
        do_sample=True,
        top_k=top_k
    )
    generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return generated_text

def load_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")
    return data

model = GPT2LMHeadModel.from_pretrained(r'C:\Users\lkh\Desktop\python\project1\gpt2_trained', local_files_only=True)
print("Model loaded from local files.")

tokenizer = BertTokenizer(vocab_file=r'C:\Users\lkh\Desktop\python\project1\gpt2\vocab.txt')
tokenizer.add_special_tokens(
    {'pad_token': '[PAD]',
     'cls_token': '[BOS]',
     'sep_token': '[EOS]',
     'mask_token': '[MASK]',
     "unk_token": "[UNK]",
     "bos_token": '[BOS]',
     "eos_token": '[EOS]'})
print("Tokenizer initialized and special tokens added.")

# Generate text for each test data entry
test_texts = load_data('C:/Users/lkh/Desktop/python/project1/gpt2/test.json')[:10]    # Sample items
generated_texts = []

# Use tqdm to add a progress bar
for item in tqdm(test_texts, desc="Generating texts"):
    prompt_text = item['text']
    generated_text = generate_text(model, tokenizer, prompt_text, top_k=50)  # Adjust top_k as needed
    generated_texts.append({'prompt': prompt_text, 'generated': generated_text})

# Save generated texts
with open('generated_texts.json', 'w', encoding='utf-8') as f:
    json.dump(generated_texts, f, ensure_ascii=False, indent=4)
print("Generated texts saved.")

