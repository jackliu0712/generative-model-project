import json
import torch
from transformers import GPT2LMHeadModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = [item['text'] for item in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Initialized TextDataset with {len(self.texts)} texts.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

def load_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")
    return data

def calculate_perplexity(model, dataloader):
    print("Calculating perplexity...")
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Calculating Perplexity"):
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    perplexity = math.exp(total_loss / total_tokens)
    print(f"Perplexity calculated: {perplexity}")
    return perplexity

if __name__ == '__main__':
    # Load and sample data
    train_texts = load_data('C:/Users/lkh/Desktop/python/project1/gpt2/train.json')[:500]  # Sample items
    test_texts = load_data('C:/Users/lkh/Desktop/python/project1/gpt2/test.json')[:10]    # Sample items

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

    model = GPT2LMHeadModel.from_pretrained('C:/Users/lkh/Desktop/python/project1/gpt2', local_files_only=True)
    print("Model loaded from local files.")

    train_dataset = TextDataset(train_texts, tokenizer, max_length=128)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    print("Training DataLoader initialized.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    losses = []
    accumulation_steps = 4  # Number of steps to accumulate gradients

    for epoch in range(1):
        print(f"Starting epoch {epoch+1}")
        epoch_loss = 0
        for i, (input_ids, attention_mask) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss = loss / accumulation_steps  # Normalize loss
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")
        losses.append(avg_loss)

    plt.plot(range(1, 2), losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    test_dataset = TextDataset(test_texts, tokenizer, max_length=128)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    perplexity = calculate_perplexity(model, test_dataloader)
    print("Perplexity:", perplexity)

    model.save_pretrained('C:/Users/lkh/Desktop/python/project1/gpt2_trained')
    tokenizer.save_pretrained('C:/Users/lkh/Desktop/python/project1/gpt2_trained')
    print("Model and tokenizer saved.")

