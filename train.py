import json
import torch
from transformers import GPT2LMHeadModel, BertTokenizer, GPT2Config
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
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    perplexity = math.exp(total_loss / total_tokens)
    print(f"Perplexity calculated: {perplexity}")
    return perplexity

if __name__ == '__main__':
    # Load and sample data
    train_texts = load_data('C:/Users/lkh/Desktop/python/project1/gpt2/train.json')
    test_texts = load_data('C:/Users/lkh/Desktop/python/project1/gpt2/test.json')

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

    # Initialize model from scratch
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()),
        n_positions=512,  
        n_ctx=512,        
        n_embd=1024,      
        n_layer=24,       
        n_head=16         
    )
    model = GPT2LMHeadModel(config)

    model.resize_token_embeddings(len(tokenizer))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print("Model initialized from scratch.")

    train_dataset = TextDataset(train_texts, tokenizer, max_length=256)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
    print("Training DataLoader initialized.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    losses = []
    accumulation_steps = 4

    for epoch in range(1):
        print(f"Starting epoch {epoch+1}")
        epoch_loss = 0
        for i, (input_ids, attention_mask) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")
        losses.append(avg_loss)

    # plt.plot(range(1, 2), losses, marker='o')
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    # test_dataset = TextDataset(test_texts, tokenizer, max_length=128)
    # test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    # perplexity = calculate_perplexity(model, test_dataloader)
    # print("Perplexity:", perplexity)

    model.save_pretrained('C:/Users/lkh/Desktop/python/project1/gpt2_trained')
    tokenizer.save_pretrained('C:/Users/lkh/Desktop/python/project1/gpt2_trained')
    print("Model and tokenizer saved.")



