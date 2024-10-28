import json
import torch
from transformers import GPT2LMHeadModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm

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

def generate_text(model, tokenizer, prompt, max_length=256):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=max_length).to(model.device)
    
    if input_ids.size(1) >= max_length:
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    remaining_tokens = max_length - input_ids.size(1)
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_new_tokens=remaining_tokens, 
            num_return_sequences=1, 
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == '__main__':
    # Load test data
    test_texts = load_data('C:/Users/lkh/Desktop/python/project1/gpt2/test.json')

    # Load trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('C:/Users/lkh/Desktop/python/project1/gpt2_trained')
    model = GPT2LMHeadModel.from_pretrained('C:/Users/lkh/Desktop/python/project1/gpt2_trained')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print("Model and tokenizer loaded.")

    test_dataset = TextDataset(test_texts, tokenizer, max_length=256)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)
    print("Test DataLoader initialized.")

    # Calculate perplexity on test data
    test_perplexity = calculate_perplexity(model, test_dataloader)
    print(f"Test Perplexity: {test_perplexity}")

    # Generate text for each test sample and save results
    results = []
    for item in tqdm(test_texts, desc="Generating Text"):
        prompt = item['text']
        try:
            generated_text = generate_text(model, tokenizer, prompt, max_length=256)
        except Exception as e:
            print(f"Error generating text for prompt: {prompt[:50]}...")
            print(f"Error message: {str(e)}")
            generated_text = prompt  # 如果生成失败，使用原文
        results.append({
            'prompt': prompt,
            'generated_text': generated_text,
        })


    # Save results to file
    with open('C:/Users/lkh/Desktop/python/project1/gpt2/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Test results saved.")



# import json
# import torch
# from transformers import GPT2LMHeadModel, BertTokenizer, GPT2Config
# from torch.utils.data import Dataset, DataLoader
# import math
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# class TextDataset(Dataset):
#     def __init__(self, texts, tokenizer, max_length):
#         self.texts = [item['text'] for item in texts]
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         print(f"Initialized TextDataset with {len(self.texts)} texts.")

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         inputs = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

# def load_data(file_path):
#     print(f"Loading data from {file_path}")
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     print(f"Loaded {len(data)} items.")
#     return data

# def calculate_perplexity_and_generate_text(model, dataloader, tokenizer, output_file):
#     print("Calculating perplexity and generating text...")
#     model.eval()
#     total_loss = 0
#     total_tokens = 0
#     generated_texts = []

#     with torch.no_grad():
#         for input_ids, attention_mask in tqdm(dataloader, desc="Calculating Perplexity"):
#             input_ids = input_ids.to(model.device)
#             attention_mask = attention_mask.to(model.device)
#             outputs = model(input_ids=input_ids, labels=input_ids)
#             loss = outputs.loss
#             total_loss += loss.item() * input_ids.size(1)
#             total_tokens += input_ids.size(1)

#             # Generate text with length 256
#             generated_ids = model.generate(input_ids=input_ids, 
#                                            max_new_tokens=256, 
#                                            attention_mask=attention_mask, 
#                                            pad_token_id=tokenizer.eos_token_id, 
#                                            num_return_sequences=1)
#             generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#             generated_texts.append(generated_text)

#     perplexity = math.exp(total_loss / total_tokens)
#     print(f"Perplexity calculated: {perplexity}")

#     # Save generated texts to file
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(generated_texts, f, ensure_ascii=False, indent=4)
#     print(f"Generated texts saved to {output_file}")

#     return perplexity

# test_texts = load_data('C:/Users/lkh/Desktop/python/project1/gpt2/test.json')
# tokenizer = BertTokenizer.from_pretrained('C:/Users/lkh/Desktop/python/project1/gpt2_trained', padding_side='left')
# model = GPT2LMHeadModel.from_pretrained('C:/Users/lkh/Desktop/python/project1/gpt2_trained')

# test_dataset = TextDataset(test_texts, tokenizer, max_length=256)
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# output_file = 'generated_texts.json'
# perplexity = calculate_perplexity_and_generate_text(model, test_dataloader, tokenizer, output_file)



