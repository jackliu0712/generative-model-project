import json
import torch
from transformers import GPT2LMHeadModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm

# 统一配置参数
version = '1.0'
output_version = '1.7'
MAX_LENGTH = 512

# 初始化tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained(f'C:/Users/lkh/Desktop/python/project1/gpt2_trained{version}')
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
    raise

CONFIG = {
    'test_data_path': 'C:/Users/lkh/Desktop/python/project1/gpt2/test.json',
    'model_path': f'C:/Users/lkh/Desktop/python/project1/gpt2_trained{version}',
    'output_path': f'C:/Users/lkh/Desktop/python/project1/gpt2/generated_texts{output_version}',
    
    'max_length': MAX_LENGTH,
    'batch_size': 16,
    'target_length': 512,  # 目标生成长度
    'num_sequences': 10,   # 生成的文本数量
    
    'num_beams': 5,
    'top_k': 50,
    'top_p': 0.95,
    'temperature': 0.8,
    'no_repeat_ngram_size': 3,
    'repetition_penalty': 1.3,
    'bad_words_ids': [[tokenizer.convert_tokens_to_ids('…')]],
    'min_length': 256,
    'diverse_penalty':0.5,
    'num_beam_groups':2
}

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = [item['text'] for item in texts if 'text' in item]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        try:
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()
        except Exception as e:
            print(f"Error processing text at index {idx}: {str(e)}")
            return torch.zeros(self.max_length, dtype=torch.long), torch.zeros(self.max_length, dtype=torch.long)

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return []

def calculate_perplexity(model, dataloader):
    print("Calculating perplexity...")
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Perplexity"):
            try:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
                
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                
                non_pad_mask = (input_ids != tokenizer.pad_token_id)
                num_tokens = non_pad_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
            except Exception as e:
                print(f"Error in perplexity calculation: {str(e)}")
                continue
    
    perplexity = math.exp(total_loss / max(total_tokens, 1))
    return perplexity

def generate_text(model, tokenizer, config):
    model.eval()
    try:
        # 使用空字符串作为起始
        input_ids = tokenizer.encode("", return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=config['target_length'],
                num_return_sequences=1,
                num_beams=config['num_beams'],
                do_sample=True,
                top_k=config['top_k'],
                top_p=config['top_p'],
                temperature=config['temperature'],
                no_repeat_ngram_size=config['no_repeat_ngram_size'],
                repetition_penalty=config['repetition_penalty'],
                bad_words_ids=config['bad_words_ids'],
                min_length=config['min_length'],
                # diversity_penalty=config['diversity_penalty'],
                # num_beam_groups=config['num_beam_groups'],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
        
    except Exception as e:
        print(f"Error in text generation: {str(e)}")
        return ""

if __name__ == '__main__':
    try:
        # 加载模型
        print("Loading model...")
        model = GPT2LMHeadModel.from_pretrained(CONFIG['model_path'])
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print("Model loaded.")

        # # 计算perplexity
        # test_texts = load_data(CONFIG['test_data_path'])
        # if test_texts:
        #     test_dataset = TextDataset(test_texts, tokenizer, CONFIG['max_length'])
        #     test_dataloader = DataLoader(
        #         test_dataset, 
        #         batch_size=CONFIG['batch_size'], 
        #         shuffle=False
        #     )
        #     test_perplexity = calculate_perplexity(model, test_dataloader)
        #     print(f"Test Perplexity: {test_perplexity}")

        # 生成新文本
        generated_texts = []
        for i in tqdm(range(CONFIG['num_sequences']), desc="Generating texts"):
            generated_text = generate_text(model, tokenizer, CONFIG)
            generated_texts.append({
                'id': i,
                'text': generated_text
            })

        # 保存生成的文本
        with open(CONFIG['output_path'], 'w', encoding='utf-8') as f:
            json.dump(generated_texts, f, ensure_ascii=False, indent=4)
        print(f"Generated {len(generated_texts)} texts and saved to {CONFIG['output_path']}")

    except Exception as e:
        print(f"An error occurred in main execution: {str(e)}")



# import json
# import torch
# from transformers import GPT2LMHeadModel, BertTokenizer
# from torch.utils.data import Dataset, DataLoader
# import math
# from tqdm import tqdm

# # 统一配置参数
# version = '1.0'
# output_version = '1.1'
# tokenizer = BertTokenizer.from_pretrained(f'C:/Users/lkh/Desktop/python/project1/gpt2_trained{version}')

# CONFIG = {
#     'test_data_path': 'C:/Users/lkh/Desktop/python/project1/gpt2/test.json',
#     'model_path': f'C:/Users/lkh/Desktop/python/project1/gpt2_trained{version}',
#     'output_path': f'C:/Users/lkh/Desktop/python/project1/gpt2/test_results{output_version}',
    
#     'max_length': 512,  # 最大新生成token数
#     'batch_size': 8,
#     'min_length': 50,  # 确保最小生成长度
    
#     # 生成参数
#     'num_return_sequences': 3,
#     'num_beams': 8,
#     'top_k': 50,
#     'top_p': 0.5,
#     'temperature': 0.8,
#     'no_repeat_ngram_size': 4,
#     'repetition_penalty': 1.2,
#     'bad_words_ids': [[tokenizer.convert_tokens_to_ids('...')]],
# }


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

# def calculate_perplexity(model, dataloader):
#     print("Calculating perplexity...")
#     model.eval()
#     total_loss = 0
#     total_tokens = 0
#     with torch.no_grad():
#         for input_ids, attention_mask in tqdm(dataloader, desc="Calculating Perplexity"):
#             input_ids = input_ids.to(model.device)
#             attention_mask = attention_mask.to(model.device)
#             outputs = model(input_ids=input_ids, labels=input_ids)
#             loss = outputs.loss
#             total_loss += loss.item() * input_ids.size(1)
#             total_tokens += input_ids.size(1)
#     perplexity = math.exp(total_loss / total_tokens)
#     print(f"Perplexity calculated: {perplexity}")
#     return perplexity

# def generate_text(model, tokenizer, prompt, config):
#     model.eval()
#     # 对输入不做截断，保留完整prompt
#     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

#     if input_ids.size(1) > MAX_LENGTH // 2:
#         input_ids = input_ids[:, -(MAX_LENGTH // 2):]
    
#     with torch.no_grad():
#         output = model.generate(
#             input_ids,
#             max_new_tokens=config['max_length'],  # 使用max_new_tokens而不是max_length
#             min_length=config['min_length'],  # 确保最小生成长度
#             num_return_sequences=config['num_return_sequences'],
#             num_beams=config['num_beams'],
#             do_sample=True,
#             # top_k=config['top_k'],
#             top_p=config['top_p'],
#             temperature=config['temperature'],
#             no_repeat_ngram_size=config['no_repeat_ngram_size'],
#             repetition_penalty=config['repetition_penalty'],
#             bad_words_ids=config['bad_words_ids'],
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     generated_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
#     # 选择最长的生成文本
#     generated_text = max(generated_texts, key=len)
    
#     return generated_text

# if __name__ == '__main__':
#     # 构建完整路径
#     model_path = f"{CONFIG['model_path']}"
#     output_path = f"{CONFIG['output_path']}.json"
    
#     # Load test data
#     test_texts = load_data(CONFIG['test_data_path'])

#     # Load trained model and tokenizer
#     tokenizer = BertTokenizer.from_pretrained(model_path)
#     model = GPT2LMHeadModel.from_pretrained(model_path)
#     model.to('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Model and tokenizer loaded.")

#     # Initialize dataset and dataloader
#     test_dataset = TextDataset(test_texts, tokenizer, CONFIG['max_length'])
#     test_dataloader = DataLoader(
#         test_dataset, 
#         batch_size=CONFIG['batch_size'], 
#         shuffle=False, 
#         pin_memory=True
#     )
#     print("Test DataLoader initialized.")

#     # Calculate perplexity
#     test_perplexity = calculate_perplexity(model, test_dataloader)
#     print(f"Test Perplexity: {test_perplexity}")

#     # Generate text for each test sample
#     results = []
#     for item in tqdm(test_texts, desc="Generating Text"):
#         prompt = item['text']
#         try:
#             generated_text = generate_text(model, tokenizer, prompt, CONFIG)
#         except Exception as e:
#             print(f"Error generating text for prompt: {prompt[:50]}...")
#             print(f"Error message: {str(e)}")
#             generated_text = prompt
#         results.append({
#             'prompt': prompt,
#             'generated_text': generated_text,
#         })

#     # Save results
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(results, f, ensure_ascii=False, indent=4)
#     print("Test results saved.")


