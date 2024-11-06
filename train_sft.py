import json
import torch
from transformers import GPT2LMHeadModel, BertTokenizer, GPT2Config
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import Dataset as HFDataset
from trl import SFTTrainer, SFTConfig
import os

# 统一参数配置
class Config:
    def __init__(self):
        # 版本号
        self.version = "1.0"
        
        # 路径配置
        self.base_path = "C:/Users/lkh/Desktop/python/project1"
        self.train_data_path = f"{self.base_path}/gpt2/train.json"
        self.test_data_path = f"{self.base_path}/gpt2/test.json"
        self.vocab_path = f"{self.base_path}/gpt2/vocab.txt"
        self.output_dir = f"{self.base_path}/gpt2_trained_sft{self.version}"
        self.loss_plot_path = f"{self.base_path}/loss_plot{self.version}.png"
        
        # 模型配置
        self.model_config = {
            'n_positions': 512,
            'n_ctx': 512,
            'n_embd': 768,
            'n_layer': 6,
            'n_head': 6
        }
        
        # 训练配置
        self.training_config = {
            'num_train_epochs': 1,
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'learning_rate': 1e-5,
            'logging_steps': 100,
            'save_strategy': "epoch",
            'max_seq_length': 512
        }
        
        # 特殊token配置
        self.special_tokens = {
            'pad_token': '[PAD]',
            'cls_token': '[BOS]',
            'sep_token': '[EOS]',
            'mask_token': '[MASK]',
            'unk_token': '[UNK]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]'
        }

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def load_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")
    return data

def prepare_dataset(texts):
    return HFDataset.from_dict({
        'text': [item['text'] for item in texts]
    })

if __name__ == '__main__':
    # 初始化配置
    config = Config()
    
    # 加载数据
    train_texts = load_data(config.train_data_path)
    test_texts = load_data(config.test_data_path)

    # 转换为HuggingFace Dataset格式
    train_dataset = prepare_dataset(train_texts)
    
    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)

    # 添加特殊tokens
    tokenizer.add_special_tokens(config.special_tokens)

    # 初始化模型配置
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        **config.model_config
    )

    # 初始化模型
    model = GPT2LMHeadModel(model_config)
    model.resize_token_embeddings(len(tokenizer))
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_parameters}")
    
    # 配置SFT训练
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        **config.training_config
    )

    # 初始化训练器
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text"
    )

    # 训练模型
    print("Starting training...")
    trainer.train()

    # 保存模型和tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    print("Model and tokenizer saved.")

    # 绘制训练损失图
    if trainer.state.log_history:
        losses = [entry['loss'] for entry in trainer.state.log_history if 'loss' in entry]
        plt.figure(figsize=(10, 6))
        plt.plot(losses, marker='.')
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(config.loss_plot_path)
        plt.close()
        print("Training loss plot saved.")
