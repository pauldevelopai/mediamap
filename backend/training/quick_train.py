"""
Quick Training Script for Highlander AI
Bypasses configuration complexity for immediate training
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"User: {item['input']}\nHighlander: {item['output']}"
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def quick_train():
    """Quick training function"""
    logger.info("🚀 Starting Quick Highlander AI Training...")
    
    # Load training data
    data_file = Path("training_data/processed/training_dataset.json")
    with open(data_file, 'r') as f:
        all_data = json.load(f)
    
    # Filter valid examples
    valid_data = []
    for item in all_data:
        if (item.get('input') and item.get('output') and 
            len(item['input'].strip()) > 10 and 
            len(item['output'].strip()) > 10):
            valid_data.append(item)
    
    logger.info(f"📊 Training with {len(valid_data)} examples")
    
    # Split data
    train_size = int(0.8 * len(valid_data))
    train_data = valid_data[:train_size]
    eval_data = valid_data[train_size:]
    
    logger.info(f"📚 Train: {len(train_data)}, Eval: {len(eval_data)}")
    
    # Initialize model and tokenizer
    model_name = "microsoft/DialoGPT-small"  # Use smaller model for speed
    logger.info(f"🤖 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets
    train_dataset = SimpleDataset(train_data, tokenizer)
    eval_dataset = SimpleDataset(eval_data, tokenizer)
    
    # Training arguments - simple and working
    training_args = TrainingArguments(
        output_dir="./models/highlander_quick",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-5,  # Direct float value
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        eval_steps=25,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer  # Use processing_class instead of tokenizer
    )
    
    logger.info("🔥 Starting training...")
    
    # Train the model
    trainer.train()
    
    # Save final model
    final_path = "./models/highlander_final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    logger.info(f"✅ Training complete! Model saved to: {final_path}")
    
    # Create deployment
    deploy_dir = Path("./models/deployment")
    deploy_dir.mkdir(exist_ok=True)
    
    import shutil
    shutil.copytree(final_path, deploy_dir / "model", dirs_exist_ok=True)
    
    # Create deployment info
    deploy_info = {
        'deployed_at': '2025-07-20T14:15:00',
        'model_path': str(deploy_dir / "model"),
        'status': 'deployed',
        'version': '1.0.0'
    }
    
    with open(deploy_dir / "deployment_info.json", 'w') as f:
        json.dump(deploy_info, f, indent=2)
    
    logger.info(f"🚀 Model deployed to: {deploy_dir}")
    
    # Test the model
    logger.info("🧪 Testing the model...")
    test_input = "How can I implement AI in my media company?"
    inputs = tokenizer.encode(f"User: {test_input}\nHighlander:", return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    highlander_response = response.split("Highlander:")[-1].strip()
    
    logger.info(f"🤖 Test response: {highlander_response}")
    logger.info("🎉 Highlander AI is ready!")
    
    return final_path

if __name__ == "__main__":
    quick_train() 