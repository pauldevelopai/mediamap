"""
Custom AI Model Training System

This module handles training of a custom AI model specialized for AI development advice.
Uses transformer architecture with custom fine-tuning on collected data.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from sklearn.model_selection import train_test_split
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighlanderDataset(Dataset):
    """Custom dataset for training Highlander AI model"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the conversation for training
        # Input: user question, Output: AI response
        prompt = f"User: {item['input']}\nHighlander: {item['output']}"
        
        # Tokenize
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

class HighlanderModelTrainer:
    """Comprehensive training system for custom AI model"""
    
    def __init__(self, 
                 config_path: str = "./training_config.yaml",
                 data_dir: str = "./training_data",
                 output_dir: str = "./models"):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load training configuration
        self.config = self.load_config(config_path)
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
        logger.info("HighlanderModelTrainer initialized")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        default_config = {
            'model': {
                'base_model': 'microsoft/DialoGPT-medium',  # Good for conversational AI
                'max_length': 512,
                'vocab_size': 50257
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 5e-5,
                'num_epochs': 3,
                'warmup_steps': 500,
                'logging_steps': 100,
                'save_steps': 1000,
                'eval_steps': 500,
                'gradient_accumulation_steps': 2,
                'max_grad_norm': 1.0,
                'weight_decay': 0.01
            },
            'data': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'min_examples': 100
            },
            'optimization': {
                'use_fp16': True,
                'use_gradient_checkpointing': True,
                'dataloader_num_workers': 4
            },
            'monitoring': {
                'use_wandb': False,
                'project_name': 'highlander-ai',
                'run_name': f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with defaults
                self._deep_update(default_config, user_config)
        else:
            # Save default config
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, indent=2)
            logger.info(f"Created default config at {config_path}")
        
        return default_config
    
    def _deep_update(self, base_dict, update_dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def prepare_data(self) -> Dict[str, int]:
        """Load and prepare training data"""
        logger.info("Preparing training data...")
        
        # Load processed training dataset
        dataset_file = self.data_dir / "processed" / "training_dataset.json"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Training dataset not found at {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter and validate data
        valid_data = []
        for item in data:
            if (item.get('input') and item.get('output') and 
                len(item['input'].strip()) > 10 and 
                len(item['output'].strip()) > 10):
                valid_data.append(item)
        
        if len(valid_data) < self.config['data']['min_examples']:
            raise ValueError(f"Insufficient training data: {len(valid_data)} < {self.config['data']['min_examples']}")
        
        # Split data
        train_size = self.config['data']['train_split']
        val_size = self.config['data']['val_split']
        
        train_data, temp_data = train_test_split(valid_data, train_size=train_size, random_state=42)
        val_data, test_data = train_test_split(temp_data, train_size=val_size/(val_size + self.config['data']['test_split']), random_state=42)
        
        # Save split data
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            output_file = self.data_dir / "processed" / f"{split_name}_split.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False, default=str)
        
        stats = {
            'total_examples': len(valid_data),
            'train_examples': len(train_data),
            'val_examples': len(val_data),
            'test_examples': len(test_data)
        }
        
        logger.info(f"Data preparation complete: {stats}")
        return stats
    
    def initialize_model(self):
        """Initialize tokenizer and model"""
        logger.info("Initializing model and tokenizer...")
        
        model_name = self.config['model']['base_model']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config['optimization']['use_fp16'] else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Model initialized: {model_name}")
    
    def create_datasets(self) -> Dict[str, Dataset]:
        """Create training datasets"""
        logger.info("Creating datasets...")
        
        datasets = {}
        max_length = self.config['model']['max_length']
        
        for split in ['train', 'validation', 'test']:
            split_file = self.data_dir / "processed" / f"{split}_split.json"
            with open(split_file, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
            
            datasets[split] = HighlanderDataset(
                data=split_data,
                tokenizer=self.tokenizer,
                max_length=max_length
            )
        
        return datasets
    
    def train_model(self) -> str:
        """Train the custom AI model"""
        logger.info("Starting model training...")
        
        # Prepare data
        data_stats = self.prepare_data()
        
        # Initialize model
        self.initialize_model()
        
        # Create datasets
        datasets = self.create_datasets()
        
        # Initialize wandb if configured and available
        if self.config['monitoring']['use_wandb'] and WANDB_AVAILABLE:
            wandb.init(
                project=self.config['monitoring']['project_name'],
                name=self.config['monitoring']['run_name'],
                config=self.config
            )
        elif self.config['monitoring']['use_wandb'] and not WANDB_AVAILABLE:
            logger.warning("Wandb is configured but not available. Training will continue without wandb logging.")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            eval_strategy="steps",  # Updated parameter name
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config['optimization']['use_fp16'],
            gradient_checkpointing=self.config['optimization']['use_gradient_checkpointing'],
            dataloader_num_workers=self.config['optimization']['dataloader_num_workers'],
            remove_unused_columns=False,
            report_to="wandb" if (self.config['monitoring']['use_wandb'] and WANDB_AVAILABLE) else None
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "highlander_final"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Evaluate on test set
        test_results = trainer.evaluate(eval_dataset=datasets['test'])
        
        # Save training metadata
        metadata = {
            'training_completed_at': datetime.now().isoformat(),
            'data_stats': data_stats,
            'config': self.config,
            'test_results': test_results,
            'model_path': str(final_model_path)
        }
        
        metadata_file = self.output_dir / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Training complete! Model saved to: {final_model_path}")
        logger.info(f"Test results: {test_results}")
        
        return str(final_model_path)
    
    def evaluate_model(self, model_path: str) -> Dict[str, float]:
        """Evaluate trained model performance"""
        logger.info("Evaluating model performance...")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load test data
        test_file = self.data_dir / "processed" / "test_split.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Evaluate on sample of test data
        sample_size = min(50, len(test_data))
        test_sample = np.random.choice(test_data, sample_size, replace=False)
        
        total_score = 0
        for item in test_sample:
            # Generate response
            input_text = f"User: {item['input']}\nHighlander:"
            inputs = tokenizer.encode(input_text, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Simple evaluation metric (can be enhanced)
            expected_response = item['output']
            generated_response = generated_text.split("Highlander:")[-1].strip()
            
            # Basic similarity score (this could be enhanced with BLEU, ROUGE, etc.)
            score = self.calculate_similarity(expected_response, generated_response)
            total_score += score
        
        avg_score = total_score / sample_size
        
        evaluation_results = {
            'average_similarity_score': avg_score,
            'samples_evaluated': sample_size,
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Save evaluation results
        eval_file = self.output_dir / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation complete: Average similarity score = {avg_score:.3f}")
        return evaluation_results
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (basic implementation)"""
        # This is a simple word overlap metric
        # In production, you'd use more sophisticated metrics like BLEU, ROUGE, or semantic similarity
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def deploy_model(self, model_path: str) -> str:
        """Deploy trained model for production use"""
        logger.info("Deploying model...")
        
        # Create deployment directory
        deploy_dir = self.output_dir / "deployment"
        deploy_dir.mkdir(exist_ok=True)
        
        # Copy model files
        import shutil
        shutil.copytree(model_path, deploy_dir / "model", dirs_exist_ok=True)
        
        # Create deployment metadata
        deploy_metadata = {
            'deployed_at': datetime.now().isoformat(),
            'model_path': str(deploy_dir / "model"),
            'status': 'deployed',
            'version': '1.0.0'
        }
        
        with open(deploy_dir / "deployment_info.json", 'w') as f:
            json.dump(deploy_metadata, f, indent=2, default=str)
        
        logger.info(f"Model deployed to: {deploy_dir}")
        return str(deploy_dir)

if __name__ == "__main__":
    # Example usage
    trainer = HighlanderModelTrainer()
    
    # Train the model
    model_path = trainer.train_model()
    
    # Evaluate the model
    eval_results = trainer.evaluate_model(model_path)
    
    # Deploy the model
    deploy_path = trainer.deploy_model(model_path)
    
    print(f"Training pipeline complete!")
    print(f"Model: {model_path}")
    print(f"Evaluation: {eval_results}")
    print(f"Deployment: {deploy_path}") 