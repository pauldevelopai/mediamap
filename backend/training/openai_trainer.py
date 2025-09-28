"""
OpenAI Fine-tuning Training System

This module implements real AI model training using OpenAI's fine-tuning API
for MediaMap and HealthPIN specialized models.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import openai
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAITrainer:
    """OpenAI fine-tuning trainer for specialized models"""
    
    def __init__(self, model_name: str, output_dir: str = "./training/models"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Model-specific configurations
        self.model_configs = {
            'mediamap': {
                'base_model': 'gpt-3.5-turbo',
                'system_prompt': "You are MediaMap AI, a specialized assistant for media industry analysis, business insights, and strategic planning. You provide expert advice on media trends, ROI optimization, and industry best practices.",
                'training_suffix': 'mediamap-v1'
            },
            'healthpin': {
                'base_model': 'gpt-3.5-turbo',
                'system_prompt': "You are HealthPIN AI, a specialized medical assistant for healthcare analysis and clinical insights. You provide evidence-based medical information, patient care guidance, and healthcare industry analysis.",
                'training_suffix': 'healthpin-v1'
            }
        }
        
        self.config = self.model_configs.get(model_name, self.model_configs['mediamap'])
        logger.info(f"OpenAITrainer initialized for {model_name}")
    
    def prepare_training_data(self, data_dir: str) -> str:
        """Prepare training data in OpenAI fine-tuning format"""
        logger.info(f"Preparing training data for {self.model_name}")
        
        data_dir = Path(data_dir)
        training_data = []
        
        # Load conversations
        conversations_file = data_dir / "conversations" / "all_conversations.json"
        if conversations_file.exists():
            with open(conversations_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            for conv in conversations:
                if self._is_relevant_conversation(conv):
                    training_data.append({
                        "messages": [
                            {"role": "system", "content": self.config['system_prompt']},
                            {"role": "user", "content": conv.get('input', '')},
                            {"role": "assistant", "content": conv.get('output', '')}
                        ]
                    })
        
        # Load PDF-derived Q&A pairs
        pdf_dir = data_dir / "pdfs"
        if pdf_dir.exists():
            for txt_file in pdf_dir.glob("*.txt"):
                qa_pairs = self._extract_qa_from_pdf_text(txt_file)
                for qa in qa_pairs:
                    training_data.append({
                        "messages": [
                            {"role": "system", "content": self.config['system_prompt']},
                            {"role": "user", "content": qa['question']},
                            {"role": "assistant", "content": qa['answer']}
                        ]
                    })
        
        # Load research papers
        research_dir = data_dir / "research"
        if research_dir.exists():
            for json_file in research_dir.glob("*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    research_data = json.load(f)
                
                if isinstance(research_data, list):
                    for item in research_data:
                        if self._is_relevant_research(item):
                            training_data.append({
                                "messages": [
                                    {"role": "system", "content": self.config['system_prompt']},
                                    {"role": "user", "content": item.get('question', '')},
                                    {"role": "assistant", "content": item.get('answer', '')}
                                ]
                            })
        
        # Save training data in OpenAI format
        training_file = self.output_dir / f"{self.model_name}_training_data.jsonl"
        with open(training_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Prepared {len(training_data)} training examples for {self.model_name}")
        return str(training_file)
    
    def _is_relevant_conversation(self, conv: Dict) -> bool:
        """Check if conversation is relevant for the specific model"""
        content = (conv.get('input', '') + ' ' + conv.get('output', '')).lower()
        
        if self.model_name == 'mediamap':
            keywords = ['media', 'marketing', 'advertising', 'roi', 'campaign', 'brand', 'audience', 'content', 'strategy']
            return any(keyword in content for keyword in keywords)
        elif self.model_name == 'healthpin':
            keywords = ['health', 'medical', 'patient', 'clinical', 'diagnosis', 'treatment', 'healthcare', 'medicine']
            return any(keyword in content for keyword in keywords)
        
        return True  # Default to including all conversations
    
    def _is_relevant_research(self, item: Dict) -> bool:
        """Check if research item is relevant for the specific model"""
        content = (str(item.get('question', '')) + ' ' + str(item.get('answer', ''))).lower()
        
        if self.model_name == 'mediamap':
            keywords = ['media', 'marketing', 'advertising', 'digital', 'content', 'brand', 'campaign']
            return any(keyword in content for keyword in keywords)
        elif self.model_name == 'healthpin':
            keywords = ['health', 'medical', 'clinical', 'patient', 'treatment', 'diagnosis', 'healthcare']
            return any(keyword in content for keyword in keywords)
        
        return True
    
    def _extract_qa_from_pdf_text(self, txt_file: Path) -> List[Dict]:
        """Extract Q&A pairs from PDF text content"""
        qa_pairs = []
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple extraction - split content into chunks and create Q&A pairs
            chunks = content.split('\n\n')
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Only use substantial chunks
                    question = f"What information is available about {txt_file.stem}?"
                    if i < len(chunks) - 1:
                        question = f"What does the document say about the topic in section {i+1}?"
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': chunk.strip()
                    })
        
        except Exception as e:
            logger.error(f"Error extracting Q&A from {txt_file}: {e}")
        
        return qa_pairs[:10]  # Limit to 10 Q&A pairs per PDF
    
    def start_fine_tuning(self, training_file: str) -> Dict[str, Any]:
        """Start OpenAI fine-tuning job"""
        logger.info(f"Starting fine-tuning for {self.model_name}")
        
        try:
            # Upload training file
            with open(training_file, 'rb') as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            logger.info(f"Uploaded training file: {file_response.id}")
            
            # Create fine-tuning job
            fine_tune_response = self.client.fine_tuning.jobs.create(
                training_file=file_response.id,
                model=self.config['base_model'],
                suffix=self.config['training_suffix']
            )
            
            job_id = fine_tune_response.id
            logger.info(f"Started fine-tuning job: {job_id}")
            
            # Save job info
            job_info = {
                'job_id': job_id,
                'model_name': self.model_name,
                'base_model': self.config['base_model'],
                'training_file': training_file,
                'started_at': datetime.now().isoformat(),
                'status': 'running'
            }
            
            job_file = self.output_dir / f"{self.model_name}_job_info.json"
            with open(job_file, 'w') as f:
                json.dump(job_info, f, indent=2)
            
            return {
                'success': True,
                'job_id': job_id,
                'status': 'running',
                'message': f'Fine-tuning started for {self.model_name}'
            }
            
        except Exception as e:
            logger.error(f"Error starting fine-tuning: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_training_status(self) -> Dict[str, Any]:
        """Check the status of the fine-tuning job"""
        job_file = self.output_dir / f"{self.model_name}_job_info.json"
        
        if not job_file.exists():
            return {
                'success': False,
                'error': 'No training job found'
            }
        
        try:
            with open(job_file, 'r') as f:
                job_info = json.load(f)
            
            job_id = job_info['job_id']
            
            # Get job status from OpenAI
            job_status = self.client.fine_tuning.jobs.retrieve(job_id)
            
            status_info = {
                'success': True,
                'job_id': job_id,
                'status': job_status.status,
                'model_name': self.model_name,
                'created_at': job_info.get('started_at'),
                'finished_at': job_status.finished_at,
                'fine_tuned_model': job_status.fine_tuned_model
            }
            
            # Update job info if completed
            if job_status.status == 'succeeded' and job_status.fine_tuned_model:
                job_info['status'] = 'completed'
                job_info['fine_tuned_model'] = job_status.fine_tuned_model
                job_info['completed_at'] = datetime.now().isoformat()
                
                with open(job_file, 'w') as f:
                    json.dump(job_info, f, indent=2)
                
                status_info['completed'] = True
                status_info['model_id'] = job_status.fine_tuned_model
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error checking training status: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        job_file = self.output_dir / f"{self.model_name}_job_info.json"
        
        if not job_file.exists():
            return {
                'model_loaded': False,
                'training_examples': 0,
                'last_training': 'Never',
                'accuracy': 'N/A',
                'openai_available': True
            }
        
        try:
            with open(job_file, 'r') as f:
                job_info = json.load(f)
            
            # Count training examples
            training_file = job_info.get('training_file', '')
            training_examples = 0
            if training_file and os.path.exists(training_file):
                with open(training_file, 'r') as f:
                    training_examples = sum(1 for line in f)
            
            return {
                'model_loaded': job_info.get('status') == 'completed',
                'training_examples': training_examples,
                'last_training': job_info.get('completed_at', job_info.get('started_at', 'Never')),
                'accuracy': 'OpenAI Managed',
                'openai_available': True,
                'model_id': job_info.get('fine_tuned_model'),
                'job_id': job_info.get('job_id')
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'model_loaded': False,
                'training_examples': 0,
                'last_training': 'Error',
                'accuracy': 'N/A',
                'openai_available': True
            }

def train_model(model_name: str, data_dir: str) -> Dict[str, Any]:
    """Main function to train a model"""
    trainer = OpenAITrainer(model_name)
    
    # Prepare training data
    training_file = trainer.prepare_training_data(data_dir)
    
    # Start fine-tuning
    result = trainer.start_fine_tuning(training_file)
    
    return result

def get_training_status(model_name: str) -> Dict[str, Any]:
    """Get training status for a model"""
    trainer = OpenAITrainer(model_name)
    return trainer.check_training_status()

def get_model_status(model_name: str) -> Dict[str, Any]:
    """Get model status and info"""
    trainer = OpenAITrainer(model_name)
    return trainer.get_model_info()
