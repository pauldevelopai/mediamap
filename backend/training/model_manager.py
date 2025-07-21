"""
Model Management System for Highlander AI

This module manages the deployment and usage of custom trained models
in the production application, including fallback to OpenAI if needed.
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import threading
import time
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighlanderModelManager:
    """Manages custom AI model deployment and inference"""
    
    def __init__(self, 
                 models_dir: str = "./models",
                 openai_api_key: Optional[str] = None,
                 use_custom_model: bool = True):
        
        self.models_dir = Path(models_dir)
        self.use_custom_model = use_custom_model
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Model components
        self.custom_model = None
        self.custom_tokenizer = None
        self.model_metadata = None
        self.is_model_loaded = False
        
        # Performance tracking
        self.inference_stats = {
            'total_requests': 0,
            'custom_model_requests': 0,
            'openai_requests': 0,
            'avg_response_time': 0,
            'errors': 0
        }
        
        # Load custom model if available
        if self.use_custom_model:
            self.load_custom_model()
        
        logger.info("HighlanderModelManager initialized")
    
    def load_custom_model(self) -> bool:
        """Load the custom trained model"""
        try:
            # Find latest deployed model
            deployment_dir = self.models_dir / "deployment"
            if not deployment_dir.exists():
                logger.warning("No deployed model found")
                return False
            
            deployment_info_file = deployment_dir / "deployment_info.json"
            if not deployment_info_file.exists():
                logger.warning("No deployment info found")
                return False
            
            # Load deployment metadata
            with open(deployment_info_file, 'r') as f:
                deployment_info = json.load(f)
            
            model_path = deployment_info['model_path']
            if not Path(model_path).exists():
                logger.warning(f"Model path does not exist: {model_path}")
                return False
            
            logger.info(f"Loading custom model from: {model_path}")
            
            # Load tokenizer and model
            self.custom_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.custom_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Load training metadata
            metadata_file = self.models_dir / "training_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
            
            self.is_model_loaded = True
            logger.info("Custom model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            self.is_model_loaded = False
            return False
    
    def generate_response(self, 
                         message: str, 
                         conversation_history: List[Dict[str, str]] = None,
                         max_length: int = 300,
                         temperature: float = 0.7) -> Tuple[str, str]:
        """Generate response using custom model or OpenAI fallback"""
        
        start_time = time.time()
        self.inference_stats['total_requests'] += 1
        
        try:
            # Try custom model first if loaded
            if self.is_model_loaded and self.use_custom_model:
                response, source = self._generate_custom_response(
                    message, conversation_history, max_length, temperature
                )
                if response:
                    self.inference_stats['custom_model_requests'] += 1
                    self._update_response_time(start_time)
                    return response, source
            
            # Fallback to OpenAI
            if self.openai_client:
                response, source = self._generate_openai_response(
                    message, conversation_history, max_length, temperature
                )
                self.inference_stats['openai_requests'] += 1
                self._update_response_time(start_time)
                return response, source
            
            # No model available
            self.inference_stats['errors'] += 1
            return "I'm currently unavailable. Please try again later.", "error"
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self.inference_stats['errors'] += 1
            return "I encountered an error. Please try again.", "error"
    
    def _generate_custom_response(self, 
                                 message: str,
                                 conversation_history: List[Dict[str, str]] = None,
                                 max_length: int = 300,
                                 temperature: float = 0.7) -> Tuple[str, str]:
        """Generate response using custom model"""
        try:
            # Build conversation context
            context = self._build_conversation_context(message, conversation_history)
            
            # Prepare input for model
            prompt = f"User: {message}\nHighlander:"
            
            # Tokenize input
            inputs = self.custom_tokenizer.encode(
                prompt, 
                return_tensors='pt',
                max_length=512,
                truncation=True
            )
            
            # Move to appropriate device
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            # Generate response
            with torch.no_grad():
                outputs = self.custom_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1,
                    pad_token_id=self.custom_tokenizer.eos_token_id,
                    eos_token_id=self.custom_tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.custom_tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Extract Highlander's response
            if "Highlander:" in generated_text:
                response = generated_text.split("Highlander:")[-1].strip()
            else:
                response = generated_text.strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            return response, "custom_model"
            
        except Exception as e:
            logger.error(f"Custom model generation failed: {e}")
            return None, "error"
    
    def _generate_openai_response(self,
                                 message: str,
                                 conversation_history: List[Dict[str, str]] = None,
                                 max_length: int = 300,
                                 temperature: float = 0.7) -> Tuple[str, str]:
        """Generate response using OpenAI as fallback"""
        try:
            # Build conversation context
            context = self._build_conversation_context(message, conversation_history)
            
            # Prepare messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": """You are Highlander, an expert AI business consultant specializing in media companies.

CONVERSATION STYLE:
- Keep responses concise and actionable (2-4 sentences max)
- Never repeat greetings or introductions unless it's truly the first message
- Be direct and professional - skip pleasantries if you've already been introduced
- Build on previous conversation context naturally
- Ask ONE focused follow-up question per response

YOUR EXPERTISE:
- Media business strategy and operations
- AI implementation for content creation, audience analysis, workflow optimization
- Digital transformation and automation
- Revenue optimization and growth strategies

APPROACH:
- Listen for business challenges and immediately suggest specific AI solutions
- Reference previous conversation points to show you remember the context
- Provide concrete, implementable advice rather than general statements
- Focus on ROI and practical business impact

NEVER say 'Hello' again after the first interaction. Always continue the conversation naturally."""
                }
            ]
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-6:]:  # Last 6 messages for context
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=temperature,
                max_tokens=max_length
            )
            
            ai_reply = response.choices[0].message.content
            return ai_reply, "openai"
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return None, "error"
    
    def _build_conversation_context(self, 
                                   message: str,
                                   conversation_history: List[Dict[str, str]] = None) -> str:
        """Build conversation context for better responses"""
        context = ""
        
        if conversation_history:
            # Add recent conversation history
            recent_messages = conversation_history[-4:]  # Last 4 messages
            for msg in recent_messages:
                context += f"{msg['role']}: {msg['content']}\n"
        
        return context
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove common artifacts
        response = response.strip()
        
        # Remove repeated patterns
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('User:'):
                cleaned_lines.append(line.strip())
        
        response = '\n'.join(cleaned_lines)
        
        # Ensure response is not too long
        if len(response) > 500:
            sentences = response.split('.')
            truncated = []
            char_count = 0
            for sentence in sentences:
                if char_count + len(sentence) < 450:
                    truncated.append(sentence)
                    char_count += len(sentence)
                else:
                    break
            response = '.'.join(truncated) + '.'
        
        return response
    
    def _update_response_time(self, start_time: float):
        """Update average response time statistics"""
        response_time = time.time() - start_time
        current_avg = self.inference_stats['avg_response_time']
        total_requests = self.inference_stats['total_requests']
        
        # Calculate new average
        new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        self.inference_stats['avg_response_time'] = new_avg
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'custom_model_loaded': self.is_model_loaded,
            'openai_available': self.openai_client is not None,
            'use_custom_model': self.use_custom_model,
            'inference_stats': self.inference_stats,
            'model_metadata': self.model_metadata
        }
        
        if self.is_model_loaded and self.model_metadata:
            info['training_date'] = self.model_metadata.get('training_completed_at')
            info['data_stats'] = self.model_metadata.get('data_stats')
        
        return info
    
    def update_model(self) -> bool:
        """Update to latest trained model"""
        logger.info("Updating to latest model...")
        
        # Unload current model
        if self.is_model_loaded:
            del self.custom_model
            del self.custom_tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Reload model
        return self.load_custom_model()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        total_requests = self.inference_stats['total_requests']
        if total_requests == 0:
            return {'message': 'No requests processed yet'}
        
        custom_percentage = (self.inference_stats['custom_model_requests'] / total_requests) * 100
        openai_percentage = (self.inference_stats['openai_requests'] / total_requests) * 100
        error_percentage = (self.inference_stats['errors'] / total_requests) * 100
        
        return {
            'total_requests': total_requests,
            'custom_model_usage': f"{custom_percentage:.1f}%",
            'openai_fallback_usage': f"{openai_percentage:.1f}%",
            'error_rate': f"{error_percentage:.1f}%",
            'avg_response_time': f"{self.inference_stats['avg_response_time']:.2f}s",
            'model_health': 'healthy' if error_percentage < 5 else 'degraded'
        }

# Global model manager instance
model_manager = None

def get_model_manager() -> HighlanderModelManager:
    """Get or create global model manager instance"""
    global model_manager
    if model_manager is None:
        openai_key = os.getenv('OPENAI_API_KEY')
        # Use absolute path to training models directory
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        model_manager = HighlanderModelManager(
            models_dir=models_dir,
            openai_api_key=openai_key,
            use_custom_model=True
        )
    return model_manager

def generate_highlander_response(message: str, 
                               conversation_history: List[Dict[str, str]] = None) -> Tuple[str, str]:
    """Main function to generate Highlander responses"""
    manager = get_model_manager()
    return manager.generate_response(message, conversation_history)

if __name__ == "__main__":
    # Test the model manager
    manager = HighlanderModelManager()
    
    # Test response generation
    response, source = manager.generate_response("Tell me about AI implementation for media companies")
    print(f"Response ({source}): {response}")
    
    # Show performance metrics
    metrics = manager.get_performance_metrics()
    print(f"Performance: {metrics}") 