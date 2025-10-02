"""
Training History Tracking System

This module tracks what data has been used for training to prevent
retraining on the same data and enable incremental training.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

class TrainingHistory:
    """Tracks training history to prevent duplicate training"""
    
    def __init__(self, history_file: str = "./models/training_history.json"):
        self.history_file = Path(history_file)
        self.history = self.load_history()
    
    def load_history(self) -> Dict[str, Any]:
        """Load training history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading training history: {e}")
        
        # Initialize empty history
        return {
            'training_sessions': [],
            'data_hashes': {},
            'last_training_date': None,
            'total_training_examples': 0,
            'model_versions': []
        }
    
    def save_history(self):
        """Save training history to file"""
        try:
            self.history_file.parent.mkdir(exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
    
    def get_data_hash(self, data_path: str) -> str:
        """Generate hash for data file to track changes"""
        try:
            with open(data_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash for {data_path}: {e}")
            return ""
    
    def has_data_changed(self, data_path: str) -> bool:
        """Check if data has changed since last training"""
        current_hash = self.get_data_hash(data_path)
        stored_hash = self.history['data_hashes'].get(data_path)
        
        if stored_hash != current_hash:
            # Update stored hash
            self.history['data_hashes'][data_path] = current_hash
            return True
        
        return False
    
    def get_new_data_since_last_training(self, data_dir: str) -> Dict[str, Any]:
        """Get only new data that hasn't been used for training"""
        data_dir = Path(data_dir)
        new_data_stats = {
            'conversations': 0,
            'pdfs': 0,
            'research_papers': 0,
            'feedback_entries': 0,
            'total_tokens': 0,
            'changed_files': []
        }
        
        # Check conversations
        conversations_file = data_dir / "conversations" / "all_conversations.json"
        if conversations_file.exists() and self.has_data_changed(str(conversations_file)):
            try:
                with open(conversations_file, 'r') as f:
                    conversations = json.load(f)
                new_data_stats['conversations'] = len(conversations)
                new_data_stats['changed_files'].append(str(conversations_file))
            except Exception as e:
                logger.error(f"Error reading conversations: {e}")
        
        # Check PDFs
        pdf_dir = data_dir / "pdfs"
        if pdf_dir.exists():
            for txt_file in pdf_dir.glob("*.txt"):
                if self.has_data_changed(str(txt_file)):
                    new_data_stats['pdfs'] += 1
                    new_data_stats['changed_files'].append(str(txt_file))
        
        # Check research papers
        research_dir = data_dir / "research"
        if research_dir.exists():
            for json_file in research_dir.glob("*.json"):
                if self.has_data_changed(str(json_file)):
                    new_data_stats['research_papers'] += 1
                    new_data_stats['changed_files'].append(str(json_file))
        
        # Check feedback
        feedback_file = data_dir / "feedback" / "user_feedback.json"
        if feedback_file.exists() and self.has_data_changed(str(feedback_file)):
            try:
                with open(feedback_file, 'r') as f:
                    feedback = json.load(f)
                new_data_stats['feedback_entries'] = len(feedback)
                new_data_stats['changed_files'].append(str(feedback_file))
            except Exception as e:
                logger.error(f"Error reading feedback: {e}")
        
        # Check processed training data
        training_data_file = data_dir / "processed" / "training_dataset.json"
        if training_data_file.exists() and self.has_data_changed(str(training_data_file)):
            try:
                with open(training_data_file, 'r') as f:
                    training_data = json.load(f)
                new_data_stats['total_tokens'] = sum(
                    len(str(item.get('input', '')).split()) + 
                    len(str(item.get('output', '')).split()) 
                    for item in training_data
                )
                new_data_stats['changed_files'].append(str(training_data_file))
            except Exception as e:
                logger.error(f"Error reading training data: {e}")
        
        return new_data_stats
    
    def record_training_session(self, 
                              training_data_path: str,
                              model_path: str,
                              training_stats: Dict[str, Any]):
        """Record a new training session"""
        session = {
            'id': len(self.history['training_sessions']) + 1,
            'date': datetime.now().isoformat(),
            'training_data_path': training_data_path,
            'model_path': model_path,
            'data_hash': self.get_data_hash(training_data_path),
            'stats': training_stats,
            'examples_processed': training_stats.get('total_examples', 0)
        }
        
        self.history['training_sessions'].append(session)
        self.history['last_training_date'] = session['date']
        self.history['total_training_examples'] += session['examples_processed']
        
        # Record model version
        model_version = {
            'version': f"v{len(self.history['model_versions']) + 1}.0",
            'date': session['date'],
            'model_path': model_path,
            'training_examples': session['examples_processed']
        }
        self.history['model_versions'].append(model_version)
        
        self.save_history()
        logger.info(f"Recorded training session {session['id']}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training history"""
        return {
            'total_sessions': len(self.history['training_sessions']),
            'last_training_date': self.history['last_training_date'],
            'total_training_examples': self.history['total_training_examples'],
            'model_versions': len(self.history['model_versions']),
            'recent_sessions': self.history['training_sessions'][-5:] if self.history['training_sessions'] else []
        }
    
    def should_retrain(self, data_dir: str, min_new_data_threshold: int = 10) -> Dict[str, Any]:
        """Determine if retraining is needed based on new data"""
        new_data = self.get_new_data_since_last_training(data_dir)
        total_new_items = (
            new_data['conversations'] + 
            new_data['pdfs'] + 
            new_data['research_papers'] + 
            new_data['feedback_entries']
        )
        
        should_retrain = total_new_items >= min_new_data_threshold
        
        return {
            'should_retrain': should_retrain,
            'new_data': new_data,
            'total_new_items': total_new_items,
            'threshold': min_new_data_threshold,
            'reason': f"Found {total_new_items} new items (threshold: {min_new_data_threshold})"
        }

# Global training history instance
training_history = None

def get_training_history() -> TrainingHistory:
    """Get or create global training history instance"""
    global training_history
    if training_history is None:
        training_history = TrainingHistory()
    return training_history 