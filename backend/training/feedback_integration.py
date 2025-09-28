"""
Continuous Learning and Feedback Integration System

This module implements continuous learning capabilities by integrating user feedback,
monitoring model performance, and automatically improving training data.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, text
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackIntegrator:
    """Integrates user feedback for continuous model improvement"""
    
    def __init__(self, model_name: str, db_path: str, data_dir: str = "./training_data"):
        self.model_name = model_name
        self.db_path = db_path
        self.data_dir = Path(data_dir) / model_name
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Create feedback directories
        (self.data_dir / "feedback").mkdir(exist_ok=True)
        (self.data_dir / "continuous_learning").mkdir(exist_ok=True)
        (self.data_dir / "performance_monitoring").mkdir(exist_ok=True)
        
        logger.info(f"FeedbackIntegrator initialized for {model_name}")
    
    def collect_user_feedback(self) -> Dict[str, Any]:
        """Collect and process user feedback from various sources"""
        logger.info("Collecting user feedback for continuous learning...")
        
        feedback_stats = {
            'positive_feedback': 0,
            'negative_feedback': 0,
            'improvement_suggestions': 0,
            'conversation_ratings': 0,
            'total_feedback_items': 0
        }
        
        try:
            engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Collect various types of feedback
            feedback_sources = [
                self._collect_direct_feedback(engine),
                self._collect_conversation_ratings(engine),
                self._collect_correction_feedback(engine),
                self._collect_usage_patterns(engine)
            ]
            
            all_feedback = []
            for feedback_list in feedback_sources:
                all_feedback.extend(feedback_list)
            
            # Process and categorize feedback
            processed_feedback = self._process_feedback(all_feedback)
            
            # Update stats
            feedback_stats.update(self._calculate_feedback_stats(processed_feedback))
            
            # Save processed feedback
            self._save_feedback_data(processed_feedback)
            
            # Generate improvement suggestions
            improvements = self._generate_improvement_suggestions(processed_feedback)
            
            logger.info(f"Collected {len(all_feedback)} feedback items")
            return {
                'stats': feedback_stats,
                'improvements': improvements,
                'processed_feedback': processed_feedback
            }
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return {'stats': feedback_stats, 'improvements': [], 'processed_feedback': []}
    
    def monitor_model_performance(self) -> Dict[str, Any]:
        """Monitor ongoing model performance"""
        logger.info("Monitoring model performance...")
        
        performance_metrics = {
            'response_quality': 0.0,
            'user_satisfaction': 0.0,
            'conversation_completion_rate': 0.0,
            'error_rate': 0.0,
            'improvement_trend': 'stable'
        }
        
        try:
            engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Analyze recent conversations
            recent_conversations = self._get_recent_conversations(engine)
            
            if recent_conversations:
                performance_metrics.update(self._analyze_conversation_performance(recent_conversations))
            
            # Track performance over time
            performance_trend = self._analyze_performance_trend()
            performance_metrics['improvement_trend'] = performance_trend
            
            # Save performance data
            self._save_performance_data(performance_metrics)
            
            logger.info(f"Performance monitoring complete. Quality score: {performance_metrics['response_quality']:.2f}")
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")
            return performance_metrics
    
    def generate_training_improvements(self) -> Dict[str, Any]:
        """Generate training data improvements based on feedback"""
        logger.info("Generating training improvements...")
        
        improvements = {
            'new_examples': [],
            'corrected_examples': [],
            'removed_examples': [],
            'quality_improvements': []
        }
        
        # Load existing feedback
        feedback_data = self._load_feedback_data()
        
        if not feedback_data:
            return improvements
        
        # Generate new training examples from positive feedback
        new_examples = self._generate_examples_from_feedback(feedback_data)
        improvements['new_examples'] = new_examples
        
        # Identify examples that need correction
        corrections = self._identify_correction_opportunities(feedback_data)
        improvements['corrected_examples'] = corrections
        
        # Identify low-quality examples to remove
        removals = self._identify_removal_candidates(feedback_data)
        improvements['removed_examples'] = removals
        
        # Generate quality improvements
        quality_improvements = self._generate_quality_improvements(feedback_data)
        improvements['quality_improvements'] = quality_improvements
        
        # Save improvements
        self._save_improvements(improvements)
        
        logger.info(f"Generated {len(new_examples)} new examples, {len(corrections)} corrections")
        return improvements
    
    def apply_continuous_learning(self) -> Dict[str, Any]:
        """Apply continuous learning improvements"""
        logger.info("Applying continuous learning improvements...")
        
        results = {
            'applied_improvements': 0,
            'new_training_examples': 0,
            'updated_examples': 0,
            'removed_examples': 0,
            'quality_score_improvement': 0.0
        }
        
        # Load improvements
        improvements = self._load_improvements()
        
        if not improvements:
            return results
        
        # Apply new examples
        if improvements.get('new_examples'):
            self._apply_new_examples(improvements['new_examples'])
            results['new_training_examples'] = len(improvements['new_examples'])
        
        # Apply corrections
        if improvements.get('corrected_examples'):
            self._apply_corrections(improvements['corrected_examples'])
            results['updated_examples'] = len(improvements['corrected_examples'])
        
        # Remove low-quality examples
        if improvements.get('removed_examples'):
            self._apply_removals(improvements['removed_examples'])
            results['removed_examples'] = len(improvements['removed_examples'])
        
        # Calculate improvement metrics
        results['applied_improvements'] = sum([
            results['new_training_examples'],
            results['updated_examples'],
            results['removed_examples']
        ])
        
        # Update training data quality
        self._update_training_data_quality()
        
        logger.info(f"Applied {results['applied_improvements']} improvements")
        return results
    
    def _collect_direct_feedback(self, engine) -> List[Dict]:
        """Collect direct user feedback"""
        feedback_list = []
        
        try:
            query = """
            SELECT feedback_text, rating, created_at, user_id
            FROM feedback
            WHERE created_at > date('now', '-30 days')
            ORDER BY created_at DESC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                for row in result:
                    feedback_list.append({
                        'type': 'direct_feedback',
                        'content': row[0],
                        'rating': row[1],
                        'created_at': row[2],
                        'user_id': row[3],
                        'source': 'feedback_table'
                    })
        
        except Exception as e:
            logger.debug(f"No direct feedback table: {e}")
        
        return feedback_list
    
    def _collect_conversation_ratings(self, engine) -> List[Dict]:
        """Collect conversation ratings and feedback"""
        feedback_list = []
        
        try:
            # Look for conversation ratings in highlander_chat
            query = """
            SELECT message, response, context, category, created_at, user_id
            FROM highlander_chat
            WHERE created_at > date('now', '-30 days')
            AND (context LIKE '%rating%' OR context LIKE '%feedback%')
            ORDER BY created_at DESC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                for row in result:
                    # Extract rating from context if available
                    context = row[2] or '{}'
                    try:
                        context_data = json.loads(context)
                        rating = context_data.get('rating', 0)
                    except:
                        rating = 0
                    
                    feedback_list.append({
                        'type': 'conversation_rating',
                        'input': row[0],
                        'output': row[1],
                        'rating': rating,
                        'category': row[3],
                        'created_at': row[4],
                        'user_id': row[5],
                        'source': 'highlander_chat'
                    })
        
        except Exception as e:
            logger.debug(f"Error collecting conversation ratings: {e}")
        
        return feedback_list
    
    def _collect_correction_feedback(self, engine) -> List[Dict]:
        """Collect user corrections and improvements"""
        feedback_list = []
        
        try:
            # Look for translation feedback which might contain corrections
            query = """
            SELECT content, rating, created_at, user_id
            FROM translation_feedback
            WHERE created_at > date('now', '-30 days')
            ORDER BY created_at DESC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                for row in result:
                    feedback_list.append({
                        'type': 'correction_feedback',
                        'content': row[0],
                        'rating': row[1],
                        'created_at': row[2],
                        'user_id': row[3],
                        'source': 'translation_feedback'
                    })
        
        except Exception as e:
            logger.debug(f"No correction feedback available: {e}")
        
        return feedback_list
    
    def _collect_usage_patterns(self, engine) -> List[Dict]:
        """Collect usage patterns and implicit feedback"""
        feedback_list = []
        
        try:
            # Analyze conversation patterns for implicit feedback
            query = """
            SELECT session_id, COUNT(*) as message_count, 
                   AVG(LENGTH(message)) as avg_message_length,
                   MAX(created_at) as last_interaction
            FROM highlander_chat
            WHERE created_at > date('now', '-30 days')
            GROUP BY session_id
            HAVING message_count > 1
            ORDER BY last_interaction DESC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                for row in result:
                    # Infer satisfaction from engagement
                    engagement_score = min(1.0, row[1] / 10)  # Normalize message count
                    
                    feedback_list.append({
                        'type': 'usage_pattern',
                        'session_id': row[0],
                        'message_count': row[1],
                        'avg_message_length': row[2],
                        'engagement_score': engagement_score,
                        'last_interaction': row[3],
                        'source': 'usage_analysis'
                    })
        
        except Exception as e:
            logger.debug(f"Error analyzing usage patterns: {e}")
        
        return feedback_list
    
    def _process_feedback(self, feedback_list: List[Dict]) -> List[Dict]:
        """Process and categorize feedback"""
        processed_feedback = []
        
        for feedback in feedback_list:
            processed_item = {
                **feedback,
                'processed_at': datetime.now().isoformat(),
                'sentiment': self._analyze_sentiment(feedback),
                'actionable': self._is_actionable_feedback(feedback),
                'priority': self._calculate_feedback_priority(feedback)
            }
            
            processed_feedback.append(processed_item)
        
        # Sort by priority
        processed_feedback.sort(key=lambda x: x['priority'], reverse=True)
        
        return processed_feedback
    
    def _analyze_sentiment(self, feedback: Dict) -> str:
        """Analyze feedback sentiment"""
        # Simple sentiment analysis based on rating and keywords
        rating = feedback.get('rating', 0)
        content = str(feedback.get('content', '')).lower()
        
        if rating >= 4 or any(word in content for word in ['good', 'great', 'excellent', 'helpful']):
            return 'positive'
        elif rating <= 2 or any(word in content for word in ['bad', 'poor', 'wrong', 'error']):
            return 'negative'
        else:
            return 'neutral'
    
    def _is_actionable_feedback(self, feedback: Dict) -> bool:
        """Determine if feedback is actionable"""
        content = str(feedback.get('content', '')).lower()
        
        # Look for actionable keywords
        actionable_keywords = [
            'should', 'could', 'improve', 'better', 'wrong', 'correct',
            'add', 'remove', 'change', 'fix', 'update'
        ]
        
        return any(keyword in content for keyword in actionable_keywords)
    
    def _calculate_feedback_priority(self, feedback: Dict) -> float:
        """Calculate feedback priority score"""
        priority = 0.0
        
        # Higher priority for negative feedback
        if feedback.get('sentiment') == 'negative':
            priority += 0.5
        
        # Higher priority for actionable feedback
        if feedback.get('actionable'):
            priority += 0.3
        
        # Higher priority for recent feedback
        try:
            created_at = datetime.fromisoformat(feedback.get('created_at', ''))
            days_ago = (datetime.now() - created_at).days
            recency_score = max(0, 1 - days_ago / 30)  # Decay over 30 days
            priority += recency_score * 0.2
        except:
            pass
        
        return priority
    
    def _generate_examples_from_feedback(self, feedback_data: List[Dict]) -> List[Dict]:
        """Generate new training examples from positive feedback"""
        new_examples = []
        
        for feedback in feedback_data:
            if feedback.get('sentiment') == 'positive' and feedback.get('type') == 'conversation_rating':
                example = {
                    'input': feedback.get('input', ''),
                    'output': feedback.get('output', ''),
                    'source': 'positive_feedback',
                    'quality_score': 0.8,  # High quality from positive feedback
                    'created_at': datetime.now().isoformat()
                }
                
                if example['input'] and example['output']:
                    new_examples.append(example)
        
        return new_examples
    
    def _identify_correction_opportunities(self, feedback_data: List[Dict]) -> List[Dict]:
        """Identify examples that need correction based on feedback"""
        corrections = []
        
        for feedback in feedback_data:
            if feedback.get('sentiment') == 'negative' and feedback.get('actionable'):
                correction = {
                    'original_input': feedback.get('input', ''),
                    'original_output': feedback.get('output', ''),
                    'feedback': feedback.get('content', ''),
                    'suggested_improvement': self._suggest_improvement(feedback),
                    'priority': feedback.get('priority', 0.0)
                }
                
                corrections.append(correction)
        
        return corrections
    
    def _suggest_improvement(self, feedback: Dict) -> str:
        """Suggest improvement based on feedback"""
        # This would use more sophisticated NLP in a real implementation
        content = feedback.get('content', '')
        
        if 'wrong' in content.lower():
            return "Review and correct the factual accuracy of the response"
        elif 'unclear' in content.lower():
            return "Improve clarity and explanation in the response"
        elif 'incomplete' in content.lower():
            return "Provide more comprehensive information"
        else:
            return "General improvement needed based on user feedback"
    
    # Additional helper methods would be implemented here...
    def _calculate_feedback_stats(self, feedback_data: List[Dict]) -> Dict[str, int]:
        """Calculate feedback statistics"""
        stats = {}
        for feedback in feedback_data:
            sentiment = feedback.get('sentiment', 'neutral')
            stats[f"{sentiment}_feedback"] = stats.get(f"{sentiment}_feedback", 0) + 1
        return stats
    
    def _save_feedback_data(self, feedback_data: List[Dict]):
        """Save processed feedback data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        feedback_file = self.data_dir / "feedback" / f"processed_feedback_{timestamp}.json"
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
    
    def _get_recent_conversations(self, engine) -> List[Dict]:
        """Get recent conversations for analysis"""
        return []
    
    def _analyze_conversation_performance(self, conversations: List[Dict]) -> Dict[str, float]:
        """Analyze conversation performance metrics"""
        return {}
    
    def _analyze_performance_trend(self) -> str:
        """Analyze performance trend over time"""
        return 'stable'
    
    def _save_performance_data(self, metrics: Dict[str, Any]):
        """Save performance monitoring data"""
        pass
    
    def _load_feedback_data(self) -> List[Dict]:
        """Load existing feedback data"""
        return []
    
    def _generate_improvement_suggestions(self, feedback_data: List[Dict]) -> List[str]:
        """Generate improvement suggestions"""
        return []
    
    def _identify_removal_candidates(self, feedback_data: List[Dict]) -> List[Dict]:
        """Identify examples to remove"""
        return []
    
    def _generate_quality_improvements(self, feedback_data: List[Dict]) -> List[str]:
        """Generate quality improvement suggestions"""
        return []
    
    def _save_improvements(self, improvements: Dict[str, Any]):
        """Save improvement suggestions"""
        pass
    
    def _load_improvements(self) -> Dict[str, Any]:
        """Load improvement suggestions"""
        return {}
    
    def _apply_new_examples(self, examples: List[Dict]):
        """Apply new training examples"""
        pass
    
    def _apply_corrections(self, corrections: List[Dict]):
        """Apply corrections to existing examples"""
        pass
    
    def _apply_removals(self, removals: List[Dict]):
        """Remove low-quality examples"""
        pass
    
    def _update_training_data_quality(self):
        """Update overall training data quality"""
        pass

def integrate_feedback(model_name: str, db_path: str) -> Dict[str, Any]:
    """Main function to integrate user feedback"""
    integrator = FeedbackIntegrator(model_name, db_path)
    
    # Collect feedback
    feedback_results = integrator.collect_user_feedback()
    
    # Monitor performance
    performance_results = integrator.monitor_model_performance()
    
    # Generate improvements
    improvement_results = integrator.generate_training_improvements()
    
    # Apply continuous learning
    learning_results = integrator.apply_continuous_learning()
    
    return {
        'feedback': feedback_results,
        'performance': performance_results,
        'improvements': improvement_results,
        'learning': learning_results
    }
