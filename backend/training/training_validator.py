"""
Training Data Validation and Quality Assessment System

This module provides comprehensive validation, quality assessment, and 
performance evaluation for training data and models.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import Counter
import statistics
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingValidator:
    """Comprehensive training data validation and quality assessment"""
    
    def __init__(self, model_name: str, data_dir: str):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_examples': 20,
            'min_avg_length': 50,
            'max_avg_length': 2000,
            'min_diversity_score': 0.3,
            'min_relevance_score': 0.7,
            'max_duplicate_ratio': 0.1
        }
        
        logger.info(f"TrainingValidator initialized for {model_name}")
    
    def validate_training_data(self) -> Dict[str, Any]:
        """Comprehensive validation of training data"""
        logger.info("Starting comprehensive training data validation...")
        
        validation_report = {
            'model_name': self.model_name,
            'validation_date': datetime.now().isoformat(),
            'data_quality': {},
            'content_analysis': {},
            'recommendations': [],
            'overall_score': 0.0,
            'ready_for_training': False
        }
        
        # Load all training data
        all_data = self._load_all_training_data()
        
        if not all_data:
            validation_report['recommendations'].append("No training data found. Please collect data first.")
            return validation_report
        
        # Perform validation checks
        validation_report['data_quality'] = self._assess_data_quality(all_data)
        validation_report['content_analysis'] = self._analyze_content(all_data)
        validation_report['diversity_analysis'] = self._analyze_diversity(all_data)
        validation_report['relevance_analysis'] = self._analyze_relevance(all_data)
        
        # Calculate overall score
        validation_report['overall_score'] = self._calculate_overall_score(validation_report)
        
        # Determine if ready for training
        validation_report['ready_for_training'] = validation_report['overall_score'] >= 0.7
        
        # Generate recommendations
        validation_report['recommendations'] = self._generate_validation_recommendations(validation_report)
        
        # Save validation report
        self._save_validation_report(validation_report)
        
        logger.info(f"Validation complete. Overall score: {validation_report['overall_score']:.2f}")
        return validation_report
    
    def evaluate_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Evaluate trained model performance"""
        logger.info(f"Evaluating model performance for {model_id}")
        
        performance_report = {
            'model_id': model_id,
            'model_name': self.model_name,
            'evaluation_date': datetime.now().isoformat(),
            'test_results': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Load test data
        test_data = self._load_test_data()
        
        if not test_data:
            performance_report['recommendations'].append("No test data available for evaluation.")
            return performance_report
        
        # Run performance tests
        performance_report['test_results'] = self._run_performance_tests(model_id, test_data)
        performance_report['performance_metrics'] = self._calculate_performance_metrics(performance_report['test_results'])
        performance_report['recommendations'] = self._generate_performance_recommendations(performance_report)
        
        # Save performance report
        self._save_performance_report(performance_report)
        
        logger.info(f"Performance evaluation complete for {model_id}")
        return performance_report
    
    def _load_all_training_data(self) -> List[Dict]:
        """Load all training data from various sources"""
        all_data = []
        
        # Load from different data sources
        data_sources = [
            'conversations/real_conversations.json',
            'conversations/highlander_conversations.json',
            'conversations/healthpin_conversations.json',
            'intelligent_examples.json',
            'research/validated_research.json'
        ]
        
        for source in data_sources:
            file_path = self.data_dir / source
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        all_data.extend(data)
                    elif isinstance(data, dict) and 'pairs' in data:
                        all_data.extend(data['pairs'])
                    
                except Exception as e:
                    logger.error(f"Error loading {source}: {e}")
        
        return all_data
    
    def _assess_data_quality(self, data: List[Dict]) -> Dict[str, Any]:
        """Assess overall data quality"""
        quality_metrics = {
            'total_examples': len(data),
            'avg_input_length': 0,
            'avg_output_length': 0,
            'empty_examples': 0,
            'duplicate_count': 0,
            'quality_score': 0.0
        }
        
        if not data:
            return quality_metrics
        
        input_lengths = []
        output_lengths = []
        seen_inputs = set()
        duplicates = 0
        empty_count = 0
        
        for item in data:
            input_text = str(item.get('input', item.get('question', '')))
            output_text = str(item.get('output', item.get('answer', '')))
            
            # Check for empty examples
            if not input_text.strip() or not output_text.strip():
                empty_count += 1
                continue
            
            # Check for duplicates
            if input_text in seen_inputs:
                duplicates += 1
            else:
                seen_inputs.add(input_text)
            
            input_lengths.append(len(input_text))
            output_lengths.append(len(output_text))
        
        # Calculate averages
        if input_lengths:
            quality_metrics['avg_input_length'] = statistics.mean(input_lengths)
            quality_metrics['avg_output_length'] = statistics.mean(output_lengths)
        
        quality_metrics['empty_examples'] = empty_count
        quality_metrics['duplicate_count'] = duplicates
        
        # Calculate quality score
        quality_score = 1.0
        
        # Penalize for too few examples
        if quality_metrics['total_examples'] < self.quality_thresholds['min_examples']:
            quality_score *= 0.5
        
        # Penalize for empty examples
        if empty_count > 0:
            quality_score *= (1 - empty_count / len(data))
        
        # Penalize for duplicates
        if duplicates > 0:
            duplicate_ratio = duplicates / len(data)
            if duplicate_ratio > self.quality_thresholds['max_duplicate_ratio']:
                quality_score *= (1 - duplicate_ratio)
        
        quality_metrics['quality_score'] = quality_score
        
        return quality_metrics
    
    def _analyze_content(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze content characteristics"""
        content_analysis = {
            'vocabulary_size': 0,
            'common_words': [],
            'domain_keywords': [],
            'content_types': {},
            'complexity_score': 0.0
        }
        
        if not data:
            return content_analysis
        
        # Collect all text
        all_text = []
        for item in data:
            input_text = str(item.get('input', item.get('question', '')))
            output_text = str(item.get('output', item.get('answer', '')))
            all_text.append(input_text + " " + output_text)
        
        # Analyze vocabulary
        all_words = []
        for text in all_text:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        content_analysis['vocabulary_size'] = len(word_counts)
        content_analysis['common_words'] = word_counts.most_common(20)
        
        # Identify domain keywords
        domain_keywords = self._identify_domain_keywords(word_counts)
        content_analysis['domain_keywords'] = domain_keywords
        
        # Analyze content types
        content_types = self._classify_content_types(data)
        content_analysis['content_types'] = content_types
        
        # Calculate complexity score
        avg_sentence_length = statistics.mean([len(text.split()) for text in all_text])
        complexity_score = min(1.0, avg_sentence_length / 20)  # Normalize to 0-1
        content_analysis['complexity_score'] = complexity_score
        
        return content_analysis
    
    def _analyze_diversity(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze data diversity"""
        diversity_analysis = {
            'topic_diversity': 0.0,
            'length_diversity': 0.0,
            'style_diversity': 0.0,
            'overall_diversity': 0.0
        }
        
        if not data:
            return diversity_analysis
        
        # Analyze topic diversity (based on keywords)
        topics = []
        for item in data:
            input_text = str(item.get('input', item.get('question', '')))
            keywords = self._extract_keywords(input_text)
            topics.append(set(keywords))
        
        # Calculate topic diversity (Jaccard similarity)
        if len(topics) > 1:
            similarities = []
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    intersection = len(topics[i] & topics[j])
                    union = len(topics[i] | topics[j])
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
            
            avg_similarity = statistics.mean(similarities)
            diversity_analysis['topic_diversity'] = 1 - avg_similarity
        
        # Analyze length diversity
        lengths = []
        for item in data:
            input_text = str(item.get('input', item.get('question', '')))
            output_text = str(item.get('output', item.get('answer', '')))
            lengths.append(len(input_text) + len(output_text))
        
        if lengths:
            length_std = statistics.stdev(lengths) if len(lengths) > 1 else 0
            length_mean = statistics.mean(lengths)
            diversity_analysis['length_diversity'] = min(1.0, length_std / length_mean) if length_mean > 0 else 0
        
        # Calculate overall diversity
        diversity_analysis['overall_diversity'] = (
            diversity_analysis['topic_diversity'] + 
            diversity_analysis['length_diversity']
        ) / 2
        
        return diversity_analysis
    
    def _analyze_relevance(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze domain relevance"""
        relevance_analysis = {
            'domain_relevance_score': 0.0,
            'relevant_examples': 0,
            'irrelevant_examples': 0,
            'relevance_distribution': {}
        }
        
        if not data:
            return relevance_analysis
        
        relevant_count = 0
        relevance_scores = []
        
        for item in data:
            input_text = str(item.get('input', item.get('question', '')))
            output_text = str(item.get('output', item.get('answer', '')))
            
            relevance_score = self._calculate_relevance_score(input_text + " " + output_text)
            relevance_scores.append(relevance_score)
            
            if relevance_score >= self.quality_thresholds['min_relevance_score']:
                relevant_count += 1
        
        relevance_analysis['relevant_examples'] = relevant_count
        relevance_analysis['irrelevant_examples'] = len(data) - relevant_count
        relevance_analysis['domain_relevance_score'] = statistics.mean(relevance_scores) if relevance_scores else 0.0
        
        return relevance_analysis
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate domain relevance score for text"""
        text_lower = text.lower()
        
        if self.model_name == 'mediamap':
            domain_keywords = [
                'media', 'marketing', 'advertising', 'campaign', 'brand', 'content',
                'roi', 'audience', 'digital', 'social media', 'analytics', 'strategy',
                'engagement', 'conversion', 'impression', 'click', 'reach', 'business'
            ]
        elif self.model_name == 'healthpin':
            domain_keywords = [
                'health', 'medical', 'patient', 'clinical', 'diagnosis', 'treatment',
                'healthcare', 'medicine', 'doctor', 'hospital', 'symptom', 'therapy',
                'prescription', 'wellness', 'disease', 'condition'
            ]
        else:
            return 1.0  # Default relevance for general models
        
        # Count keyword matches
        matches = sum(1 for keyword in domain_keywords if keyword in text_lower)
        
        # Calculate relevance score
        relevance_score = min(1.0, matches / 5)  # Normalize to 0-1, max at 5 keywords
        
        return relevance_score
    
    def _calculate_overall_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate overall validation score"""
        weights = {
            'data_quality': 0.3,
            'content_analysis': 0.2,
            'diversity_analysis': 0.25,
            'relevance_analysis': 0.25
        }
        
        scores = {
            'data_quality': validation_report['data_quality'].get('quality_score', 0.0),
            'content_analysis': min(1.0, validation_report['content_analysis'].get('complexity_score', 0.0)),
            'diversity_analysis': validation_report['diversity_analysis'].get('overall_diversity', 0.0),
            'relevance_analysis': validation_report['relevance_analysis'].get('domain_relevance_score', 0.0)
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        return overall_score
    
    def _generate_validation_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        data_quality = validation_report['data_quality']
        diversity = validation_report['diversity_analysis']
        relevance = validation_report['relevance_analysis']
        
        # Data quality recommendations
        if data_quality['total_examples'] < self.quality_thresholds['min_examples']:
            recommendations.append(f"Collect more training examples (current: {data_quality['total_examples']}, recommended: {self.quality_thresholds['min_examples']}+)")
        
        if data_quality['empty_examples'] > 0:
            recommendations.append(f"Remove {data_quality['empty_examples']} empty examples")
        
        if data_quality['duplicate_count'] > 0:
            recommendations.append(f"Remove {data_quality['duplicate_count']} duplicate examples")
        
        # Diversity recommendations
        if diversity['overall_diversity'] < self.quality_thresholds['min_diversity_score']:
            recommendations.append("Increase data diversity by adding examples from different topics and contexts")
        
        # Relevance recommendations
        if relevance['domain_relevance_score'] < self.quality_thresholds['min_relevance_score']:
            recommendations.append(f"Improve domain relevance (current: {relevance['domain_relevance_score']:.2f}, target: {self.quality_thresholds['min_relevance_score']})")
        
        if relevance['irrelevant_examples'] > 0:
            recommendations.append(f"Remove or improve {relevance['irrelevant_examples']} irrelevant examples")
        
        return recommendations
    
    def _save_validation_report(self, report: Dict[str, Any]):
        """Save validation report"""
        reports_dir = self.data_dir / "validation_reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"validation_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved: {report_file}")
    
    # Helper methods
    def _identify_domain_keywords(self, word_counts: Counter) -> List[Tuple[str, int]]:
        """Identify domain-specific keywords"""
        # This would be more sophisticated in a real implementation
        return word_counts.most_common(10)
    
    def _classify_content_types(self, data: List[Dict]) -> Dict[str, int]:
        """Classify content into types"""
        # Simplified classification
        return {"question_answer": len(data)}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [word for word in words if word not in stop_words and len(word) > 3]
    
    def _load_test_data(self) -> List[Dict]:
        """Load test data for evaluation"""
        # This would load a separate test dataset
        return []
    
    def _run_performance_tests(self, model_id: str, test_data: List[Dict]) -> Dict[str, Any]:
        """Run performance tests on the model"""
        # This would test the model against test data
        return {}
    
    def _calculate_performance_metrics(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        return {}
    
    def _generate_performance_recommendations(self, performance_report: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        return []
    
    def _save_performance_report(self, report: Dict[str, Any]):
        """Save performance report"""
        pass

def validate_training_data(model_name: str, data_dir: str) -> Dict[str, Any]:
    """Main function to validate training data"""
    validator = TrainingValidator(model_name, data_dir)
    return validator.validate_training_data()

def evaluate_model_performance(model_name: str, model_id: str, data_dir: str) -> Dict[str, Any]:
    """Main function to evaluate model performance"""
    validator = TrainingValidator(model_name, data_dir)
    return validator.evaluate_model_performance(model_id)
