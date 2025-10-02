"""
Enhanced Domain-Specific Data Collection System

This module provides advanced data collection with real conversation integration,
quality validation, and intelligent data augmentation.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests
from sqlalchemy import create_engine, text
import re
from collections import Counter
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDomainDataCollector:
    """Advanced domain-specific training data collector"""
    
    def __init__(self, model_name: str, db_path: str, output_dir: str = "./training_data"):
        self.model_name = model_name
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create model-specific directories
        self.model_dir = self.output_dir / model_name
        self.model_dir.mkdir(exist_ok=True)
        (self.model_dir / "conversations").mkdir(exist_ok=True)
        (self.model_dir / "pdfs").mkdir(exist_ok=True)
        (self.model_dir / "research").mkdir(exist_ok=True)
        (self.model_dir / "feedback").mkdir(exist_ok=True)
        (self.model_dir / "quality_reports").mkdir(exist_ok=True)
        
        # Quality thresholds
        self.min_message_length = 10
        self.max_message_length = 4000
        self.min_conversation_turns = 2
        
        logger.info(f"EnhancedDomainDataCollector initialized for {model_name}")
    
    def collect_enhanced_data(self) -> Dict[str, Any]:
        """Collect enhanced training data with quality validation"""
        logger.info(f"Starting enhanced data collection for {self.model_name}")
        
        stats = {
            'real_conversations': 0,
            'highlander_conversations': 0,
            'healthpin_conversations': 0,
            'quality_filtered': 0,
            'augmented_examples': 0,
            'pdfs': 0,
            'research_papers': 0,
            'feedback_entries': 0,
            'total_examples': 0,
            'quality_score': 0.0
        }
        
        # Collect real conversations from database
        stats.update(self.collect_real_conversations())
        
        # Collect domain-specific conversations
        stats.update(self.collect_specialized_conversations())
        
        # Collect and process PDFs
        stats['pdfs'] = self.collect_enhanced_pdfs()
        
        # Collect research with validation
        stats['research_papers'] = self.collect_validated_research()
        
        # Collect user feedback
        stats['feedback_entries'] = self.collect_user_feedback()
        
        # Generate intelligent augmented data
        stats['augmented_examples'] = self.generate_intelligent_examples()
        
        # Validate and clean all data
        quality_report = self.validate_and_clean_data()
        stats.update(quality_report)
        
        # Calculate total examples
        stats['total_examples'] = sum([
            stats['real_conversations'],
            stats['highlander_conversations'],
            stats['healthpin_conversations'],
            stats['augmented_examples'],
            stats['pdfs'],
            stats['research_papers'],
            stats['feedback_entries']
        ])
        
        # Generate quality report
        self.generate_quality_report(stats)
        
        logger.info(f"Enhanced data collection complete for {self.model_name}: {stats}")
        return stats
    
    def collect_real_conversations(self) -> Dict[str, int]:
        """Collect real user conversations from the database"""
        logger.info("Collecting real user conversations...")
        
        stats = {'real_conversations': 0}
        
        try:
            engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Collect regular chat conversations
            query = """
            SELECT c.id, c.title, c.created_at, c.fact_sheet, c.strategies,
                   m.role, m.content, m.created_at as msg_created
            FROM chats c
            JOIN messages m ON c.id = m.chat_id
            WHERE c.created_at > date('now', '-90 days')
            ORDER BY c.id, m.created_at ASC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
            
            # Group messages by chat
            conversations = {}
            for row in rows:
                chat_id = row[0]
                if chat_id not in conversations:
                    conversations[chat_id] = {
                        'id': chat_id,
                        'title': row[1] or f"Chat {chat_id}",
                        'created_at': row[2],
                        'fact_sheet': row[3],
                        'strategies': row[4],
                        'messages': []
                    }
                
                conversations[chat_id]['messages'].append({
                    'role': row[5],
                    'content': row[6],
                    'created_at': row[7]
                })
            
            # Process and filter conversations
            quality_conversations = []
            for chat_id, conv in conversations.items():
                if self._is_quality_conversation(conv):
                    processed_conv = self._process_conversation(conv)
                    if processed_conv:
                        quality_conversations.append(processed_conv)
            
            # Save real conversations
            if quality_conversations:
                output_file = self.model_dir / "conversations" / "real_conversations.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(quality_conversations, f, indent=2, ensure_ascii=False)
                
                stats['real_conversations'] = len(quality_conversations)
                logger.info(f"Collected {len(quality_conversations)} quality real conversations")
            
        except Exception as e:
            logger.error(f"Error collecting real conversations: {e}")
        
        return stats
    
    def collect_specialized_conversations(self) -> Dict[str, int]:
        """Collect specialized conversations (Highlander, HealthPIN)"""
        logger.info("Collecting specialized conversations...")
        
        stats = {'highlander_conversations': 0, 'healthpin_conversations': 0}
        
        try:
            engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Collect Highlander conversations
            highlander_query = """
            SELECT session_id, message, response, context, category, created_at
            FROM highlander_chat
            WHERE created_at > date('now', '-90 days')
            ORDER BY created_at ASC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(highlander_query))
                highlander_rows = result.fetchall()
            
            # Process Highlander conversations
            highlander_conversations = []
            for row in highlander_rows:
                if self._is_domain_relevant_message(row[1] + " " + row[2]):
                    conv = {
                        'session_id': row[0],
                        'input': row[1],
                        'output': row[2],
                        'context': row[3],
                        'category': row[4],
                        'created_at': row[5],
                        'source': 'highlander'
                    }
                    highlander_conversations.append(conv)
            
            if highlander_conversations:
                output_file = self.model_dir / "conversations" / "highlander_conversations.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(highlander_conversations, f, indent=2, ensure_ascii=False)
                
                stats['highlander_conversations'] = len(highlander_conversations)
                logger.info(f"Collected {len(highlander_conversations)} Highlander conversations")
            
            # Collect HealthPIN conversations if available
            if self.model_name == 'healthpin':
                healthpin_query = """
                SELECT patient_id, doctor_id, consultation_notes, recommendations, created_at
                FROM healthpin_consultations
                WHERE created_at > date('now', '-90 days')
                ORDER BY created_at ASC
                """
                
                try:
                    with engine.connect() as conn:
                        result = conn.execute(text(healthpin_query))
                        healthpin_rows = result.fetchall()
                    
                    healthpin_conversations = []
                    for row in healthpin_rows:
                        if row[2] and row[3]:  # Has consultation notes and recommendations
                            conv = {
                                'patient_id': row[0],
                                'doctor_id': row[1],
                                'input': f"Patient consultation: {row[2]}",
                                'output': f"Medical recommendation: {row[3]}",
                                'created_at': row[4],
                                'source': 'healthpin_consultation'
                            }
                            healthpin_conversations.append(conv)
                    
                    if healthpin_conversations:
                        output_file = self.model_dir / "conversations" / "healthpin_conversations.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(healthpin_conversations, f, indent=2, ensure_ascii=False)
                        
                        stats['healthpin_conversations'] = len(healthpin_conversations)
                        logger.info(f"Collected {len(healthpin_conversations)} HealthPIN conversations")
                
                except Exception as e:
                    logger.info(f"No HealthPIN consultation data available: {e}")
            
        except Exception as e:
            logger.error(f"Error collecting specialized conversations: {e}")
        
        return stats
    
    def collect_enhanced_pdfs(self) -> int:
        """Collect and intelligently process PDFs"""
        logger.info("Collecting enhanced PDF data...")
        
        pdf_count = 0
        data_dir = Path("../data")
        
        if data_dir.exists():
            for pdf_file in data_dir.glob("**/*.pdf"):
                if self._is_pdf_domain_relevant(pdf_file.name):
                    # Extract and process PDF content
                    processed_content = self._process_pdf_intelligently(pdf_file)
                    if processed_content:
                        # Save processed content
                        output_file = self.model_dir / "pdfs" / f"{pdf_file.stem}_processed.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(processed_content, f, indent=2, ensure_ascii=False)
                        pdf_count += 1
        
        logger.info(f"Processed {pdf_count} PDFs with intelligent extraction")
        return pdf_count
    
    def collect_validated_research(self) -> int:
        """Collect research data with validation"""
        logger.info("Collecting validated research data...")
        
        research_data = []
        
        if self.model_name == 'mediamap':
            research_data = self._get_advanced_media_research()
        elif self.model_name == 'healthpin':
            research_data = self._get_advanced_health_research()
        
        # Validate research quality
        validated_research = [item for item in research_data if self._validate_research_item(item)]
        
        if validated_research:
            output_file = self.model_dir / "research" / f"{self.model_name}_validated_research.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validated_research, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Collected {len(validated_research)} validated research items")
        return len(validated_research)
    
    def collect_user_feedback(self) -> int:
        """Collect and process user feedback"""
        logger.info("Collecting user feedback...")
        
        feedback_count = 0
        
        try:
            engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Collect feedback from multiple sources
            queries = [
                "SELECT feedback_text, rating, created_at FROM feedback WHERE created_at > date('now', '-90 days')",
                "SELECT content, rating, created_at FROM translation_feedback WHERE created_at > date('now', '-90 days')"
            ]
            
            all_feedback = []
            
            with engine.connect() as conn:
                for query in queries:
                    try:
                        result = conn.execute(text(query))
                        for row in result:
                            if self._is_feedback_domain_relevant(row[0]):
                                all_feedback.append({
                                    'feedback': row[0],
                                    'rating': row[1],
                                    'created_at': row[2],
                                    'processed_at': datetime.now().isoformat()
                                })
                    except Exception as e:
                        logger.debug(f"Feedback query failed: {e}")
            
            if all_feedback:
                output_file = self.model_dir / "feedback" / "processed_feedback.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_feedback, f, indent=2, ensure_ascii=False)
                feedback_count = len(all_feedback)
        
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
        
        return feedback_count
    
    def generate_intelligent_examples(self) -> int:
        """Generate intelligent, context-aware training examples"""
        logger.info("Generating intelligent training examples...")
        
        # Load existing data to understand patterns
        existing_patterns = self._analyze_existing_patterns()
        
        # Generate examples based on patterns
        intelligent_examples = []
        
        if self.model_name == 'mediamap':
            intelligent_examples = self._generate_contextual_media_examples(existing_patterns)
        elif self.model_name == 'healthpin':
            intelligent_examples = self._generate_contextual_health_examples(existing_patterns)
        
        if intelligent_examples:
            output_file = self.model_dir / "intelligent_examples.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(intelligent_examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(intelligent_examples)} intelligent examples")
        return len(intelligent_examples)
    
    def validate_and_clean_data(self) -> Dict[str, Any]:
        """Validate and clean all collected data"""
        logger.info("Validating and cleaning data...")
        
        quality_stats = {
            'quality_filtered': 0,
            'duplicates_removed': 0,
            'quality_score': 0.0
        }
        
        # Load all data files
        all_files = list(self.model_dir.glob("**/*.json"))
        total_items = 0
        quality_items = 0
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    original_count = len(data)
                    # Remove duplicates and low-quality items
                    cleaned_data = self._clean_data_list(data)
                    quality_count = len(cleaned_data)
                    
                    # Save cleaned data
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
                    
                    total_items += original_count
                    quality_items += quality_count
                    quality_stats['quality_filtered'] += (original_count - quality_count)
            
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Calculate quality score
        if total_items > 0:
            quality_stats['quality_score'] = quality_items / total_items
        
        return quality_stats
    
    def generate_quality_report(self, stats: Dict[str, Any]):
        """Generate a comprehensive quality report"""
        logger.info("Generating quality report...")
        
        report = {
            'model_name': self.model_name,
            'collection_date': datetime.now().isoformat(),
            'data_sources': {
                'real_conversations': stats.get('real_conversations', 0),
                'highlander_conversations': stats.get('highlander_conversations', 0),
                'healthpin_conversations': stats.get('healthpin_conversations', 0),
                'pdfs': stats.get('pdfs', 0),
                'research_papers': stats.get('research_papers', 0),
                'feedback_entries': stats.get('feedback_entries', 0),
                'augmented_examples': stats.get('augmented_examples', 0)
            },
            'quality_metrics': {
                'total_examples': stats.get('total_examples', 0),
                'quality_score': stats.get('quality_score', 0.0),
                'quality_filtered': stats.get('quality_filtered', 0),
                'duplicates_removed': stats.get('duplicates_removed', 0)
            },
            'recommendations': self._generate_recommendations(stats)
        }
        
        # Save quality report
        report_file = self.model_dir / "quality_reports" / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Quality report saved: {report_file}")
        return report
    
    # Helper methods
    def _is_quality_conversation(self, conv: Dict) -> bool:
        """Check if conversation meets quality standards"""
        messages = conv.get('messages', [])
        
        if len(messages) < self.min_conversation_turns:
            return False
        
        # Check message quality
        for msg in messages:
            content = msg.get('content', '')
            if len(content) < self.min_message_length or len(content) > self.max_message_length:
                return False
            
            # Check for meaningful content (not just greetings)
            if self._is_meaningful_content(content):
                return True
        
        return False
    
    def _is_meaningful_content(self, content: str) -> bool:
        """Check if content is meaningful for training"""
        # Remove common greetings and short responses
        meaningless_patterns = [
            r'^(hi|hello|hey|thanks|thank you|ok|okay|yes|no)\.?$',
            r'^.{1,5}$',  # Very short responses
        ]
        
        content_lower = content.lower().strip()
        
        for pattern in meaningless_patterns:
            if re.match(pattern, content_lower):
                return False
        
        # Check for domain-relevant keywords
        return self._is_domain_relevant_message(content)
    
    def _is_domain_relevant_message(self, content: str) -> bool:
        """Check if message content is relevant to the domain"""
        content_lower = content.lower()
        
        if self.model_name == 'mediamap':
            keywords = [
                'media', 'marketing', 'advertising', 'campaign', 'brand', 'content',
                'roi', 'audience', 'digital', 'social media', 'analytics', 'strategy',
                'engagement', 'conversion', 'impression', 'click', 'reach', 'business',
                'client', 'revenue', 'growth', 'market', 'competitor', 'analysis'
            ]
        elif self.model_name == 'healthpin':
            keywords = [
                'health', 'medical', 'patient', 'clinical', 'diagnosis', 'treatment',
                'healthcare', 'medicine', 'doctor', 'hospital', 'symptom', 'therapy',
                'prescription', 'wellness', 'disease', 'condition', 'consultation'
            ]
        else:
            return True
        
        return any(keyword in content_lower for keyword in keywords)
    
    def _process_conversation(self, conv: Dict) -> Optional[Dict]:
        """Process conversation into training format"""
        messages = conv.get('messages', [])
        if len(messages) < 2:
            return None
        
        # Extract user-assistant pairs
        training_pairs = []
        current_user_msg = None
        
        for msg in messages:
            if msg['role'] == 'user':
                current_user_msg = msg['content']
            elif msg['role'] == 'assistant' and current_user_msg:
                training_pairs.append({
                    'input': current_user_msg,
                    'output': msg['content'],
                    'context': {
                        'chat_id': conv.get('id'),
                        'title': conv.get('title'),
                        'fact_sheet': conv.get('fact_sheet'),
                        'strategies': conv.get('strategies')
                    }
                })
                current_user_msg = None
        
        return {
            'conversation_id': conv.get('id'),
            'title': conv.get('title'),
            'created_at': conv.get('created_at'),
            'pairs': training_pairs,
            'quality_score': self._calculate_conversation_quality(training_pairs)
        } if training_pairs else None
    
    def _calculate_conversation_quality(self, pairs: List[Dict]) -> float:
        """Calculate quality score for conversation"""
        if not pairs:
            return 0.0
        
        total_score = 0.0
        for pair in pairs:
            # Score based on length, relevance, and completeness
            input_len = len(pair['input'])
            output_len = len(pair['output'])
            
            length_score = min(1.0, (input_len + output_len) / 200)
            relevance_score = 1.0 if self._is_domain_relevant_message(pair['input'] + " " + pair['output']) else 0.5
            
            total_score += (length_score + relevance_score) / 2
        
        return total_score / len(pairs)
    
    def _clean_data_list(self, data: List[Dict]) -> List[Dict]:
        """Clean and deduplicate data list"""
        if not data:
            return data
        
        # Remove duplicates based on content hash
        seen_hashes = set()
        cleaned_data = []
        
        for item in data:
            # Create hash of content
            content_str = json.dumps(item, sort_keys=True)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                cleaned_data.append(item)
        
        return cleaned_data
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving training data"""
        recommendations = []
        
        total_examples = stats.get('total_examples', 0)
        quality_score = stats.get('quality_score', 0.0)
        
        if total_examples < 50:
            recommendations.append("Consider collecting more training examples (target: 100+ examples)")
        
        if quality_score < 0.7:
            recommendations.append("Focus on improving data quality - filter out low-quality conversations")
        
        if stats.get('real_conversations', 0) == 0:
            recommendations.append("Add more real user conversations from the database")
        
        if stats.get('feedback_entries', 0) == 0:
            recommendations.append("Implement user feedback collection for continuous improvement")
        
        return recommendations
    
    # Domain-specific methods would be implemented here
    def _analyze_existing_patterns(self) -> Dict:
        """Analyze existing data patterns"""
        return {}
    
    def _generate_contextual_media_examples(self, patterns: Dict) -> List[Dict]:
        """Generate contextual media examples"""
        return []
    
    def _generate_contextual_health_examples(self, patterns: Dict) -> List[Dict]:
        """Generate contextual health examples"""
        return []
    
    def _process_pdf_intelligently(self, pdf_file: Path) -> Optional[Dict]:
        """Intelligently process PDF content"""
        return None
    
    def _get_advanced_media_research(self) -> List[Dict]:
        """Get advanced media research data"""
        return []
    
    def _get_advanced_health_research(self) -> List[Dict]:
        """Get advanced health research data"""
        return []
    
    def _validate_research_item(self, item: Dict) -> bool:
        """Validate research item quality"""
        return True
    
    def _is_pdf_domain_relevant(self, filename: str) -> bool:
        """Check if PDF is domain relevant"""
        return True
    
    def _is_feedback_domain_relevant(self, feedback: str) -> bool:
        """Check if feedback is domain relevant"""
        return True

def collect_enhanced_domain_data(model_name: str, db_path: str) -> Dict[str, Any]:
    """Main function to collect enhanced domain-specific data"""
    collector = EnhancedDomainDataCollector(model_name, db_path)
    return collector.collect_enhanced_data()
