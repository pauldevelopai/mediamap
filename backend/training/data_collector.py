"""
Data Collection System for Custom AI Model Training

This module collects and aggregates data from multiple sources:
1. User conversations from the database
2. PDF documents and research papers
3. External knowledge sources
4. Real-time feedback and corrections
"""

import os
import json
import sqlite3
import PyPDF2
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import logging
from sqlalchemy import create_engine, text, inspect
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Comprehensive data collection system for AI model training"""
    
    def __init__(self, 
                 db_path: str = "instance/media_analysis.db",
                 data_dir: str = "../data",
                 output_dir: str = "./training_data"):
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different data types
        (self.output_dir / "conversations").mkdir(exist_ok=True)
        (self.output_dir / "pdfs").mkdir(exist_ok=True)
        (self.output_dir / "research").mkdir(exist_ok=True)
        (self.output_dir / "feedback").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "datasafe").mkdir(exist_ok=True)
        
        logger.info("DataCollector initialized")
    
    def collect_all_data(self) -> Dict[str, int]:
        """Main method to collect all training data"""
        logger.info("Starting comprehensive data collection...")
        
        stats = {
            'conversations': 0,
            'pdfs': 0,
            'research_papers': 0,
            'feedback_entries': 0,
            'total_tokens': 0
        }
        
        # Collect user conversations
        stats['conversations'] = self.collect_user_conversations()
        
        # Collect PDF documents
        stats['pdfs'] = self.collect_pdf_documents()
        
        # Collect external research
        stats['research_papers'] = self.collect_external_research()
        
        # Collect user feedback
        stats['feedback_entries'] = self.collect_user_feedback()
        
        # Process and consolidate data
        stats['total_tokens'] = self.process_and_consolidate()
        
        logger.info(f"Data collection complete: {stats}")
        return stats
    
    def collect_user_conversations(self) -> int:
        """Collect all user conversations from database"""
        logger.info("Collecting user conversations...")
        
        try:
            engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Check if tables exist first
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            
            # Collect from both regular chats and Highlander chats
            conversations_collected = 0
            
            # Collect Highlander chats (current active data)
            if 'highlander_chat' in table_names:
                conversations_collected += self._collect_highlander_chats(engine)
            
            # Collect regular chats (if any)
            if 'chats' in table_names:
                conversations_collected += self._collect_regular_chats(engine)
            
            if conversations_collected == 0:
                logger.info("No chat tables found - no conversations to collect")
                
            return conversations_collected
            
        except Exception as e:
            logger.error(f"Error collecting conversations: {e}")
            return 0
    
    def _collect_highlander_chats(self, engine) -> int:
        """Collect Highlander AI chat conversations"""
        logger.info("Collecting Highlander chats...")
        
        try:
            query = """
            SELECT 
                h.id,
                h.user_id,
                h.session_id,
                h.message as user_message,
                h.response as ai_response,
                h.context,
                h.category,
                h.created_at,
                u.username
            FROM highlander_chat h
            LEFT JOIN users u ON h.user_id = u.id
            ORDER BY h.created_at
            """
            
            df = pd.read_sql_query(query, engine)
            
            if df.empty:
                logger.info("No Highlander chats found")
                return 0
            
            conversations = []
            for _, row in df.iterrows():
                # Create training examples from Highlander conversations
                conversation = {
                    'conversation_id': f"highlander_{row['id']}",
                    'user_id': row['user_id'],
                    'username': row['username'] or 'anonymous',
                    'created_at': row['created_at'],
                    'category': row['category'] or 'AI Consultation',
                    'messages': [
                        {'role': 'user', 'content': row['user_message']},
                        {'role': 'assistant', 'content': row['ai_response']}
                    ]
                }
                
                # Add context if available
                if row['context']:
                    conversation['context'] = row['context']
                
                conversations.append(conversation)
            
            # Save to conversations directory
            output_file = self.output_dir / "conversations" / "highlander_conversations.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, default=str)
            
            logger.info(f"Collected {len(conversations)} Highlander conversations")
            return len(conversations)
            
        except Exception as e:
            logger.error(f"Error collecting Highlander chats: {e}")
            return 0
    
    def _collect_regular_chats(self, engine) -> int:
        """Collect regular chat conversations"""
        logger.info("Collecting regular chats...")
        
        try:
            # Get all chats with messages
            query = """
            SELECT 
                c.id as chat_id,
                c.user_id,
                c.title,
                c.created_at as chat_created,
                c.fact_sheet,
                c.strategies,
                m.id as message_id,
                m.role,
                m.content,
                m.created_at as message_created,
                u.username
            FROM chats c
            LEFT JOIN messages m ON c.id = m.chat_id
            LEFT JOIN users u ON c.user_id = u.id
            ORDER BY c.id, m.created_at
            """
            
            df = pd.read_sql_query(query, engine)
            
            # Group by chat and create conversation records
            conversations = []
            for chat_id, group in df.groupby('chat_id'):
                if group['message_id'].isna().all():
                    continue  # Skip empty chats
                
                conversation = {
                    'chat_id': chat_id,
                    'user_id': group['user_id'].iloc[0],
                    'username': group['username'].iloc[0],
                    'title': group['title'].iloc[0],
                    'created_at': group['chat_created'].iloc[0],
                    'fact_sheet': group['fact_sheet'].iloc[0],
                    'strategies': group['strategies'].iloc[0],
                    'messages': []
                }
                
                for _, msg in group.iterrows():
                    if pd.notna(msg['message_id']):
                        conversation['messages'].append({
                            'role': msg['role'],
                            'content': msg['content'],
                            'timestamp': msg['message_created']
                        })
                
                conversations.append(conversation)
            
            # Save conversations
            output_file = self.output_dir / "conversations" / "all_conversations.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Collected {len(conversations)} conversations")
            return len(conversations)
            
        except Exception as e:
            logger.error(f"Error collecting conversations: {e}")
            return 0
    
    def collect_pdf_documents(self) -> int:
        """Extract text from all PDF documents"""
        logger.info("Collecting PDF documents...")
        
        pdf_count = 0
        pdf_dirs = [
            self.data_dir / "gpt_pdfs",
            self.data_dir / "strategy_pdfs",
            self.data_dir
        ]
        
        for pdf_dir in pdf_dirs:
            if not pdf_dir.exists():
                continue
                
            for pdf_file in pdf_dir.glob("*.pdf"):
                try:
                    text_content = self.extract_pdf_text(pdf_file)
                    if text_content:
                        # Save extracted text
                        output_file = self.output_dir / "pdfs" / f"{pdf_file.stem}.txt"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(text_content)
                        
                        # Create metadata
                        metadata = {
                            'filename': pdf_file.name,
                            'source_path': str(pdf_file),
                            'extracted_at': datetime.now().isoformat(),
                            'text_length': len(text_content),
                            'hash': hashlib.md5(text_content.encode()).hexdigest()
                        }
                        
                        metadata_file = self.output_dir / "pdfs" / f"{pdf_file.stem}_metadata.json"
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        pdf_count += 1
                        logger.info(f"Processed PDF: {pdf_file.name}")
                
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
        
        return pdf_count
    
    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from a PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def collect_external_research(self) -> int:
        """Collect external research papers and articles"""
        logger.info("Collecting external research...")
        
        # Define key research topics for AI development advice
        research_topics = [
            "AI model training best practices",
            "Large language model development",
            "AI business strategy implementation", 
            "Machine learning deployment patterns",
            "AI ethics and responsible development",
            "Neural network architecture design",
            "AI product management frameworks",
            "Transformer model optimization",
            "AI team organization structures",
            "AI ROI measurement methods"
        ]
        
        research_count = 0
        
        # For now, create placeholders for research integration
        # In production, this would integrate with:
        # - arXiv API for research papers
        # - Google Scholar API
        # - Industry blogs and whitepapers
        # - Technical documentation
        
        for topic in research_topics:
            placeholder_content = {
                'topic': topic,
                'status': 'placeholder',
                'note': 'Integration with research APIs to be implemented',
                'sources_to_integrate': [
                    'arXiv papers',
                    'Google Scholar results',
                    'Industry whitepapers',
                    'Technical blog posts',
                    'Documentation and guides'
                ]
            }
            
            output_file = self.output_dir / "research" / f"{topic.replace(' ', '_')}.json"
            with open(output_file, 'w') as f:
                json.dump(placeholder_content, f, indent=2)
            
            research_count += 1
        
        return research_count
    
    def collect_user_feedback(self) -> int:
        """Collect user feedback for model improvement"""
        logger.info("Collecting user feedback...")
        
        try:
            engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Get all feedback
            query = """
            SELECT 
                f.id,
                f.user_id,
                f.username,
                f.feedback_type,
                f.subject,
                f.message,
                f.allow_followup,
                f.created_at,
                f.status,
                f.admin_notes
            FROM feedback f
            ORDER BY f.created_at DESC
            """
            
            df = pd.read_sql_query(query, engine)
            
            if not df.empty:
                feedback_data = df.to_dict('records')
                
                output_file = self.output_dir / "feedback" / "user_feedback.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(feedback_data, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"Collected {len(feedback_data)} feedback entries")
                return len(feedback_data)
            else:
                logger.info("No feedback data found")
                return 0
                
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return 0
    
    def process_and_consolidate(self, include_datasafe: bool = False) -> int:
        """Process and consolidate all collected data into training format
        include_datasafe: if True, append DataSafe threat summaries as alert examples
        """
        logger.info("Processing and consolidating data...")
        
        training_data = []
        total_tokens = 0
        
        # Process conversations into training examples
        conversations_file = self.output_dir / "conversations" / "all_conversations.json"
        if conversations_file.exists():
            with open(conversations_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            for conv in conversations:
                if conv['messages']:
                    # Create training examples from conversation pairs
                    for i in range(len(conv['messages']) - 1):
                        user_msg = conv['messages'][i]
                        ai_msg = conv['messages'][i + 1]
                        
                        if user_msg['role'] == 'user' and ai_msg['role'] == 'assistant':
                            training_example = {
                                'input': user_msg['content'],
                                'output': ai_msg['content'],
                                'context': {
                                    'chat_id': conv['chat_id'],
                                    'user_type': 'real_user',
                                    'timestamp': user_msg['timestamp'],
                                    'fact_sheet': conv['fact_sheet'],
                                    'strategies': conv['strategies']
                                },
                                'source': 'user_conversation'
                            }
                            training_data.append(training_example)
                            total_tokens += len(user_msg['content'].split()) + len(ai_msg['content'].split())
        
        # Process PDFs into knowledge base entries
        pdf_dir = self.output_dir / "pdfs"
        for txt_file in pdf_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks for training
                chunks = self.split_text_into_chunks(content, max_tokens=512)
                for i, chunk in enumerate(chunks):
                    training_example = {
                        'input': f"Provide insights about AI development based on this content: {chunk[:200]}...",
                        'output': chunk,
                        'context': {
                            'source_file': txt_file.name,
                            'chunk_index': i
                        },
                        'source': 'pdf_document'
                    }
                    training_data.append(training_example)
                    total_tokens += len(chunk.split())
            
            except Exception as e:
                logger.error(f"Error processing {txt_file}: {e}")
        
        # Optionally include DataSafe threats as training examples (alert-style)
        if include_datasafe:
            try:
                ds_examples = self._export_datasafe_threats()
                training_data.extend(ds_examples)
                total_tokens += sum(len(ex['input'].split()) + len(ex['output'].split()) for ex in ds_examples)
            except Exception as e:
                logger.error(f"Error exporting DataSafe threats: {e}")

        # Save consolidated training data
        output_file = self.output_dir / "processed" / "training_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Create dataset statistics - only real data
        stats = {
            'total_examples': len(training_data),
            'total_tokens': total_tokens,
            'sources': {
                'user_conversations': len([x for x in training_data if x['source'] == 'user_conversation']),
                'pdf_documents': len([x for x in training_data if x['source'] == 'pdf_document']),
                'datasafe_threats': len([x for x in training_data if x.get('source') == 'datasafe_threat'])
            },
            'created_at': datetime.now().isoformat()
        }
        
        # If no real data, return early
        if len(training_data) == 0:
            logger.info("No real training data found - returning empty dataset")
            stats_file = self.output_dir / "processed" / "dataset_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            return 0
        
        stats_file = self.output_dir / "processed" / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Created {len(training_data)} training examples with {total_tokens} total tokens")
        return total_tokens

    def _export_datasafe_threats(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Export recent DataSafe threats into instruction-output pairs.
        Uses threat summary as the target output (no fabricated labels).
        """
        examples: List[Dict[str, Any]] = []
        try:
            import sys
            # Add project root to path for DataSafe imports
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            sys.path.insert(0, project_root)
            
            # Temporarily change to project root for correct DB path resolution
            original_cwd = os.getcwd()
            os.chdir(project_root)
            
            from datasafe.storage.sqlite_store import query_recent, ensure_schema  # type: ignore
            ensure_schema()
            threats = query_recent(limit=limit)
            
            # Restore original working directory
            os.chdir(original_cwd)
        except Exception as e:
            logger.error(f"Failed to read DataSafe threats: {e}")
            return examples

        # Save a raw export for traceability
        export_path = self.output_dir / "datasafe" / "threats_export.json"
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(threats, f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

        for t in threats:
            title = t.get('title') or ''
            summary = t.get('summary') or ''
            if not summary:
                continue
            severity = t.get('severity') or ''
            risk = t.get('risk') or {}
            threats_list = t.get('threats') or []
            sectors = t.get('sectors') or []
            iocs = t.get('iocs') or {}

            input_text = (
                f"Create a concise newsroom security alert based on the following threat record.\n"
                f"Title: {title}\n"
                f"Severity: {severity}\n"
                f"Risk: digital={risk.get('digital', '')} physical={risk.get('physical', '')}\n"
                f"Threats: {', '.join(threats_list)}\n"
                f"Sectors: {', '.join(sectors)}\n"
                f"IOCs: {', '.join([k for k in (iocs.keys() if isinstance(iocs, dict) else [])]) or 'None'}\n"
                f"Context summary: {summary[:500]}"
            )

            examples.append({
                'input': input_text,
                'output': summary,  # use provided summary; do not fabricate
                'context': {
                    'severity': severity,
                    'risk': risk,
                    'source': t.get('source'),
                    'url': t.get('url')
                },
                'source': 'datasafe_threat'
            })

        logger.info(f"Exported {len(examples)} DataSafe threat examples")
        return examples
    
    def split_text_into_chunks(self, text: str, max_tokens: int = 512) -> List[str]:
        """Split text into smaller chunks for training"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_tokens):
            chunk = ' '.join(words[i:i + max_tokens])
            chunks.append(chunk)
        
        return chunks
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about collected data"""
        stats_file = self.output_dir / "processed" / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {}

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    results = collector.collect_all_data()
    print(f"Data collection complete: {results}") 