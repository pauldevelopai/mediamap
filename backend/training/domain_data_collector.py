"""
Domain-Specific Data Collection System

This module collects specialized training data for MediaMap and HealthPIN models
from various sources including databases, PDFs, and external APIs.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from sqlalchemy import create_engine, text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainDataCollector:
    """Collects domain-specific training data"""
    
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
        
        logger.info(f"DomainDataCollector initialized for {model_name}")
    
    def collect_all_data(self) -> Dict[str, Any]:
        """Collect all domain-specific training data"""
        logger.info(f"Starting data collection for {self.model_name}")
        
        stats = {
            'conversations': 0,
            'pdfs': 0,
            'research_papers': 0,
            'feedback_entries': 0,
            'total_examples': 0
        }
        
        # Collect conversations
        stats['conversations'] = self.collect_domain_conversations()
        
        # Collect PDFs
        stats['pdfs'] = self.collect_domain_pdfs()
        
        # Collect research
        stats['research_papers'] = self.collect_domain_research()
        
        # Collect feedback
        stats['feedback_entries'] = self.collect_domain_feedback()
        
        # Generate synthetic training data
        synthetic_count = self.generate_synthetic_data()
        stats['synthetic_examples'] = synthetic_count
        
        stats['total_examples'] = sum([
            stats['conversations'],
            stats['pdfs'],
            stats['research_papers'],
            stats['feedback_entries'],
            stats['synthetic_examples']
        ])
        
        logger.info(f"Data collection complete for {self.model_name}: {stats}")
        return stats
    
    def collect_domain_conversations(self) -> int:
        """Collect conversations relevant to the domain"""
        try:
            engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Get conversations from database
            query = """
            SELECT c.title, m.content, m.role, c.created_at
            FROM chats c
            JOIN messages m ON c.id = m.chat_id
            ORDER BY c.created_at DESC, m.created_at ASC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
            
            # Group messages by chat
            conversations = {}
            for row in rows:
                chat_title = row[0] or "Untitled"
                if chat_title not in conversations:
                    conversations[chat_title] = {
                        'title': chat_title,
                        'messages': [],
                        'created_at': row[3]
                    }
                conversations[chat_title]['messages'].append({
                    'content': row[1],
                    'role': row[2]
                })
            
            # Filter and format conversations for domain
            domain_conversations = []
            for chat_title, chat_data in conversations.items():
                if self._is_domain_relevant(chat_data['messages']):
                    formatted_conv = self._format_conversation(chat_data)
                    if formatted_conv:
                        domain_conversations.append(formatted_conv)
            
            # Save conversations
            output_file = self.model_dir / "conversations" / "all_conversations.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(domain_conversations, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Collected {len(domain_conversations)} domain conversations")
            return len(domain_conversations)
            
        except Exception as e:
            logger.error(f"Error collecting conversations: {e}")
            return 0
    
    def collect_domain_pdfs(self) -> int:
        """Collect and process domain-relevant PDFs"""
        pdf_count = 0
        
        # Look for PDFs in data directory
        data_dir = Path("../data")
        if data_dir.exists():
            for pdf_file in data_dir.glob("**/*.pdf"):
                if self._is_pdf_domain_relevant(pdf_file.name):
                    # Extract text and create training examples
                    text_content = self._extract_pdf_text(pdf_file)
                    if text_content:
                        # Save extracted text
                        output_file = self.model_dir / "pdfs" / f"{pdf_file.stem}.txt"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(text_content)
                        pdf_count += 1
        
        logger.info(f"Collected {pdf_count} domain PDFs")
        return pdf_count
    
    def collect_domain_research(self) -> int:
        """Collect domain-specific research data"""
        research_data = []
        
        if self.model_name == 'mediamap':
            research_data = self._collect_media_research()
        elif self.model_name == 'healthpin':
            research_data = self._collect_health_research()
        
        if research_data:
            output_file = self.model_dir / "research" / f"{self.model_name}_research.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(research_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Collected {len(research_data)} research items")
        return len(research_data)
    
    def collect_domain_feedback(self) -> int:
        """Collect domain-specific user feedback"""
        feedback_count = 0
        
        try:
            engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Get feedback from database if feedback table exists
            query = """
            SELECT feedback_text, rating, created_at
            FROM user_feedback
            WHERE created_at > date('now', '-30 days')
            """
            
            with engine.connect() as conn:
                try:
                    result = conn.execute(text(query))
                    feedback_data = []
                    
                    for row in result:
                        if self._is_feedback_domain_relevant(row[0]):
                            feedback_data.append({
                                'feedback': row[0],
                                'rating': row[1],
                                'created_at': row[2]
                            })
                    
                    if feedback_data:
                        output_file = self.model_dir / "feedback" / "user_feedback.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
                        feedback_count = len(feedback_data)
                
                except Exception:
                    # Feedback table might not exist
                    logger.info("No feedback table found")
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
        
        return feedback_count
    
    def generate_synthetic_data(self) -> int:
        """Generate synthetic training data for the domain"""
        synthetic_data = []
        
        if self.model_name == 'mediamap':
            synthetic_data = self._generate_mediamap_examples()
        elif self.model_name == 'healthpin':
            synthetic_data = self._generate_healthpin_examples()
        
        if synthetic_data:
            output_file = self.model_dir / "synthetic_training_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(synthetic_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(synthetic_data)} synthetic examples")
        return len(synthetic_data)
    
    def _is_domain_relevant(self, messages: List[Dict]) -> bool:
        """Check if conversation is relevant to the domain"""
        content = ' '.join([msg.get('content', '') for msg in messages]).lower()
        
        if self.model_name == 'mediamap':
            keywords = [
                'media', 'marketing', 'advertising', 'campaign', 'brand', 'content',
                'roi', 'audience', 'digital', 'social media', 'analytics', 'strategy',
                'engagement', 'conversion', 'impression', 'click', 'reach'
            ]
        elif self.model_name == 'healthpin':
            keywords = [
                'health', 'medical', 'patient', 'clinical', 'diagnosis', 'treatment',
                'healthcare', 'medicine', 'doctor', 'hospital', 'symptom', 'therapy',
                'prescription', 'wellness', 'disease', 'condition'
            ]
        else:
            return True
        
        return any(keyword in content for keyword in keywords)
    
    def _is_pdf_domain_relevant(self, filename: str) -> bool:
        """Check if PDF is relevant to the domain"""
        filename_lower = filename.lower()
        
        if self.model_name == 'mediamap':
            keywords = ['media', 'marketing', 'advertising', 'digital', 'content', 'brand']
        elif self.model_name == 'healthpin':
            keywords = ['health', 'medical', 'clinical', 'healthcare', 'medicine']
        else:
            return True
        
        return any(keyword in filename_lower for keyword in keywords)
    
    def _is_feedback_domain_relevant(self, feedback_text: str) -> bool:
        """Check if feedback is relevant to the domain"""
        return self._is_domain_relevant([{'content': feedback_text}])
    
    def _format_conversation(self, chat_data: Dict) -> Optional[Dict]:
        """Format conversation for training"""
        messages = chat_data['messages']
        if len(messages) < 2:
            return None
        
        # Find user-assistant pairs
        formatted_pairs = []
        current_user_msg = None
        
        for msg in messages:
            if msg['role'] == 'user':
                current_user_msg = msg['content']
            elif msg['role'] == 'assistant' and current_user_msg:
                formatted_pairs.append({
                    'input': current_user_msg,
                    'output': msg['content']
                })
                current_user_msg = None
        
        return {
            'title': chat_data['title'],
            'pairs': formatted_pairs,
            'created_at': chat_data['created_at']
        } if formatted_pairs else None
    
    def _extract_pdf_text(self, pdf_file: Path) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text from {pdf_file}: {e}")
            return ""
    
    def _collect_media_research(self) -> List[Dict]:
        """Collect media industry research data"""
        return [
            {
                'question': 'What are the key metrics for measuring digital marketing ROI?',
                'answer': 'Key digital marketing ROI metrics include: Cost Per Acquisition (CPA), Customer Lifetime Value (CLV), Return on Ad Spend (ROAS), Click-Through Rate (CTR), Conversion Rate, and Attribution modeling across touchpoints.'
            },
            {
                'question': 'How do you optimize content for different social media platforms?',
                'answer': 'Platform optimization involves: Instagram - visual storytelling with hashtags; LinkedIn - professional insights and industry content; Twitter - real-time engagement and trending topics; Facebook - community building and detailed targeting; TikTok - short-form creative content with trending sounds.'
            },
            {
                'question': 'What are effective strategies for brand awareness campaigns?',
                'answer': 'Effective brand awareness strategies include: consistent visual identity across channels, influencer partnerships, content marketing, social media engagement, PR initiatives, event marketing, and targeted advertising with broad reach objectives.'
            },
            {
                'question': 'How do you measure campaign effectiveness across multiple channels?',
                'answer': 'Multi-channel campaign measurement requires: unified tracking with UTM parameters, cross-device attribution modeling, marketing mix modeling, incrementality testing, and integrated analytics platforms that provide holistic performance views.'
            },
            {
                'question': 'What are best practices for audience segmentation?',
                'answer': 'Audience segmentation best practices include: demographic analysis, behavioral tracking, psychographic profiling, purchase history analysis, engagement patterns, lookalike modeling, and dynamic segmentation based on real-time data.'
            }
        ]
    
    def _collect_health_research(self) -> List[Dict]:
        """Collect healthcare research data"""
        return [
            {
                'question': 'What are the key components of patient-centered care?',
                'answer': 'Patient-centered care includes: respect for patient preferences and values, coordination of care across providers, information sharing and communication, physical comfort, emotional support, involvement of family and friends, and continuity of care.'
            },
            {
                'question': 'How do you implement evidence-based medicine in clinical practice?',
                'answer': 'Evidence-based medicine implementation involves: systematic literature review, critical appraisal of research, integration of clinical expertise with best evidence, consideration of patient values and circumstances, and continuous quality improvement.'
            },
            {
                'question': 'What are effective strategies for chronic disease management?',
                'answer': 'Chronic disease management strategies include: patient education and self-management support, care coordination, regular monitoring and follow-up, medication management, lifestyle interventions, and multidisciplinary team approaches.'
            },
            {
                'question': 'How do you ensure healthcare quality and safety?',
                'answer': 'Healthcare quality and safety measures include: standardized protocols, error reporting systems, staff training and competency assessment, patient safety indicators, quality metrics monitoring, and continuous improvement processes.'
            },
            {
                'question': 'What are key considerations for healthcare technology adoption?',
                'answer': 'Healthcare technology adoption considerations include: clinical workflow integration, user training and support, data security and privacy, interoperability with existing systems, cost-benefit analysis, and patient acceptance and engagement.'
            }
        ]
    
    def _generate_mediamap_examples(self) -> List[Dict]:
        """Generate synthetic MediaMap training examples"""
        return [
            {
                'input': 'How can I improve my social media engagement rates?',
                'output': 'To improve social media engagement rates: 1) Post consistently at optimal times for your audience, 2) Create interactive content like polls and questions, 3) Use relevant hashtags and trending topics, 4) Respond promptly to comments and messages, 5) Share user-generated content, 6) Post high-quality visuals and videos, 7) Tell authentic stories that resonate with your brand values.'
            },
            {
                'input': 'What\'s the best way to track marketing attribution?',
                'output': 'Effective marketing attribution tracking involves: 1) Implement UTM parameters for all campaigns, 2) Use multi-touch attribution models, 3) Set up conversion tracking across all channels, 4) Utilize marketing analytics platforms like Google Analytics 4, 5) Consider incrementality testing, 6) Track both online and offline conversions, 7) Regular attribution model validation and optimization.'
            },
            {
                'input': 'How do I calculate customer lifetime value for my business?',
                'output': 'Calculate Customer Lifetime Value (CLV) using: CLV = (Average Purchase Value × Purchase Frequency × Customer Lifespan). For subscription businesses: CLV = (Monthly Recurring Revenue × Gross Margin %) / Monthly Churn Rate. Consider factors like acquisition costs, retention rates, and upselling opportunities to optimize CLV.'
            },
            {
                'input': 'What are effective content marketing strategies for B2B companies?',
                'output': 'B2B content marketing strategies include: 1) Create educational whitepapers and case studies, 2) Develop thought leadership content, 3) Use LinkedIn for professional networking, 4) Host webinars and virtual events, 5) Optimize for search with industry keywords, 6) Create solution-focused content addressing pain points, 7) Leverage email marketing for nurturing leads.'
            }
        ]
    
    def _generate_healthpin_examples(self) -> List[Dict]:
        """Generate synthetic HealthPIN training examples"""
        return [
            {
                'input': 'What are the key principles of patient safety in healthcare?',
                'output': 'Key patient safety principles include: 1) Create a culture of safety with open communication, 2) Implement standardized protocols and checklists, 3) Use technology to reduce errors (e.g., electronic prescribing), 4) Encourage error reporting without blame, 5) Provide ongoing staff training, 6) Ensure proper hand hygiene and infection control, 7) Verify patient identity before procedures, 8) Maintain clear communication during handoffs.'
            },
            {
                'input': 'How do you implement effective chronic disease management programs?',
                'output': 'Effective chronic disease management includes: 1) Patient education and self-management training, 2) Regular monitoring and follow-up appointments, 3) Care coordination among healthcare providers, 4) Medication adherence support, 5) Lifestyle modification programs, 6) Use of remote monitoring technology, 7) Multidisciplinary care teams, 8) Patient engagement tools and resources.'
            },
            {
                'input': 'What are best practices for healthcare data security?',
                'output': 'Healthcare data security best practices: 1) Implement HIPAA-compliant systems, 2) Use encryption for data at rest and in transit, 3) Regular security audits and vulnerability assessments, 4) Staff training on privacy and security, 5) Access controls and user authentication, 6) Secure backup and disaster recovery plans, 7) Monitor for unauthorized access, 8) Regular software updates and patches.'
            },
            {
                'input': 'How do you measure quality of care in healthcare settings?',
                'output': 'Quality of care measurement involves: 1) Clinical outcome indicators (mortality, readmission rates), 2) Patient satisfaction surveys, 3) Process measures (adherence to guidelines), 4) Safety indicators (infection rates, medication errors), 5) Efficiency metrics (length of stay, wait times), 6) Preventive care measures, 7) Care coordination effectiveness, 8) Use of evidence-based practices.'
            }
        ]

def collect_domain_data(model_name: str, db_path: str) -> Dict[str, Any]:
    """Main function to collect domain-specific data"""
    collector = DomainDataCollector(model_name, db_path)
    return collector.collect_all_data()
