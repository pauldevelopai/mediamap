"""
Agent Training Bridge
====================

Connects AI agents to the training pipeline by converting agent-collected data
into training-ready formats for custom model fine-tuning.

This module:
1. Extracts data from agent storage files
2. Converts agent insights and knowledge into training examples
3. Creates conversation pairs from agent interactions
4. Formats data for OpenAI fine-tuning
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """A single training example for model fine-tuning"""
    system_prompt: str
    user_input: str
    assistant_output: str
    category: str
    source: str
    confidence: float
    metadata: Dict[str, Any]

class AgentTrainingBridge:
    """Bridge between AI agents and training pipeline"""
    
    def __init__(self, 
                 agent_storage_path: str = "backend/agents/storage",
                 training_output_path: str = "backend/training_data"):
        self.agent_storage_path = Path(agent_storage_path)
        self.training_output_path = Path(training_output_path)
        self.training_output_path.mkdir(exist_ok=True)
        
        # Create subdirectories for agent-derived training data
        (self.training_output_path / "agent_conversations").mkdir(exist_ok=True)
        (self.training_output_path / "agent_insights").mkdir(exist_ok=True)
        (self.training_output_path / "agent_knowledge").mkdir(exist_ok=True)
        
        logger.info("AgentTrainingBridge initialized")
    
    def collect_all_agent_training_data(self) -> Dict[str, int]:
        """Collect training data from all agents"""
        logger.info("🚀 Starting agent training data collection...")
        
        stats = {
            'mediamap_examples': 0,
            'healthpin_examples': 0,
            'total_examples': 0,
            'conversation_pairs': 0,
            'insight_examples': 0,
            'knowledge_examples': 0
        }
        
        # Process MediaMap agent data
        mediamap_examples = self.process_agent_data('mediamap')
        stats['mediamap_examples'] = len(mediamap_examples)
        
        # Process HealthPIN agent data
        healthpin_examples = self.process_agent_data('healthpin')
        stats['healthpin_examples'] = len(healthpin_examples)
        
        # Combine all examples
        all_examples = mediamap_examples + healthpin_examples
        stats['total_examples'] = len(all_examples)
        
        # Generate conversation pairs from agent insights
        conversation_pairs = self.generate_conversation_pairs(all_examples)
        stats['conversation_pairs'] = len(conversation_pairs)
        
        # Generate insight-based examples
        insight_examples = self.generate_insight_examples()
        stats['insight_examples'] = len(insight_examples)
        
        # Generate knowledge-based examples
        knowledge_examples = self.generate_knowledge_examples()
        stats['knowledge_examples'] = len(knowledge_examples)
        
        # Save all training data
        self.save_training_data(all_examples, conversation_pairs, insight_examples, knowledge_examples)
        
        logger.info(f"✅ Agent training data collection complete: {stats}")
        return stats
    
    def process_agent_data(self, agent_name: str) -> List[TrainingExample]:
        """Process data from a specific agent"""
        logger.info(f"📊 Processing {agent_name} agent data...")
        
        examples = []
        agent_dir = self.agent_storage_path / agent_name
        
        if not agent_dir.exists():
            logger.warning(f"⚠️ Agent directory not found: {agent_dir}")
            return examples
        
        # Load agent data file
        data_file = agent_dir / f"{agent_name.title()}Agent_data.json"
        knowledge_file = agent_dir / f"{agent_name.title()}Agent_knowledge.json"
        
        # Process data points
        if data_file.exists():
            examples.extend(self._process_data_file(data_file, agent_name))
        
        # Process knowledge base
        if knowledge_file.exists():
            examples.extend(self._process_knowledge_file(knowledge_file, agent_name))
        
        logger.info(f"✅ Processed {len(examples)} examples from {agent_name}")
        return examples
    
    def _process_data_file(self, data_file: Path, agent_name: str) -> List[TrainingExample]:
        """Process agent data file into training examples"""
        examples = []
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data_points = json.load(f)
            
            for point in data_points:
                if not isinstance(point, dict):
                    continue
                
                content = point.get('content', '').strip()
                category = point.get('category', 'General')
                relevance_score = point.get('relevance_score', 0.5)
                
                if len(content) < 50:  # Skip very short content
                    continue
                
                # Create training examples based on content type
                if category == 'Business_Model':
                    examples.extend(self._create_business_examples(content, point, agent_name))
                elif category == 'Technology':
                    examples.extend(self._create_technology_examples(content, point, agent_name))
                elif category == 'Health':
                    examples.extend(self._create_health_examples(content, point, agent_name))
                else:
                    examples.extend(self._create_general_examples(content, point, agent_name))
        
        except Exception as e:
            logger.error(f"❌ Error processing data file {data_file}: {e}")
        
        return examples
    
    def _process_knowledge_file(self, knowledge_file: Path, agent_name: str) -> List[TrainingExample]:
        """Process agent knowledge file into training examples"""
        examples = []
        
        try:
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                knowledge = json.load(f)
            
            # Process insights
            insights = knowledge.get('insights', [])
            for insight in insights:
                if isinstance(insight, dict) and 'content' in insight:
                    examples.extend(self._create_insight_examples(insight, agent_name))
            
            # Process patterns
            patterns = knowledge.get('patterns', [])
            for pattern in patterns:
                if isinstance(pattern, dict) and 'description' in pattern:
                    examples.extend(self._create_pattern_examples(pattern, agent_name))
        
        except Exception as e:
            logger.error(f"❌ Error processing knowledge file {knowledge_file}: {e}")
        
        return examples
    
    def _create_business_examples(self, content: str, point: Dict, agent_name: str) -> List[TrainingExample]:
        """Create business-focused training examples"""
        examples = []
        
        # Business analysis example
        examples.append(TrainingExample(
            system_prompt=f"You are {agent_name.title()}, an AI assistant specialized in media business analysis. Provide insights based on industry data and trends.",
            user_input=f"What can you tell me about this business development: {content[:200]}...",
            assistant_output=f"Based on the latest industry data, this development shows {content}. This indicates important trends in media business models and market dynamics.",
            category="business_analysis",
            source=f"{agent_name}_agent_data",
            confidence=point.get('relevance_score', 0.7),
            metadata={
                'agent': agent_name,
                'original_category': point.get('category'),
                'timestamp': point.get('timestamp'),
                'source_url': point.get('metadata', {}).get('url', '')
            }
        ))
        
        # Strategic insight example
        examples.append(TrainingExample(
            system_prompt=f"You are {agent_name.title()}, providing strategic business insights for media organizations.",
            user_input="What are the key business trends I should be aware of?",
            assistant_output=f"One significant trend is: {content}. This reflects broader changes in how media organizations are adapting their business models.",
            category="strategic_insights",
            source=f"{agent_name}_agent_data",
            confidence=point.get('relevance_score', 0.7),
            metadata={
                'agent': agent_name,
                'original_category': point.get('category'),
                'timestamp': point.get('timestamp')
            }
        ))
        
        return examples
    
    def _create_technology_examples(self, content: str, point: Dict, agent_name: str) -> List[TrainingExample]:
        """Create technology-focused training examples"""
        examples = []
        
        examples.append(TrainingExample(
            system_prompt=f"You are {agent_name.title()}, an AI assistant with expertise in media technology and digital transformation.",
            user_input="What technology developments should media organizations pay attention to?",
            assistant_output=f"A key technology development is: {content}. This has significant implications for how media organizations operate and deliver content.",
            category="technology_analysis",
            source=f"{agent_name}_agent_data",
            confidence=point.get('relevance_score', 0.7),
            metadata={
                'agent': agent_name,
                'original_category': point.get('category'),
                'timestamp': point.get('timestamp')
            }
        ))
        
        return examples
    
    def _create_health_examples(self, content: str, point: Dict, agent_name: str) -> List[TrainingExample]:
        """Create health-focused training examples"""
        examples = []
        
        examples.append(TrainingExample(
            system_prompt=f"You are {agent_name.title()}, an AI assistant specialized in healthcare information and medical insights.",
            user_input="What health-related developments should I be aware of?",
            assistant_output=f"An important health development is: {content}. This information can help inform healthcare decisions and understanding.",
            category="health_analysis",
            source=f"{agent_name}_agent_data",
            confidence=point.get('relevance_score', 0.7),
            metadata={
                'agent': agent_name,
                'original_category': point.get('category'),
                'timestamp': point.get('timestamp')
            }
        ))
        
        return examples
    
    def _create_general_examples(self, content: str, point: Dict, agent_name: str) -> List[TrainingExample]:
        """Create general training examples"""
        examples = []
        
        examples.append(TrainingExample(
            system_prompt=f"You are {agent_name.title()}, an AI assistant that provides helpful information and analysis.",
            user_input=f"Can you explain this topic: {content[:100]}...",
            assistant_output=f"Here's what I can tell you: {content}",
            category="general_knowledge",
            source=f"{agent_name}_agent_data",
            confidence=point.get('relevance_score', 0.6),
            metadata={
                'agent': agent_name,
                'original_category': point.get('category'),
                'timestamp': point.get('timestamp')
            }
        ))
        
        return examples
    
    def _create_insight_examples(self, insight: Dict, agent_name: str) -> List[TrainingExample]:
        """Create training examples from agent insights"""
        examples = []
        
        content = insight.get('content', '')
        if len(content) < 30:
            return examples
        
        examples.append(TrainingExample(
            system_prompt=f"You are {agent_name.title()}, providing expert insights and analysis.",
            user_input="What insights can you share based on your analysis?",
            assistant_output=content,
            category="agent_insights",
            source=f"{agent_name}_agent_knowledge",
            confidence=insight.get('confidence', 0.8),
            metadata={
                'agent': agent_name,
                'insight_type': insight.get('type', 'general'),
                'timestamp': insight.get('timestamp')
            }
        ))
        
        return examples
    
    def _create_pattern_examples(self, pattern: Dict, agent_name: str) -> List[TrainingExample]:
        """Create training examples from agent patterns"""
        examples = []
        
        description = pattern.get('description', '')
        if len(description) < 30:
            return examples
        
        examples.append(TrainingExample(
            system_prompt=f"You are {agent_name.title()}, identifying and explaining patterns in data.",
            user_input="What patterns are you seeing in the data?",
            assistant_output=f"I've identified this pattern: {description}",
            category="pattern_analysis",
            source=f"{agent_name}_agent_knowledge",
            confidence=pattern.get('confidence', 0.7),
            metadata={
                'agent': agent_name,
                'pattern_type': pattern.get('type', 'general'),
                'frequency': pattern.get('frequency', 1)
            }
        ))
        
        return examples
    
    def generate_conversation_pairs(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Generate conversation pairs for training"""
        logger.info("💬 Generating conversation pairs...")
        
        conversations = []
        
        for example in examples:
            conversation = {
                "messages": [
                    {"role": "system", "content": example.system_prompt},
                    {"role": "user", "content": example.user_input},
                    {"role": "assistant", "content": example.assistant_output}
                ],
                "metadata": {
                    "category": example.category,
                    "source": example.source,
                    "confidence": example.confidence,
                    "agent_metadata": example.metadata
                }
            }
            conversations.append(conversation)
        
        return conversations
    
    def generate_insight_examples(self) -> List[Dict[str, Any]]:
        """Generate training examples from agent insights"""
        logger.info("💡 Generating insight examples...")
        
        examples = []
        
        # This would typically query a database or API for agent insights
        # For now, we'll create some example patterns
        
        insight_templates = [
            {
                "system": "You are MediaMap, providing media industry insights.",
                "user": "What trends are emerging in digital media?",
                "assistant": "Based on my analysis of recent data, I'm seeing increased investment in subscription models, with many organizations moving away from ad-dependent revenue streams."
            },
            {
                "system": "You are HealthPIN, providing healthcare insights.",
                "user": "What developments should healthcare professionals be aware of?",
                "assistant": "Recent data shows growing adoption of telemedicine platforms and increased focus on preventive care technologies."
            }
        ]
        
        for template in insight_templates:
            examples.append({
                "messages": [
                    {"role": "system", "content": template["system"]},
                    {"role": "user", "content": template["user"]},
                    {"role": "assistant", "content": template["assistant"]}
                ],
                "metadata": {
                    "category": "agent_insights",
                    "source": "generated_insights",
                    "confidence": 0.8
                }
            })
        
        return examples
    
    def generate_knowledge_examples(self) -> List[Dict[str, Any]]:
        """Generate training examples from agent knowledge bases"""
        logger.info("📚 Generating knowledge examples...")
        
        examples = []
        
        # Load knowledge from agent files and convert to training examples
        for agent_name in ['mediamap', 'healthpin']:
            knowledge_file = self.agent_storage_path / agent_name / f"{agent_name.title()}Agent_knowledge.json"
            
            if knowledge_file.exists():
                try:
                    with open(knowledge_file, 'r', encoding='utf-8') as f:
                        knowledge = json.load(f)
                    
                    # Convert knowledge entries to training examples
                    for category, items in knowledge.items():
                        if isinstance(items, list):
                            for item in items[:5]:  # Limit to prevent overwhelming
                                if isinstance(item, dict) and 'content' in item:
                                    examples.append({
                                        "messages": [
                                            {"role": "system", "content": f"You are {agent_name.title()}, sharing knowledge from your database."},
                                            {"role": "user", "content": f"What do you know about {category}?"},
                                            {"role": "assistant", "content": item['content']}
                                        ],
                                        "metadata": {
                                            "category": f"knowledge_{category}",
                                            "source": f"{agent_name}_knowledge",
                                            "confidence": item.get('confidence', 0.7)
                                        }
                                    })
                
                except Exception as e:
                    logger.error(f"❌ Error processing knowledge file for {agent_name}: {e}")
        
        return examples
    
    def save_training_data(self, 
                          examples: List[TrainingExample], 
                          conversations: List[Dict[str, Any]], 
                          insights: List[Dict[str, Any]], 
                          knowledge: List[Dict[str, Any]]):
        """Save all training data to files"""
        logger.info("💾 Saving training data...")
        
        # Save raw examples
        examples_data = []
        for example in examples:
            examples_data.append({
                "system_prompt": example.system_prompt,
                "user_input": example.user_input,
                "assistant_output": example.assistant_output,
                "category": example.category,
                "source": example.source,
                "confidence": example.confidence,
                "metadata": example.metadata
            })
        
        with open(self.training_output_path / "agent_raw_examples.json", 'w', encoding='utf-8') as f:
            json.dump(examples_data, f, indent=2, ensure_ascii=False)
        
        # Save conversation pairs (OpenAI format)
        with open(self.training_output_path / "agent_conversations.json", 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        # Save insight examples
        with open(self.training_output_path / "agent_insights.json", 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        
        # Save knowledge examples
        with open(self.training_output_path / "agent_knowledge.json", 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, indent=2, ensure_ascii=False)
        
        # Create consolidated training file
        all_training_data = conversations + insights + knowledge
        with open(self.training_output_path / "agent_consolidated_training.json", 'w', encoding='utf-8') as f:
            json.dump(all_training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(all_training_data)} total training examples")
    
    def get_training_data_stats(self) -> Dict[str, Any]:
        """Get statistics about available training data"""
        stats = {
            'agents_available': [],
            'total_data_points': 0,
            'data_by_agent': {},
            'categories': set(),
            'last_updated': None
        }
        
        # Check available agents
        if self.agent_storage_path.exists():
            for agent_dir in self.agent_storage_path.iterdir():
                if agent_dir.is_dir():
                    agent_name = agent_dir.name
                    stats['agents_available'].append(agent_name)
                    
                    # Count data points
                    data_file = agent_dir / f"{agent_name.title()}Agent_data.json"
                    if data_file.exists():
                        try:
                            with open(data_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            count = len(data) if isinstance(data, list) else 0
                            stats['data_by_agent'][agent_name] = count
                            stats['total_data_points'] += count
                            
                            # Extract categories
                            for item in data:
                                if isinstance(item, dict) and 'category' in item:
                                    stats['categories'].add(item['category'])
                        
                        except Exception as e:
                            logger.error(f"Error reading data for {agent_name}: {e}")
        
        stats['categories'] = list(stats['categories'])
        stats['last_updated'] = datetime.now().isoformat()
        
        return stats
    
    def start_continuous_training_collection(self, interval_hours: int = 24):
        """Start continuous collection of training data from agents"""
        logger.info(f"🔄 Starting continuous training data collection (every {interval_hours} hours)")
        
        # This would typically be implemented with a scheduler like APScheduler
        # For now, we'll just log the intention
        logger.info("Continuous collection would be implemented with a background scheduler")
        
        return {
            'status': 'scheduled',
            'interval_hours': interval_hours,
            'next_collection': (datetime.now() + timedelta(hours=interval_hours)).isoformat()
        }

# Convenience function for easy import
def create_training_bridge() -> AgentTrainingBridge:
    """Create and return an AgentTrainingBridge instance"""
    return AgentTrainingBridge()

if __name__ == "__main__":
    # Test the bridge
    bridge = AgentTrainingBridge()
    stats = bridge.collect_all_agent_training_data()
    print(f"Training data collection complete: {stats}")
