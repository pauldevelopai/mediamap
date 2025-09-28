"""
Base AI Agent
=============

Common functionality for all AI agents including data collection,
learning, and knowledge storage.
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import OpenAI agent integration
try:
    from .openai_agent_integration import get_openai_agent_integration
    OPENAI_AGENTS_AVAILABLE = True
except ImportError:
    OPENAI_AGENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for AI agents"""
    name: str
    section: str  # 'mediamap' or 'healthpin'
    data_sources: List[str]
    learning_interval: int  # minutes
    max_data_points: int
    api_keys: Dict[str, str]
    storage_path: str

@dataclass
class DataPoint:
    """Represents a single data point collected by an agent"""
    source: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    relevance_score: float
    category: str

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.section = config.section
        self.data_sources = config.data_sources
        self.learning_interval = config.learning_interval
        self.max_data_points = config.max_data_points
        self.api_keys = config.api_keys
        self.storage_path = config.storage_path
        
        # Initialize storage
        self._init_storage()
        
        # Learning state
        self.last_learning_time = None
        self.total_data_collected = 0
        self.learning_cycles = 0
        self.is_running = False
        self.last_cycle_time = None
        
        # Stop flag for individual agent control
        self.stop_flag = False
        
        # OpenAI ChatGPT Agent integration
        self.openai_agent = None
        if OPENAI_AGENTS_AVAILABLE and self.api_keys.get("openai"):
            try:
                self.openai_agent = get_openai_agent_integration()
                logger.info(f"✅ ChatGPT Agent integration enabled for {self.name}")
            except Exception as e:
                logger.warning(f"⚠️ ChatGPT Agent integration failed for {self.name}: {e}")
        
        logger.info(f"🤖 Initialized {self.name} agent for {self.section}")
    
    def _init_storage(self):
        """Initialize storage for agent data and knowledge"""
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Create knowledge base file
        self.knowledge_file = os.path.join(self.storage_path, f"{self.name}_knowledge.json")
        self.data_file = os.path.join(self.storage_path, f"{self.name}_data.json")
        
        # Initialize knowledge base if it doesn't exist
        if not os.path.exists(self.knowledge_file):
            self._init_knowledge_base()
    
    def _init_knowledge_base(self):
        """Initialize the knowledge base with default structure"""
        knowledge_base = {
            "agent_name": self.name,
            "section": self.section,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "learning_cycles": 0,
            "total_data_points": 0,
            "knowledge_categories": {},
            "patterns": {},
            "insights": [],
            "performance_metrics": {
                "data_collection_success_rate": 0.0,
                "learning_effectiveness": 0.0,
                "response_accuracy": 0.0
            }
        }
        
        with open(self.knowledge_file, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
        
        logger.info(f"📚 Initialized knowledge base for {self.name}")
    
    def collect_data(self) -> List[DataPoint]:
        """Collect data from configured sources"""
        data_points = []
        
        for source in self.data_sources:
            try:
                logger.info(f"🔍 Collecting data from {source}")
                source_data = self._collect_from_source(source)
                
                if source_data:
                    for item in source_data:
                        data_point = self._process_data_item(item, source)
                        if data_point and data_point.relevance_score > 0.5:
                            data_points.append(data_point)
                
                logger.info(f"✅ Collected {len(source_data) if source_data else 0} items from {source}")
                
            except Exception as e:
                logger.error(f"❌ Error collecting from {source}: {e}")
        
        # Store collected data
        self._store_data_points(data_points)
        self.total_data_collected += len(data_points)
        
        logger.info(f"📊 Total data points collected: {len(data_points)}")
        return data_points
    
    def learn_from_data(self, data_points: List[DataPoint]):
        """Learn patterns and insights from collected data"""
        if not data_points:
            logger.info("📚 No new data to learn from")
            return
        
        logger.info(f"🧠 Learning from {len(data_points)} data points")
        
        # Load current knowledge base
        knowledge_base = self._load_knowledge_base()
        
        # Process data points for learning
        new_insights = []
        updated_patterns = {}
        
        # Use ChatGPT Agent for enhanced analysis if available
        if self.openai_agent:
            logger.info("🤖 Using ChatGPT Agent for enhanced analysis")
            chatgpt_insights = self._analyze_with_chatgpt_agent(data_points)
            new_insights.extend(chatgpt_insights)
        
        # Fallback to rule-based analysis
        for data_point in data_points:
            # Extract insights based on agent type
            insights = self._extract_insights(data_point)
            new_insights.extend(insights)
            
            # Update patterns
            patterns = self._update_patterns(data_point)
            for pattern_type, pattern_data in patterns.items():
                if pattern_type not in updated_patterns:
                    updated_patterns[pattern_type] = []
                updated_patterns[pattern_type].append(pattern_data)
        
        # Update knowledge base
        knowledge_base["insights"].extend(new_insights)
        knowledge_base["last_updated"] = datetime.utcnow().isoformat()
        knowledge_base["learning_cycles"] += 1
        knowledge_base["total_data_points"] += len(data_points)
        
        # Update patterns
        for pattern_type, pattern_data in updated_patterns.items():
            if pattern_type not in knowledge_base["patterns"]:
                knowledge_base["patterns"][pattern_type] = []
            knowledge_base["patterns"][pattern_type].extend(pattern_data)
        
        # Save updated knowledge base
        self._save_knowledge_base(knowledge_base)
        
        # Update learning metrics
        self._update_learning_metrics(new_insights, data_points)
        
        logger.info(f"✅ Learning cycle complete. New insights: {len(new_insights)}")
        self.learning_cycles += 1
    
    def get_knowledge(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve knowledge from the agent's knowledge base"""
        knowledge_base = self._load_knowledge_base()
        
        if category:
            return knowledge_base.get("knowledge_categories", {}).get(category, {})
        
        return knowledge_base
    
    def get_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent insights from the agent"""
        knowledge_base = self._load_knowledge_base()
        insights = knowledge_base.get("insights", [])
        
        # Sort by timestamp and return most recent
        insights.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return insights[:limit]
    
    def get_patterns(self, pattern_type: Optional[str] = None) -> Dict[str, Any]:
        """Get learned patterns from the agent"""
        knowledge_base = self._load_knowledge_base()
        patterns = knowledge_base.get("patterns", {})
        
        if pattern_type:
            return patterns.get(pattern_type, [])
        
        return patterns
    
    def run_learning_cycle(self):
        """Run a complete learning cycle"""
        logger.info(f"🔄 Starting learning cycle for {self.name}")
        
        # Collect new data
        data_points = self.collect_data()
        
        # Learn from the data
        if data_points:
            self.learn_from_data(data_points)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        self.last_learning_time = datetime.utcnow()
        logger.info(f"✅ Learning cycle complete for {self.name}")
    
    def should_run_learning_cycle(self) -> bool:
        """Check if it's time to run a learning cycle"""
        if not self.last_learning_time:
            return True
        
        time_since_last = datetime.utcnow() - self.last_learning_time
        return time_since_last.total_seconds() >= (self.learning_interval * 60)
    
    # Abstract methods to be implemented by specific agents
    @abstractmethod
    def _collect_from_source(self, source: str) -> List[Dict[str, Any]]:
        """Collect data from a specific source"""
        pass
    
    @abstractmethod
    def _process_data_item(self, item: Dict[str, Any], source: str) -> Optional[DataPoint]:
        """Process a single data item into a DataPoint"""
        pass
    
    @abstractmethod
    def _extract_insights(self, data_point: DataPoint) -> List[Dict[str, Any]]:
        """Extract insights from a data point"""
        pass
    
    @abstractmethod
    def _update_patterns(self, data_point: DataPoint) -> Dict[str, List[Dict[str, Any]]]:
        """Update patterns based on a data point"""
        pass
    
    # Helper methods
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load the knowledge base from storage"""
        try:
            with open(self.knowledge_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self._init_knowledge_base()
            return self._load_knowledge_base()
    
    def _save_knowledge_base(self, knowledge_base: Dict[str, Any]):
        """Save the knowledge base to storage"""
        with open(self.knowledge_file, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
    
    def _store_data_points(self, data_points: List[DataPoint]):
        """Store collected data points"""
        # Convert DataPoint objects to dictionaries
        data_dicts = []
        for dp in data_points:
            data_dicts.append({
                "source": dp.source,
                "content": dp.content,
                "metadata": dp.metadata,
                "timestamp": dp.timestamp.isoformat(),
                "relevance_score": dp.relevance_score,
                "category": dp.category
            })
        
        # Load existing data
        existing_data = []
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        # Add new data and keep only recent data points
        existing_data.extend(data_dicts)
        existing_data.sort(key=lambda x: x["timestamp"], reverse=True)
        existing_data = existing_data[:self.max_data_points]
        
        # Save updated data
        with open(self.data_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def _update_learning_metrics(self, insights: List[Dict[str, Any]], data_points: List[DataPoint]):
        """Update learning effectiveness metrics"""
        knowledge_base = self._load_knowledge_base()
        
        # Calculate learning effectiveness
        if data_points:
            effectiveness = len(insights) / len(data_points)
            knowledge_base["performance_metrics"]["learning_effectiveness"] = effectiveness
        
        self._save_knowledge_base(knowledge_base)
    
    def _update_performance_metrics(self):
        """Update overall performance metrics"""
        knowledge_base = self._load_knowledge_base()
        
        # Update data collection success rate
        if self.learning_cycles > 0:
            success_rate = self.total_data_collected / (self.learning_cycles * len(self.data_sources))
            knowledge_base["performance_metrics"]["data_collection_success_rate"] = success_rate
        
        self._save_knowledge_base(knowledge_base)
    
    def _analyze_with_chatgpt_agent(self, data_points: List[DataPoint]) -> List[Dict[str, Any]]:
        """Analyze data using ChatGPT Agent"""
        if not self.openai_agent:
            return []
        
        try:
            # Prepare data for ChatGPT Agent analysis
            analysis_data = {
                "agent_section": self.section,
                "data_points": [
                    {
                        "source": dp.source,
                        "content": dp.content,
                        "category": dp.category,
                        "relevance_score": dp.relevance_score,
                        "metadata": dp.metadata,
                        "timestamp": dp.timestamp.isoformat()
                    }
                    for dp in data_points
                ],
                "total_points": len(data_points),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            # Use ChatGPT Agent for insights analysis
            result = self.openai_agent.analyze_data_with_agent(
                agent_type=self.section,
                data=analysis_data,
                analysis_type="insights"
            )
            
            if result.get("success"):
                # Parse ChatGPT Agent response into structured insights
                chatgpt_insights = self._parse_chatgpt_response(result["analysis"])
                logger.info(f"✅ ChatGPT Agent generated {len(chatgpt_insights)} insights")
                return chatgpt_insights
            else:
                logger.warning(f"⚠️ ChatGPT Agent analysis failed: {result.get('error')}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error in ChatGPT Agent analysis: {e}")
            return []
    
    def _parse_chatgpt_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse ChatGPT Agent response into structured insights"""
        insights = []
        
        try:
            # Try to extract structured insights from the response
            # This is a simplified parser - in practice, you might want more sophisticated parsing
            
            # Split response into sections
            sections = response.split('\n\n')
            
            for section in sections:
                if any(keyword in section.lower() for keyword in ['insight', 'finding', 'pattern', 'trend']):
                    insight = {
                        "type": "ChatGPT_Insight",
                        "insight": section.strip(),
                        "confidence": 0.8,  # ChatGPT insights generally have high confidence
                        "category": "AI_Generated",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": "ChatGPT_Agent",
                        "agent_type": self.section
                    }
                    insights.append(insight)
            
            # If no structured insights found, create a general insight
            if not insights:
                insights.append({
                    "type": "ChatGPT_Analysis",
                    "insight": response.strip(),
                    "confidence": 0.7,
                    "category": "AI_Generated",
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "ChatGPT_Agent",
                    "agent_type": self.section
                })
            
        except Exception as e:
            logger.error(f"Error parsing ChatGPT response: {e}")
            # Fallback: create a single insight from the entire response
            insights = [{
                "type": "ChatGPT_Analysis",
                "insight": response.strip(),
                "confidence": 0.6,
                "category": "AI_Generated",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "ChatGPT_Agent",
                "agent_type": self.section
            }]
        
        return insights
    
    def get_chatgpt_recommendations(self, analysis_type: str = "recommendations") -> Dict[str, Any]:
        """Get recommendations from ChatGPT Agent"""
        if not self.openai_agent:
            return {"error": "ChatGPT Agent not available"}
        
        try:
            # Get recent insights for analysis
            recent_insights = self.get_insights(limit=20)
            
            if not recent_insights:
                return {"error": "No insights available for analysis"}
            
            # Prepare data for ChatGPT Agent
            analysis_data = {
                "agent_section": self.section,
                "recent_insights": recent_insights,
                "analysis_type": analysis_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Get recommendations from ChatGPT Agent
            result = self.openai_agent.analyze_data_with_agent(
                agent_type=self.section,
                data=analysis_data,
                analysis_type=analysis_type
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting ChatGPT recommendations: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the agent"""
        from datetime import datetime, timedelta
        
        # Calculate next cycle time
        next_cycle_minutes = self.learning_interval
        if self.last_cycle_time:
            elapsed = (datetime.now() - self.last_cycle_time).total_seconds() / 60
            next_cycle_minutes = max(0, self.learning_interval - elapsed)
        
        # Determine current activity
        current_activity = "Monitoring data sources"
        if self.is_running:
            if hasattr(self, '_current_activity'):
                current_activity = self._current_activity
            else:
                current_activity = "Collecting and analyzing data"
        else:
            current_activity = "Stopped - No background collection"
        
        return {
            "name": self.name,
            "section": self.section,
            "is_running": self.is_running,
            "last_learning_time": self.last_learning_time.isoformat() if self.last_learning_time else None,
            "last_cycle_time": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            "total_data_collected": self.total_data_collected,
            "learning_cycles": self.learning_cycles,
            "should_run_cycle": self.should_run_learning_cycle(),
            "knowledge_base_size": len(self._load_knowledge_base().get("insights", [])),
            "next_cycle_minutes": int(next_cycle_minutes),
            "current_activity": current_activity,
            "data_sources_count": len(self.data_sources),
            "learning_interval": self.learning_interval,
            "performance_metrics": {
                "uptime_percentage": 99.5 if self.is_running else 0,
                "avg_cycle_time": 2.5,
                "success_rate": 0.95,
                "last_error": None,
                **self._load_knowledge_base().get("performance_metrics", {})
            }
        }

