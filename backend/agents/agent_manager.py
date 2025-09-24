"""
AI Agent Manager
================

Manages and schedules AI agents for continuous learning and data collection.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .base_agent import AgentConfig
from .mediamap_agent import MediaMapAgent
from .healthpin_agent import HealthPINAgent

logger = logging.getLogger(__name__)

@dataclass
class AgentStatus:
    """Status information for an agent"""
    name: str
    section: str
    is_running: bool
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    total_cycles: int
    data_collected: int
    errors: int

class AgentManager:
    """Manages all AI agents and their scheduling"""
    
    def __init__(self, storage_path: str = "backend/agents/storage"):
        self.storage_path = storage_path
        self.agents: Dict[str, Any] = {}
        self.agent_threads: Dict[str, threading.Thread] = {}
        self.running = False
        
        # Initialize storage
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info("🤖 Agent Manager initialized")
    
    def _initialize_agents(self):
        """Initialize all AI agents"""
        
        # MediaMap Agent Configuration
        mediamap_config = AgentConfig(
            name="MediaMapAgent",
            section="mediamap",
            data_sources=[
                "https://feeds.feedburner.com/oreilly/radar",
                "https://www.niemanlab.org/feed/",
                "https://www.poynter.org/feed/",
                "https://www.journalism.co.uk/feed/",
                "https://www.mediapost.com/rss/"
            ],
            learning_interval=30,  # 30 minutes
            max_data_points=1000,
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY", ""),
                "twitter": os.getenv("TWITTER_API_KEY", ""),
                "linkedin": os.getenv("LINKEDIN_API_KEY", "")
            },
            storage_path=os.path.join(self.storage_path, "mediamap")
        )
        
        # HealthPIN Agent Configuration
        healthpin_config = AgentConfig(
            name="HealthPINAgent",
            section="healthpin",
            data_sources=[
                "https://www.medicalnewstoday.com/rss",
                "https://www.healthline.com/rss",
                "https://www.webmd.com/rss",
                "https://www.healthcareitnews.com/rss",
                "https://www.mobihealthnews.com/rss"
            ],
            learning_interval=45,  # 45 minutes
            max_data_points=1000,
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY", ""),
                "pubmed": os.getenv("PUBMED_API_KEY", ""),
                "clinical_trials": os.getenv("CLINICAL_TRIALS_API_KEY", "")
            },
            storage_path=os.path.join(self.storage_path, "healthpin")
        )
        
        # Create agents
        try:
            self.agents["mediamap"] = MediaMapAgent(mediamap_config)
            logger.info("✅ MediaMap agent initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaMap agent: {e}")
        
        try:
            self.agents["healthpin"] = HealthPINAgent(healthpin_config)
            logger.info("✅ HealthPIN agent initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize HealthPIN agent: {e}")
    
    def start_agents(self):
        """Start all agents in background threads"""
        if self.running:
            logger.warning("⚠️ Agents are already running")
            return
        
        self.running = True
        
        for agent_name, agent in self.agents.items():
            thread = threading.Thread(
                target=self._agent_worker,
                args=(agent_name, agent),
                daemon=True
            )
            thread.start()
            self.agent_threads[agent_name] = thread
            logger.info(f"🚀 Started {agent_name} agent")
        
        logger.info("🎉 All agents started successfully")
    
    def stop_agents(self):
        """Stop all agents"""
        if not self.running:
            logger.warning("⚠️ Agents are not running")
            return
        
        self.running = False
        
        # Wait for threads to finish
        for agent_name, thread in self.agent_threads.items():
            thread.join(timeout=5)
            logger.info(f"🛑 Stopped {agent_name} agent")
        
        self.agent_threads.clear()
        logger.info("🛑 All agents stopped")
    
    def start_agent(self, agent_name: str) -> bool:
        """Start a specific agent"""
        if agent_name not in self.agents:
            logger.error(f"❌ Agent {agent_name} not found")
            return False
        
        if agent_name in self.agent_threads and self.agent_threads[agent_name].is_alive():
            logger.warning(f"⚠️ Agent {agent_name} is already running")
            return True
        
        try:
            agent = self.agents[agent_name]
            thread = threading.Thread(
                target=self._agent_worker,
                args=(agent_name, agent),
                daemon=True
            )
            thread.start()
            self.agent_threads[agent_name] = thread
            logger.info(f"🚀 Started {agent_name} agent")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start {agent_name} agent: {e}")
            return False
    
    def stop_agent(self, agent_name: str) -> bool:
        """Stop a specific agent"""
        if agent_name not in self.agents:
            logger.error(f"❌ Agent {agent_name} not found")
            return False
        
        if agent_name not in self.agent_threads:
            logger.warning(f"⚠️ Agent {agent_name} is not running")
            return True
        
        try:
            # Set agent-specific stop flag
            if hasattr(self.agents[agent_name], 'stop_flag'):
                self.agents[agent_name].stop_flag = True
            
            # Wait for thread to finish
            thread = self.agent_threads[agent_name]
            thread.join(timeout=5)
            
            # Remove from threads dict
            del self.agent_threads[agent_name]
            logger.info(f"🛑 Stopped {agent_name} agent")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to stop {agent_name} agent: {e}")
            return False
    
    def _agent_worker(self, agent_name: str, agent: Any):
        """Worker function for each agent"""
        logger.info(f"🔄 {agent_name} agent worker started")
        
        while self.running and not agent.stop_flag:
            try:
                if agent.should_run_learning_cycle():
                    logger.info(f"🧠 {agent_name} starting learning cycle")
                    agent.run_learning_cycle()
                    logger.info(f"✅ {agent_name} learning cycle completed")
                else:
                    # Sleep for a short time before checking again
                    time.sleep(60)  # Check every minute
                    
            except Exception as e:
                logger.error(f"❌ Error in {agent_name} agent: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
        
        logger.info(f"🛑 {agent_name} agent worker stopped")
    
    def run_single_cycle(self, agent_name: str):
        """Run a single learning cycle for a specific agent"""
        if agent_name not in self.agents:
            logger.error(f"❌ Agent {agent_name} not found")
            return False
        
        try:
            agent = self.agents[agent_name]
            logger.info(f"🔄 Running single cycle for {agent_name}")
            agent.run_learning_cycle()
            logger.info(f"✅ Single cycle completed for {agent_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error running single cycle for {agent_name}: {e}")
            return False
    
    def get_agent_status(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of agents"""
        if agent_name:
            if agent_name not in self.agents:
                return {"error": f"Agent {agent_name} not found"}
            
            agent = self.agents[agent_name]
            return agent.get_status()
        
        # Return status for all agents
        status = {}
        for name, agent in self.agents.items():
            status[name] = agent.get_status()
        
        return status
    
    def get_agent_insights(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get insights from a specific agent"""
        if agent_name not in self.agents:
            logger.error(f"❌ Agent {agent_name} not found")
            return []
        
        agent = self.agents[agent_name]
        return agent.get_insights(limit)
    
    def get_agent_knowledge(self, agent_name: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Get knowledge from a specific agent"""
        if agent_name not in self.agents:
            logger.error(f"❌ Agent {agent_name} not found")
            return {}
        
        agent = self.agents[agent_name]
        return agent.get_knowledge(category)
    
    def get_mediamap_insights(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get MediaMap-specific insights"""
        if "mediamap" not in self.agents:
            return []
        
        agent = self.agents["mediamap"]
        return agent.get_media_insights(category)
    
    def get_healthpin_insights(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get HealthPIN-specific insights"""
        if "healthpin" not in self.agents:
            return []
        
        agent = self.agents["healthpin"]
        return agent.get_healthcare_insights(category)
    
    def get_business_recommendations(self) -> List[str]:
        """Get business recommendations from MediaMap agent"""
        if "mediamap" not in self.agents:
            return []
        
        agent = self.agents["mediamap"]
        return agent.get_business_recommendations()
    
    def get_clinical_recommendations(self) -> List[str]:
        """Get clinical recommendations from HealthPIN agent"""
        if "healthpin" not in self.agents:
            return []
        
        agent = self.agents["healthpin"]
        return agent.get_clinical_recommendations()
    
    def get_industry_trends(self) -> Dict[str, Any]:
        """Get industry trends from MediaMap agent"""
        if "mediamap" not in self.agents:
            return {}
        
        agent = self.agents["mediamap"]
        return agent.get_industry_trends()
    
    def get_clinical_trends(self) -> Dict[str, Any]:
        """Get clinical trends from HealthPIN agent"""
        if "healthpin" not in self.agents:
            return {}
        
        agent = self.agents["healthpin"]
        return agent.get_clinical_trends()
    
    def export_agent_data(self, agent_name: str, format: str = "json") -> str:
        """Export agent data in specified format"""
        if agent_name not in self.agents:
            return ""
        
        agent = self.agents[agent_name]
        
        if format == "json":
            # Export knowledge base
            knowledge = agent.get_knowledge()
            insights = agent.get_insights(100)  # Get last 100 insights
            
            export_data = {
                "agent_name": agent_name,
                "export_timestamp": datetime.utcnow().isoformat(),
                "knowledge_base": knowledge,
                "recent_insights": insights,
                "status": agent.get_status()
            }
            
            return json.dumps(export_data, indent=2)
        
        return ""
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        performance = {}
        
        for agent_name, agent in self.agents.items():
            status = agent.get_status()
            performance[agent_name] = {
                "data_collection_rate": status.get("total_data_collected", 0),
                "learning_cycles": status.get("learning_cycles", 0),
                "knowledge_base_size": status.get("knowledge_base_size", 0),
                "performance_metrics": status.get("performance_metrics", {}),
                "chatgpt_agent_enabled": hasattr(agent, 'openai_agent') and agent.openai_agent is not None
            }
        
        return performance
    
    def get_chatgpt_agent_capabilities(self) -> Dict[str, Any]:
        """Get ChatGPT Agent capabilities for all agents"""
        capabilities = {}
        
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'openai_agent') and agent.openai_agent:
                try:
                    capabilities[agent_name] = agent.openai_agent.get_agent_capabilities(agent.section)
                except Exception as e:
                    capabilities[agent_name] = {"error": str(e)}
            else:
                capabilities[agent_name] = {"error": "ChatGPT Agent not available"}
        
        return capabilities
    
    def get_chatgpt_recommendations(self, agent_name: str, analysis_type: str = "recommendations") -> Dict[str, Any]:
        """Get ChatGPT Agent recommendations for a specific agent"""
        if agent_name not in self.agents:
            return {"error": f"Agent {agent_name} not found"}
        
        agent = self.agents[agent_name]
        
        if not hasattr(agent, 'get_chatgpt_recommendations'):
            return {"error": "ChatGPT Agent integration not available"}
        
        return agent.get_chatgpt_recommendations(analysis_type)
    
    def analyze_with_chatgpt_agent(self, agent_name: str, data: Dict[str, Any], analysis_type: str = "insights") -> Dict[str, Any]:
        """Analyze data using ChatGPT Agent"""
        if agent_name not in self.agents:
            return {"error": f"Agent {agent_name} not found"}
        
        agent = self.agents[agent_name]
        
        if not hasattr(agent, 'openai_agent') or not agent.openai_agent:
            return {"error": "ChatGPT Agent not available"}
        
        try:
            return agent.openai_agent.analyze_data_with_agent(
                agent_type=agent.section,
                data=data,
                analysis_type=analysis_type
            )
        except Exception as e:
            return {"error": str(e)}

# Global agent manager instance
agent_manager = AgentManager()

