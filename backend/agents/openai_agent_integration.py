"""
OpenAI ChatGPT Agents Integration
================================

Integration with OpenAI's ChatGPT Agents framework using the Assistants API
with built-in tools for web search, code execution, and data analysis.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIAgentIntegration:
    """Integration with OpenAI ChatGPT Agents framework"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.assistants = {}
        
        # Initialize assistants for different domains
        self._initialize_assistants()
    
    def _initialize_assistants(self):
        """Initialize ChatGPT assistants for different domains"""
        
        # MediaMap Assistant
        try:
            mediamap_assistant = self.client.beta.assistants.create(
                name="MediaMap Business Intelligence Agent",
                instructions="""
                You are a specialized AI agent for media industry business intelligence. Your role is to:

                1. Analyze media industry data, trends, and business patterns
                2. Provide strategic insights for media companies
                3. Identify opportunities for AI adoption and digital transformation
                4. Generate actionable business recommendations
                5. Monitor industry developments and competitive landscape

                Focus on:
                - Media business models and revenue strategies
                - AI and technology adoption in media
                - Audience engagement and analytics
                - Content creation and distribution
                - Industry trends and market opportunities

                Always provide data-driven insights with specific recommendations.
                """,
                model="gpt-4-turbo-preview",
                tools=[
                    {"type": "code_interpreter"},
                    {"type": "file_search"}
                ]
            )
            self.assistants["mediamap"] = mediamap_assistant
            logger.info("✅ MediaMap ChatGPT Agent initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaMap assistant: {e}")
        
        # HealthPIN Assistant
        try:
            healthpin_assistant = self.client.beta.assistants.create(
                name="HealthPIN Clinical Intelligence Agent",
                instructions="""
                You are a specialized AI agent for healthcare clinical intelligence. Your role is to:

                1. Analyze healthcare data, clinical trends, and patient care patterns
                2. Provide clinical insights and evidence-based recommendations
                3. Identify opportunities for AI adoption in healthcare
                4. Generate actionable clinical and operational recommendations
                5. Monitor medical research and healthcare technology developments

                Focus on:
                - Clinical care protocols and patient outcomes
                - Healthcare technology and AI applications
                - Medical research and evidence-based practices
                - Patient safety and quality improvement
                - Healthcare operations and efficiency

                Always provide evidence-based insights with clinical recommendations.
                Ensure all recommendations comply with medical best practices and regulations.
                """,
                model="gpt-4-turbo-preview",
                tools=[
                    {"type": "code_interpreter"},
                    {"type": "file_search"}
                ]
            )
            self.assistants["healthpin"] = healthpin_assistant
            logger.info("✅ HealthPIN ChatGPT Agent initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize HealthPIN assistant: {e}")
    
    def analyze_data_with_agent(self, agent_type: str, data: Dict[str, Any], analysis_type: str = "insights") -> Dict[str, Any]:
        """Analyze data using ChatGPT Agent"""
        
        if agent_type not in self.assistants:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        assistant = self.assistants[agent_type]
        
        # Prepare the analysis prompt based on type
        if analysis_type == "insights":
            prompt = self._create_insights_prompt(data)
        elif analysis_type == "recommendations":
            prompt = self._create_recommendations_prompt(data)
        elif analysis_type == "trends":
            prompt = self._create_trends_prompt(data)
        else:
            prompt = self._create_general_analysis_prompt(data, analysis_type)
        
        try:
            # Create a thread for this analysis
            thread = self.client.beta.threads.create()
            
            # Add the message to the thread
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for completion
            while run.status in ['queued', 'in_progress', 'cancelling']:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                # Get the response
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                
                response = messages.data[0].content[0].text.value
                
                return {
                    "success": True,
                    "analysis": response,
                    "agent_type": agent_type,
                    "analysis_type": analysis_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"Analysis failed with status: {run.status}",
                    "agent_type": agent_type
                }
                
        except Exception as e:
            logger.error(f"Error in ChatGPT Agent analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_type": agent_type
            }
    
    def _create_insights_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for insights analysis"""
        return f"""
        Analyze the following data and extract key insights:

        Data: {json.dumps(data, indent=2)}

        Please provide:
        1. Key insights and patterns identified
        2. Confidence levels for each insight
        3. Supporting evidence from the data
        4. Implications and significance
        5. Areas requiring further investigation

        Format your response as structured insights with clear categories and actionable findings.
        """
    
    def _create_recommendations_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for recommendations analysis"""
        return f"""
        Based on the following data, generate actionable recommendations:

        Data: {json.dumps(data, indent=2)}

        Please provide:
        1. Strategic recommendations with clear rationale
        2. Implementation priorities and timelines
        3. Expected outcomes and benefits
        4. Potential risks and mitigation strategies
        5. Success metrics and KPIs

        Focus on practical, implementable recommendations that drive value.
        """
    
    def _create_trends_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for trends analysis"""
        return f"""
        Analyze the following data for trends and patterns:

        Data: {json.dumps(data, indent=2)}

        Please identify:
        1. Emerging trends and their significance
        2. Historical patterns and evolution
        3. Future projections and implications
        4. Market dynamics and competitive factors
        5. Opportunities and threats

        Provide trend analysis with supporting data and forward-looking insights.
        """
    
    def _create_general_analysis_prompt(self, data: Dict[str, Any], analysis_type: str) -> str:
        """Create general analysis prompt"""
        return f"""
        Perform {analysis_type} analysis on the following data:

        Data: {json.dumps(data, indent=2)}

        Provide comprehensive analysis including:
        1. Key findings and observations
        2. Data interpretation and context
        3. Implications and significance
        4. Recommendations or next steps
        5. Areas for further investigation

        Ensure the analysis is thorough, accurate, and actionable.
        """
    
    def get_agent_capabilities(self, agent_type: str) -> Dict[str, Any]:
        """Get capabilities of a specific agent"""
        if agent_type not in self.assistants:
            return {"error": f"Agent {agent_type} not found"}
        
        assistant = self.assistants[agent_type]
        
        return {
            "name": assistant.name,
            "model": assistant.model,
            "tools": [tool.type for tool in assistant.tools],
            "instructions": assistant.instructions,
            "created_at": assistant.created_at,
                "capabilities": [
                "Code execution for data analysis",
                "File search and analysis",
                "Natural language processing",
                "Pattern recognition",
                "Insight generation",
                "Recommendation creation",
                "Data interpretation and analysis"
            ]
        }
    
    def list_available_agents(self) -> List[Dict[str, Any]]:
        """List all available agents and their capabilities"""
        agents = []
        
        for agent_type, assistant in self.assistants.items():
            agents.append({
                "type": agent_type,
                "name": assistant.name,
                "model": assistant.model,
                "tools": [tool.type for tool in assistant.tools],
                "created_at": assistant.created_at
            })
        
        return agents
    
    def update_agent_instructions(self, agent_type: str, new_instructions: str) -> bool:
        """Update agent instructions"""
        if agent_type not in self.assistants:
            return False
        
        try:
            assistant = self.assistants[agent_type]
            self.client.beta.assistants.update(
                assistant_id=assistant.id,
                instructions=new_instructions
            )
            logger.info(f"✅ Updated instructions for {agent_type} agent")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update {agent_type} agent instructions: {e}")
            return False
    
    def delete_agent(self, agent_type: str) -> bool:
        """Delete an agent"""
        if agent_type not in self.assistants:
            return False
        
        try:
            assistant = self.assistants[agent_type]
            self.client.beta.assistants.delete(assistant_id=assistant.id)
            del self.assistants[agent_type]
            logger.info(f"✅ Deleted {agent_type} agent")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete {agent_type} agent: {e}")
            return False

# Global instance
openai_agent_integration = None

def get_openai_agent_integration() -> OpenAIAgentIntegration:
    """Get or create the global OpenAI agent integration instance"""
    global openai_agent_integration
    
    if openai_agent_integration is None:
        try:
            openai_agent_integration = OpenAIAgentIntegration()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI agent integration: {e}")
            return None
    
    return openai_agent_integration
