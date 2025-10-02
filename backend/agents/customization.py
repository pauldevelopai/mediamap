"""
ChatGPT Agents Customization System
==================================

System for customizing ChatGPT Agent instructions, behavior, and analysis parameters.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class AgentCustomization:
    """Customization settings for a ChatGPT Agent"""
    agent_name: str
    instructions: str
    analysis_focus: List[str]
    data_sources: List[str]
    analysis_types: List[str]
    response_style: str
    confidence_threshold: float
    max_insights_per_cycle: int
    custom_prompts: Dict[str, str]
    last_updated: datetime
    updated_by: str

@dataclass
class AnalysisTemplate:
    """Template for different types of analysis"""
    name: str
    description: str
    prompt_template: str
    expected_output_format: str
    use_cases: List[str]

class AgentCustomizationManager:
    """Manages ChatGPT Agent customizations"""
    
    def __init__(self, storage_path: str = "backend/agents/customization"):
        self.storage_path = storage_path
        self.customizations: Dict[str, AgentCustomization] = {}
        self.analysis_templates: Dict[str, AnalysisTemplate] = {}
        
        # Initialize storage
        os.makedirs(storage_path, exist_ok=True)
        self.customizations_file = os.path.join(storage_path, "customizations.json")
        self.templates_file = os.path.join(storage_path, "analysis_templates.json")
        
        # Load existing data
        self._load_customizations()
        self._load_analysis_templates()
        
        # Initialize default customizations if none exist
        if not self.customizations:
            self._initialize_default_customizations()
        
        logger.info("🎨 Agent Customization Manager initialized")
    
    def _initialize_default_customizations(self):
        """Initialize default customizations for agents"""
        
        # MediaMap Agent Default Customization
        mediamap_customization = AgentCustomization(
            agent_name="mediamap",
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
            analysis_focus=[
                "business_models",
                "technology_adoption",
                "market_trends",
                "competitive_analysis",
                "revenue_strategies"
            ],
            data_sources=[
                "rss_feeds",
                "news_sites",
                "social_media",
                "industry_reports"
            ],
            analysis_types=[
                "insights",
                "recommendations",
                "trends",
                "competitive_analysis"
            ],
            response_style="professional_business",
            confidence_threshold=0.7,
            max_insights_per_cycle=10,
            custom_prompts={
                "insights": "Analyze the following media industry data and extract key business insights:",
                "recommendations": "Based on the media industry analysis, provide strategic recommendations:",
                "trends": "Identify emerging trends and patterns in the media industry:"
            },
            last_updated=datetime.utcnow(),
            updated_by="system"
        )
        
        # HealthPIN Agent Default Customization
        healthpin_customization = AgentCustomization(
            agent_name="healthpin",
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
            analysis_focus=[
                "clinical_care",
                "healthcare_technology",
                "patient_outcomes",
                "medical_research",
                "operational_efficiency"
            ],
            data_sources=[
                "medical_news",
                "research_feeds",
                "healthcare_tech",
                "clinical_guidelines"
            ],
            analysis_types=[
                "clinical_insights",
                "treatment_recommendations",
                "research_trends",
                "quality_improvement"
            ],
            response_style="clinical_professional",
            confidence_threshold=0.8,
            max_insights_per_cycle=8,
            custom_prompts={
                "clinical_insights": "Analyze the following healthcare data and extract key clinical insights:",
                "treatment_recommendations": "Based on the clinical analysis, provide evidence-based treatment recommendations:",
                "research_trends": "Identify emerging trends in medical research and clinical practice:"
            },
            last_updated=datetime.utcnow(),
            updated_by="system"
        )
        
        self.customizations["mediamap"] = mediamap_customization
        self.customizations["healthpin"] = healthpin_customization
        
        # Save default customizations
        self._save_customizations()
        
        logger.info("🎨 Initialized default customizations for MediaMap and HealthPIN agents")
    
    def _initialize_analysis_templates(self):
        """Initialize default analysis templates"""
        
        templates = {
            "business_analysis": AnalysisTemplate(
                name="Business Analysis",
                description="Comprehensive business analysis with strategic insights",
                prompt_template="""
                Analyze the following business data and provide:
                1. Key business insights and patterns
                2. Strategic opportunities and threats
                3. Actionable recommendations
                4. Market implications
                
                Data: {data}
                """,
                expected_output_format="Structured analysis with numbered insights and recommendations",
                use_cases=["market_analysis", "competitive_intelligence", "strategic_planning"]
            ),
            
            "clinical_analysis": AnalysisTemplate(
                name="Clinical Analysis",
                description="Evidence-based clinical analysis and recommendations",
                prompt_template="""
                Analyze the following clinical data and provide:
                1. Clinical insights and patterns
                2. Evidence-based recommendations
                3. Patient safety considerations
                4. Quality improvement opportunities
                
                Data: {data}
                """,
                expected_output_format="Clinical analysis with evidence-based recommendations",
                use_cases=["patient_care", "clinical_decision_support", "quality_improvement"]
            ),
            
            "trend_analysis": AnalysisTemplate(
                name="Trend Analysis",
                description="Identify and analyze emerging trends",
                prompt_template="""
                Analyze the following data for trends and provide:
                1. Emerging trends and patterns
                2. Trend significance and implications
                3. Future projections
                4. Strategic recommendations based on trends
                
                Data: {data}
                """,
                expected_output_format="Trend analysis with future projections and recommendations",
                use_cases=["market_trends", "technology_trends", "industry_analysis"]
            )
        }
        
        self.analysis_templates = templates
        self._save_analysis_templates()
        
        logger.info("🎨 Initialized analysis templates")
    
    def get_agent_customization(self, agent_name: str) -> Optional[AgentCustomization]:
        """Get customization for a specific agent"""
        return self.customizations.get(agent_name)
    
    def update_agent_instructions(self, agent_name: str, new_instructions: str, updated_by: str = "user") -> bool:
        """Update agent instructions"""
        try:
            if agent_name not in self.customizations:
                logger.error(f"❌ Agent {agent_name} not found for customization")
                return False
            
            customization = self.customizations[agent_name]
            customization.instructions = new_instructions
            customization.last_updated = datetime.utcnow()
            customization.updated_by = updated_by
            
            self._save_customizations()
            
            # Update the actual ChatGPT Agent
            self._update_chatgpt_agent_instructions(agent_name, new_instructions)
            
            logger.info(f"✅ Updated instructions for {agent_name} agent")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating agent instructions: {e}")
            return False
    
    def update_analysis_focus(self, agent_name: str, analysis_focus: List[str], updated_by: str = "user") -> bool:
        """Update agent analysis focus areas"""
        try:
            if agent_name not in self.customizations:
                return False
            
            customization = self.customizations[agent_name]
            customization.analysis_focus = analysis_focus
            customization.last_updated = datetime.utcnow()
            customization.updated_by = updated_by
            
            self._save_customizations()
            
            logger.info(f"✅ Updated analysis focus for {agent_name} agent")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating analysis focus: {e}")
            return False
    
    def update_custom_prompts(self, agent_name: str, custom_prompts: Dict[str, str], updated_by: str = "user") -> bool:
        """Update custom prompts for agent"""
        try:
            if agent_name not in self.customizations:
                return False
            
            customization = self.customizations[agent_name]
            customization.custom_prompts.update(custom_prompts)
            customization.last_updated = datetime.utcnow()
            customization.updated_by = updated_by
            
            self._save_customizations()
            
            logger.info(f"✅ Updated custom prompts for {agent_name} agent")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating custom prompts: {e}")
            return False
    
    def update_analysis_parameters(self, agent_name: str, parameters: Dict[str, Any], updated_by: str = "user") -> bool:
        """Update analysis parameters"""
        try:
            if agent_name not in self.customizations:
                return False
            
            customization = self.customizations[agent_name]
            
            # Update specific parameters
            if "confidence_threshold" in parameters:
                customization.confidence_threshold = parameters["confidence_threshold"]
            if "max_insights_per_cycle" in parameters:
                customization.max_insights_per_cycle = parameters["max_insights_per_cycle"]
            if "response_style" in parameters:
                customization.response_style = parameters["response_style"]
            
            customization.last_updated = datetime.utcnow()
            customization.updated_by = updated_by
            
            self._save_customizations()
            
            logger.info(f"✅ Updated analysis parameters for {agent_name} agent")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating analysis parameters: {e}")
            return False
    
    def get_analysis_template(self, template_name: str) -> Optional[AnalysisTemplate]:
        """Get analysis template by name"""
        return self.analysis_templates.get(template_name)
    
    def create_analysis_template(self, template: AnalysisTemplate) -> bool:
        """Create new analysis template"""
        try:
            self.analysis_templates[template.name] = template
            self._save_analysis_templates()
            
            logger.info(f"✅ Created analysis template: {template.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating analysis template: {e}")
            return False
    
    def get_customized_prompt(self, agent_name: str, analysis_type: str, data: str) -> str:
        """Get customized prompt for analysis"""
        try:
            customization = self.customizations.get(agent_name)
            if not customization:
                return f"Analyze the following data: {data}"
            
            # Get custom prompt or use default
            custom_prompt = customization.custom_prompts.get(analysis_type, f"Analyze the following {analysis_type} data:")
            
            # Format the prompt with data
            return custom_prompt + f"\n\nData: {data}"
            
        except Exception as e:
            logger.error(f"❌ Error getting customized prompt: {e}")
            return f"Analyze the following data: {data}"
    
    def get_all_customizations(self) -> Dict[str, Dict[str, Any]]:
        """Get all customizations"""
        return {
            agent_name: asdict(customization) 
            for agent_name, customization in self.customizations.items()
        }
    
    def get_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all analysis templates"""
        return {
            template_name: asdict(template)
            for template_name, template in self.analysis_templates.items()
        }
    
    def reset_to_defaults(self, agent_name: str, updated_by: str = "user") -> bool:
        """Reset agent customization to defaults"""
        try:
            if agent_name == "mediamap":
                self._initialize_default_customizations()
                logger.info(f"✅ Reset {agent_name} to default customization")
                return True
            elif agent_name == "healthpin":
                self._initialize_default_customizations()
                logger.info(f"✅ Reset {agent_name} to default customization")
                return True
            else:
                logger.error(f"❌ Unknown agent: {agent_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error resetting to defaults: {e}")
            return False
    
    def _update_chatgpt_agent_instructions(self, agent_name: str, instructions: str):
        """Update the actual ChatGPT Agent instructions"""
        try:
            from .openai_agent_integration import get_openai_agent_integration
            
            openai_integration = get_openai_agent_integration()
            if openai_integration:
                success = openai_integration.update_agent_instructions(agent_name, instructions)
                if success:
                    logger.info(f"✅ Updated ChatGPT Agent instructions for {agent_name}")
                else:
                    logger.warning(f"⚠️ Failed to update ChatGPT Agent instructions for {agent_name}")
            
        except Exception as e:
            logger.error(f"❌ Error updating ChatGPT Agent instructions: {e}")
    
    def _load_customizations(self):
        """Load customizations from file"""
        try:
            if os.path.exists(self.customizations_file):
                with open(self.customizations_file, 'r') as f:
                    data = json.load(f)
                    
                for agent_name, custom_data in data.items():
                    custom_data["last_updated"] = datetime.fromisoformat(custom_data["last_updated"])
                    self.customizations[agent_name] = AgentCustomization(**custom_data)
                
                logger.info(f"🎨 Loaded {len(self.customizations)} customizations")
                
        except Exception as e:
            logger.error(f"❌ Error loading customizations: {e}")
    
    def _load_analysis_templates(self):
        """Load analysis templates from file"""
        try:
            if os.path.exists(self.templates_file):
                with open(self.templates_file, 'r') as f:
                    data = json.load(f)
                    
                for template_name, template_data in data.items():
                    self.analysis_templates[template_name] = AnalysisTemplate(**template_data)
                
                logger.info(f"🎨 Loaded {len(self.analysis_templates)} analysis templates")
            else:
                # Initialize default templates if file doesn't exist
                self._initialize_analysis_templates()
                
        except Exception as e:
            logger.error(f"❌ Error loading analysis templates: {e}")
    
    def _save_customizations(self):
        """Save customizations to file"""
        try:
            data = {}
            for agent_name, customization in self.customizations.items():
                custom_dict = asdict(customization)
                custom_dict["last_updated"] = custom_dict["last_updated"].isoformat()
                data[agent_name] = custom_dict
            
            with open(self.customizations_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error saving customizations: {e}")
    
    def _save_analysis_templates(self):
        """Save analysis templates to file"""
        try:
            data = {
                template_name: asdict(template)
                for template_name, template in self.analysis_templates.items()
            }
            
            with open(self.templates_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error saving analysis templates: {e}")

# Global customization manager instance
customization_manager = AgentCustomizationManager()



