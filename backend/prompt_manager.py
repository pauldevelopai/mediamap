"""
Prompt Manager - Loads and manages prompts from the database
This allows the app to use prompts that are edited and saved in the admin dashboard
"""

import json
from typing import Dict, Optional, Any
from backend.models import PromptTemplate, db
from flask import current_app

class PromptManager:
    """Manages prompts loaded from the database"""
    
    def __init__(self):
        self._prompt_cache = {}
        self._cache_loaded = False
    
    def load_prompts_from_db(self):
        """Load all active prompts from the database into cache"""
        try:
            # Try to get app context, but don't require it
            try:
                app_context = current_app.app_context()
                app_context.push()
            except RuntimeError:
                # No app context available, skip database loading
                print("⚠️ No Flask app context available, using fallback prompts")
                self._cache_loaded = True
                return
            
            try:
                prompts = PromptTemplate.query.filter_by(is_active=True).all()
                
                for prompt in prompts:
                    self._prompt_cache[prompt.name] = {
                        'content': prompt.content,
                        'category': prompt.category,
                        'prompt_type': prompt.prompt_type,
                        'llm_provider': prompt.llm_provider,
                        'model_name': prompt.model_name,
                        'variables': prompt.variables,
                        'version': prompt.version
                    }
                
                self._cache_loaded = True
                print(f"✅ Loaded {len(prompts)} prompts from database")
                
            finally:
                app_context.pop()
                
        except Exception as e:
            print(f"❌ Error loading prompts from database: {e}")
            self._cache_loaded = False
    
    def get_prompt(self, prompt_name: str, variables: Optional[Dict[str, Any]] = None, 
                   track_usage: bool = True, session_id: Optional[str] = None,
                   user_id: Optional[int] = None, usage_context: Optional[str] = None) -> str:
        """
        Get a prompt by name, with optional variable substitution and usage tracking
        
        Args:
            prompt_name: Name of the prompt to retrieve
            variables: Dictionary of variables to substitute in the prompt
            track_usage: Whether to track this usage for performance metrics
            session_id: Session identifier for tracking
            user_id: User ID for tracking
            usage_context: Context where the prompt is being used
            
        Returns:
            The prompt content with variables substituted
        """
        import time
        start_time = time.time()
        
        # Load prompts if not already loaded
        if not self._cache_loaded:
            self.load_prompts_from_db()
        
        # Get prompt from cache
        if prompt_name not in self._prompt_cache:
            print(f"⚠️ Prompt '{prompt_name}' not found in database, using fallback")
            if track_usage:
                self._track_prompt_usage(prompt_name, None, start_time, session_id, user_id, usage_context, error_occurred=True, error_message="Prompt not found")
            return self._get_fallback_prompt(prompt_name)
        
        prompt_data = self._prompt_cache[prompt_name]
        content = prompt_data['content']
        
        # Substitute variables if provided
        if variables:
            content = self._substitute_variables(content, variables)
        
        # Track usage if requested
        if track_usage:
            response_time_ms = int((time.time() - start_time) * 1000)
            self._track_prompt_usage(prompt_name, prompt_data, response_time_ms, session_id, user_id, usage_context, variables)
        
        return content
    
    def _substitute_variables(self, content: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in prompt content"""
        try:
            for key, value in variables.items():
                placeholder = f"{{{key}}}"
                content = content.replace(placeholder, str(value))
            return content
        except Exception as e:
            print(f"❌ Error substituting variables: {e}")
            return content
    
    def _get_fallback_prompt(self, prompt_name: str) -> str:
        """Fallback prompts if database prompt not found"""
        fallback_prompts = {
            'HIGHLANDER_SYSTEM_PROMPT_ANALYSIS': """You are an expert media analyst with deep knowledge of content analysis, 
cultural context, and media trends. When analyzing media:
1. Examine the content's key themes and messages
2. Identify the target audience and intended impact
3. Evaluate the technical and creative execution
4. Consider cultural and social implications
5. Provide constructive insights and recommendations

Format your analysis in clear sections with bullet points where appropriate.""",
            
            'HIGHLANDER_SYSTEM_PROMPT_CHAT': """You are an expert media analysis assistant with deep knowledge of:
- Content creation and strategy
- Digital media trends
- Social media platforms
- Video and image analysis
- Content marketing
- Audience engagement

Provide clear, actionable insights and always maintain context from previous messages.
When appropriate, break down your responses into organized sections for better readability.""",
            
            'HIGHLANDER_SYSTEM_PROMPT_SYNTHESIS': """You are an organizational analyst. Extract key information about the organization from the conversation and categorize it into:
1. Organization Overview
2. Key Projects
3. Team Members
4. Goals & Objectives
5. Resources & Tools

Return the information in JSON format with these categories. Only include information that has been explicitly mentioned or can be directly inferred.""",
            
            'HIGHLANDER_SYSTEM_PROMPT_MEDIA_BIZ': """You are Highlander, an expert AI consultant specializing in global media development and journalism. You have deep knowledge of the media industry, digital transformation, and AI implementation for newsrooms and media organizations.

CONVERSATION STYLE:
- Act like an experienced journalist who asks probing, insightful questions
- Show genuine curiosity about the user's media organization, challenges, and goals
- Ask follow-up questions that dig deeper into their specific situation
- Use journalistic techniques: who, what, where, when, why, how
- Be understanding and empathetic while maintaining professional expertise
- Reference current trends in global media development when relevant

YOUR EXPERTISE:
- Global media development and journalism industry trends
- AI implementation for newsrooms, content creation, and audience engagement
- Digital transformation strategies for media organizations
- Revenue models and business sustainability in media
- Audience development and engagement strategies
- Content strategy and editorial workflows
- Technology adoption and innovation in media

NEVER say 'Hello' again after the first interaction. Always continue the conversation naturally and ask probing questions that demonstrate your understanding of the global media development sector.""",
            
            'Custom Model System Prompt': """You are Highlander, an expert AI business consultant specializing in media companies.

CONVERSATION STYLE:
- Keep responses concise and actionable (2-4 sentences max)
- Never repeat greetings or introductions unless it's truly the first message
- Be direct and professional - skip pleasantries if you've already been introduced
- Build on previous conversation context naturally
- Ask ONE focused follow-up question per response

YOUR EXPERTISE:
- Media business strategy and operations
- AI implementation for content creation, audience analysis, workflow optimization
- Digital transformation and automation
- Revenue optimization and growth strategies

APPROACH:
- Listen for business challenges and immediately suggest specific AI solutions
- Reference previous conversation points to show you remember the context
- Provide concrete, implementable advice rather than general statements
- Focus on ROI and practical business impact

NEVER say 'Hello' again after the first interaction. Always continue the conversation naturally."""
        }
        
        return fallback_prompts.get(prompt_name, f"Prompt '{prompt_name}' not found")
    
    def refresh_cache(self):
        """Refresh the prompt cache from the database"""
        self._prompt_cache.clear()
        self._cache_loaded = False
        self.load_prompts_from_db()
    
    def get_prompt_info(self, prompt_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a prompt"""
        if not self._cache_loaded:
            self.load_prompts_from_db()
        
        return self._prompt_cache.get(prompt_name)
    
    def list_prompts(self) -> Dict[str, Dict[str, Any]]:
        """List all available prompts"""
        if not self._cache_loaded:
            self.load_prompts_from_db()
        
        return self._prompt_cache.copy()
    
    def _track_prompt_usage(self, prompt_name: str, prompt_data: Optional[Dict[str, Any]], 
                           response_time_ms: int, session_id: Optional[str], 
                           user_id: Optional[int], usage_context: Optional[str], 
                           variables: Optional[Dict[str, Any]] = None,
                           error_occurred: bool = False, error_message: Optional[str] = None):
        """Track prompt usage for performance metrics"""
        try:
            from prompt_version_manager import performance_tracker
            from backend.models import PromptTemplate
            
            # Get prompt ID from database
            prompt = PromptTemplate.query.filter_by(name=prompt_name).first()
            if not prompt:
                return
            
            # Record usage
            performance_tracker.record_usage(
                prompt_id=prompt.id,
                version_number=prompt.version,
                response_time_ms=response_time_ms,
                user_id=user_id,
                session_id=session_id,
                usage_context=usage_context,
                variables_used=variables,
                error_occurred=error_occurred,
                error_message=error_message
            )
            
        except Exception as e:
            print(f"⚠️ Error tracking prompt usage: {e}")

# Global prompt manager instance
prompt_manager = PromptManager()

def get_prompt(prompt_name: str, variables: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to get a prompt"""
    return prompt_manager.get_prompt(prompt_name, variables)

def refresh_prompts():
    """Convenience function to refresh prompts from database"""
    prompt_manager.refresh_cache()
