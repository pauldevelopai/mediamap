"""
Prompt Version Manager - Handles versioning and rollback functionality
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from backend.models import PromptTemplate, PromptVersion, PromptPerformance, db
from flask import current_app

class PromptVersionManager:
    """Manages prompt versioning and rollback functionality"""
    
    def __init__(self):
        pass
    
    def create_version(self, prompt_id: int, change_reason: str, created_by: int) -> Optional[PromptVersion]:
        """
        Create a new version of a prompt
        
        Args:
            prompt_id: ID of the prompt to version
            change_reason: Reason for creating this version
            created_by: User ID who created this version
            
        Returns:
            PromptVersion object if successful, None otherwise
        """
        try:
            with current_app.app_context():
                # Get the current prompt
                prompt = PromptTemplate.query.get(prompt_id)
                if not prompt:
                    print(f"❌ Prompt with ID {prompt_id} not found")
                    return None
                
                # Get the next version number
                next_version = self._get_next_version_number(prompt_id)
                
                # Deactivate all current versions
                PromptVersion.query.filter_by(prompt_id=prompt_id, is_active=True).update({'is_active': False})
                
                # Create new version
                version = PromptVersion(
                    prompt_id=prompt_id,
                    version_number=next_version,
                    content=prompt.content,
                    description=prompt.description,
                    variables=prompt.variables,
                    change_reason=change_reason,
                    is_active=True,
                    created_by=created_by
                )
                
                db.session.add(version)
                db.session.commit()
                
                print(f"✅ Created version {next_version} for prompt '{prompt.name}'")
                return version
                
        except Exception as e:
            print(f"❌ Error creating version: {e}")
            db.session.rollback()
            return None
    
    def rollback_to_version(self, prompt_id: int, version_number: str, created_by: int) -> bool:
        """
        Rollback a prompt to a previous version
        
        Args:
            prompt_id: ID of the prompt to rollback
            version_number: Version to rollback to
            created_by: User ID performing the rollback
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with current_app.app_context():
                # Find the version to rollback to
                version = PromptVersion.query.filter_by(
                    prompt_id=prompt_id, 
                    version_number=version_number
                ).first()
                
                if not version:
                    print(f"❌ Version {version_number} not found for prompt {prompt_id}")
                    return False
                
                # Get the current prompt
                prompt = PromptTemplate.query.get(prompt_id)
                if not prompt:
                    print(f"❌ Prompt with ID {prompt_id} not found")
                    return False
                
                # Create a new version with the rollback reason
                rollback_reason = f"Rollback to version {version_number}"
                self.create_version(prompt_id, rollback_reason, created_by)
                
                # Update the prompt with the version content
                prompt.content = version.content
                prompt.description = version.description
                prompt.variables = version.variables
                prompt.version = version_number
                prompt.updated_at = datetime.utcnow()
                
                db.session.commit()
                
                print(f"✅ Rolled back prompt '{prompt.name}' to version {version_number}")
                return True
                
        except Exception as e:
            print(f"❌ Error rolling back: {e}")
            db.session.rollback()
            return False
    
    def get_version_history(self, prompt_id: int) -> List[Dict[str, Any]]:
        """
        Get version history for a prompt
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            List of version dictionaries
        """
        try:
            with current_app.app_context():
                versions = PromptVersion.query.filter_by(prompt_id=prompt_id).order_by(
                    PromptVersion.created_at.desc()
                ).all()
                
                return [version.to_dict() for version in versions]
                
        except Exception as e:
            print(f"❌ Error getting version history: {e}")
            return []
    
    def get_current_version(self, prompt_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the current active version of a prompt
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            Version dictionary if found, None otherwise
        """
        try:
            with current_app.app_context():
                version = PromptVersion.query.filter_by(
                    prompt_id=prompt_id, 
                    is_active=True
                ).first()
                
                return version.to_dict() if version else None
                
        except Exception as e:
            print(f"❌ Error getting current version: {e}")
            return None
    
    def _get_next_version_number(self, prompt_id: int) -> str:
        """
        Get the next version number for a prompt
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            Next version number string
        """
        try:
            with current_app.app_context():
                # Get the latest version
                latest_version = PromptVersion.query.filter_by(prompt_id=prompt_id).order_by(
                    PromptVersion.created_at.desc()
                ).first()
                
                if not latest_version:
                    return "1.0"
                
                # Parse current version and increment
                try:
                    major, minor = map(int, latest_version.version_number.split('.'))
                    minor += 1
                    return f"{major}.{minor}"
                except ValueError:
                    # If version format is invalid, start from 1.0
                    return "1.0"
                    
        except Exception as e:
            print(f"❌ Error getting next version number: {e}")
            return "1.0"


class PromptPerformanceTracker:
    """Tracks performance metrics for prompts"""
    
    def __init__(self):
        pass
    
    def record_usage(self, 
                    prompt_id: int, 
                    version_number: str,
                    response_time_ms: Optional[int] = None,
                    token_count_input: Optional[int] = None,
                    token_count_output: Optional[int] = None,
                    cost_estimate: Optional[float] = None,
                    user_id: Optional[int] = None,
                    session_id: Optional[str] = None,
                    usage_context: Optional[str] = None,
                    variables_used: Optional[Dict[str, Any]] = None,
                    error_occurred: bool = False,
                    error_message: Optional[str] = None) -> bool:
        """
        Record a prompt usage with performance metrics
        
        Args:
            prompt_id: ID of the prompt used
            version_number: Version of the prompt used
            response_time_ms: Response time in milliseconds
            token_count_input: Number of input tokens
            token_count_output: Number of output tokens
            cost_estimate: Estimated cost in USD
            user_id: ID of the user (if available)
            session_id: Session identifier
            usage_context: Where the prompt was used
            variables_used: Variables that were substituted
            error_occurred: Whether an error occurred
            error_message: Error message if any
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with current_app.app_context():
                performance = PromptPerformance(
                    prompt_id=prompt_id,
                    version_number=version_number,
                    session_id=session_id,
                    user_id=user_id,
                    response_time_ms=response_time_ms,
                    token_count_input=token_count_input,
                    token_count_output=token_count_output,
                    cost_estimate=cost_estimate,
                    usage_context=usage_context,
                    variables_used=json.dumps(variables_used) if variables_used else None,
                    error_occurred=error_occurred,
                    error_message=error_message
                )
                
                db.session.add(performance)
                db.session.commit()
                
                return True
                
        except Exception as e:
            print(f"❌ Error recording performance: {e}")
            db.session.rollback()
            return False
    
    def record_user_feedback(self, 
                           prompt_id: int, 
                           version_number: str,
                           user_rating: Optional[int] = None,
                           user_feedback: Optional[str] = None,
                           was_helpful: Optional[bool] = None,
                           user_id: Optional[int] = None,
                           session_id: Optional[str] = None) -> bool:
        """
        Record user feedback for a prompt usage
        
        Args:
            prompt_id: ID of the prompt
            version_number: Version of the prompt
            user_rating: 1-5 star rating
            user_feedback: User comments
            was_helpful: Binary helpful/not helpful
            user_id: ID of the user
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with current_app.app_context():
                # Find the most recent usage record for this session
                performance = PromptPerformance.query.filter_by(
                    prompt_id=prompt_id,
                    version_number=version_number,
                    session_id=session_id
                ).order_by(PromptPerformance.created_at.desc()).first()
                
                if not performance:
                    print(f"❌ No performance record found for prompt {prompt_id}, version {version_number}")
                    return False
                
                # Update with feedback
                performance.user_rating = user_rating
                performance.user_feedback = user_feedback
                performance.was_helpful = was_helpful
                performance.user_id = user_id
                
                db.session.commit()
                
                return True
                
        except Exception as e:
            print(f"❌ Error recording feedback: {e}")
            db.session.rollback()
            return False
    
    def get_performance_stats(self, prompt_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Get performance statistics for a prompt
        
        Args:
            prompt_id: ID of the prompt
            days: Number of days to look back
            
        Returns:
            Dictionary with performance statistics
        """
        try:
            with current_app.app_context():
                from datetime import timedelta
                
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Get performance records
                records = PromptPerformance.query.filter(
                    PromptPerformance.prompt_id == prompt_id,
                    PromptPerformance.created_at >= cutoff_date
                ).all()
                
                if not records:
                    return {
                        'total_uses': 0,
                        'avg_response_time_ms': 0,
                        'total_tokens_input': 0,
                        'total_tokens_output': 0,
                        'total_cost': 0.0,
                        'error_rate': 0.0,
                        'avg_user_rating': 0.0,
                        'helpful_rate': 0.0
                    }
                
                # Calculate statistics
                total_uses = len(records)
                response_times = [r.response_time_ms for r in records if r.response_time_ms]
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                
                total_tokens_input = sum(r.token_count_input or 0 for r in records)
                total_tokens_output = sum(r.token_count_output or 0 for r in records)
                total_cost = sum(r.cost_estimate or 0 for r in records)
                
                error_count = sum(1 for r in records if r.error_occurred)
                error_rate = (error_count / total_uses) * 100 if total_uses > 0 else 0
                
                ratings = [r.user_rating for r in records if r.user_rating]
                avg_rating = sum(ratings) / len(ratings) if ratings else 0
                
                helpful_count = sum(1 for r in records if r.was_helpful is True)
                helpful_rate = (helpful_count / total_uses) * 100 if total_uses > 0 else 0
                
                return {
                    'total_uses': total_uses,
                    'avg_response_time_ms': round(avg_response_time, 2),
                    'total_tokens_input': total_tokens_input,
                    'total_tokens_output': total_tokens_output,
                    'total_cost': round(total_cost, 4),
                    'error_rate': round(error_rate, 2),
                    'avg_user_rating': round(avg_rating, 2),
                    'helpful_rate': round(helpful_rate, 2)
                }
                
        except Exception as e:
            print(f"❌ Error getting performance stats: {e}")
            return {}


# Global instances
version_manager = PromptVersionManager()
performance_tracker = PromptPerformanceTracker()




