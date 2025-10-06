"""
HealthPIN Model Management System
================================

Extends the existing AIMAP model manager for HealthPIN-specific use cases:
1. Medical conversation understanding
2. Doctor-patient matching
3. Health record analysis
4. Family notification generation
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import existing model manager
from training.model_manager import HighlanderModelManager

logger = logging.getLogger(__name__)

class HealthPINModelManager(HighlanderModelManager):
    """HealthPIN-specific model manager extending the base manager"""
    
    def __init__(self, 
                 models_dir: str = "./healthpin_models",
                 openai_api_key: Optional[str] = None,
                 use_custom_model: bool = True):
        
        super().__init__(
            models_dir=models_dir,
            openai_api_key=openai_api_key,
            use_custom_model=use_custom_model
        )
        
        self.model_type = "HealthPIN Medical Assistant"
        self.specialization = [
            "Medical conversation understanding",
            "Doctor-patient matching", 
            "Health record analysis",
            "Family notification generation",
            "Multilingual medical support"
        ]
        
        # HealthPIN-specific performance tracking
        self.healthpin_stats = {
            'medical_conversations': 0,
            'doctor_matches': 0,
            'health_record_analyses': 0,
            'family_notifications': 0,
            'multilingual_requests': 0
        }
    
    def generate_medical_response(self, 
                                patient_message: str,
                                conversation_history: List[Dict[str, str]] = None,
                                patient_context: Dict[str, Any] = None,
                                max_length: int = 300,
                                temperature: float = 0.7) -> Tuple[str, str]:
        """Generate medical conversation response"""
        
        # Add medical context to the prompt
        if patient_context:
            context_prompt = self._build_medical_context_prompt(patient_context)
            enhanced_message = f"{context_prompt}\n\nPatient: {patient_message}"
        else:
            enhanced_message = patient_message
        
        response, source = self.generate_response(
            message=enhanced_message,
            conversation_history=conversation_history,
            max_length=max_length,
            temperature=temperature
        )
        
        self.healthpin_stats['medical_conversations'] += 1
        return response, source
    
    def generate_doctor_match(self, 
                            patient_profile: Dict[str, Any],
                            available_doctors: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
        """Generate doctor-patient matching recommendation"""
        
        # Build matching prompt
        patient_prompt = self._build_patient_profile_prompt(patient_profile)
        doctors_prompt = self._build_doctors_list_prompt(available_doctors)
        
        matching_prompt = f"""
        Patient Profile:
        {patient_prompt}
        
        Available Doctors:
        {doctors_prompt}
        
        Please recommend the best doctor match with reasoning.
        """
        
        response, source = self.generate_response(
            message=matching_prompt,
            max_length=400,
            temperature=0.3  # Lower temperature for more consistent matching
        )
        
        # Parse the response to extract match information
        match_result = self._parse_doctor_match_response(response, available_doctors)
        
        self.healthpin_stats['doctor_matches'] += 1
        return match_result, source
    
    def generate_health_record_summary(self, 
                                     health_record: Dict[str, Any],
                                     family_context: Dict[str, Any] = None) -> Tuple[str, str]:
        """Generate family-friendly health record summary"""
        
        # Build health record prompt
        record_prompt = self._build_health_record_prompt(health_record)
        
        if family_context:
            family_prompt = f"\nFamily Context: {family_context.get('relationship', 'family member')}"
            summary_prompt = f"{record_prompt}{family_prompt}\n\nGenerate a family-friendly summary of this health record."
        else:
            summary_prompt = f"{record_prompt}\n\nGenerate a family-friendly summary of this health record."
        
        response, source = self.generate_response(
            message=summary_prompt,
            max_length=250,
            temperature=0.5
        )
        
        self.healthpin_stats['health_record_analyses'] += 1
        return response, source
    
    def generate_family_notification(self, 
                                   patient_name: str,
                                   notification_type: str,
                                   health_update: str,
                                   urgency_level: str = "normal",
                                   language: str = "English") -> Tuple[str, str]:
        """Generate family notification message"""
        
        notification_prompt = f"""
        Generate a family notification for {patient_name}:
        
        Type: {notification_type}
        Update: {health_update}
        Urgency: {urgency_level}
        Language: {language}
        
        Make it clear, caring, and appropriate for family members.
        """
        
        response, source = self.generate_response(
            message=notification_prompt,
            max_length=200,
            temperature=0.6
        )
        
        self.healthpin_stats['family_notifications'] += 1
        return response, source
    
    def generate_multilingual_response(self, 
                                     message: str,
                                     target_language: str,
                                     context: str = "medical") -> Tuple[str, str]:
        """Generate response in target language"""
        
        language_prompts = {
            "isiZulu": "Respond in isiZulu (Zulu language)",
            "isiXhosa": "Respond in isiXhosa (Xhosa language)", 
            "Shona": "Respond in Shona language",
            "English": "Respond in English"
        }
        
        language_prompt = language_prompts.get(target_language, "Respond in English")
        
        multilingual_prompt = f"""
        {language_prompt}
        Context: {context}
        Message: {message}
        """
        
        response, source = self.generate_response(
            message=multilingual_prompt,
            max_length=300,
            temperature=0.7
        )
        
        self.healthpin_stats['multilingual_requests'] += 1
        return response, source
    
    def _build_medical_context_prompt(self, patient_context: Dict[str, Any]) -> str:
        """Build medical context prompt from patient information"""
        context_parts = []
        
        if patient_context.get('age'):
            context_parts.append(f"Age: {patient_context['age']}")
        if patient_context.get('gender'):
            context_parts.append(f"Gender: {patient_context['gender']}")
        if patient_context.get('medical_history'):
            context_parts.append(f"Medical History: {patient_context['medical_history']}")
        if patient_context.get('current_medications'):
            context_parts.append(f"Current Medications: {patient_context['current_medications']}")
        if patient_context.get('allergies'):
            context_parts.append(f"Allergies: {patient_context['allergies']}")
        
        return "Patient Context:\n" + "\n".join(context_parts)
    
    def _build_patient_profile_prompt(self, patient_profile: Dict[str, Any]) -> str:
        """Build patient profile prompt for doctor matching"""
        profile_parts = []
        
        if patient_profile.get('age'):
            profile_parts.append(f"Age: {patient_profile['age']}")
        if patient_profile.get('gender'):
            profile_parts.append(f"Gender: {patient_profile['gender']}")
        if patient_profile.get('symptoms'):
            profile_parts.append(f"Symptoms: {', '.join(patient_profile['symptoms'])}")
        if patient_profile.get('language_preference'):
            profile_parts.append(f"Language Preference: {patient_profile['language_preference']}")
        if patient_profile.get('location'):
            profile_parts.append(f"Location: {patient_profile['location']}")
        if patient_profile.get('cultural_preferences'):
            profile_parts.append(f"Cultural Preferences: {patient_profile['cultural_preferences']}")
        
        return "\n".join(profile_parts)
    
    def _build_doctors_list_prompt(self, doctors: List[Dict[str, Any]]) -> str:
        """Build doctors list prompt for matching"""
        doctors_parts = []
        
        for i, doctor in enumerate(doctors, 1):
            doctor_parts = [f"Doctor {i}:"]
            if doctor.get('name'):
                doctor_parts.append(f"  Name: {doctor['name']}")
            if doctor.get('specialty'):
                doctor_parts.append(f"  Specialty: {doctor['specialty']}")
            if doctor.get('languages'):
                doctor_parts.append(f"  Languages: {', '.join(doctor['languages'])}")
            if doctor.get('location'):
                doctor_parts.append(f"  Location: {doctor['location']}")
            if doctor.get('rating'):
                doctor_parts.append(f"  Rating: {doctor['rating']}")
            if doctor.get('experience_years'):
                doctor_parts.append(f"  Experience: {doctor['experience_years']} years")
            
            doctors_parts.append("\n".join(doctor_parts))
        
        return "\n\n".join(doctors_parts)
    
    def _build_health_record_prompt(self, health_record: Dict[str, Any]) -> str:
        """Build health record prompt for analysis"""
        record_parts = []
        
        if health_record.get('title'):
            record_parts.append(f"Title: {health_record['title']}")
        if health_record.get('date'):
            record_parts.append(f"Date: {health_record['date']}")
        if health_record.get('doctor'):
            record_parts.append(f"Doctor: {health_record['doctor']}")
        if health_record.get('findings'):
            record_parts.append(f"Findings: {health_record['findings']}")
        if health_record.get('diagnosis'):
            record_parts.append(f"Diagnosis: {health_record['diagnosis']}")
        if health_record.get('recommendations'):
            record_parts.append(f"Recommendations: {health_record['recommendations']}")
        
        return "Health Record:\n" + "\n".join(record_parts)
    
    def _parse_doctor_match_response(self, response: str, available_doctors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse doctor match response to extract structured information"""
        # This is a simplified parser - in production, you'd want more sophisticated parsing
        
        match_result = {
            'recommended_doctor': None,
            'match_score': 0.0,
            'reasoning': response,
            'alternatives': []
        }
        
        # Try to extract doctor name from response
        for doctor in available_doctors:
            if doctor.get('name') and doctor['name'] in response:
                match_result['recommended_doctor'] = doctor
                match_result['match_score'] = 0.8  # Default score
                break
        
        return match_result
    
    def get_healthpin_stats(self) -> Dict[str, Any]:
        """Get HealthPIN-specific statistics"""
        base_stats = self.inference_stats.copy()
        base_stats.update(self.healthpin_stats)
        
        return {
            'model_type': self.model_type,
            'specialization': self.specialization,
            'stats': base_stats,
            'last_updated': datetime.now().isoformat()
        }

# Global HealthPIN model manager instance
healthpin_model_manager = None

def get_healthpin_model_manager() -> HealthPINModelManager:
    """Get or create HealthPIN model manager instance"""
    global healthpin_model_manager
    
    if healthpin_model_manager is None:
        healthpin_model_manager = HealthPINModelManager(
            models_dir="./healthpin_models",
            use_custom_model=True
        )
    
    return healthpin_model_manager

def initialize_healthpin_model():
    """Initialize HealthPIN model for use in the application"""
    logger.info("Initializing HealthPIN model...")
    
    manager = get_healthpin_model_manager()
    
    if manager.is_model_loaded:
        logger.info("✅ HealthPIN model loaded successfully")
        return True
    else:
        logger.warning("⚠️ HealthPIN model not loaded, will use OpenAI fallback")
        return False
