"""
HealthPIN AI Training System
===========================

Extends the existing AIMAP training infrastructure for HealthPIN-specific use cases:
1. Medical conversation understanding
2. Doctor-patient matching
3. Health record analysis
4. Family notification generation
5. Multi-language support (English, isiZulu, isiXhosa, Shona)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import existing training infrastructure
from backend.training.model_trainer import HighlanderModelTrainer, HighlanderDataset
from backend.training.data_collector import DataCollector
from backend.training.enhanced_data_collector import EnhancedDataCollector

logger = logging.getLogger(__name__)

class HealthPINDataCollector:
    """HealthPIN-specific data collection system"""
    
    def __init__(self, output_dir: str = "./healthpin_training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create HealthPIN-specific subdirectories
        (self.output_dir / "medical_conversations").mkdir(exist_ok=True)
        (self.output_dir / "doctor_matching").mkdir(exist_ok=True)
        (self.output_dir / "health_records").mkdir(exist_ok=True)
        (self.output_dir / "family_notifications").mkdir(exist_ok=True)
        (self.output_dir / "multilingual").mkdir(exist_ok=True)
        (self.output_dir / "medical_literature").mkdir(exist_ok=True)
        
        logger.info("HealthPINDataCollector initialized")
    
    def collect_healthpin_data(self) -> Dict[str, int]:
        """Collect HealthPIN-specific training data"""
        logger.info("Starting HealthPIN data collection...")
        
        stats = {
            'medical_conversations': 0,
            'doctor_matching_examples': 0,
            'health_record_summaries': 0,
            'family_notifications': 0,
            'multilingual_examples': 0,
            'medical_literature': 0
        }
        
        # Collect medical conversation patterns
        stats['medical_conversations'] = self._collect_medical_conversations()
        
        # Collect doctor-patient matching examples
        stats['doctor_matching_examples'] = self._collect_doctor_matching_data()
        
        # Collect health record analysis examples
        stats['health_record_summaries'] = self._collect_health_record_data()
        
        # Collect family notification examples
        stats['family_notifications'] = self._collect_family_notification_data()
        
        # Collect multilingual medical content
        stats['multilingual_examples'] = self._collect_multilingual_data()
        
        # Collect medical literature and guidelines
        stats['medical_literature'] = self._collect_medical_literature()
        
        logger.info(f"HealthPIN data collection complete: {stats}")
        return stats
    
    def _collect_medical_conversations(self) -> int:
        """Collect medical conversation patterns"""
        # This would collect real patient-doctor conversations
        # For now, we'll create structured examples
        
        medical_conversations = [
            {
                "input": "I have been experiencing chest pain for the past 3 days. It gets worse when I breathe deeply.",
                "output": "I understand you're experiencing chest pain that worsens with deep breathing. This could be related to several conditions. I recommend you see a cardiologist or visit the emergency room if the pain is severe. In the meantime, try to rest and avoid strenuous activities. Would you like me to help you find a nearby cardiologist?",
                "category": "symptom_assessment",
                "urgency": "high",
                "specialty": "cardiology"
            },
            {
                "input": "My blood pressure has been high lately. What should I do?",
                "output": "High blood pressure is a serious condition that needs monitoring. I recommend you see a general practitioner or cardiologist for proper evaluation. They may suggest lifestyle changes like reducing salt intake, regular exercise, and possibly medication. Let me help you find a suitable doctor in your area.",
                "category": "chronic_condition",
                "urgency": "medium",
                "specialty": "general_practice"
            },
            {
                "input": "I need help finding a doctor who speaks isiZulu",
                "output": "I can help you find a doctor who speaks isiZulu. Let me search for healthcare providers in your area who are fluent in isiZulu. This will ensure better communication and understanding during your consultations.",
                "category": "language_preference",
                "urgency": "low",
                "specialty": "general"
            }
        ]
        
        # Save to file
        output_file = self.output_dir / "medical_conversations" / "conversations.json"
        with open(output_file, 'w') as f:
            json.dump(medical_conversations, f, indent=2)
        
        return len(medical_conversations)
    
    def _collect_doctor_matching_data(self) -> int:
        """Collect doctor-patient matching examples"""
        matching_examples = [
            {
                "patient_profile": {
                    "age": 45,
                    "gender": "female",
                    "symptoms": ["chest pain", "shortness of breath"],
                    "language_preference": "English",
                    "location": "Johannesburg",
                    "cultural_preferences": "Western medicine"
                },
                "doctor_profile": {
                    "specialty": "Cardiology",
                    "languages": ["English", "Afrikaans"],
                    "location": "Johannesburg",
                    "cultural_competence": "Western medicine",
                    "rating": 4.8,
                    "experience_years": 15
                },
                "match_score": 0.92,
                "match_reasoning": "High match due to specialty alignment, location proximity, language compatibility, and cultural fit. The doctor has extensive experience in cardiology and is located in the same city."
            },
            {
                "patient_profile": {
                    "age": 28,
                    "gender": "male",
                    "symptoms": ["anxiety", "depression"],
                    "language_preference": "isiZulu",
                    "location": "Durban",
                    "cultural_preferences": "Traditional and Western medicine"
                },
                "doctor_profile": {
                    "specialty": "Psychiatry",
                    "languages": ["isiZulu", "English"],
                    "location": "Durban",
                    "cultural_competence": "Traditional and Western medicine",
                    "rating": 4.6,
                    "experience_years": 8
                },
                "match_score": 0.88,
                "match_reasoning": "Excellent match for mental health needs with cultural sensitivity. The doctor speaks isiZulu and understands traditional medicine approaches, which is important for this patient's cultural preferences."
            }
        ]
        
        output_file = self.output_dir / "doctor_matching" / "matching_examples.json"
        with open(output_file, 'w') as f:
            json.dump(matching_examples, f, indent=2)
        
        return len(matching_examples)
    
    def _collect_health_record_data(self) -> int:
        """Collect health record analysis examples"""
        health_record_examples = [
            {
                "health_record": {
                    "title": "Annual Physical Examination",
                    "date": "2024-01-15",
                    "doctor": "Dr. Sarah Johnson",
                    "findings": "Blood pressure: 140/90, Weight: 75kg, Height: 170cm, BMI: 26.0",
                    "diagnosis": "Hypertension, Overweight",
                    "recommendations": "Lose 5kg, reduce salt intake, regular exercise, follow-up in 3 months"
                },
                "family_summary": "Patient had their annual check-up. Doctor found high blood pressure and recommended lifestyle changes including weight loss and exercise. Follow-up appointment scheduled in 3 months.",
                "urgency": "medium",
                "key_points": ["High blood pressure detected", "Weight management needed", "Follow-up in 3 months"]
            },
            {
                "health_record": {
                    "title": "Emergency Room Visit",
                    "date": "2024-02-10",
                    "doctor": "Dr. Michael Brown",
                    "findings": "Chest pain, elevated heart rate, normal EKG",
                    "diagnosis": "Anxiety-related chest pain",
                    "recommendations": "Stress management, follow-up with GP, consider counseling"
                },
                "family_summary": "Patient visited emergency room for chest pain. Tests showed it was anxiety-related. Doctor recommended stress management and follow-up care.",
                "urgency": "low",
                "key_points": ["Chest pain was anxiety-related", "No heart problems found", "Stress management recommended"]
            }
        ]
        
        output_file = self.output_dir / "health_records" / "record_examples.json"
        with open(output_file, 'w') as f:
            json.dump(health_record_examples, f, indent=2)
        
        return len(health_record_examples)
    
    def _collect_family_notification_data(self) -> int:
        """Collect family notification examples"""
        notification_examples = [
            {
                "patient_name": "Thabo Mthembu",
                "notification_type": "health_update",
                "health_record": "Annual check-up completed",
                "family_message": "Thabo had his annual check-up today. The doctor found his blood pressure is slightly high and recommended some lifestyle changes. He's doing well overall and will have a follow-up in 3 months.",
                "urgency": "normal",
                "language": "English"
            },
            {
                "patient_name": "Nomsa Dlamini",
                "notification_type": "appointment_reminder",
                "health_record": "Cardiology appointment scheduled",
                "family_message": "Reminder: Nomsa has a cardiology appointment tomorrow at 2:00 PM at Johannesburg Heart Clinic. Please ensure she takes her medication as prescribed.",
                "urgency": "normal",
                "language": "isiZulu"
            }
        ]
        
        output_file = self.output_dir / "family_notifications" / "notification_examples.json"
        with open(output_file, 'w') as f:
            json.dump(notification_examples, f, indent=2)
        
        return len(notification_examples)
    
    def _collect_multilingual_data(self) -> int:
        """Collect multilingual medical content"""
        multilingual_examples = [
            {
                "language": "isiZulu",
                "english": "I have a headache",
                "translation": "Nginekhanda elibuhlungu",
                "context": "symptom_description"
            },
            {
                "language": "isiXhosa",
                "english": "My blood pressure is high",
                "translation": "I-blood pressure yam iphezulu",
                "context": "medical_condition"
            },
            {
                "language": "Shona",
                "english": "I need to see a doctor",
                "translation": "Ndinoda kuona chiremba",
                "context": "appointment_request"
            }
        ]
        
        output_file = self.output_dir / "multilingual" / "multilingual_examples.json"
        with open(output_file, 'w') as f:
            json.dump(multilingual_examples, f, indent=2)
        
        return len(multilingual_examples)
    
    def _collect_medical_literature(self) -> int:
        """Collect medical literature and guidelines"""
        # This would integrate with medical databases, WHO guidelines, etc.
        # For now, we'll create structured examples
        
        medical_literature = [
            {
                "title": "WHO Guidelines for Hypertension Management",
                "content": "Hypertension is a major risk factor for cardiovascular disease. Management includes lifestyle modifications and medication when necessary.",
                "category": "guidelines",
                "source": "WHO"
            },
            {
                "title": "South African Diabetes Guidelines",
                "content": "Diabetes management in South Africa requires consideration of cultural factors and access to healthcare resources.",
                "category": "guidelines",
                "source": "SA Medical Association"
            }
        ]
        
        output_file = self.output_dir / "medical_literature" / "literature.json"
        with open(output_file, 'w') as f:
            json.dump(medical_literature, f, indent=2)
        
        return len(medical_literature)

class HealthPINModelTrainer(HighlanderModelTrainer):
    """HealthPIN-specific model trainer extending the base trainer"""
    
    def __init__(self, 
                 models_dir: str = "./healthpin_models",
                 base_model: str = "microsoft/DialoGPT-medium",
                 **kwargs):
        
        # Initialize with HealthPIN-specific configuration
        super().__init__(
            models_dir=models_dir,
            base_model=base_model,
            **kwargs
        )
        
        self.model_type = "HealthPIN Medical Assistant"
        self.specialization = [
            "Medical conversation understanding",
            "Doctor-patient matching",
            "Health record analysis",
            "Family notification generation",
            "Multilingual medical support"
        ]
    
    def prepare_healthpin_training_data(self) -> List[Dict[str, str]]:
        """Prepare HealthPIN-specific training data"""
        logger.info("Preparing HealthPIN training data...")
        
        training_data = []
        
        # Load medical conversations
        conversations_file = Path("healthpin_training_data/medical_conversations/conversations.json")
        if conversations_file.exists():
            with open(conversations_file, 'r') as f:
                conversations = json.load(f)
                for conv in conversations:
                    training_data.append({
                        "input": conv["input"],
                        "output": conv["output"],
                        "category": conv.get("category", "general"),
                        "urgency": conv.get("urgency", "normal")
                    })
        
        # Load doctor matching examples
        matching_file = Path("healthpin_training_data/doctor_matching/matching_examples.json")
        if matching_file.exists():
            with open(matching_file, 'r') as f:
                matches = json.load(f)
                for match in matches:
                    patient = match["patient_profile"]
                    doctor = match["doctor_profile"]
                    input_text = f"Patient: {patient['age']} year old {patient['gender']} with {', '.join(patient['symptoms'])}. Language: {patient['language_preference']}. Location: {patient['location']}"
                    output_text = f"Recommended: {doctor['specialty']} doctor with {doctor['experience_years']} years experience. Match score: {match['match_score']:.2f}. {match['match_reasoning']}"
                    training_data.append({
                        "input": input_text,
                        "output": output_text,
                        "category": "doctor_matching",
                        "urgency": "normal"
                    })
        
        # Load health record examples
        records_file = Path("healthpin_training_data/health_records/record_examples.json")
        if records_file.exists():
            with open(records_file, 'r') as f:
                records = json.load(f)
                for record in records:
                    health_record = record["health_record"]
                    input_text = f"Health record: {health_record['title']} on {health_record['date']}. Findings: {health_record['findings']}. Diagnosis: {health_record['diagnosis']}"
                    output_text = f"Family summary: {record['family_summary']}. Key points: {', '.join(record['key_points'])}"
                    training_data.append({
                        "input": input_text,
                        "output": output_text,
                        "category": "health_record_analysis",
                        "urgency": record.get("urgency", "normal")
                    })
        
        logger.info(f"Prepared {len(training_data)} HealthPIN training examples")
        return training_data
    
    def train_healthpin_model(self, 
                            data_dir: str = "./healthpin_training_data",
                            num_epochs: int = 3,
                            learning_rate: float = 3e-5) -> bool:
        """Train HealthPIN-specific model"""
        logger.info("Starting HealthPIN model training...")
        
        # Collect HealthPIN data
        collector = HealthPINDataCollector(data_dir)
        collector.collect_healthpin_data()
        
        # Prepare training data
        training_data = self.prepare_healthpin_training_data()
        
        if len(training_data) < 50:
            logger.warning(f"Only {len(training_data)} training examples available. Consider collecting more data.")
        
        # Train the model
        success = self.train_model(
            training_data=training_data,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        if success:
            # Save HealthPIN-specific metadata
            metadata = {
                "model_type": "HealthPIN Medical Assistant",
                "training_completed_at": datetime.now().isoformat(),
                "base_model": self.base_model,
                "training_examples": len(training_data),
                "specialization": self.specialization,
                "capabilities": [
                    "Medical conversation understanding",
                    "Doctor-patient matching",
                    "Health record analysis",
                    "Family notification generation",
                    "Multilingual medical support (English, isiZulu, isiXhosa, Shona)"
                ],
                "data_sources": [
                    "Medical conversation patterns",
                    "Doctor-patient matching examples",
                    "Health record analysis",
                    "Family notification templates",
                    "Multilingual medical content"
                ]
            }
            
            metadata_file = Path(self.models_dir) / "healthpin_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("HealthPIN model training completed successfully!")
        
        return success

def train_healthpin_model():
    """Main function to train HealthPIN model"""
    logger.info("🚀 Starting HealthPIN AI Model Training...")
    
    trainer = HealthPINModelTrainer(
        models_dir="./healthpin_models",
        base_model="microsoft/DialoGPT-medium"
    )
    
    success = trainer.train_healthpin_model(
        data_dir="./healthpin_training_data",
        num_epochs=3,
        learning_rate=3e-5
    )
    
    if success:
        logger.info("✅ HealthPIN model training completed successfully!")
        return True
    else:
        logger.error("❌ HealthPIN model training failed!")
        return False

if __name__ == "__main__":
    train_healthpin_model()
