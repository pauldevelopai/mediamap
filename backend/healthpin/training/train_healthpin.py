#!/usr/bin/env python3
"""
HealthPIN Model Training Script
==============================

This script trains a HealthPIN-specific AI model using the existing AIMAP training infrastructure.
It extends the base training system with HealthPIN-specific data and capabilities.

Usage:
    python train_healthpin.py [options]

Options:
    --data-dir: Directory for training data (default: ./healthpin_training_data)
    --models-dir: Directory for trained models (default: ./healthpin_models)
    --epochs: Number of training epochs (default: 3)
    --learning-rate: Learning rate (default: 2e-5)
    --base-model: Base model to use (default: microsoft/DialoGPT-medium)
    --quick: Quick training with minimal data (default: False)
    --collect-data: Collect new training data (default: True)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .healthpin_trainer import HealthPINModelTrainer, HealthPINDataCollector
from .healthpin_model_manager import HealthPINModelManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train HealthPIN AI Model')
    parser.add_argument('--data-dir', default='./healthpin_training_data', 
                       help='Directory for training data')
    parser.add_argument('--models-dir', default='./healthpin_models',
                       help='Directory for trained models')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate for training')
    parser.add_argument('--base-model', default='microsoft/DialoGPT-medium',
                       help='Base model to use for training')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with minimal data')
    parser.add_argument('--collect-data', action='store_true', default=True,
                       help='Collect new training data')
    parser.add_argument('--no-collect-data', dest='collect_data', action='store_false',
                       help='Skip data collection')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting HealthPIN AI Model Training...")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Training epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Base model: {args.base_model}")
    
    try:
        # Step 1: Collect training data
        if args.collect_data:
            logger.info("📊 Collecting HealthPIN training data...")
            collector = HealthPINDataCollector(args.data_dir)
            data_stats = collector.collect_healthpin_data()
            logger.info(f"✅ Data collection complete: {data_stats}")
        else:
            logger.info("⏭️ Skipping data collection")
        
        # Step 2: Initialize trainer
        logger.info("🤖 Initializing HealthPIN model trainer...")
        trainer = HealthPINModelTrainer(
            models_dir=args.models_dir,
            base_model=args.base_model
        )
        
        # Step 3: Train the model
        logger.info("🎯 Starting model training...")
        success = trainer.train_healthpin_model(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        if success:
            logger.info("✅ HealthPIN model training completed successfully!")
            
            # Step 4: Test the model
            logger.info("🧪 Testing trained model...")
            test_model(args.models_dir)
            
            # Step 5: Initialize model manager
            logger.info("🔧 Initializing model manager...")
            manager = HealthPINModelManager(
                models_dir=args.models_dir,
                use_custom_model=True
            )
            
            if manager.is_model_loaded:
                logger.info("✅ HealthPIN model manager initialized successfully!")
                logger.info("🎉 HealthPIN training pipeline completed!")
                return True
            else:
                logger.warning("⚠️ Model manager initialization failed, but training was successful")
                return True
        else:
            logger.error("❌ HealthPIN model training failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        return False

def test_model(models_dir: str):
    """Test the trained model with sample inputs"""
    logger.info("🧪 Testing HealthPIN model...")
    
    try:
        manager = HealthPINModelManager(
            models_dir=models_dir,
            use_custom_model=True
        )
        
        if not manager.is_model_loaded:
            logger.warning("⚠️ Model not loaded, skipping tests")
            return
        
        # Test medical conversation
        logger.info("Testing medical conversation...")
        response, source = manager.generate_medical_response(
            "I have been experiencing chest pain for the past 3 days.",
            patient_context={
                'age': 45,
                'gender': 'male',
                'medical_history': 'hypertension'
            }
        )
        logger.info(f"Medical response: {response[:100]}... (Source: {source})")
        
        # Test doctor matching
        logger.info("Testing doctor matching...")
        patient_profile = {
            'age': 45,
            'gender': 'male',
            'symptoms': ['chest pain', 'shortness of breath'],
            'language_preference': 'English',
            'location': 'Johannesburg'
        }
        
        available_doctors = [
            {
                'name': 'Dr. Sarah Johnson',
                'specialty': 'Cardiology',
                'languages': ['English'],
                'location': 'Johannesburg',
                'rating': 4.8,
                'experience_years': 15
            }
        ]
        
        match_result, source = manager.generate_doctor_match(patient_profile, available_doctors)
        logger.info(f"Doctor match: {match_result.get('reasoning', 'No reasoning')[:100]}... (Source: {source})")
        
        # Test health record summary
        logger.info("Testing health record summary...")
        health_record = {
            'title': 'Annual Physical Examination',
            'date': '2024-01-15',
            'doctor': 'Dr. Sarah Johnson',
            'findings': 'Blood pressure: 140/90, Weight: 75kg',
            'diagnosis': 'Hypertension',
            'recommendations': 'Lose weight, reduce salt intake'
        }
        
        summary, source = manager.generate_health_record_summary(health_record)
        logger.info(f"Health record summary: {summary[:100]}... (Source: {source})")
        
        # Test family notification
        logger.info("Testing family notification...")
        notification, source = manager.generate_family_notification(
            patient_name="John Doe",
            notification_type="health_update",
            health_update="Annual check-up completed with good results",
            urgency_level="normal"
        )
        logger.info(f"Family notification: {notification[:100]}... (Source: {source})")
        
        logger.info("✅ Model testing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Model testing failed: {str(e)}")

def quick_train():
    """Quick training function for development/testing"""
    logger.info("🚀 Starting Quick HealthPIN Training...")
    
    trainer = HealthPINModelTrainer(
        models_dir="./healthpin_models_quick",
        base_model="microsoft/DialoGPT-small"  # Use smaller model for speed
    )
    
    success = trainer.train_healthpin_model(
        data_dir="./healthpin_training_data",
        num_epochs=1,  # Just 1 epoch for quick training
        learning_rate=5e-5
    )
    
    if success:
        logger.info("✅ Quick HealthPIN training completed!")
        return True
    else:
        logger.error("❌ Quick HealthPIN training failed!")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick training mode
        success = quick_train()
    else:
        # Full training mode
        success = main()
    
    if success:
        print("\n🎉 HealthPIN training completed successfully!")
        print("You can now use the trained model in your HealthPIN application.")
    else:
        print("\n❌ HealthPIN training failed!")
        sys.exit(1)
