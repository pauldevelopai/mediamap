#!/usr/bin/env python3
"""
Prompt Cleanup Script
====================

This script cleans up the prompt database by:
1. Deactivating unused/redundant prompts
2. Ensuring only actually used prompts are active
3. Updating prompt data to be real and accurate
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app import app, db
from backend.models import PromptTemplate

def cleanup_prompts():
    """Clean up the prompt database"""
    
    with app.app_context():
        print("🧹 Starting prompt cleanup...")
        
        # List of actually used prompts (based on code analysis)
        actually_used_prompts = {
            'SYSTEM_PROMPT_MEDIA_BIZ': {
                'category': 'Business',
                'usage_context': 'Main chat interface - Highlander AI in app.py',
                'description': 'Highlander AI business consultant prompt for media industry discussions'
            },
            'SYSTEM_PROMPT_ANALYSIS': {
                'category': 'Analysis', 
                'usage_context': 'Media analysis feature in app.py',
                'description': 'System prompt for media content analysis'
            },
            'SYSTEM_PROMPT_SYNTHESIS': {
                'category': 'Business',
                'usage_context': 'Organization synthesis in app.py', 
                'description': 'System prompt for organization synthesis'
            },
            'SYSTEM_PROMPT_CHAT': {
                'category': 'Chat',
                'usage_context': 'Chat interface in app.py',
                'description': 'System prompt for chat assistance'
            },
            'DOC_SYSTEM_PROMPT_MAIN': {
                'category': 'Healthcare',
                'usage_context': 'Main Doc chatbot interface in HealthPIN',
                'description': 'Main system prompt for Doc chatbot - healthcare AI assistant'
            },
            'DOC_SYSTEM_PROMPT_PATIENT_LOOKUP': {
                'category': 'Healthcare',
                'usage_context': 'Patient lookup and identification in Doc chatbot',
                'description': 'System prompt for patient lookup and identification in Doc chatbot'
            },
            'DOC_SYSTEM_PROMPT_PATIENT_ANALYSIS': {
                'category': 'Healthcare',
                'usage_context': 'Patient data analysis and clinical insights in Doc chatbot',
                'description': 'System prompt for patient data analysis and clinical insights'
            },
            'DOC_SYSTEM_PROMPT_PATIENT_CARE_PLAN': {
                'category': 'Healthcare',
                'usage_context': 'Patient-specific care plan development in Doc chatbot',
                'description': 'System prompt for developing patient-specific care plans'
            }
        }
        
        # Get all prompts
        all_prompts = PromptTemplate.query.all()
        print(f"📊 Found {len(all_prompts)} total prompts")
        
        # Deactivate unused prompts
        unused_prompts = []
        for prompt in all_prompts:
            if prompt.name not in actually_used_prompts:
                unused_prompts.append(prompt.name)
                prompt.is_active = False
                print(f"❌ Deactivated unused prompt: {prompt.name}")
        
        # Update active prompts with correct information
        for prompt in all_prompts:
            if prompt.name in actually_used_prompts:
                prompt_info = actually_used_prompts[prompt.name]
                prompt.is_active = True
                prompt.category = prompt_info['category']
                prompt.usage_context = prompt_info['usage_context']
                prompt.description = prompt_info['description']
                print(f"✅ Updated active prompt: {prompt.name}")
        
        # Commit changes
        db.session.commit()
        
        print(f"\n📈 Cleanup Summary:")
        print(f"✅ Active prompts: {len(actually_used_prompts)}")
        print(f"❌ Deactivated prompts: {len(unused_prompts)}")
        print(f"📊 Total prompts: {len(all_prompts)}")
        
        print(f"\n🗑️ Deactivated prompts:")
        for prompt_name in unused_prompts:
            print(f"   - {prompt_name}")
        
        print(f"\n✅ Active prompts:")
        for prompt_name in actually_used_prompts.keys():
            print(f"   - {prompt_name}")
        
        print("\n🎉 Prompt cleanup completed successfully!")

if __name__ == "__main__":
    cleanup_prompts()
