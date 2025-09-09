#!/usr/bin/env python3
"""
AI Tools Database Seeder
Populates the database with comprehensive AI tools from around the world
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the app and models
from backend.app import app
from backend.models import db, AITool, AIToolRecommendation, AIToolCategory
from datetime import datetime

def seed_ai_tools():
    """Seed the database with comprehensive AI tools data"""
    
    with app.app_context():
        
        # Clear existing data
        print("Clearing existing AI tools data...")
        AITool.query.delete()
        AIToolRecommendation.query.delete()
        AIToolCategory.query.delete()
        db.session.commit()
        
        # Create categories
        categories = [
            {'name': 'Text Generation', 'description': 'AI tools for writing, content creation, and text processing', 'icon': 'bi-file-text', 'color': '#007bff'},
            {'name': 'Image Generation', 'description': 'AI tools for creating and editing images', 'icon': 'bi-image', 'color': '#28a745'},
            {'name': 'Video Generation', 'description': 'AI tools for video creation and editing', 'icon': 'bi-camera-video', 'color': '#dc3545'},
            {'name': 'Audio Generation', 'description': 'AI tools for audio creation and processing', 'icon': 'bi-music-note', 'color': '#ffc107'},
            {'name': 'Data Analysis', 'description': 'AI tools for data processing and analytics', 'icon': 'bi-graph-up', 'color': '#17a2b8'},
            {'name': 'Automation', 'description': 'AI tools for workflow automation', 'icon': 'bi-gear', 'color': '#6c757d'},
            {'name': 'Translation', 'description': 'AI tools for language translation', 'icon': 'bi-translate', 'color': '#fd7e14'},
            {'name': 'Summarization', 'description': 'AI tools for content summarization', 'icon': 'bi-file-earmark-text', 'color': '#e83e8c'},
            {'name': 'Code Generation', 'description': 'AI tools for programming and code generation', 'icon': 'bi-code-slash', 'color': '#6f42c1'},
            {'name': 'Research', 'description': 'AI tools for research and analysis', 'icon': 'bi-search', 'color': '#20c997'}
        ]
        
        for cat_data in categories:
            category = AIToolCategory(**cat_data)
            db.session.add(category)
        
        # Comprehensive AI Tools Database
        ai_tools = [
            # Text Generation Tools
            {
                'name': 'ChatGPT',
                'description': 'Advanced language model for conversation, writing, and content creation',
                'company': 'OpenAI',
                'category': 'text-generation',
                'subcategory': 'Conversational AI',
                'website_url': 'https://chat.openai.com',
                'pricing_model': 'freemium',
                'pricing_details': 'Free tier available, Plus plan $20/month',
                'data_safety_score': 7.5,
                'data_safety_assessment': 'Good privacy practices, but data may be used for training',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.8,
                'review_count': 15000,
                'recommendation_score': 8.5,
                'recommendation_reason': 'Excellent for content creation, research, and brainstorming. Widely adopted and reliable.',
                'use_cases': '["Content writing", "Research assistance", "Code generation", "Translation"]',
                'limitations': 'May generate incorrect information, limited to training data cutoff',
                'alternatives': '["Claude", "Gemini", "Perplexity"]'
            },
            {
                'name': 'Claude',
                'description': 'Advanced AI assistant with strong reasoning and writing capabilities',
                'company': 'Anthropic',
                'category': 'text-generation',
                'subcategory': 'Conversational AI',
                'website_url': 'https://claude.ai',
                'pricing_model': 'freemium',
                'pricing_details': 'Free tier available, Pro plan $20/month',
                'data_safety_score': 8.5,
                'data_safety_assessment': 'Strong privacy focus, constitutional AI approach',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.7,
                'review_count': 8000,
                'recommendation_score': 9.0,
                'recommendation_reason': 'Excellent for sensitive content, strong reasoning capabilities, privacy-focused',
                'use_cases': '["Content writing", "Analysis", "Research", "Sensitive data processing"]',
                'limitations': 'Limited image capabilities, smaller context window than some competitors',
                'alternatives': '["ChatGPT", "Gemini", "Perplexity"]'
            },
            {
                'name': 'Jasper',
                'description': 'AI writing assistant for marketing and business content',
                'company': 'Jasper',
                'category': 'text-generation',
                'subcategory': 'Marketing Content',
                'website_url': 'https://jasper.ai',
                'pricing_model': 'subscription',
                'pricing_details': 'Starter $39/month, Professional $125/month',
                'data_safety_score': 7.0,
                'data_safety_assessment': 'Standard business practices, data may be used for improvement',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.5,
                'review_count': 5000,
                'recommendation_score': 7.5,
                'recommendation_reason': 'Excellent for marketing content, social media, and business writing',
                'use_cases': '["Marketing copy", "Social media posts", "Blog writing", "Email campaigns"]',
                'limitations': 'Higher cost, primarily focused on marketing content',
                'alternatives': '["Copy.ai", "Writesonic", "ChatGPT"]'
            },
            
            # Image Generation Tools
            {
                'name': 'Midjourney',
                'description': 'Advanced AI image generation with artistic and photorealistic capabilities',
                'company': 'Midjourney',
                'category': 'image-generation',
                'subcategory': 'Artistic Generation',
                'website_url': 'https://midjourney.com',
                'pricing_model': 'subscription',
                'pricing_details': 'Basic $10/month, Standard $30/month, Pro $60/month',
                'data_safety_score': 6.5,
                'data_safety_assessment': 'Standard practices, images may be used for training',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': False,
                'rating': 4.6,
                'review_count': 12000,
                'recommendation_score': 8.0,
                'recommendation_reason': 'Excellent artistic quality, great for creative projects and visual content',
                'use_cases': '["Art creation", "Marketing visuals", "Concept art", "Social media images"]',
                'limitations': 'No API access, Discord-only interface, limited control over output',
                'alternatives': '["DALL-E", "Stable Diffusion", "Adobe Firefly"]'
            },
            {
                'name': 'DALL-E',
                'description': 'OpenAI\'s image generation model with high-quality output',
                'company': 'OpenAI',
                'category': 'image-generation',
                'subcategory': 'General Generation',
                'website_url': 'https://openai.com/dall-e-2',
                'pricing_model': 'pay-per-use',
                'pricing_details': '$0.02 per image (1024x1024)',
                'data_safety_score': 7.0,
                'data_safety_assessment': 'Good safety filters, content moderation in place',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.4,
                'review_count': 8000,
                'recommendation_score': 7.5,
                'recommendation_reason': 'High-quality output, good API integration, reasonable pricing',
                'use_cases': '["Marketing visuals", "Product mockups", "Illustrations", "Social media"]',
                'limitations': 'Limited artistic style control, safety filters may be restrictive',
                'alternatives': '["Midjourney", "Stable Diffusion", "Adobe Firefly"]'
            },
            
            # Video Generation Tools
            {
                'name': 'Runway',
                'description': 'AI-powered video editing and generation platform',
                'company': 'Runway',
                'category': 'video-generation',
                'subcategory': 'Video Editing',
                'website_url': 'https://runwayml.com',
                'pricing_model': 'subscription',
                'pricing_details': 'Creator $15/month, Pro $35/month, Team $95/month',
                'data_safety_score': 7.5,
                'data_safety_assessment': 'Good privacy practices, professional-grade security',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.3,
                'review_count': 3000,
                'recommendation_score': 8.0,
                'recommendation_reason': 'Excellent for video editing, motion graphics, and creative video content',
                'use_cases': '["Video editing", "Motion graphics", "Background removal", "Video generation"]',
                'limitations': 'Learning curve, some features require subscription',
                'alternatives': '["CapCut", "Adobe Premiere", "DaVinci Resolve"]'
            },
            {
                'name': 'Synthesia',
                'description': 'AI video generation with virtual presenters and avatars',
                'company': 'Synthesia',
                'category': 'video-generation',
                'subcategory': 'Avatar Generation',
                'website_url': 'https://synthesia.io',
                'pricing_model': 'subscription',
                'pricing_details': 'Starter $30/month, Creator $89/month, Enterprise custom',
                'data_safety_score': 8.0,
                'data_safety_assessment': 'Strong enterprise focus, good data protection',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.2,
                'review_count': 2000,
                'recommendation_score': 7.5,
                'recommendation_reason': 'Excellent for training videos, presentations, and educational content',
                'use_cases': '["Training videos", "Presentations", "Educational content", "Marketing videos"]',
                'limitations': 'Limited to avatar-based videos, higher cost for advanced features',
                'alternatives': '["Lumen5", "InVideo", "Pictory"]'
            },
            
            # Audio Generation Tools
            {
                'name': 'ElevenLabs',
                'description': 'AI voice generation and text-to-speech platform',
                'company': 'ElevenLabs',
                'category': 'audio-generation',
                'subcategory': 'Voice Generation',
                'website_url': 'https://elevenlabs.io',
                'pricing_model': 'subscription',
                'pricing_details': 'Starter $22/month, Creator $99/month, Pro $330/month',
                'data_safety_score': 7.0,
                'data_safety_assessment': 'Good practices, voice cloning requires consent',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.5,
                'review_count': 4000,
                'recommendation_score': 8.0,
                'recommendation_reason': 'High-quality voice generation, excellent for podcasts and audio content',
                'use_cases': '["Podcast narration", "Video voiceovers", "Audiobooks", "Voice cloning"]',
                'limitations': 'Voice cloning requires consent, higher cost for commercial use',
                'alternatives': '["Murf", "Play.ht", "Amazon Polly"]'
            },
            
            # Data Analysis Tools
            {
                'name': 'Tableau',
                'description': 'Data visualization and business intelligence platform',
                'company': 'Salesforce',
                'category': 'data-analysis',
                'subcategory': 'Business Intelligence',
                'website_url': 'https://tableau.com',
                'pricing_model': 'subscription',
                'pricing_details': 'Creator $70/month, Explorer $42/month, Viewer $15/month',
                'data_safety_score': 8.5,
                'data_safety_assessment': 'Enterprise-grade security, strong data protection',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.4,
                'review_count': 6000,
                'recommendation_score': 8.5,
                'recommendation_reason': 'Industry standard for data visualization, excellent for newsroom analytics',
                'use_cases': '["Data visualization", "Business intelligence", "Analytics dashboards", "Reporting"]',
                'limitations': 'Steep learning curve, higher cost for advanced features',
                'alternatives': '["Power BI", "Looker", "Qlik"]'
            },
            
            # Automation Tools
            {
                'name': 'Zapier',
                'description': 'Workflow automation platform connecting apps and services',
                'company': 'Zapier',
                'category': 'automation',
                'subcategory': 'Workflow Automation',
                'website_url': 'https://zapier.com',
                'pricing_model': 'subscription',
                'pricing_details': 'Free tier, Starter $20/month, Professional $50/month',
                'data_safety_score': 8.0,
                'data_safety_assessment': 'Good security practices, data encryption in transit',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.6,
                'review_count': 10000,
                'recommendation_score': 8.5,
                'recommendation_reason': 'Excellent for automating repetitive tasks and workflows',
                'use_cases': '["Workflow automation", "Data synchronization", "Task automation", "Integration"]',
                'limitations': 'Complex workflows can be expensive, limited free tier',
                'alternatives': '["IFTTT", "Make", "n8n"]'
            },
            
            # Translation Tools
            {
                'name': 'DeepL',
                'description': 'AI-powered translation service with high accuracy',
                'company': 'DeepL',
                'category': 'translation',
                'subcategory': 'Language Translation',
                'website_url': 'https://deepl.com',
                'pricing_model': 'subscription',
                'pricing_details': 'Free tier, Pro $8.74/month, Advanced $28.74/month',
                'data_safety_score': 8.5,
                'data_safety_assessment': 'Strong privacy focus, data deletion options',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.7,
                'review_count': 7000,
                'recommendation_score': 9.0,
                'recommendation_reason': 'Excellent translation quality, strong privacy practices',
                'use_cases': '["Document translation", "Website localization", "Content translation", "Real-time translation"]',
                'limitations': 'Limited language pairs compared to Google Translate',
                'alternatives': '["Google Translate", "Microsoft Translator", "Lingvanex"]'
            },
            
            # Summarization Tools
            {
                'name': 'Otter.ai',
                'description': 'AI-powered transcription and meeting summarization',
                'company': 'Otter.ai',
                'category': 'summarization',
                'subcategory': 'Meeting Summarization',
                'website_url': 'https://otter.ai',
                'pricing_model': 'subscription',
                'pricing_details': 'Basic free, Pro $10/month, Business $20/month',
                'data_safety_score': 7.5,
                'data_safety_assessment': 'Good security, data may be used for improvement',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.3,
                'review_count': 5000,
                'recommendation_score': 7.5,
                'recommendation_reason': 'Excellent for meeting transcription and summarization',
                'use_cases': '["Meeting transcription", "Interview notes", "Content summarization", "Voice notes"]',
                'limitations': 'Accuracy varies with audio quality, limited free tier',
                'alternatives': '["Rev", "Temi", "Trint"]'
            },
            
            # Code Generation Tools
            {
                'name': 'GitHub Copilot',
                'description': 'AI-powered code completion and generation',
                'company': 'Microsoft',
                'category': 'code-generation',
                'subcategory': 'Code Completion',
                'website_url': 'https://github.com/features/copilot',
                'pricing_model': 'subscription',
                'pricing_details': '$10/month for individuals, $19/month for business',
                'data_safety_score': 7.0,
                'data_safety_assessment': 'Standard Microsoft security practices',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': False,
                'rating': 4.4,
                'review_count': 8000,
                'recommendation_score': 8.0,
                'recommendation_reason': 'Excellent for developers, integrates well with GitHub',
                'use_cases': '["Code completion", "Bug fixing", "Documentation", "Code review"]',
                'limitations': 'GitHub integration required, may suggest insecure code',
                'alternatives': '["Tabnine", "Kite", "IntelliCode"]'
            },
            
            # Research Tools
            {
                'name': 'Perplexity',
                'description': 'AI-powered research and information search',
                'company': 'Perplexity',
                'category': 'research',
                'subcategory': 'Information Search',
                'website_url': 'https://perplexity.ai',
                'pricing_model': 'freemium',
                'pricing_details': 'Free tier, Pro $20/month',
                'data_safety_score': 7.5,
                'data_safety_assessment': 'Good privacy practices, search history available',
                'gdpr_compliant': True,
                'ccpa_compliant': True,
                'data_encryption': True,
                'api_available': True,
                'rating': 4.5,
                'review_count': 6000,
                'recommendation_score': 8.0,
                'recommendation_reason': 'Excellent for research, provides sources and citations',
                'use_cases': '["Research", "Fact-checking", "Information gathering", "Academic writing"]',
                'limitations': 'Limited to search-based responses, may not be as creative as other AI',
                'alternatives': '["ChatGPT", "Claude", "Google Bard"]'
            }
        ]
        
        print(f"Adding {len(ai_tools)} AI tools...")
        for tool_data in ai_tools:
            tool = AITool(**tool_data)
            db.session.add(tool)
        
        # Create recommendations
        recommendations = [
            {
                'title': 'Best AI Tools for Small Newsrooms',
                'description': 'Curated selection of affordable AI tools perfect for small newsrooms with limited budgets',
                'target_audience': 'Small newsrooms (1-10 staff)',
                'use_case': 'Content creation and basic automation',
                'budget_range': '$10-50/month',
                'recommended_tools': '[{"tool_id": 1, "priority": 1, "reason": "Free tier available, excellent for content creation"}, {"tool_id": 2, "priority": 2, "reason": "Strong privacy focus, good for sensitive content"}]',
                'implementation_steps': '1. Start with ChatGPT free tier for content creation\n2. Add Claude for sensitive content\n3. Implement basic automation with Zapier\n4. Train staff on AI best practices',
                'timeline': '2-4 weeks',
                'estimated_cost': '$30-80/month',
                'training_requirements': 'Basic AI literacy training, content guidelines'
            },
            {
                'title': 'Data-Safe AI Tools for Investigative Journalism',
                'description': 'AI tools with strong privacy and data protection for sensitive investigative work',
                'target_audience': 'Investigative journalists and newsrooms',
                'use_case': 'Research and analysis with privacy protection',
                'budget_range': '$50-150/month',
                'recommended_tools': '[{"tool_id": 2, "priority": 1, "reason": "Strong privacy focus and constitutional AI approach"}, {"tool_id": 9, "priority": 2, "reason": "Enterprise-grade security for data analysis"}]',
                'implementation_steps': '1. Implement Claude for sensitive research\n2. Use Tableau for data visualization\n3. Establish data handling protocols\n4. Regular security audits',
                'timeline': '4-6 weeks',
                'estimated_cost': '$100-200/month',
                'training_requirements': 'Advanced privacy training, data protection protocols'
            },
            {
                'title': 'AI Tools for Digital-First Publishers',
                'description': 'Comprehensive AI toolkit for modern digital publishers focusing on content and audience engagement',
                'target_audience': 'Digital-first publishers and online media',
                'use_case': 'Content creation, automation, and audience analytics',
                'budget_range': '$100-300/month',
                'recommended_tools': '[{"tool_id": 1, "priority": 1, "reason": "Versatile content creation"}, {"tool_id": 3, "priority": 2, "reason": "Specialized marketing content"}, {"tool_id": 10, "priority": 3, "reason": "Workflow automation"}]',
                'implementation_steps': '1. Deploy ChatGPT for general content\n2. Add Jasper for marketing content\n3. Implement Zapier for automation\n4. Set up analytics tracking',
                'timeline': '6-8 weeks',
                'estimated_cost': '$200-400/month',
                'training_requirements': 'Comprehensive AI training, content strategy development'
            }
        ]
        
        print(f"Adding {len(recommendations)} recommendations...")
        for rec_data in recommendations:
            recommendation = AIToolRecommendation(**rec_data)
            db.session.add(recommendation)
        
        # Commit all changes
        db.session.commit()
        print("✅ AI Tools database seeded successfully!")
        print(f"📊 Added {len(ai_tools)} AI tools and {len(recommendations)} recommendations")

if __name__ == '__main__':
    seed_ai_tools()
