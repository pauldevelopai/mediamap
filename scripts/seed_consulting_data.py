#!/usr/bin/env python3
"""
Consulting Database Seeder
Populates the database with sample consulting clients, sessions, and progress data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the app and models
from backend.app import app
from backend.models import db, ConsultingClient, ConsultingSession, ConsultingProgressReport, ConsultingProgressEntry, ConsultingSuccessMetric
from datetime import datetime, timedelta
import json

def seed_consulting_data():
    """Seed the database with sample consulting data"""
    
    with app.app_context():
        
        # Clear existing consulting data
        print("Clearing existing consulting data...")
        ConsultingProgressEntry.query.delete()
        ConsultingSuccessMetric.query.delete()
        ConsultingProgressReport.query.delete()
        ConsultingSession.query.delete()
        ConsultingClient.query.delete()
        db.session.commit()
        
        # Create sample clients
        clients = [
            {
                'name': 'Sarah Johnson',
                'organization': 'Digital News Network',
                'email': 'sarah.johnson@digitalnews.com',
                'phone': '+1-555-0123',
                'website': 'https://digitalnews.com',
                'industry': 'Media',
                'organization_size': 'Medium (11-50)',
                'location': 'New York, NY',
                'timezone': 'UTC-5',
                'engagement_type': 'Ongoing',
                'contract_value': 15000.0,
                'start_date': datetime.now() - timedelta(days=60),
                'end_date': datetime.now() + timedelta(days=120),
                'status': 'Active',
                'contact_person': 'Sarah Johnson',
                'contact_role': 'Editor-in-Chief',
                'contact_email': 'sarah.johnson@digitalnews.com',
                'notes': 'Digital-first news organization looking to implement AI tools for content creation and audience engagement.',
                'goals': 'Implement AI-powered content generation, improve audience engagement, increase publishing efficiency',
                'challenges': 'Limited technical expertise, budget constraints, resistance to change from traditional journalists',
                'success_metrics': '50% increase in content output, 30% improvement in engagement metrics, 25% reduction in publishing time'
            },
            {
                'name': 'Michael Chen',
                'organization': 'TechStart Media',
                'email': 'michael.chen@techstart.com',
                'phone': '+1-555-0456',
                'website': 'https://techstart.com',
                'industry': 'Technology',
                'organization_size': 'Small (1-10)',
                'location': 'San Francisco, CA',
                'timezone': 'UTC-8',
                'engagement_type': 'One-time',
                'contract_value': 5000.0,
                'start_date': datetime.now() - timedelta(days=30),
                'end_date': datetime.now() + timedelta(days=30),
                'status': 'Active',
                'contact_person': 'Michael Chen',
                'contact_role': 'Founder & CEO',
                'contact_email': 'michael.chen@techstart.com',
                'notes': 'Startup media company focused on technology news and analysis.',
                'goals': 'Establish AI workflow for content creation, build audience from scratch, monetize through subscriptions',
                'challenges': 'Limited resources, no existing audience, need to differentiate from established tech media',
                'success_metrics': 'Launch with 1000 subscribers, achieve 10,000 monthly page views, generate $5000 monthly revenue'
            },
            {
                'name': 'Emily Rodriguez',
                'organization': 'Community Voice',
                'email': 'emily.rodriguez@communityvoice.org',
                'phone': '+1-555-0789',
                'website': 'https://communityvoice.org',
                'industry': 'Non-Profit',
                'organization_size': 'Medium (11-50)',
                'location': 'Chicago, IL',
                'timezone': 'UTC-6',
                'engagement_type': 'Retainer',
                'contract_value': 8000.0,
                'start_date': datetime.now() - timedelta(days=90),
                'end_date': datetime.now() + timedelta(days=180),
                'status': 'Active',
                'contact_person': 'Emily Rodriguez',
                'contact_role': 'Executive Director',
                'contact_email': 'emily.rodriguez@communityvoice.org',
                'notes': 'Non-profit community news organization serving underserved communities.',
                'goals': 'Improve community engagement, increase local coverage, develop sustainable funding model',
                'challenges': 'Limited funding, volunteer-based staff, need to serve diverse community needs',
                'success_metrics': 'Increase community engagement by 40%, publish 50% more local stories, secure $50,000 in grants'
            }
        ]
        
        print(f"Adding {len(clients)} consulting clients...")
        created_clients = []
        for client_data in clients:
            client = ConsultingClient(**client_data)
            db.session.add(client)
            created_clients.append(client)
        
        db.session.commit()
        
        # Create sample sessions
        sessions = [
            {
                'client_id': created_clients[0].id,  # Sarah Johnson
                'title': 'AI Content Strategy Workshop',
                'description': 'Initial workshop to understand current content workflow and identify AI implementation opportunities',
                'session_type': 'Strategy',
                'session_date': datetime.now() - timedelta(days=45),
                'duration_hours': 2.0,
                'session_notes': 'Discussed current content creation process, identified bottlenecks, explored AI tools for newsrooms',
                'topics_covered': json.dumps(['Content workflow analysis', 'AI tool evaluation', 'Implementation roadmap']),
                'action_items': json.dumps(['Research ChatGPT and Claude for content generation', 'Evaluate data safety of AI tools', 'Create pilot program proposal']),
                'next_steps': 'Schedule follow-up session to review AI tool research and discuss pilot program',
                'client_satisfaction': 5,
                'client_feedback': 'Excellent session! Very helpful in understanding how AI can improve our workflow.',
                'status': 'Completed'
            },
            {
                'client_id': created_clients[0].id,  # Sarah Johnson
                'title': 'AI Tool Implementation Planning',
                'description': 'Detailed planning session for implementing AI tools in the newsroom',
                'session_type': 'Implementation',
                'session_date': datetime.now() - timedelta(days=30),
                'duration_hours': 1.5,
                'session_notes': 'Reviewed AI tool research, discussed pilot program details, created implementation timeline',
                'topics_covered': json.dumps(['AI tool selection', 'Pilot program design', 'Training requirements']),
                'action_items': json.dumps(['Set up ChatGPT and Claude accounts', 'Create training materials', 'Select pilot team members']),
                'next_steps': 'Begin pilot program with selected team members',
                'client_satisfaction': 4,
                'client_feedback': 'Good planning session. Looking forward to starting the pilot program.',
                'status': 'Completed'
            },
            {
                'client_id': created_clients[1].id,  # Michael Chen
                'title': 'Startup Media Strategy Session',
                'description': 'Strategic planning session for launching a new technology media startup',
                'session_type': 'Strategy',
                'session_date': datetime.now() - timedelta(days=20),
                'duration_hours': 2.5,
                'session_notes': 'Discussed market positioning, content strategy, audience development, and monetization approaches',
                'topics_covered': json.dumps(['Market analysis', 'Content strategy', 'Audience development', 'Monetization models']),
                'action_items': json.dumps(['Finalize content categories', 'Set up social media accounts', 'Create launch timeline']),
                'next_steps': 'Begin content creation and social media setup',
                'client_satisfaction': 5,
                'client_feedback': 'Incredibly valuable session! Clear roadmap for launching our startup.',
                'status': 'Completed'
            },
            {
                'client_id': created_clients[2].id,  # Emily Rodriguez
                'title': 'Community Engagement Strategy',
                'description': 'Workshop focused on improving community engagement for local news organization',
                'session_type': 'Strategy',
                'session_date': datetime.now() - timedelta(days=60),
                'duration_hours': 2.0,
                'session_notes': 'Analyzed current community engagement efforts, identified gaps, developed improvement strategies',
                'topics_covered': json.dumps(['Community analysis', 'Engagement strategies', 'Partnership opportunities']),
                'action_items': json.dumps(['Conduct community survey', 'Identify potential partners', 'Develop engagement calendar']),
                'next_steps': 'Implement community survey and begin partnership outreach',
                'client_satisfaction': 4,
                'client_feedback': 'Great insights into improving our community connections.',
                'status': 'Completed'
            },
            {
                'client_id': created_clients[0].id,  # Sarah Johnson
                'title': 'Pilot Program Review',
                'description': 'Review session for the AI tools pilot program',
                'session_type': 'Review',
                'session_date': datetime.now() - timedelta(days=7),
                'duration_hours': 1.0,
                'session_notes': 'Reviewed pilot program results, discussed challenges and successes, planned next steps',
                'topics_covered': json.dumps(['Pilot results', 'User feedback', 'Implementation challenges']),
                'action_items': json.dumps(['Address technical issues', 'Expand pilot to more team members', 'Develop best practices guide']),
                'next_steps': 'Expand pilot program and create training documentation',
                'client_satisfaction': 4,
                'client_feedback': 'Pilot program is showing promising results. Team is excited about the possibilities.',
                'status': 'Completed'
            },
            {
                'client_id': created_clients[1].id,  # Michael Chen
                'title': 'Launch Preparation Session',
                'description': 'Final preparation session before website launch',
                'session_type': 'Implementation',
                'session_date': datetime.now() + timedelta(days=5),
                'duration_hours': 1.5,
                'session_notes': 'Final review of launch checklist, content preparation, and marketing strategy',
                'topics_covered': json.dumps(['Launch checklist', 'Content preparation', 'Marketing strategy']),
                'action_items': json.dumps(['Complete website setup', 'Finalize launch content', 'Prepare social media campaign']),
                'next_steps': 'Launch website and begin marketing campaign',
                'client_satisfaction': None,
                'client_feedback': None,
                'status': 'Scheduled'
            }
        ]
        
        print(f"Adding {len(sessions)} consulting sessions...")
        for session_data in sessions:
            session = ConsultingSession(**session_data)
            db.session.add(session)
        
        db.session.commit()
        
        # Create sample progress reports
        progress_reports = [
            {
                'client_id': created_clients[0].id,
                'report_date': datetime.now() - timedelta(days=15),
                'report_period': 'Monthly',
                'goals_progress': json.dumps({
                    'ai_implementation': '75%',
                    'content_efficiency': '60%',
                    'audience_engagement': '40%'
                }),
                'achievements': json.dumps([
                    'Successfully launched AI pilot program',
                    'Trained 5 team members on AI tools',
                    'Increased content output by 25%'
                ]),
                'challenges_faced': json.dumps([
                    'Technical integration issues',
                    'Resistance from some team members',
                    'Limited budget for premium AI tools'
                ]),
                'lessons_learned': json.dumps([
                    'Start with small pilot programs',
                    'Provide comprehensive training',
                    'Address concerns about job security'
                ]),
                'key_metrics': json.dumps({
                    'content_output': '+25%',
                    'publishing_time': '-20%',
                    'team_satisfaction': '4.2/5'
                }),
                'recommendations': json.dumps([
                    'Expand pilot program to more team members',
                    'Invest in premium AI tools',
                    'Create best practices documentation'
                ]),
                'overall_satisfaction': 4
            }
        ]
        
        print(f"Adding {len(progress_reports)} progress reports...")
        for report_data in progress_reports:
            report = ConsultingProgressReport(**report_data)
            db.session.add(report)
        
        db.session.commit()
        
        # Create sample success metrics
        success_metrics = [
            {
                'client_id': created_clients[0].id,
                'metric_name': 'Content Output Increase',
                'metric_description': 'Percentage increase in content published per week',
                'metric_type': 'Quantitative',
                'unit': '%',
                'baseline_value': 0.0,
                'target_value': 50.0,
                'current_value': 25.0,
                'measurement_frequency': 'Weekly',
                'priority': 'High'
            },
            {
                'client_id': created_clients[0].id,
                'metric_name': 'Publishing Time Reduction',
                'metric_description': 'Percentage reduction in time to publish articles',
                'metric_type': 'Quantitative',
                'unit': '%',
                'baseline_value': 0.0,
                'target_value': 30.0,
                'current_value': 20.0,
                'measurement_frequency': 'Weekly',
                'priority': 'High'
            },
            {
                'client_id': created_clients[1].id,
                'metric_name': 'Website Launch',
                'metric_description': 'Successful launch of new website',
                'metric_type': 'Binary',
                'unit': 'Yes/No',
                'baseline_value': 0.0,
                'target_value': 1.0,
                'current_value': 0.0,
                'measurement_frequency': 'Once',
                'priority': 'Critical'
            }
        ]
        
        print(f"Adding {len(success_metrics)} success metrics...")
        for metric_data in success_metrics:
            metric = ConsultingSuccessMetric(**metric_data)
            db.session.add(metric)
        
        db.session.commit()
        
        print("✅ Consulting database seeded successfully!")
        print(f"📊 Added {len(clients)} clients, {len(sessions)} sessions, {len(progress_reports)} progress reports, and {len(success_metrics)} success metrics")

if __name__ == '__main__':
    seed_consulting_data()






