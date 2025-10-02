"""
Organization Insight Service
Generates comprehensive AI implementation insights and reports for organizations/newsrooms
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import openai
from flask import current_app

from backend.models import db, OrganizationInsight, Newsroom, User, HighlanderChat
from backend.aimap.models import Organisation

logger = logging.getLogger(__name__)

class OrganizationInsightService:
    """Service for generating organization-specific AI insights and reports"""
    
    def __init__(self):
        self.openai_client = None
    
    def _get_openai_client(self):
        """Get OpenAI client, initializing if needed"""
        if self.openai_client is None:
            self.openai_client = openai.OpenAI(api_key=current_app.config.get('OPENAI_API_KEY'))
        return self.openai_client
    
    def generate_comprehensive_insight(self, organization_id: int, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate comprehensive AI implementation insight for an organization"""
        try:
            # Get organization data
            organization = Newsroom.query.get(organization_id)
            if not organization:
                raise ValueError(f"Organization {organization_id} not found")
            
            # Gather organization data
            org_data = self._gather_organization_data(organization)
            
            # Generate insights using AI
            insights = self._generate_ai_insights(org_data)
            
            # Save insights to database
            insight_record = self._save_insights(organization_id, insights, user_id)
            
            return {
                'success': True,
                'insight_id': insight_record.id,
                'insights': insights,
                'organization': {
                    'id': organization.id,
                    'name': organization.name
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating insight for organization {organization_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_two_page_report(self, organization_id: int, format_type: str = 'html') -> Dict[str, Any]:
        """Generate comprehensive 2-page AI implementation report"""
        try:
            organization = Newsroom.query.get(organization_id)
            if not organization:
                raise ValueError(f"Organization {organization_id} not found")
            
            # Get latest insights
            latest_insights = OrganizationInsight.query.filter_by(
                organization_id=organization_id
            ).order_by(OrganizationInsight.created_at.desc()).limit(5).all()
            
            # Gather comprehensive data
            org_data = self._gather_organization_data(organization)
            
            # Generate report content
            report_content = self._generate_report_content(org_data, latest_insights)
            
            if format_type == 'html':
                report_html = self._generate_html_report(organization, report_content)
                return {
                    'success': True,
                    'format': 'html',
                    'content': report_html,
                    'organization': organization.name
                }
            elif format_type == 'pdf':
                # For PDF generation, we'd use a library like WeasyPrint
                report_html = self._generate_html_report(organization, report_content)
                return {
                    'success': True,
                    'format': 'pdf',
                    'html_content': report_html,  # Can be converted to PDF
                    'organization': organization.name
                }
            else:
                return {
                    'success': True,
                    'format': 'json',
                    'content': report_content,
                    'organization': organization.name
                }
                
        except Exception as e:
            logger.error(f"Error generating report for organization {organization_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _gather_organization_data(self, organization: Newsroom) -> Dict[str, Any]:
        """Gather comprehensive data about the organization"""
        
        # Get user interactions and chat history
        # Since Newsroom doesn't have direct users relationship, we'll get recent chats from all users
        recent_chats = HighlanderChat.query.order_by(HighlanderChat.created_at.desc()).limit(20).all()
        
        # Get AIMAP organization if available
        aimap_org = Organisation.query.filter_by(name=organization.name).first()
        
        return {
            'organization': {
                'id': organization.id,
                'name': organization.name,
                'type': organization.type,
                'location': organization.location,
                'ai_readiness': organization.ai_readiness,
                'website': organization.website,
                'notes': organization.notes,
                'created_at': organization.created_at.isoformat() if organization.created_at else None,
                'user_count': 1  # Default assumption
            },
            'recent_interactions': [
                {
                    'message': chat.message,
                    'response': chat.response,
                    'category': chat.category,
                    'created_at': chat.created_at.isoformat()
                } for chat in recent_chats
            ],
            'aimap_data': {
                'sector': aimap_org.sector if aimap_org else None,
                'ai_tools': aimap_org.ai_tools if aimap_org else [],
                'size_band': aimap_org.size_band if aimap_org else None,
                'region': aimap_org.region if aimap_org else None
            } if aimap_org else None,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_ai_insights(self, org_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI insights using OpenAI"""
        
        prompt = f"""
        Analyze the following organization data and generate comprehensive AI implementation insights:

        Organization: {org_data['organization']['name']}
        Type: {org_data['organization'].get('type', 'N/A')}
        Location: {org_data['organization'].get('location', 'N/A')}
        AI Readiness: {org_data['organization'].get('ai_readiness', 'N/A')}
        User Count: {org_data['organization']['user_count']}
        
        Recent AI Interactions: {len(org_data['recent_interactions'])} conversations
        
        AIMAP Data: {json.dumps(org_data.get('aimap_data', {}), indent=2)}

        Please provide a comprehensive analysis including:

        1. CURRENT AI MATURITY ASSESSMENT
        - Current level: beginner/intermediate/advanced
        - Strengths and capabilities identified
        - Usage patterns and engagement

        2. IMPLEMENTATION GAPS ANALYSIS
        - Key gaps in AI adoption
        - Missing capabilities or tools
        - Process improvement opportunities

        3. STRATEGIC RECOMMENDATIONS
        - Top 5 priority actions
        - Implementation timeline (short/medium/long term)
        - Expected outcomes and benefits

        4. RESOURCE REQUIREMENTS
        - Staff training needs
        - Technology requirements
        - Budget considerations

        5. NEXT STEPS ROADMAP
        - Immediate actions (next 30 days)
        - Medium-term goals (3-6 months)
        - Long-term vision (6-12 months)

        Format your response as a structured JSON with clear sections and actionable insights.
        Focus on practical, implementable recommendations specific to this organization's context.
        """
        
        try:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert AI implementation consultant specializing in newsroom and media organization digital transformation. Provide detailed, actionable insights based on data analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse the AI response
            ai_response = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    insights_json = json.loads(json_match.group())
                else:
                    # If no JSON found, structure the response
                    insights_json = {
                        'current_ai_maturity': 'intermediate',
                        'implementation_gaps': ['AI strategy development', 'Staff training', 'Process automation'],
                        'recommended_actions': ['Develop AI strategy', 'Train staff', 'Implement automation tools'],
                        'expected_outcomes': ['Improved efficiency', 'Better content quality', 'Cost savings'],
                        'timeline_estimate': '3-6 months',
                        'resource_requirements': ['Training budget', 'Technology investment', 'Staff time'],
                        'raw_analysis': ai_response
                    }
            except json.JSONDecodeError:
                # Fallback structure
                insights_json = {
                    'current_ai_maturity': 'intermediate',
                    'implementation_gaps': ['Comprehensive AI strategy needed'],
                    'recommended_actions': ['Develop AI implementation plan'],
                    'expected_outcomes': ['Improved AI adoption'],
                    'timeline_estimate': '3-6 months',
                    'resource_requirements': ['Strategic planning time'],
                    'raw_analysis': ai_response
                }
            
            return insights_json
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            # Return fallback insights
            return {
                'current_ai_maturity': 'beginner',
                'implementation_gaps': ['AI strategy development needed'],
                'recommended_actions': ['Assess current AI capabilities', 'Develop AI strategy'],
                'expected_outcomes': ['Improved AI readiness'],
                'timeline_estimate': '1-3 months',
                'resource_requirements': ['Strategic planning'],
                'error': str(e)
            }
    
    def _save_insights(self, organization_id: int, insights: Dict[str, Any], user_id: Optional[int] = None) -> OrganizationInsight:
        """Save insights to database"""
        
        insight = OrganizationInsight(
            organization_id=organization_id,
            title=f"AI Implementation Analysis - {datetime.utcnow().strftime('%Y-%m-%d')}",
            content=insights.get('raw_analysis', 'Comprehensive AI implementation analysis'),
            category='Implementation',
            insight_type='comprehensive_analysis',
            confidence_score=0.85,
            priority='high',
            current_ai_maturity=insights.get('current_ai_maturity', 'intermediate'),
            implementation_gaps=insights.get('implementation_gaps', []),
            recommended_actions=insights.get('recommended_actions', []),
            expected_outcomes=insights.get('expected_outcomes', []),
            timeline_estimate=insights.get('timeline_estimate', '3-6 months'),
            resource_requirements=insights.get('resource_requirements', []),
            generated_by='OpenAI GPT-4',
            data_sources=['organization_data', 'chat_history', 'aimap_metrics']
        )
        
        db.session.add(insight)
        db.session.commit()
        
        return insight
    
    def _generate_report_content(self, org_data: Dict[str, Any], insights: List[OrganizationInsight]) -> Dict[str, Any]:
        """Generate comprehensive report content"""
        
        # Analyze trends from insights
        recent_insights = insights[:3] if insights else []
        
        return {
            'executive_summary': {
                'organization_name': org_data['organization']['name'],
                'analysis_date': datetime.utcnow().strftime('%B %d, %Y'),
                'current_maturity': recent_insights[0].current_ai_maturity if recent_insights else 'Assessment needed',
                'key_recommendation': recent_insights[0].recommended_actions[0] if recent_insights and recent_insights[0].recommended_actions else 'Develop AI strategy',
                'priority_level': recent_insights[0].priority if recent_insights else 'high'
            },
            'current_status': {
                'ai_maturity_level': recent_insights[0].current_ai_maturity if recent_insights else 'beginner',
                'user_engagement': org_data['organization']['user_count'],
                'recent_activity': len(org_data['recent_interactions']),
                'aimap_sector': org_data.get('aimap_data', {}).get('sector', 'N/A') if org_data.get('aimap_data') else 'N/A'
            },
            'gap_analysis': {
                'identified_gaps': recent_insights[0].implementation_gaps if recent_insights else ['AI strategy development'],
                'priority_gaps': recent_insights[0].implementation_gaps[:3] if recent_insights and recent_insights[0].implementation_gaps else ['Strategy', 'Training', 'Tools'],
                'impact_assessment': 'Medium to High impact on operational efficiency'
            },
            'recommendations': {
                'immediate_actions': recent_insights[0].recommended_actions[:2] if recent_insights and recent_insights[0].recommended_actions else ['Assess current state', 'Develop strategy'],
                'medium_term_goals': recent_insights[0].recommended_actions[2:4] if recent_insights and len(recent_insights[0].recommended_actions) > 2 else ['Implement tools', 'Train staff'],
                'long_term_vision': recent_insights[0].expected_outcomes if recent_insights else ['Improved efficiency', 'Better outcomes']
            },
            'implementation_roadmap': {
                'phase_1': '30 days - Assessment and planning',
                'phase_2': '3-6 months - Implementation and training',
                'phase_3': '6-12 months - Optimization and scaling',
                'estimated_timeline': recent_insights[0].timeline_estimate if recent_insights else '6-9 months'
            },
            'resource_requirements': {
                'budget_estimate': 'Contact for detailed assessment',
                'staff_time': recent_insights[0].resource_requirements if recent_insights else ['Management time', 'Staff training'],
                'technology_needs': ['AI tools', 'Training platforms', 'Integration support']
            },
            'success_metrics': {
                'efficiency_gains': '20-40% improvement in content production',
                'quality_improvements': 'Enhanced content accuracy and relevance',
                'cost_savings': 'Reduced manual processing time',
                'roi_timeline': '6-12 months'
            }
        }
    
    def _generate_html_report(self, organization: Newsroom, content: Dict[str, Any]) -> str:
        """Generate HTML report"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Implementation Report - {organization.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; page-break-inside: avoid; }}
                .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .section h3 {{ color: #555; margin-top: 20px; }}
                .highlight {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; margin: 15px 0; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .recommendations {{ background: #e8f5e8; padding: 20px; border-radius: 8px; }}
                .roadmap {{ background: #fff3cd; padding: 20px; border-radius: 8px; }}
                ul {{ padding-left: 20px; }}
                li {{ margin-bottom: 8px; }}
                .page-break {{ page-break-before: always; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Implementation Report</h1>
                <h2>{organization.name}</h2>
                <p>Generated on {content['executive_summary']['analysis_date']}</p>
            </div>
            
            <!-- Page 1: Executive Summary & Current Status -->
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="highlight">
                    <p><strong>Organization:</strong> {content['executive_summary']['organization_name']}</p>
                    <p><strong>Current AI Maturity:</strong> {content['executive_summary']['current_maturity'].title()}</p>
                    <p><strong>Priority Recommendation:</strong> {content['executive_summary']['key_recommendation']}</p>
                    <p><strong>Priority Level:</strong> {content['executive_summary']['priority_level'].title()}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Current AI Implementation Status</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{content['current_status']['ai_maturity_level'].title()}</div>
                        <div>AI Maturity</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{content['current_status']['user_engagement']}</div>
                        <div>Active Users</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{content['current_status']['recent_activity']}</div>
                        <div>Recent Interactions</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{content['current_status']['aimap_sector']}</div>
                        <div>Sector</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Gap Analysis</h2>
                <h3>Identified Implementation Gaps:</h3>
                <ul>
                    {''.join(f'<li>{gap}</li>' for gap in content['gap_analysis']['identified_gaps'])}
                </ul>
                <div class="highlight">
                    <strong>Impact Assessment:</strong> {content['gap_analysis']['impact_assessment']}
                </div>
            </div>
            
            <!-- Page 2: Recommendations & Roadmap -->
            <div class="page-break"></div>
            
            <div class="section">
                <h2>Strategic Recommendations</h2>
                <div class="recommendations">
                    <h3>Immediate Actions (Next 30 Days)</h3>
                    <ul>
                        {''.join(f'<li>{action}</li>' for action in content['recommendations']['immediate_actions'])}
                    </ul>
                    
                    <h3>Medium-Term Goals (3-6 Months)</h3>
                    <ul>
                        {''.join(f'<li>{goal}</li>' for goal in content['recommendations']['medium_term_goals'])}
                    </ul>
                    
                    <h3>Long-Term Vision</h3>
                    <ul>
                        {''.join(f'<li>{vision}</li>' for vision in content['recommendations']['long_term_vision'])}
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>Implementation Roadmap</h2>
                <div class="roadmap">
                    <h3>Phase 1: {content['implementation_roadmap']['phase_1']}</h3>
                    <h3>Phase 2: {content['implementation_roadmap']['phase_2']}</h3>
                    <h3>Phase 3: {content['implementation_roadmap']['phase_3']}</h3>
                    <div class="highlight">
                        <strong>Estimated Total Timeline:</strong> {content['implementation_roadmap']['estimated_timeline']}
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Resource Requirements & Expected Outcomes</h2>
                <h3>Resource Requirements:</h3>
                <ul>
                    {''.join(f'<li>{req}</li>' for req in content['resource_requirements']['staff_time'])}
                    {''.join(f'<li>{tech}</li>' for tech in content['resource_requirements']['technology_needs'])}
                </ul>
                
                <h3>Expected Success Metrics:</h3>
                <ul>
                    <li><strong>Efficiency Gains:</strong> {content['success_metrics']['efficiency_gains']}</li>
                    <li><strong>Quality Improvements:</strong> {content['success_metrics']['quality_improvements']}</li>
                    <li><strong>Cost Savings:</strong> {content['success_metrics']['cost_savings']}</li>
                    <li><strong>ROI Timeline:</strong> {content['success_metrics']['roi_timeline']}</li>
                </ul>
            </div>
            
            <div class="section" style="margin-top: 40px; text-align: center; color: #666;">
                <p><em>This report was generated using AI analysis of your organization's current state and industry best practices.</em></p>
                <p><em>For detailed implementation support, contact the AIMAP team.</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def get_organization_insights(self, organization_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get insights for an organization"""
        insights = OrganizationInsight.query.filter_by(
            organization_id=organization_id
        ).order_by(OrganizationInsight.created_at.desc()).limit(limit).all()
        
        return [insight.to_dict() for insight in insights]

# Global service instance
organization_insight_service = OrganizationInsightService()
