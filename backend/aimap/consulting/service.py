"""
AIMAP Consulting Service
Main service for AI consulting intelligence and strategy generation
"""
from typing import Dict, List, Optional
from datetime import datetime
from .processes import AIProcessLibrary
from .strategy_generator import AIStrategyGenerator
from .success_tracker import AISuccessTracker

class AIMAPConsultingService:
    """Main service for AI consulting intelligence"""
    
    def __init__(self):
        self.process_library = AIProcessLibrary()
        self.strategy_generator = AIStrategyGenerator()
        self.success_tracker = AISuccessTracker()
    
    def generate_comprehensive_consulting_package(self, org_profile: Dict) -> Dict:
        """Generate complete consulting package for an organization"""
        
        # Generate AI strategy
        strategy = self.strategy_generator.generate_strategy(org_profile)
        
        # Create success tracking plan
        success_plan = self.success_tracker.create_success_plan(org_profile, strategy)
        
        # Get process recommendations
        process_recommendations = self.process_library.get_process_recommendations(org_profile)
        
        return {
            "organization": {
                "name": org_profile.get("name", "Unknown"),
                "sector": org_profile.get("sector", ""),
                "current_score": org_profile.get("ai_adoption_score", 0),
                "team_size": org_profile.get("team_size", 5),
                "budget": org_profile.get("budget", 100000)
            },
            "consulting_package": {
                "strategy": strategy,
                "success_plan": success_plan,
                "process_recommendations": process_recommendations,
                "implementation_roadmap": strategy.get("roadmap", {}),
                "risk_assessment": strategy.get("risk_mitigation", []),
                "success_metrics": success_plan.get("success_metrics", [])
            },
            "deliverables": self._generate_deliverables(org_profile, strategy, success_plan),
            "timeline": self._generate_timeline(strategy),
            "investment_summary": self._generate_investment_summary(strategy),
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_deliverables(self, org_profile: Dict, strategy: Dict, success_plan: Dict) -> List[Dict]:
        """Generate consulting deliverables"""
        deliverables = []
        
        # Strategy deliverables
        deliverables.append({
            "type": "Strategy Document",
            "name": f"AI Strategy for {org_profile.get('name', 'Organization')}",
            "description": "Comprehensive AI implementation strategy with roadmap",
            "content": {
                "executive_summary": strategy.get("strategy_summary", {}),
                "phases": strategy.get("phases", []),
                "process_recommendations": strategy.get("process_recommendations", [])
            },
            "format": "PDF Report",
            "estimated_pages": 25
        })
        
        # Implementation roadmap
        deliverables.append({
            "type": "Implementation Roadmap",
            "name": "AI Implementation Roadmap",
            "description": "Detailed timeline and milestones for AI implementation",
            "content": strategy.get("roadmap", {}),
            "format": "Interactive Timeline + PDF",
            "estimated_pages": 15
        })
        
        # Success tracking plan
        deliverables.append({
            "type": "Success Tracking Plan",
            "name": "Success Metrics & Tracking Framework",
            "description": "Comprehensive success measurement and tracking plan",
            "content": success_plan,
            "format": "Dashboard + PDF Report",
            "estimated_pages": 20
        })
        
        # Process recommendations
        deliverables.append({
            "type": "Process Recommendations",
            "name": "AI Process Implementation Guide",
            "description": "Detailed implementation guides for recommended AI processes",
            "content": {
                "processes": strategy.get("process_recommendations", []),
                "implementation_steps": self._extract_implementation_steps(strategy),
                "tool_recommendations": self._extract_tool_recommendations(strategy)
            },
            "format": "Interactive Guide + PDF",
            "estimated_pages": 30
        })
        
        # Risk assessment
        deliverables.append({
            "type": "Risk Assessment",
            "name": "AI Implementation Risk Assessment",
            "description": "Comprehensive risk analysis and mitigation strategies",
            "content": {
                "risks": strategy.get("risk_mitigation", []),
                "mitigation_strategies": self._extract_mitigation_strategies(strategy),
                "contingency_plans": self._generate_contingency_plans(strategy)
            },
            "format": "Risk Matrix + PDF Report",
            "estimated_pages": 15
        })
        
        return deliverables
    
    def _extract_implementation_steps(self, strategy: Dict) -> List[Dict]:
        """Extract implementation steps from strategy"""
        steps = []
        
        for rec in strategy.get("process_recommendations", []):
            process_steps = rec.get("implementation_steps", [])
            steps.append({
                "process": rec.get("name", ""),
                "steps": process_steps,
                "timeline_weeks": rec.get("timeline_weeks", 0)
            })
        
        return steps
    
    def _extract_tool_recommendations(self, strategy: Dict) -> List[Dict]:
        """Extract tool recommendations from strategy"""
        tools = []
        
        for rec in strategy.get("process_recommendations", []):
            process_tools = rec.get("tools", [])
            tools.append({
                "process": rec.get("name", ""),
                "tools": process_tools,
                "budget_range": rec.get("budget_range", "")
            })
        
        return tools
    
    def _extract_mitigation_strategies(self, strategy: Dict) -> List[Dict]:
        """Extract mitigation strategies from strategy"""
        mitigations = []
        
        for rec in strategy.get("process_recommendations", []):
            process_mitigations = rec.get("mitigation_strategies", [])
            mitigations.append({
                "process": rec.get("name", ""),
                "strategies": process_mitigations
            })
        
        return mitigations
    
    def _generate_contingency_plans(self, strategy: Dict) -> List[Dict]:
        """Generate contingency plans for high-risk scenarios"""
        contingency_plans = [
            {
                "scenario": "Budget Overruns",
                "trigger": "Project costs exceed 120% of budget",
                "response": "Implement cost optimization measures and prioritize high-ROI initiatives",
                "actions": [
                    "Review and optimize tool subscriptions",
                    "Focus on quick-win implementations",
                    "Extend timeline for non-critical projects"
                ]
            },
            {
                "scenario": "Team Resistance",
                "trigger": "Employee satisfaction scores drop below 60%",
                "response": "Intensify change management and communication efforts",
                "actions": [
                    "Increase training and support resources",
                    "Implement feedback collection system",
                    "Adjust implementation pace"
                ]
            },
            {
                "scenario": "Technology Integration Issues",
                "trigger": "Integration delays exceed 2 weeks",
                "response": "Implement alternative solutions and technical workarounds",
                "actions": [
                    "Engage technical consultants",
                    "Implement manual workarounds",
                    "Consider alternative tools"
                ]
            }
        ]
        
        return contingency_plans
    
    def _generate_timeline(self, strategy: Dict) -> Dict:
        """Generate implementation timeline"""
        roadmap = strategy.get("roadmap", {})
        timeline = roadmap.get("timeline", [])
        
        total_weeks = sum(phase.get("duration_weeks", 0) for phase in timeline)
        total_months = max(6, total_weeks // 4)
        
        return {
            "total_duration_weeks": total_weeks,
            "total_duration_months": total_months,
            "phases": timeline,
            "milestones": roadmap.get("milestones", []),
            "critical_path": self._identify_critical_path(timeline)
        }
    
    def _identify_critical_path(self, timeline: List[Dict]) -> List[str]:
        """Identify critical path activities"""
        critical_activities = []
        
        for phase in timeline:
            if phase.get("phase") == 1:
                critical_activities.append("Foundation & Assessment")
            elif phase.get("phase") == 2:
                critical_activities.append("Pilot Implementation")
            elif phase.get("phase") == 3:
                critical_activities.append("Strategic Expansion")
        
        return critical_activities
    
    def _generate_investment_summary(self, strategy: Dict) -> Dict:
        """Generate investment summary"""
        strategy_summary = strategy.get("strategy_summary", {})
        expected_roi = strategy_summary.get("expected_roi", {})
        
        return {
            "total_investment_range": strategy_summary.get("total_investment_range", ""),
            "expected_roi_percentage": expected_roi.get("roi_percentage", 0),
            "payback_period_months": expected_roi.get("payback_months", 0),
            "success_probability": strategy_summary.get("success_probability", 0),
            "risk_level": self._calculate_risk_level(strategy),
            "investment_breakdown": self._breakdown_investment(strategy)
        }
    
    def _calculate_risk_level(self, strategy: Dict) -> str:
        """Calculate overall risk level"""
        success_probability = strategy.get("strategy_summary", {}).get("success_probability", 0.7)
        
        if success_probability >= 0.8:
            return "Low"
        elif success_probability >= 0.6:
            return "Medium"
        else:
            return "High"
    
    def _breakdown_investment(self, strategy: Dict) -> List[Dict]:
        """Break down investment by process"""
        breakdown = []
        
        for rec in strategy.get("process_recommendations", []):
            budget_range = rec.get("budget_range", "")
            if "-" in budget_range:
                min_budget = int(budget_range.split("-")[0].replace("$", "").replace(",", ""))
                max_budget = int(budget_range.split("-")[1].replace("$", "").replace(",", ""))
                avg_budget = (min_budget + max_budget) / 2
                
                breakdown.append({
                    "process": rec.get("name", ""),
                    "budget_range": budget_range,
                    "average_budget": f"${avg_budget:,.0f}",
                    "timeline_weeks": rec.get("timeline_weeks", 0)
                })
        
        return breakdown
    
    def get_consulting_insights(self, org_profile: Dict) -> Dict:
        """Get consulting insights and recommendations"""
        
        # Generate comprehensive package
        package = self.generate_comprehensive_consulting_package(org_profile)
        
        # Extract key insights
        insights = {
            "organization": package["organization"],
            "key_insights": [
                {
                    "insight": "Strategic Approach",
                    "value": package["consulting_package"]["strategy"]["strategy_summary"]["overall_approach"],
                    "priority": "High"
                },
                {
                    "insight": "Success Probability",
                    "value": f"{package['consulting_package']['strategy']['strategy_summary']['success_probability']:.1%}",
                    "priority": "High"
                },
                {
                    "insight": "Expected ROI",
                    "value": f"{package['investment_summary']['expected_roi_percentage']}%",
                    "priority": "High"
                },
                {
                    "insight": "Payback Period",
                    "value": f"{package['investment_summary']['payback_period_months']} months",
                    "priority": "Medium"
                },
                {
                    "insight": "Risk Level",
                    "value": package["investment_summary"]["risk_level"],
                    "priority": "Medium"
                }
            ],
            "top_recommendations": [
                rec["name"] for rec in package["consulting_package"]["process_recommendations"][:3]
            ],
            "critical_success_factors": self._identify_critical_success_factors(package),
            "next_steps": self._generate_next_steps(package)
        }
        
        return insights
    
    def _identify_critical_success_factors(self, package: Dict) -> List[str]:
        """Identify critical success factors"""
        factors = [
            "Executive sponsorship and commitment",
            "Clear communication and change management",
            "Adequate budget allocation",
            "Team training and support",
            "Realistic timeline expectations"
        ]
        
        # Add sector-specific factors
        sector = package["organization"]["sector"]
        if sector == "Media":
            factors.append("Content quality maintenance during automation")
        elif sector == "Finance":
            factors.append("Regulatory compliance throughout implementation")
        elif sector == "Healthcare":
            factors.append("Patient safety and data privacy protection")
        
        return factors
    
    def _generate_next_steps(self, package: Dict) -> List[Dict]:
        """Generate next steps for implementation"""
        return [
            {
                "step": 1,
                "action": "Executive Strategy Review",
                "description": "Present strategy to executive team for approval",
                "timeline": "1-2 weeks",
                "owner": "Consultant + Executive Sponsor"
            },
            {
                "step": 2,
                "action": "Team Assessment & Training",
                "description": "Assess team readiness and begin training program",
                "timeline": "2-4 weeks",
                "owner": "HR + IT Teams"
            },
            {
                "step": 3,
                "action": "Pilot Project Selection",
                "description": "Select and initiate first pilot project",
                "timeline": "4-6 weeks",
                "owner": "Project Manager"
            },
            {
                "step": 4,
                "action": "Infrastructure Setup",
                "description": "Set up required infrastructure and tools",
                "timeline": "6-8 weeks",
                "owner": "IT Team"
            },
            {
                "step": 5,
                "action": "Implementation Launch",
                "description": "Begin full implementation program",
                "timeline": "8-12 weeks",
                "owner": "Project Team"
            }
        ]
