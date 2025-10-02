"""
AI Strategy Generator
Generate real consulting strategies and implementation roadmaps
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .processes import AIProcessLibrary

class AIStrategyGenerator:
    """Generate comprehensive AI implementation strategies"""
    
    def __init__(self):
        self.process_library = AIProcessLibrary()
        
    def generate_strategy(self, org_profile: Dict) -> Dict:
        """Generate comprehensive AI strategy for an organization"""
        
        current_score = org_profile.get("ai_adoption_score", 0)
        sector = org_profile.get("sector", "")
        budget = org_profile.get("budget", 100000)
        team_size = org_profile.get("team_size", 5)
        
        # Get process recommendations
        process_recommendations = self.process_library.get_process_recommendations(org_profile)
        
        # Generate strategy phases
        strategy_phases = self._generate_phases(org_profile, process_recommendations)
        
        # Calculate success probability
        success_probability = self._calculate_success_probability(org_profile, process_recommendations)
        
        # Generate implementation roadmap
        roadmap = self._generate_roadmap(strategy_phases)
        
        return {
            "organization": {
                "name": org_profile.get("name", "Unknown"),
                "sector": sector,
                "current_score": current_score,
                "team_size": team_size,
                "budget": budget
            },
            "strategy_summary": {
                "overall_approach": self._determine_approach(current_score),
                "success_probability": success_probability,
                "estimated_timeline_months": self._calculate_timeline(strategy_phases),
                "total_investment_range": self._calculate_investment_range(process_recommendations),
                "expected_roi": self._estimate_roi(org_profile, process_recommendations)
            },
            "phases": strategy_phases,
            "roadmap": roadmap,
            "process_recommendations": process_recommendations,
            "risk_mitigation": self._generate_risk_mitigation(org_profile, process_recommendations),
            "success_metrics": self._define_success_metrics(org_profile, process_recommendations),
            "generated_at": datetime.now().isoformat()
        }
    
    def _determine_approach(self, current_score: float) -> str:
        """Determine the overall strategic approach based on current AI maturity"""
        if current_score < 20:
            return "Foundation Building - Focus on basic AI literacy and pilot projects"
        elif current_score < 40:
            return "Accelerated Adoption - Implement proven AI solutions with quick wins"
        elif current_score < 60:
            return "Strategic Integration - Connect AI initiatives to business objectives"
        elif current_score < 80:
            return "Optimization & Scale - Refine existing AI implementations and expand"
        else:
            return "Innovation Leadership - Pioneer advanced AI applications and capabilities"
    
    def _generate_phases(self, org_profile: Dict, recommendations: List[Dict]) -> List[Dict]:
        """Generate implementation phases"""
        current_score = org_profile.get("ai_adoption_score", 0)
        
        if current_score < 30:
            # Foundation phase for low-maturity organizations
            return [
                {
                    "phase": 1,
                    "name": "Foundation & Assessment",
                    "duration_weeks": 4,
                    "focus": "AI readiness assessment and team preparation",
                    "deliverables": [
                        "AI maturity assessment report",
                        "Team training plan",
                        "Pilot project selection",
                        "Infrastructure requirements"
                    ],
                    "processes": [rec for rec in recommendations if rec.get("complexity") == "Low"][:2]
                },
                {
                    "phase": 2,
                    "name": "Pilot Implementation",
                    "duration_weeks": 8,
                    "focus": "Execute pilot projects and measure results",
                    "deliverables": [
                        "Pilot project results",
                        "Process documentation",
                        "Team feedback and lessons learned",
                        "Scaling recommendations"
                    ],
                    "processes": [rec for rec in recommendations if rec.get("complexity") in ["Low", "Medium"]][:2]
                },
                {
                    "phase": 3,
                    "name": "Strategic Expansion",
                    "duration_weeks": 12,
                    "focus": "Scale successful pilots and implement strategic initiatives",
                    "deliverables": [
                        "Scaled AI implementations",
                        "Performance metrics dashboard",
                        "ROI analysis report",
                        "Future roadmap"
                    ],
                    "processes": recommendations[:3]
                }
            ]
        else:
            # Advanced phase for higher-maturity organizations
            return [
                {
                    "phase": 1,
                    "name": "Strategic Assessment & Planning",
                    "duration_weeks": 6,
                    "focus": "Advanced AI strategy development and optimization planning",
                    "deliverables": [
                        "Advanced AI strategy document",
                        "Performance optimization plan",
                        "Innovation roadmap",
                        "Competitive analysis"
                    ],
                    "processes": recommendations[:2]
                },
                {
                    "phase": 2,
                    "name": "Advanced Implementation",
                    "duration_weeks": 16,
                    "focus": "Implement advanced AI capabilities and integrations",
                    "deliverables": [
                        "Advanced AI implementations",
                        "Integration documentation",
                        "Performance benchmarks",
                        "Innovation metrics"
                    ],
                    "processes": recommendations[:3]
                },
                {
                    "phase": 3,
                    "name": "Optimization & Innovation",
                    "duration_weeks": 12,
                    "focus": "Continuous optimization and innovation initiatives",
                    "deliverables": [
                        "Optimization results",
                        "Innovation pipeline",
                        "Future technology roadmap",
                        "Leadership positioning strategy"
                    ],
                    "processes": recommendations[2:]
                }
            ]
    
    def _generate_roadmap(self, phases: List[Dict]) -> Dict:
        """Generate detailed implementation roadmap"""
        roadmap = {
            "timeline": [],
            "milestones": [],
            "dependencies": []
        }
        
        current_date = datetime.now()
        
        for phase in phases:
            phase_start = current_date
            phase_end = current_date + timedelta(weeks=phase["duration_weeks"])
            
            roadmap["timeline"].append({
                "phase": phase["phase"],
                "name": phase["name"],
                "start_date": phase_start.strftime("%Y-%m-%d"),
                "end_date": phase_end.strftime("%Y-%m-%d"),
                "duration_weeks": phase["duration_weeks"],
                "focus": phase["focus"]
            })
            
            # Add milestones
            for i, deliverable in enumerate(phase["deliverables"]):
                milestone_date = phase_start + timedelta(weeks=(i + 1) * phase["duration_weeks"] // len(phase["deliverables"]))
                roadmap["milestones"].append({
                    "phase": phase["phase"],
                    "deliverable": deliverable,
                    "target_date": milestone_date.strftime("%Y-%m-%d"),
                    "status": "Planned"
                })
            
            current_date = phase_end
        
        return roadmap
    
    def _calculate_success_probability(self, org_profile: Dict, recommendations: List[Dict]) -> float:
        """Calculate probability of successful AI implementation"""
        base_probability = 0.7
        
        # Factors that increase success probability
        if org_profile.get("team_size", 0) >= 5:
            base_probability += 0.1
        
        if org_profile.get("budget", 0) >= 100000:
            base_probability += 0.1
        
        if org_profile.get("ai_adoption_score", 0) > 30:
            base_probability += 0.1
        
        # Factors that decrease success probability
        if len([r for r in recommendations if r.get("complexity") == "High"]) > 2:
            base_probability -= 0.1
        
        return min(0.95, max(0.3, base_probability))
    
    def _calculate_timeline(self, phases: List[Dict]) -> int:
        """Calculate total timeline in months"""
        total_weeks = sum(phase["duration_weeks"] for phase in phases)
        return max(6, total_weeks // 4)  # Minimum 6 months
    
    def _calculate_investment_range(self, recommendations: List[Dict]) -> str:
        """Calculate total investment range"""
        total_min = 0
        total_max = 0
        
        for rec in recommendations[:3]:  # Top 3 recommendations
            budget_range = rec.get("budget_range", "")
            if "-" in budget_range:
                min_budget = int(budget_range.split("-")[0].replace("$", "").replace(",", ""))
                max_budget = int(budget_range.split("-")[1].replace("$", "").replace(",", ""))
                total_min += min_budget
                total_max += max_budget
        
        return f"${total_min:,}-${total_max:,}"
    
    def _estimate_roi(self, org_profile: Dict, recommendations: List[Dict]) -> Dict:
        """Estimate ROI for the strategy"""
        total_investment = 0
        for rec in recommendations[:3]:
            budget_range = rec.get("budget_range", "")
            if "-" in budget_range:
                avg_budget = sum(int(x.replace("$", "").replace(",", "")) for x in budget_range.split("-")) / 2
                total_investment += avg_budget
        
        # Estimate benefits based on sector and current score
        sector = org_profile.get("sector", "")
        current_score = org_profile.get("ai_adoption_score", 0)
        
        # ROI multipliers by sector
        roi_multipliers = {
            "Media": 2.5,
            "Communications": 2.2,
            "Finance": 3.0,
            "Healthcare": 2.8,
            "Manufacturing": 2.3,
            "Retail": 2.6
        }
        
        multiplier = roi_multipliers.get(sector, 2.0)
        estimated_benefits = total_investment * multiplier
        roi_percentage = ((estimated_benefits - total_investment) / total_investment) * 100
        
        return {
            "total_investment": f"${total_investment:,.0f}",
            "estimated_benefits": f"${estimated_benefits:,.0f}",
            "roi_percentage": round(roi_percentage, 1),
            "payback_months": round(total_investment / (estimated_benefits / 12), 1)
        }
    
    def _generate_risk_mitigation(self, org_profile: Dict, recommendations: List[Dict]) -> List[Dict]:
        """Generate risk mitigation strategies"""
        risks = []
        
        # Common risks
        risks.append({
            "risk": "Change Management Resistance",
            "probability": "Medium",
            "impact": "High",
            "mitigation": "Comprehensive change management program with stakeholder engagement"
        })
        
        risks.append({
            "risk": "Budget Overruns",
            "probability": "Medium",
            "impact": "Medium",
            "mitigation": "Phased implementation with regular budget reviews and contingency planning"
        })
        
        risks.append({
            "risk": "Technology Integration Issues",
            "probability": "High",
            "impact": "Medium",
            "mitigation": "Thorough technical assessment and pilot testing before full deployment"
        })
        
        # Add process-specific risks
        for rec in recommendations[:2]:
            for risk in rec.get("risks", []):
                risks.append({
                    "risk": f"{rec['name']}: {risk}",
                    "probability": "Medium",
                    "impact": "Medium",
                    "mitigation": "Follow recommended mitigation strategies and best practices"
                })
        
        return risks
    
    def _define_success_metrics(self, org_profile: Dict, recommendations: List[Dict]) -> List[Dict]:
        """Define success metrics for the strategy"""
        metrics = []
        
        # Overall AI adoption metrics
        metrics.append({
            "category": "AI Maturity",
            "metric": "AI Adoption Score Improvement",
            "target": f"From {org_profile.get('ai_adoption_score', 0)} to {min(100, org_profile.get('ai_adoption_score', 0) + 25)}",
            "measurement": "Quarterly assessment"
        })
        
        metrics.append({
            "category": "Business Impact",
            "metric": "ROI Achievement",
            "target": "Meet or exceed projected ROI within 18 months",
            "measurement": "Monthly financial review"
        })
        
        # Process-specific metrics
        for rec in recommendations[:3]:
            for metric in rec.get("success_metrics", []):
                metrics.append({
                    "category": rec["name"],
                    "metric": metric,
                    "target": "TBD based on baseline assessment",
                    "measurement": "Monthly tracking"
                })
        
        return metrics
