"""
ChatGPT Agents Monitoring System
===============================

Comprehensive monitoring and performance tracking for ChatGPT Agents.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    """Metrics for agent performance monitoring"""
    agent_name: str
    timestamp: datetime
    data_collected: int
    learning_cycles: int
    insights_generated: int
    chatgpt_requests: int
    chatgpt_success_rate: float
    response_time_avg: float
    error_count: int
    knowledge_base_size: int
    last_learning_time: Optional[datetime]
    status: str

@dataclass
class SystemMetrics:
    """Overall system metrics"""
    timestamp: datetime
    total_agents: int
    active_agents: int
    total_data_collected: int
    total_insights: int
    total_chatgpt_requests: int
    system_health_score: float
    uptime_hours: float

class AgentMonitor:
    """Monitor and track ChatGPT Agents performance"""
    
    def __init__(self, storage_path: str = "backend/agents/monitoring"):
        self.storage_path = storage_path
        self.metrics_history: List[AgentMetrics] = []
        self.system_metrics_history: List[SystemMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        
        # Initialize storage
        os.makedirs(storage_path, exist_ok=True)
        self.metrics_file = os.path.join(storage_path, "metrics_history.json")
        self.alerts_file = os.path.join(storage_path, "alerts.json")
        
        # Load existing data
        self._load_metrics_history()
        self._load_alerts()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_time = datetime.utcnow()
        
        logger.info("📊 Agent Monitor initialized")
    
    def start_monitoring(self, interval_minutes: int = 5):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("⚠️ Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"📊 Started monitoring with {interval_minutes} minute intervals")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("📊 Monitoring stopped")
    
    def _monitoring_loop(self, interval_minutes: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_metrics()
                self._check_alerts()
                self._save_metrics_history()
                
                # Sleep for the specified interval
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _collect_metrics(self):
        """Collect current metrics from agents"""
        try:
            from .agent_manager import agent_manager
            
            # Get agent status
            agent_status = agent_manager.get_agent_status()
            performance = agent_manager.get_agent_performance()
            
            # Collect metrics for each agent
            for agent_name, status in agent_status.items():
                perf = performance.get(agent_name, {})
                
                metrics = AgentMetrics(
                    agent_name=agent_name,
                    timestamp=datetime.utcnow(),
                    data_collected=status.get("total_data_collected", 0),
                    learning_cycles=status.get("learning_cycles", 0),
                    insights_generated=len(agent_manager.get_agent_insights(agent_name, limit=100)),
                    chatgpt_requests=perf.get("chatgpt_requests", 0),
                    chatgpt_success_rate=perf.get("chatgpt_success_rate", 0.0),
                    response_time_avg=perf.get("response_time_avg", 0.0),
                    error_count=perf.get("error_count", 0),
                    knowledge_base_size=status.get("knowledge_base_size", 0),
                    last_learning_time=datetime.fromisoformat(status["last_learning_time"]) if status.get("last_learning_time") else None,
                    status="active" if status.get("should_run_cycle", False) else "idle"
                )
                
                self.metrics_history.append(metrics)
            
            # Collect system metrics
            system_metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                total_agents=len(agent_status),
                active_agents=sum(1 for s in agent_status.values() if s.get("should_run_cycle", False)),
                total_data_collected=sum(s.get("total_data_collected", 0) for s in agent_status.values()),
                total_insights=sum(len(agent_manager.get_agent_insights(name, limit=100)) for name in agent_status.keys()),
                total_chatgpt_requests=sum(p.get("chatgpt_requests", 0) for p in performance.values()),
                system_health_score=self._calculate_health_score(agent_status, performance),
                uptime_hours=(datetime.utcnow() - self.start_time).total_seconds() / 3600
            )
            
            self.system_metrics_history.append(system_metrics)
            
            # Keep only recent metrics (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            self.system_metrics_history = [m for m in self.system_metrics_history if m.timestamp > cutoff_time]
            
            logger.info(f"📊 Collected metrics for {len(agent_status)} agents")
            
        except Exception as e:
            logger.error(f"❌ Error collecting metrics: {e}")
    
    def _calculate_health_score(self, agent_status: Dict, performance: Dict) -> float:
        """Calculate overall system health score"""
        try:
            if not agent_status:
                return 0.0
            
            scores = []
            
            for agent_name, status in agent_status.items():
                perf = performance.get(agent_name, {})
                
                # Data collection score (0-1)
                data_score = min(status.get("total_data_collected", 0) / 100, 1.0)
                
                # Learning activity score (0-1)
                learning_score = min(status.get("learning_cycles", 0) / 10, 1.0)
                
                # ChatGPT success rate (0-1)
                chatgpt_score = perf.get("chatgpt_success_rate", 0.0)
                
                # Error rate score (inverted, 0-1)
                error_score = max(0, 1.0 - (perf.get("error_count", 0) / 10))
                
                # Combined score
                agent_score = (data_score + learning_score + chatgpt_score + error_score) / 4
                scores.append(agent_score)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating health score: {e}")
            return 0.0
    
    def _check_alerts(self):
        """Check for alert conditions"""
        try:
            if not self.metrics_history:
                return
            
            # Get latest metrics
            latest_metrics = self.metrics_history[-1]
            
            # Check for various alert conditions
            alerts_to_add = []
            
            # High error rate alert
            if latest_metrics.error_count > 5:
                alerts_to_add.append({
                    "type": "high_error_rate",
                    "severity": "warning",
                    "message": f"High error rate detected for {latest_metrics.agent_name}: {latest_metrics.error_count} errors",
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": latest_metrics.agent_name
                })
            
            # Low ChatGPT success rate alert
            if latest_metrics.chatgpt_success_rate < 0.8:
                alerts_to_add.append({
                    "type": "low_success_rate",
                    "severity": "warning",
                    "message": f"Low ChatGPT success rate for {latest_metrics.agent_name}: {latest_metrics.chatgpt_success_rate:.2%}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": latest_metrics.agent_name
                })
            
            # No data collection alert
            if latest_metrics.data_collected == 0 and latest_metrics.learning_cycles > 0:
                alerts_to_add.append({
                    "type": "no_data_collection",
                    "severity": "error",
                    "message": f"No data collected by {latest_metrics.agent_name} despite learning cycles",
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": latest_metrics.agent_name
                })
            
            # Add new alerts
            for alert in alerts_to_add:
                if not self._is_duplicate_alert(alert):
                    self.alerts.append(alert)
            
            # Keep only recent alerts (last 7 days)
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            self.alerts = [a for a in self.alerts if datetime.fromisoformat(a["timestamp"]) > cutoff_time]
            
            if alerts_to_add:
                logger.warning(f"⚠️ Generated {len(alerts_to_add)} new alerts")
            
        except Exception as e:
            logger.error(f"❌ Error checking alerts: {e}")
    
    def _is_duplicate_alert(self, new_alert: Dict[str, Any]) -> bool:
        """Check if alert is a duplicate of recent alerts"""
        try:
            recent_time = datetime.utcnow() - timedelta(hours=1)
            
            for existing_alert in self.alerts[-10:]:  # Check last 10 alerts
                if (existing_alert["type"] == new_alert["type"] and
                    existing_alert["agent"] == new_alert["agent"] and
                    datetime.fromisoformat(existing_alert["timestamp"]) > recent_time):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking duplicate alerts: {e}")
            return False
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        try:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            latest_metrics = self.metrics_history[-1]
            latest_system = self.system_metrics_history[-1] if self.system_metrics_history else None
            
            return {
                "agent_metrics": asdict(latest_metrics),
                "system_metrics": asdict(latest_system) if latest_system else None,
                "monitoring_active": self.monitoring_active,
                "uptime_hours": (datetime.utcnow() - self.start_time).total_seconds() / 3600
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting current metrics: {e}")
            return {"error": str(e)}
    
    def get_metrics_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics history for specified time period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            filtered_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            filtered_system = [m for m in self.system_metrics_history if m.timestamp > cutoff_time]
            
            return {
                "agent_metrics": [asdict(m) for m in filtered_metrics],
                "system_metrics": [asdict(m) for m in filtered_system],
                "time_period_hours": hours,
                "total_records": len(filtered_metrics)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting metrics history: {e}")
            return {"error": str(e)}
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by severity"""
        try:
            if severity:
                return [a for a in self.alerts if a.get("severity") == severity]
            return self.alerts
            
        except Exception as e:
            logger.error(f"❌ Error getting alerts: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            # Calculate averages and trends
            recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
            
            avg_data_collected = sum(m.data_collected for m in recent_metrics) / len(recent_metrics)
            avg_insights = sum(m.insights_generated for m in recent_metrics) / len(recent_metrics)
            avg_success_rate = sum(m.chatgpt_success_rate for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics)
            
            # Calculate trends
            if len(recent_metrics) >= 2:
                data_trend = recent_metrics[-1].data_collected - recent_metrics[0].data_collected
                insights_trend = recent_metrics[-1].insights_generated - recent_metrics[0].insights_generated
            else:
                data_trend = 0
                insights_trend = 0
            
            return {
                "summary": {
                    "avg_data_collected": avg_data_collected,
                    "avg_insights_generated": avg_insights,
                    "avg_chatgpt_success_rate": avg_success_rate,
                    "avg_response_time_ms": avg_response_time,
                    "data_trend": data_trend,
                    "insights_trend": insights_trend
                },
                "alerts": {
                    "total": len(self.alerts),
                    "warnings": len([a for a in self.alerts if a.get("severity") == "warning"]),
                    "errors": len([a for a in self.alerts if a.get("severity") == "error"])
                },
                "system_health": self.system_metrics_history[-1].system_health_score if self.system_metrics_history else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def _load_metrics_history(self):
        """Load metrics history from file"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = [
                        AgentMetrics(**{**m, "timestamp": datetime.fromisoformat(m["timestamp"])})
                        for m in data.get("agent_metrics", [])
                    ]
                    self.system_metrics_history = [
                        SystemMetrics(**{**m, "timestamp": datetime.fromisoformat(m["timestamp"])})
                        for m in data.get("system_metrics", [])
                    ]
                logger.info(f"📊 Loaded {len(self.metrics_history)} metrics records")
        except Exception as e:
            logger.error(f"❌ Error loading metrics history: {e}")
    
    def _load_alerts(self):
        """Load alerts from file"""
        try:
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r') as f:
                    self.alerts = json.load(f)
                logger.info(f"📊 Loaded {len(self.alerts)} alerts")
        except Exception as e:
            logger.error(f"❌ Error loading alerts: {e}")
    
    def _save_metrics_history(self):
        """Save metrics history to file"""
        try:
            data = {
                "agent_metrics": [asdict(m) for m in self.metrics_history],
                "system_metrics": [asdict(m) for m in self.system_metrics_history],
                "last_updated": datetime.utcnow().isoformat()
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error saving metrics history: {e}")
        
        try:
            with open(self.alerts_file, 'w') as f:
                json.dump(self.alerts, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error saving alerts: {e}")

# Global monitor instance
agent_monitor = AgentMonitor()
