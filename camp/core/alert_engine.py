"""
alert_engine.py — Priority-scheduled debounced alert engine for CAMP.

Surfaces alerts only when the momentum-damped belief state violates constraints.
Uses value-weighted priority (value * surprise) to schedule and route alerts
when compute/alert budget is saturated.
"""

import time
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from RealityOS.kernel.cognitive_state import CognitiveState


@dataclass(order=True)
class Alert:
    priority: float = field(compare=True)  # negative priority for max-heap usage
    agent_id: str = field(compare=False)
    metric: str = field(compare=False)
    message: str = field(compare=False)
    value: float = field(compare=False)
    limit: float = field(compare=False)
    timestamp: float = field(compare=False, default_factory=time.time)
    resolved: bool = field(compare=False, default=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": abs(self.priority),
            "agent_id": self.agent_id,
            "metric": self.metric,
            "message": self.message,
            "value": self.value,
            "limit": self.limit,
            "timestamp": self.timestamp,
            "resolved": self.resolved
        }


class AlertEngine:
    def __init__(self, alert_budget_per_tick: int = 5):
        self.alert_budget_per_tick = alert_budget_per_tick
        self.alert_history: List[Alert] = []
        # Track active alert states to prevent duplicates
        self.active_alerts: Dict[Tuple[str, str], Alert] = {}

    def check_agent(self, s: CognitiveState) -> List[Alert]:
        """
        Check agent smoothed belief against constraints.
        Surfaces alerts sorted by priority (agent.value * metric surprise).
        """
        s.ensure_dimensions()
        tick_alerts = []
        
        # Metric maps: 0=cost, 1=latency, 2=error, 3=tokens
        metric_names = ["cost", "latency", "error_rate", "tokens"]
        
        for c in s.constraints:
            dim = c.dimension
            if dim >= len(s.belief) or dim >= len(metric_names):
                continue
                
            val = s.belief[dim]
            limit = c.upper
            
            # Check for violation in the belief (not raw observation!)
            if val > limit:
                metric_name = metric_names[dim]
                
                # Priority = agent importance * magnitude of surprise/violation
                surprise_factor = (val - limit) / (limit if limit > 0 else 1.0)
                priority_score = s.value * (1.0 + surprise_factor)
                
                msg = f"Agent '{s.uid}' {metric_name} exceeded threshold: {val:.3f} > {limit:.3f}"
                
                # We use negative priority because heapq in Python is a min-heap
                alert = Alert(
                    priority=-priority_score,
                    agent_id=s.uid,
                    metric=metric_name,
                    message=msg,
                    value=val,
                    limit=limit
                )
                tick_alerts.append(alert)
                
        return tick_alerts

    def process_tick_alerts(self, agents: List[CognitiveState]) -> List[Alert]:
        """
        Gathers all candidate alerts across all agents, schedules them by priority,
        enforces the alert budget limit, and updates history.
        """
        candidates: List[Alert] = []
        for agent in agents:
            candidates.extend(self.check_agent(agent))

        # Sort candidates by priority (highest value * surprise first)
        # Note: Alert objects compare by their .priority attribute (which is negative)
        heapq.heapify(candidates)

        surfaced_alerts = []
        processed_count = 0
        
        while candidates and processed_count < self.alert_budget_per_tick:
            alert = heapq.heappop(candidates)
            key = (alert.agent_id, alert.metric)
            
            # Update history and active alerts
            self.active_alerts[key] = alert
            self.alert_history.append(alert)
            surfaced_alerts.append(alert)
            processed_count += 1

        # Check resolution for alerts no longer active
        active_keys = {(a.agent_id, a.metric) for a in surfaced_alerts}
        resolved_keys = []
        for key, active_alert in self.active_alerts.items():
            if key not in active_keys:
                # Metric recovered below threshold
                agent_id, metric = key
                active_alert.resolved = True
                resolved_keys.append(key)
                
        for key in resolved_keys:
            del self.active_alerts[key]

        return surfaced_alerts

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        # Return newest first
        sorted_history = sorted(self.alert_history, key=lambda a: a.timestamp, reverse=True)
        return [a.to_dict() for a in sorted_history[:limit]]
