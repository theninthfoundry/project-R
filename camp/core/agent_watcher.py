"""
agent_watcher.py — Agent watcher core engine for CAMP.

Manages the registration and state tracking of multiple AI agents.
Each agent is mapped to a 4D CognitiveState:
    x[0] = cost_per_call ($)
    x[1] = latency_ms (ms)
    x[2] = error_rate (0.0 to 1.0)
    x[3] = token_count (thousands of tokens)
"""

import time
from typing import Dict, List, Any, Optional
from RealityOS.kernel.cognitive_state import CognitiveState, Constraint
from RealityOS.kernel.evolution import AnalyticEvolution


class AgentWatcher:
    """
    Monitors a fleet of AI agents. Uses the validated Cognitive OS engine
    to track belief states, detect anomalies (surprise), and optimize compute.
    """

    def __init__(self, eta: float = 0.1, alpha: float = 0.15):
        self.evolution_engine = AnalyticEvolution(eta=eta, alpha=alpha, lambda_c=15.0, lambda_g=0.5)
        self.agents: Dict[str, CognitiveState] = {}
        # Keep track of total logs processed
        self.processed_event_count = 0

    def register_agent(
        self,
        agent_id: str,
        value: float = 0.5,
        cost_limit: float = 1.0,      # max cost per call ($)
        latency_limit: float = 5000.0, # max latency per call (ms)
        error_limit: float = 0.2,      # max error rate (0.0 to 1.0)
        token_limit: float = 20.0,     # max token count (thousands)
    ) -> CognitiveState:
        """Register a new agent for active monitoring."""
        # Create standard constraints based on limits
        constraints = [
            Constraint(name="cost_limit", dimension=0, upper=cost_limit, weight=20.0),
            Constraint(name="latency_limit", dimension=1, upper=latency_limit, weight=0.01),
            Constraint(name="error_limit", dimension=2, upper=error_limit, weight=50.0),
            Constraint(name="token_limit", dimension=3, upper=token_limit, weight=0.5)
        ]

        # Initial state is 0.0 for all metrics
        initial_x = [0.0, 0.0, 0.0, 0.0]
        s = CognitiveState(
            uid=agent_id,
            type_tag="agent",
            x=initial_x,
            value=value,
            constraints=constraints
        )
        s.ensure_dimensions()
        self.agents[agent_id] = s
        return s

    def get_agent(self, agent_id: str) -> Optional[CognitiveState]:
        return self.agents.get(agent_id)

    def observe(
        self,
        agent_id: str,
        cost: float,
        latency: float,
        error: float,
        tokens: float
    ) -> Dict[str, Any]:
        """
        Record a new log measurement (x) for the given agent and
        evolve its cognitive state one step.
        """
        self.processed_event_count += 1
        
        # Auto-register if not seen yet
        if agent_id not in self.agents:
            self.register_agent(agent_id)

        s = self.agents[agent_id]
        obs = [cost, latency, error, tokens]
        
        # Run evolution step (Eq 1)
        telemetry = self.evolution_engine.step(s, obs)
        
        # Add metrics helper attributes to the return payload
        telemetry["belief"] = list(s.belief)
        telemetry["raw"] = obs
        telemetry["confused"] = s.meta.get("confused", False)
        telemetry["needs_help"] = s.meta.get("needs_help", False)
        return telemetry

    def get_all_states(self) -> List[Dict[str, Any]]:
        """Return list of serialized snapshots of all monitored agents."""
        return [s.snapshot() for s in self.agents.values()]
