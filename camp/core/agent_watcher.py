"""
agent_watcher.py — Agent watcher core engine for CAMP.

Manages the registration and state tracking of multiple AI agents using
AdaptiveInteraction (AIx) coupling links.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from RealityOS.kernel.adaptive_interaction import AdaptiveInteraction
from RealityOS.kernel.evolution import DecentralizedEvolution
from RealityOS.kernel.calculus import Calculus


class AgentWatcher:
    """
    Monitors a fleet of AI agents. Uses the first-principles AIx interaction
    primitives to track coupling strength, predict spikes, and calculate surprise.
    """

    def __init__(self, eta: float = 0.15, alpha: float = 0.15):
        # We use alpha as a momentum term inside the watcher if needed
        self.evolution_engine = DecentralizedEvolution(eta_base=eta)
        self.interactions: Dict[str, AdaptiveInteraction] = {}
        self.processed_event_count = 0

    def register_agent(
        self,
        agent_id: str,
        value: float = 0.5,
        cost_limit: float = 1.0,
        latency_limit: float = 5000.0,
        error_limit: float = 0.2,
        token_limit: float = 20.0,
    ) -> AdaptiveInteraction:
        """Register a new agent as an AdaptiveInteraction between observations and targets."""
        # Create an interaction coupling state_a (observation) and state_b (target/goal)
        ix = AdaptiveInteraction(
            state_a_id=f"{agent_id}_obs",
            state_b_id=f"{agent_id}_target",
            dim=4,
            uid=agent_id,
            capacity=value  # Capacity scales directly with agent importance value
        )
        
        # Define constraints as penalty functions: violation = weight * max(0, val - limit)^2
        def cost_constraint(z: List[float]) -> float:
            return 20.0 * (max(0.0, z[0] - cost_limit) ** 2)

        def latency_constraint(z: List[float]) -> float:
            return 0.01 * (max(0.0, z[1] - latency_limit) ** 2)

        def error_constraint(z: List[float]) -> float:
            return 50.0 * (max(0.0, z[2] - error_limit) ** 2)

        def token_constraint(z: List[float]) -> float:
            return 0.5 * (max(0.0, z[3] - token_limit) ** 2)

        Calculus.Constrain(ix, cost_constraint)
        Calculus.Constrain(ix, latency_constraint)
        Calculus.Constrain(ix, error_constraint)
        Calculus.Constrain(ix, token_constraint)

        # Store limits in metadata for UI constraints rendering
        ix.meta["constraints"] = [
            {"name": "cost_limit", "upper": cost_limit},
            {"name": "latency_limit", "upper": latency_limit},
            {"name": "error_limit", "upper": error_limit},
            {"name": "token_limit", "upper": token_limit}
        ]

        self.interactions[agent_id] = ix
        return ix

    def get_agent(self, agent_id: str) -> Optional[AdaptiveInteraction]:
        return self.interactions.get(agent_id)

    def observe(
        self,
        agent_id: str,
        cost: float,
        latency: float,
        error: float,
        tokens: float
    ) -> Dict[str, Any]:
        """
        Ingest a new metrics sample, run Observe and step the evolution engine.
        """
        self.processed_event_count += 1
        
        # Auto-register if not seen yet
        if agent_id not in self.interactions:
            self.register_agent(agent_id)

        ix = self.interactions[agent_id]
        
        # Observe: update state A (actual observation) and B (target baseline, which relaxes)
        obs_a = [cost, latency, error, tokens]
        obs_b = [cost * 0.9, latency * 0.9, error * 0.9, tokens * 0.9] # target relaxes slightly below
        
        Calculus.Observe(ix, obs_a, obs_b, confidence=1.0, source="watcher")

        # Step evolution local gradient update
        telemetry = self.evolution_engine.step(ix)

        # Re-attach metrics vector fields for API / UI rendering
        telemetry["belief"] = list(ix.z_b) # z_b represents the smoothed/predicted target belief coordinate
        telemetry["raw"] = list(ix.z_a)
        telemetry["confused"] = ix.meta["surprise_ema"] > 1.5
        telemetry["needs_help"] = ix.dt < 0.05
        return telemetry

    def get_all_states(self) -> List[Dict[str, Any]]:
        """Return snapshots of all monitored agents."""
        states = []
        for ix in self.interactions.values():
            states.append({
                "uid": ix.uid,
                "x": list(ix.z_a),
                "belief": list(ix.z_b),
                "surprise": ix.meta.get("surprise", 0.0),
                "energy": ix.capacity,  # capacity replaces energy budget representation
                "tau": list(ix.precision),
                "timestamp": time.time(),
                "version": len(ix.evidence),
                "meta": ix.meta
            })
        return states

