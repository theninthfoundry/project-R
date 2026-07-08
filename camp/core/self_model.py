"""
self_model.py — System self-monitoring and health engine for CAMP.

Implements a dedicated CognitiveState that monitors the health of the
CAMP observability service itself (memory, scheduler load, processed event count).
"""

import time
import os
from typing import Dict, Any
from RealityOS.kernel.cognitive_state import CognitiveState, Constraint
from RealityOS.kernel.evolution import AnalyticEvolution


class SelfModel:
    def __init__(self, uid: str = "camp_system"):
        self.evolution_engine = AnalyticEvolution(eta=0.05, alpha=0.2)
        
        # State vector representation for the system:
        # x[0] = memory_usage_mb
        # x[1] = event_queue_depth
        # x[2] = ticks_per_second
        # x[3] = alerts_raised_total
        self.state = CognitiveState(
            uid=uid,
            type_tag="system_self",
            x=[0.0, 0.0, 0.0, 0.0],
            value=1.0,  # High value self-model
            constraints=[
                Constraint(name="memory_ceiling", dimension=0, upper=512.0, weight=10.0),
                Constraint(name="queue_backlog", dimension=1, upper=1000.0, weight=5.0)
            ]
        )
        self.state.ensure_dimensions()
        self.start_time = time.time()
        self.tick_count = 0

    def get_system_metrics(self, queue_depth: int, alerts_count: int) -> Dict[str, Any]:
        """Collect current system status metrics."""
        self.tick_count += 1
        
        # Estimate memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback mock memory estimation
            mem_mb = 42.5 + (self.tick_count % 10) * 0.1
            
        elapsed = time.time() - self.start_time
        ticks_per_sec = self.tick_count / max(1.0, elapsed)

        obs = [mem_mb, float(queue_depth), ticks_per_sec, float(alerts_count)]
        
        # Evolve system self-state
        self.evolution_engine.step(self.state, obs)

        # Meta flags update
        self.state.meta["uptime"] = elapsed
        self.state.meta["stable"] = mem_mb < 256.0 and queue_depth < 100

        return {
            "uptime": elapsed,
            "memory_mb": mem_mb,
            "queue_depth": queue_depth,
            "ticks_per_sec": ticks_per_sec,
            "alerts_raised": alerts_count,
            "system_state": self.state.snapshot()
        }
