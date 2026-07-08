"""
evolution.py — The universal evolution operator for Cognitive States.

Implements the 7-step cognitive cycle:
    1. Predict   → generate x_hat from current state + learned dynamics
    2. Observe   → receive observation from environment
    3. Surprise  → compute ε = ||x_obs - x_hat||_tau
    4. Update    → gradient-like correction: belief ← belief - η·∇(surprise + λ·constraints²)
    5. Trust     → adjust tau based on surprise history
    6. Energy    → deduct computational cost; if exhausted → maintenance mode
    7. Predict   → generate next prediction for the next cycle

Two implementations:
    - AnalyticEvolution: Pure Python, rule-based (zero dependencies)
    - (Future) NeuralEvolution: PyTorch Neural ODE wrapper
"""

import math
from typing import List, Optional, Dict, Any
from RealityOS.kernel.cognitive_state import CognitiveState, Constraint


class AnalyticEvolution:
    """
    Pure-Python evolution operator. Uses linear prediction + Euler
    integration to evolve CognitiveState without any ML library.

    This is the default engine for Phase 0 / CAMP. It can be replaced
    by NeuralEvolution (PyTorch Neural ODE) in later phases without
    changing any calling code.
    """

    def __init__(
        self,
        eta: float = 0.1,          # base learning rate / temperature
        alpha: float = 0.15,       # belief momentum coefficient
        lambda_c: float = 10.0,    # constraint penalty weight
        lambda_g: float = 1.0,     # intent/goal weight
        energy_cost_per_step: float = 0.005,
    ):
        self.eta = eta
        self.alpha = alpha
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g
        self.energy_cost_per_step = energy_cost_per_step

    def predict(self, s: CognitiveState):
        """
        Step 1: Generate one-step-ahead prediction from current belief
        and velocity (xdot). Simple linear extrapolation.
        """
        s.ensure_dimensions()
        d = s.dim()
        for i in range(d):
            # Linear prediction: next = belief + velocity
            s.prediction[i] = s.belief[i] + s.xdot[i]

    def observe(self, s: CognitiveState, observation: List[float]):
        """
        Step 2: Receive new observation from environment.
        Updates the observable state x.
        """
        s.x = list(observation)
        s.ensure_dimensions()
        s.timestamp = __import__('time').time()

    def compute_surprise(self, s: CognitiveState) -> float:
        """
        Step 3: Compute prediction error (surprise).
        Mahalanobis distance weighted by precision tau.
        """
        return s.compute_surprise()

    def update_belief(self, s: CognitiveState, surprise: float):
        """
        Step 4: Gradient-like correction of belief toward observation.

        The update combines:
        - Momentum-damped belief update (α-weighted EMA)
        - Constraint projection (push state back into feasible region)
        - Goal attraction (pull toward goal attractor if present)
        """
        s.ensure_dimensions()
        d = s.dim()

        # 4a. Momentum update: belief moves toward observation
        s.momentum_update(s.x, self.alpha)

        # 4b. Constraint projection
        for c in s.constraints:
            violation = c.violation(s.belief)
            if violation > 0.0 and c.dimension < d:
                dim = c.dimension
                val = s.belief[dim]
                if val < c.lower:
                    s.belief[dim] += self.eta * self.lambda_c * (c.lower - val)
                elif val > c.upper:
                    s.belief[dim] -= self.eta * self.lambda_c * (val - c.upper)

        # 4c. Goal attraction
        if s.goal:
            for i in range(min(d, len(s.goal))):
                direction = s.goal[i] - s.belief[i]
                s.belief[i] += self.eta * self.lambda_g * s.value * direction

        # 4d. Update velocity estimate (finite difference)
        for i in range(d):
            new_xdot = s.x[i] - (s.belief[i] if i < len(s.belief) else 0.0)
            # Smooth the velocity estimate
            s.xdot[i] = s.xdot[i] * 0.7 + new_xdot * 0.3

    def update_trust(self, s: CognitiveState, surprise: float):
        """
        Step 5: Adapt precision/trust based on surprise.
        """
        s.update_trust(surprise)

        # Update meta-cognition EMA of surprise
        ema = s.meta.get("surprise_ema", 0.0)
        s.meta["surprise_ema"] = ema * 0.9 + surprise * 0.1

    def update_energy(self, s: CognitiveState, surprise: float) -> bool:
        """
        Step 6: Deduct energy. Higher surprise = more energy consumed.
        Returns False if state is in maintenance mode.
        """
        cost = self.energy_cost_per_step + surprise * 0.001
        return s.consume_energy(cost)

    def update_metacognition(self, s: CognitiveState, surprise: float):
        """
        Evaluate meta-cognition flags based on surprise history.
        """
        ema = s.meta.get("surprise_ema", 0.0)
        ticks_confused = s.meta.get("ticks_confused", 0)

        if ema > 2.0:
            ticks_confused += 1
        else:
            ticks_confused = max(0, ticks_confused - 1)

        s.meta["ticks_confused"] = ticks_confused
        s.meta["confused"] = ticks_confused > 5
        s.meta["stable"] = ema < 0.5 and s.energy_budget > 0.1

        # Overconfident: trust is high but surprise is rising
        avg_tau = sum(s.tau) / max(len(s.tau), 1)
        s.meta["overconfident"] = avg_tau > 3.0 and surprise > 1.0

        # Needs help: energy low AND surprise high
        s.meta["needs_help"] = s.energy_budget < 0.1 and ema > 1.0

    def step(self, s: CognitiveState, observation: List[float]) -> Dict[str, Any]:
        """
        Execute one full evolution cycle: predict → observe → surprise →
        update → trust → energy → metacognition → record → predict_next.

        Returns a dict of telemetry for this step.
        """
        # 1. Predict (from previous state)
        self.predict(s)

        # 2. Observe
        self.observe(s, observation)

        # 3. Compute surprise
        surprise = self.compute_surprise(s)
        s.meta["surprise"] = surprise

        # 4. Update belief
        self.update_belief(s, surprise)

        # 5. Update trust
        self.update_trust(s, surprise)

        # 6. Update energy
        alive = self.update_energy(s, surprise)

        # 7. Meta-cognition
        self.update_metacognition(s, surprise)

        # Record history
        s.record_history()

        # 8. Predict next (for next cycle)
        self.predict(s)

        return {
            "uid": s.uid,
            "surprise": surprise,
            "energy": s.energy_budget,
            "stable": s.meta["stable"],
            "confused": s.meta["confused"],
            "alive": alive,
        }
