"""
cognitive_state.py — The Cognitive State primitive for Project S / CAMP.

This is the mathematical object at the center of the Cognitive Dynamics
Architecture. Every monitored entity (agent, sensor, service) is
represented as a CognitiveState that evolves under the predict-observe-
compare-update loop.

The key design choice: this is pure Python with zero ML dependencies.
All fields use plain lists/floats so the primitive can be serialized,
distributed via CRDT, and run on any hardware without torch/numpy.
"""

import time
import math
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Constraint:
    """A soft or hard boundary condition on the state space."""
    name: str
    dimension: int          # which dimension of x this constrains
    lower: float = float('-inf')
    upper: float = float('inf')
    weight: float = 1.0     # penalty multiplier

    def violation(self, x: List[float]) -> float:
        """Returns squared violation penalty. Zero if within bounds."""
        if self.dimension >= len(x):
            return 0.0
        val = x[self.dimension]
        if val < self.lower:
            return self.weight * (self.lower - val) ** 2
        if val > self.upper:
            return self.weight * (val - self.upper) ** 2
        return 0.0


@dataclass
class Evidence:
    """A single piece of evidence supporting a belief."""
    source_id: str
    data: Any
    confidence: float       # [0, 1]
    timestamp: float
    verification: str = "unverified"  # "unverified", "verified", "disputed"


class RingBuffer:
    """Fixed-capacity circular buffer for state history."""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.head: int = 0

    def append(self, snapshot: Dict[str, Any]):
        if len(self.buffer) < self.capacity:
            self.buffer.append(snapshot)
        else:
            self.buffer[self.head] = snapshot
            self.head = (self.head + 1) % self.capacity

    def latest(self) -> Optional[Dict[str, Any]]:
        if not self.buffer:
            return None
        idx = (self.head - 1) % len(self.buffer)
        return self.buffer[idx]

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class CognitiveState:
    """
    The universal cognitive primitive.

    Formal tuple: S = (I, x, x_dot, belief, tau, prediction, goal, V, C, E, meta, R, history, t)

    Every entity in the system — an AI agent, a sensor, a service endpoint,
    or the system itself — is represented as one of these. The Evolution
    operator updates it each tick.
    """

    # === Identity (immutable after creation) ===
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    type_tag: str = "unknown"

    # === Observable state ===
    x: List[float] = field(default_factory=list)
    xdot: List[float] = field(default_factory=list)

    # === Belief (momentum-damped estimate of true state) ===
    belief: List[float] = field(default_factory=list)
    belief_variance: List[float] = field(default_factory=list)

    # === Prediction (one-step-ahead forecast) ===
    prediction: List[float] = field(default_factory=list)
    prediction_variance: List[float] = field(default_factory=list)

    # === Goal attractor ===
    goal: List[float] = field(default_factory=list)

    # === Value / utility ===
    value: float = 0.5

    # === Trust / precision ===
    tau: List[float] = field(default_factory=list)

    # === Constraints ===
    constraints: List[Constraint] = field(default_factory=list)

    # === Evidence provenance ===
    evidence: List[Evidence] = field(default_factory=list)

    # === Energy budget ===
    energy_budget: float = 1.0
    energy_used: float = 0.0

    # === Meta-cognition flags ===
    meta: Dict[str, Any] = field(default_factory=lambda: {
        "surprise": 0.0,
        "surprise_ema": 0.0,
        "confused": False,
        "overconfident": False,
        "needs_help": False,
        "stable": True,
        "ticks_confused": 0,
    })

    # === Relations ===
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)

    # === History ===
    history: RingBuffer = field(default_factory=lambda: RingBuffer(100))

    # === Time ===
    timestamp: float = field(default_factory=time.time)
    version: int = 0

    # ─────────────────────────────────────────────
    # Core operations
    # ─────────────────────────────────────────────

    def dim(self) -> int:
        return len(self.x)

    def ensure_dimensions(self):
        """Ensure all vector fields match the dimension of x."""
        d = self.dim()
        if len(self.xdot) != d:
            self.xdot = [0.0] * d
        if len(self.belief) != d:
            self.belief = list(self.x)
        if len(self.belief_variance) != d:
            self.belief_variance = [1.0] * d
        if len(self.prediction) != d:
            self.prediction = list(self.x)
        if len(self.prediction_variance) != d:
            self.prediction_variance = [1.0] * d
        if len(self.tau) != d:
            self.tau = [1.0] * d
        if len(self.goal) != d:
            self.goal = [0.0] * d

    def compute_surprise(self) -> float:
        """
        Mahalanobis-style surprise: weighted squared distance between
        observation x and prediction, scaled by precision tau.

        surprise = sum_i tau_i * (x_i - prediction_i)^2
        """
        if not self.prediction or not self.x:
            return 0.0
        s = 0.0
        for i in range(min(len(self.x), len(self.prediction))):
            tau_i = self.tau[i] if i < len(self.tau) else 1.0
            s += tau_i * (self.x[i] - self.prediction[i]) ** 2
        return s

    def momentum_update(self, observation: List[float], alpha: float = 0.15):
        """
        Damped belief update: belief moves partway toward observation.
        alpha = 1.0 means snap-to-latest (no momentum).
        alpha = 0.05 means heavy smoothing (slow to react, very stable).
        """
        self.ensure_dimensions()
        for i in range(min(len(self.belief), len(observation))):
            self.belief[i] = self.belief[i] * (1.0 - alpha) + observation[i] * alpha

    def update_trust(self, surprise: float, decay_rate: float = 0.05):
        """
        Adapt precision/trust based on surprise history.
        High surprise → lower trust (wider variance).
        Low surprise → higher trust (tighter variance).
        """
        for i in range(len(self.tau)):
            if surprise > 1.0:
                # Surprise is high: reduce trust
                self.tau[i] = max(0.1, self.tau[i] * (1.0 - decay_rate))
            else:
                # Surprise is low: build trust
                self.tau[i] = min(5.0, self.tau[i] * (1.0 + decay_rate * 0.5))

    def consume_energy(self, cost: float) -> bool:
        """
        Deduct from energy budget. Returns False if budget exhausted
        (maintenance mode — skip heavy computation).
        """
        self.energy_used += cost
        self.energy_budget = max(0.0, self.energy_budget - cost)
        return self.energy_budget > 0.01

    def snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot for history/persistence."""
        return {
            "uid": self.uid,
            "x": list(self.x),
            "belief": list(self.belief),
            "surprise": self.meta.get("surprise", 0.0),
            "energy": self.energy_budget,
            "tau": list(self.tau),
            "timestamp": self.timestamp,
            "version": self.version,
        }

    def record_history(self):
        """Push current state to history ring buffer."""
        self.history.append(self.snapshot())
        self.version += 1
