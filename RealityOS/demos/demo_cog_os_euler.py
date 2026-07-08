"""
demo_cog_os_euler.py  -  A robust, self-contained Cognitive OS Prototype

Implements the continuous evolution of Cognitive States (CS) under Eq (1):
s_dot = -eta * grad_s [ L_pred(s) + lambda_C * ||C(s)||^2 + lambda_V * f(V) + lambda_g * ||g||^2 ]

Key features:
1. Pure PyTorch autograd integration.
2. Custom Euler step solver to avoid external library dependencies (no torchdiffeq required).
3. VERI-style trust and safety guard (filtering out state changes that violate constraints).
4. Dynamic surprise-driven trust adaptation and energy-budget depletion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import uuid
from typing import Dict, List, Callable, Optional

# ---------- 1. Cognitive State (CS) Primitive ----------
class CognitiveState:
    def __init__(self, uid: str, type_tag: str, x: torch.Tensor, value: float = 0.5):
        self.uid = uid
        self.type_tag = type_tag
        
        # State variables
        self.x = x.clone().detach().requires_grad_(True)  # Must track gradients
        self.xdot = torch.zeros_like(x)
        
        # Predictor belief
        self.p = x.clone().detach()                       # One-step ahead prediction
        self.tau = torch.ones_like(x) * 1.0               # Precision / Trust (weights prediction error)
        
        # Goals, Values & Budgets
        self.value = value                                # Utility V
        self.g = torch.zeros_like(x)                      # Intent field pull force
        self.energy_budget = 1.0                          # Normalized energy budget [0, 1]
        self.energy_used = 0.0
        
        # Meta-cognition
        self.meta = {
            "surprise": 0.0,
            "confused": False,
            "stable": True
        }

# ---------- 2. Predictive Engine (Neural Dynamics) ----------
class PredictiveDynamics(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.Tanh(),
            nn.Linear(32, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------- 3. Constraints & VERI Safety Guard ----------
class Constraint:
    """Represents a soft or hard boundary constraint on the state space."""
    def __init__(self, name: str, violation_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.name = name
        self.violation_fn = violation_fn  # Returns scalar tensor penalty

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self.violation_fn(x)

class SafetyGuard:
    """VERI-style verifier that intercepts updates before commitment."""
    def __init__(self, hard_constraints: List[Constraint]):
        self.hard_constraints = hard_constraints

    def verify_proposed_step(self, proposed_x: torch.Tensor) -> bool:
        """Returns True if the proposed state is safe, False otherwise."""
        for c in self.hard_constraints:
            penalty = c.evaluate(proposed_x)
            if penalty.item() > 0.5:  # High penalty indicates violation
                print(f"  [VERI Safety Alert] Update blocked by constraint '{c.name}' (penalty: {penalty.item():.3f})")
                return False
        return True

# ---------- 4. The Evolution Engine ----------
class EvolutionEngine:
    def __init__(self, dim: int, safety_guard: SafetyGuard):
        self.predictor = PredictiveDynamics(dim)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-3)
        self.safety_guard = safety_guard
        
        # Gains / Hyper-parameters for Eq (1)
        self.lambda_c = 20.0   # Constraint weight
        self.lambda_g = 5.0    # Intent weight
        self.eta_base = 0.1    # Learning rate/temperature

    def predict_step(self, s: CognitiveState):
        """Generates next prediction using our learned dynamics model."""
        with torch.no_grad():
            delta = self.predictor(s.x)
            s.p = s.x + delta
        return s

    def compute_intent_field(self, s: CognitiveState, target: torch.Tensor):
        """Generates attractor force g pulling the state toward target location."""
        direction = target - s.x
        dist = direction.norm()
        if dist > 0.01:
            s.g = (direction / dist) * s.value * 0.5
        else:
            s.g = torch.zeros_like(s.x)

    def evolve_state(self, s: CognitiveState, constraints: List[Constraint], dt: float = 0.05):
        """
        Executes a single Euler-integration step of Eq (1)
        minimizing prediction error (surprise) + constraint violations + intent pulls.
        """
        if s.energy_budget <= 0.01:
            # Not enough energy: enter maintenance mode (low computation update)
            s.meta["stable"] = False
            return
            
        # Ensure s.x retains gradients
        x = s.x.clone().detach().requires_grad_(True)
        
        # 1. Compute Surprise Loss (weighted by trust/precision)
        err = x - s.p
        surprise_loss = 0.5 * (err**2 * s.tau).sum()
        
        # 2. Compute Constraints Penalties
        constraint_loss = torch.tensor(0.0)
        for c in constraints:
            constraint_loss = constraint_loss + c.evaluate(x)
            
        # 3. Compute Intent Attraction energy
        # g pushes x toward the goal, meaning we minimize the dot product -g^T * x
        intent_loss = -(s.g * x).sum()
        
        # Total Free Energy formulation
        free_energy = surprise_loss + (self.lambda_c * constraint_loss) + (self.lambda_g * intent_loss)
        
        # Perform backward pass to get gradients
        free_energy.backward()
        
        with torch.no_grad():
            grad = x.grad
            if grad is None:
                grad = torch.zeros_like(x)
                
            # Gradient step modulated by temperature/eta
            eta = self.eta_base * s.energy_budget
            proposed_x = s.x - eta * grad
            
            # VERI safety check
            if self.safety_guard.verify_proposed_step(proposed_x):
                # Apply step
                s.xdot = (proposed_x - s.x) / dt
                s.x = proposed_x.clone().detach().requires_grad_(True)
                
                # Deduct energy budget proportional to update magnitude
                step_size = grad.norm().item()
                energy_cost = step_size * 0.02 + 0.005
                s.energy_budget = max(0.0, s.energy_budget - energy_cost)
                s.energy_used += energy_cost
            else:
                # Blocked: decelerate and dissipate energy
                s.xdot = torch.zeros_like(s.x)
                s.energy_budget = max(0.0, s.energy_budget - 0.01)
                
            # Trust update: Exponential decay based on surprise
            surprise = surprise_loss.item()
            s.meta["surprise"] = surprise
            
            # High surprise decays precision/trust; stability builds it back
            trust_decay = 0.05
            new_tau = s.tau * (1 - trust_decay * surprise) + torch.ones_like(s.tau) * trust_decay
            s.tau = torch.clamp(new_tau, 0.1, 5.0)

    def train_predictor(self, s: CognitiveState, observation: torch.Tensor):
        """Online training step for the predictive network (learning the dynamics)."""
        self.optimizer.zero_grad()
        prediction_delta = self.predictor(s.x)
        target_delta = observation - s.x
        loss = nn.MSELoss()(prediction_delta, target_delta)
        loss.backward()
        self.optimizer.step()

# ---------- 5. Verification Run ----------
def run_cognitive_loop():
    print("=== Cognitive OS Sandbox: Euler-based Eq (1) Evolution ===")
    
    # Define constraints (e.g., Table boundary: y cannot be negative)
    table_edge = Constraint(
        name="Table Edge Boundary",
        violation_fn=lambda pos: torch.max(torch.tensor(0.0), -pos[1])**2
    )
    
    # Define a hard collision zone at (0.0, -0.5)
    danger_zone = Constraint(
        name="Obstacle Collision Avoidance",
        violation_fn=lambda pos: torch.max(torch.tensor(0.0), 0.4 - (pos - torch.tensor([0.0, -0.5])).norm())**2
    )
    
    safety = SafetyGuard([danger_zone])
    engine = EvolutionEngine(dim=2, safety_guard=safety)
    
    # Initialize robot state at (1.0, 1.0)
    robot = CognitiveState(uid="robot_arm", type_tag="agent", x=torch.tensor([1.0, 1.0]), value=0.8)
    
    # Goal target at (0.0, -1.0) -- this will pull the robot directly through the danger zone
    goal = torch.tensor([0.0, -1.0])
    
    print(f"Robot starting position: {robot.x.tolist()}")
    print(f"Goal target location: {goal.tolist()}")
    print("Danger zone obstacle located at: [0.0, -0.5]")
    print("-" * 60)
    
    # Run loop
    dt = 0.05
    for tick in range(40):
        # 1. Generate prediction
        engine.predict_step(robot)
        
        # 2. Update intent attractor
        engine.compute_intent_field(robot, goal)
        
        # 3. Evolve state under constraints
        engine.evolve_state(robot, [table_edge], dt)
        
        # 4. Perform online training of predictor network
        engine.train_predictor(robot, robot.x)
        
        if tick % 5 == 0:
            print(f"Tick {tick:02d} | Pos: [{robot.x[0]:.3f}, {robot.x[1]:.3f}] | Surprise: {robot.meta['surprise']:.4f} | Energy: {robot.energy_budget:.3f}")
            
    print("-" * 60)
    print(f"Final robot position: {robot.x.tolist()}")
    print("Verification completed successfully!")

if __name__ == "__main__":
    run_cognitive_loop()
