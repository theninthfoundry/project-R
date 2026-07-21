"""
sdk.py — High-Level Developer SDK for State Computing.
Provides a clean wrapper (State, Universe) around the KKT RelationalEngine.
"""

import copy
from typing import List, Callable, Dict, Any, Optional

class State:
    """
    State: A first-class computational primitive representing a dynamic entity.
    Coordinates, velocity, constraints, and pressure are managed in a shared Universe.
    """
    def __init__(self, name: str, dim: int = 2):
        self.name = name
        self.dim = dim
        self.universe: Optional['Universe'] = None
        self.index: Optional[int] = None
        self._initial_coords = [0.0] * dim
        self._goal_force = [0.0] * dim

    @property
    def coords(self) -> List[float]:
        if self.universe and self.index is not None:
            return self.universe.engine.G[self.index]
        return self._initial_coords

    @coords.setter
    def coords(self, values: List[float]):
        if len(values) != self.dim:
            raise ValueError(f"Expected coordinates of dimension {self.dim}, got {len(values)}")
        if self.universe and self.index is not None:
            self.universe.engine.observe(self.index, values)
        else:
            self._initial_coords = list(values)

    @property
    def velocity(self) -> List[float]:
        if self.universe and self.index is not None:
            return self.universe.engine.delta_G[self.index]
        return [0.0] * self.dim

    def observe(self, values: List[float]):
        """Ingest new sensor or environmental evidence, updating state coordinates."""
        self.coords = values

    def goal(self, force: List[float]):
        """Apply an intent or driving force vector toward an attractor target."""
        if len(force) != self.dim:
            raise ValueError(f"Expected force vector of dimension {self.dim}, got {len(force)}")
        self._goal_force = list(force)

    def intervene(self, force: List[float]):
        """Injects a counterfactual intervention impulse directly into the coordinates."""
        if len(force) != self.dim:
            raise ValueError(f"Expected force vector of dimension {self.dim}")
        if self.universe and self.universe.is_initialized:
            for d in range(self.dim):
                self.universe.engine.G[self.index][d] += force[d]
        else:
            for d in range(self.dim):
                self._initial_coords[d] += force[d]

    def get_pressure(self, constraint_name: str) -> float:
        """Get the current KKT Lagrange multiplier (pressure) of a constraint."""
        if self.universe:
            return self.universe.engine.lambdas.get(constraint_name, 0.0)
        return 0.0


class Universe:
    """
    Universe: An orchestrator that manages state coordinate evolution under KKT constraints.
    Acts as the state space substrate / scheduling runtime.
    """
    def __init__(self, eta: float = 0.08, alpha_dual: float = 0.1):
        self.states: List[State] = []
        self.state_map: Dict[str, State] = {}
        self.engine: Optional[RelationalEngine] = None
        self.eta = eta
        self.alpha_dual = alpha_dual
        self.is_initialized = False
        self.use_negotiation = False

    def create_state(self, name: str, dim: int = 2) -> State:
        """Factory method to instantiate and register a state object in the universe."""
        state = State(name, dim)
        self.add_state(state)
        return state

    def add_state(self, state: State):
        """Register an existing state object in this universe."""
        if state.name in self.state_map:
            raise ValueError(f"State named '{state.name}' already registered.")
        if self.is_initialized:
            raise RuntimeError("Cannot add states after initializing the universe.")
            
        state.universe = self
        state.index = len(self.states)
        self.states.append(state)
        self.state_map[state.name] = state

    def initialize(self):
        """Construct the relational engine and copy initial coordinates."""
        if self.is_initialized:
            return
            
        dims = [s.dim for s in self.states]
        if not dims:
            raise RuntimeError("Cannot initialize a universe with no states.")
        if len(set(dims)) > 1:
            raise ValueError("All states in the universe must share the same coordinate dimension.")
            
        dim = dims[0]
        self.engine = RelationalEngine(size=len(self.states), eta=self.eta, alpha_dual=self.alpha_dual)
        self.engine.dim = dim
        
        # Copy initial coordinates to engine
        for i, s in enumerate(self.states):
            self.engine.observe(i, s._initial_coords)
            
        self.is_initialized = True

    def apply_constraint(self, name: str, constraint_fn: Callable[[List[List[float]]], float]):
        """Apply a constraint function over all coordinates in the universe."""
        if not self.is_initialized:
            self.initialize()
        self.engine.perturb(name, constraint_fn)

    def step(self, dt: float = 0.1):
        """Execute one step of the primal-dual constraint evolution loop."""
        if not self.is_initialized:
            self.initialize()
            
        # Apply goal forces (intents) directly to gradient updates if present
        for i, s in enumerate(self.states):
            if any(f != 0.0 for f in s._goal_force):
                # Drive G directly towards intent force
                for d in range(s.dim):
                    self.engine.G[i][d] += self.eta * s._goal_force[d]
                if hasattr(self.engine, "node_active") and i < len(self.engine.node_active):
                    self.engine.node_active[i] = True
                    self.engine.node_sleep_ticks[i] = 0
                    
        # Solve KKT constraints & evolve ACE candidates
        if getattr(self, "use_negotiation", False):
            self.engine.resolve_negotiation(dt)
        else:
            self.engine.resolve(dt)

    def simulate(self, steps: int, dt: float = 0.1) -> List[List[List[float]]]:
        """Project future trajectories on a branched timeline without altering the active coordinates."""
        cloned = self.fork()
        trajectory = []
        for _ in range(steps):
            cloned.step(dt)
            trajectory.append(copy.deepcopy(cloned.engine.G))
        return trajectory

    def fork(self) -> 'Universe':
        """Deep copy the universe and state configurations to fork a branched simulation."""
        cloned = Universe(eta=self.eta, alpha_dual=self.alpha_dual)
        cloned.use_negotiation = getattr(self, "use_negotiation", False)
        for s in self.states:
            cloned_state = State(s.name, s.dim)
            cloned_state._initial_coords = list(s.coords)
            cloned_state._goal_force = list(s._goal_force)
            cloned.states.append(cloned_state)
            cloned.state_map[s.name] = cloned_state
            
        if self.is_initialized:
            cloned.initialize()
            cloned.engine.G = copy.deepcopy(self.engine.G)
            cloned.engine.delta_G = copy.deepcopy(self.engine.delta_G)
            cloned.engine.lambdas = copy.deepcopy(self.engine.lambdas)
            cloned.engine.constraints = dict(self.engine.constraints)
            cloned.engine.trajectory_history = copy.deepcopy(self.engine.trajectory_history)
            if hasattr(self.engine, "node_active"):
                cloned.engine.node_active = list(self.engine.node_active)
                cloned.engine.node_sleep_ticks = list(self.engine.node_sleep_ticks)
                cloned.engine.update_count = self.engine.update_count
        return cloned

    def merge(self, other: 'Universe') -> 'Universe':
        """Compose this universe with another, combining states and merging active constraints."""
        if not self.is_initialized:
            self.initialize()
        if not other.is_initialized:
            other.initialize()
            
        merged = Universe(eta=self.eta, alpha_dual=self.alpha_dual)
        
        # Add states from self
        for s in self.states:
            cloned_s = State(s.name, s.dim)
            cloned_s._initial_coords = list(s.coords)
            merged.add_state(cloned_s)
            
        # Add states from other (prefixing name to prevent conflicts)
        for s in other.states:
            cloned_s = State(f"merged_{s.name}", s.dim)
            cloned_s._initial_coords = list(s.coords)
            merged.add_state(cloned_s)
            
        merged.initialize()
        
        # Merge constraints
        for name, c_fn in self.engine.constraints.items():
            merged.apply_constraint(name, c_fn)
            
        offset = len(self.states)
        for name, c_fn in other.engine.constraints.items():
            shifted_fn = lambda G, fn=c_fn, o=offset: fn(G[o:])
            merged.apply_constraint(f"merged_{name}", shifted_fn)
            
        return merged

    def rewind(self, ticks: int):
        """Rollback the active coordinate timeline to a historical state."""
        if not self.engine or len(self.engine.trajectory_history) < ticks:
            raise RuntimeError("Insufficient history to rewind.")
        target_state = self.engine.trajectory_history[-ticks]
        self.engine.G = [list(row) for row in target_state]
        self.engine.trajectory_history = self.engine.trajectory_history[:-ticks]

    def replay(self, trajectory: List[List[List[float]]], tick_callback: Callable = None):
        """Replay a specific sequence of historical coordinate states."""
        if not self.engine:
            self.initialize()
        for G_t in trajectory:
            self.engine.G = [list(row) for row in G_t]
            if tick_callback:
                tick_callback(self)

    def get_rules(self) -> Dict[str, Any]:
        """Query active and retired rules from the constraint compiler."""
        if self.engine:
            return self.engine.ace.get_summary()
        return {}
