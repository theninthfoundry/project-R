"""
relational_engine.py — Rigorous Constraint-Transition (CT) Engine with KKT Dual Ascent.

Implements the Constraint-Transition (CT) triplet:
    tau = (G, delta_G, Lambda)
and the 6 universal operators of the Relational Calculus:
    1. Observe
    2. Perturb
    3. Resolve (Euler-Lagrange update under KKT shadow prices)
    4. Compose
    5. Propagate
    6. Explain (Symbolic invariant compiler / MDL-based model reduction)
"""

import math
from typing import List, Dict, Any, Callable, Tuple

class RelationalEngine:
    """
    The Relational Engine with KKT dual ascent and symbolic invariant discovery.
    """
    def __init__(self, size: int = 20, eta: float = 0.05, alpha_dual: float = 0.1):
        self.size = size
        self.eta = eta
        self.alpha_dual = alpha_dual  # Dual learning rate for KKT multiplier updates

        self.dim = 2
        self.G = [[0.0 for _ in range(self.dim)] for _ in range(size)]
        self.delta_G = [[0.0 for _ in range(self.dim)] for _ in range(size)]
        
        # lambdas: Lagrange multipliers (KKT dual variables representing constraint pressure)
        self.lambdas: Dict[str, float] = {}
        self.constraints: Dict[str, Callable[[List[List[float]]], float]] = {}
        
        # Local time derived dynamically from change
        self.local_time = 0.0

        # Trajectory history and ACE Engine integration
        self.trajectory_history: List[List[List[float]]] = []
        from RealityOS.kernel.ace_engine import ACEEngine
        self.ace = ACEEngine(size=self.size, dim=self.dim)

        # Active states and sleep trackers for ACN solver
        self.node_active = [True] * size
        self.node_sleep_ticks = [0] * size
        self.update_count = 0


    def observe(self, node_idx: int, y: List[float]):
        """Observe: updates G's coordinates with observed y."""
        if 0 <= node_idx < self.size:
            self.G[node_idx] = list(y)
            if hasattr(self, "node_active") and node_idx < len(self.node_active):
                self.node_active[node_idx] = True
                self.node_sleep_ticks[node_idx] = 0

    def perturb(self, name: str, constraint_fn: Callable[[List[List[float]]], float], initial_lambda: float = 1.0):
        """Perturb: registers a new constraint and initial Lagrange pressure."""
        self.constraints[name] = constraint_fn
        self.lambdas[name] = initial_lambda

    def resolve(self, dt: float = 0.1):
        """
        Resolve: runs one KKT primal-dual update step.
        Primal update: G = G - eta * grad_G (L)
        Dual update: lambda = max(0, lambda + alpha_dual * C(G))
        """
        eps = 1e-4
        grad = [[0.0 for _ in range(self.dim)] for _ in range(self.size)]
        
        # 1. Compute primal gradient of the Lagrangian: L = Surprise + sum(lambda * C(G)^2)
        for name, c_fn in self.constraints.items():
            l_val = self.lambdas.get(name, 1.0)
            base_val = c_fn(self.G)
            
            if abs(base_val) < 1e-9:
                continue
                
            for i in range(self.size):
                for d in range(self.dim):
                    # Numerical gradient of C(G)
                    self.G[i][d] += eps
                    val_plus = c_fn(self.G)
                    self.G[i][d] -= 2 * eps
                    val_minus = c_fn(self.G)
                    self.G[i][d] += eps # Restore
                    
                    dC_dG = (val_plus - val_minus) / (2 * eps)
                    # d/dG (lambda * C(G)^2) = 2 * lambda * C(G) * dC_dG
                    grad[i][d] += 2 * l_val * base_val * dC_dG

        # 2. Update G and estimate delta_G using adaptive eta (dynamic scaling under constraint violation)
        max_violation = max([abs(c_fn(self.G)) for c_fn in self.constraints.values()] + [0.0])
        adaptive_eta = self.eta * (1.0 + min(2.0, max_violation))
        
        displacement = 0.0
        for i in range(self.size):
            for d in range(self.dim):
                step = -adaptive_eta * grad[i][d]
                self.delta_G[i][d] = step / dt
                self.G[i][d] += step
                displacement += step ** 2

        # 3. KKT Dual Ascent: update multipliers (shadow prices) based on constraint violation
        for name, c_fn in self.constraints.items():
            violation = c_fn(self.G)
            # Update dual variable: lambda = max(0.01, lambda + alpha_dual * violation^2)
            self.lambdas[name] = max(0.01, self.lambdas[name] + self.alpha_dual * (violation ** 2))

        # 4. Event-driven time: dt emerges from coordinate displacement magnitude
        # If nothing changes, time does not advance.
        self.local_time += math.sqrt(displacement) + 0.01

        # 5. Track trajectory history and step the ACE Engine
        import copy
        self.trajectory_history.append(copy.deepcopy(self.G))
        if len(self.trajectory_history) > 100:
            self.trajectory_history.pop(0)
        self.ace.evolve(self, self.trajectory_history)


    def resolve_negotiation(self, dt: float = 0.1, threshold: float = 0.01):
        """
        Resolve negotiation: Decentralized Active Constraint Negotiation solver.
        Constraints autonomously propose local adjustment force vectors,
        nodes locally broker updates, and inactive nodes sleep to conserve energy.
        """
        import math
        eps = 1e-4
        proposals = [[] for _ in range(self.size)] # list of (force_vector, constraint_name) per node
        
        # 1. Active Constraint processes calculate local force proposals
        for name, c_fn in self.constraints.items():
            l_val = self.lambdas.get(name, 1.0)
            base_val = c_fn(self.G)
            
            if abs(base_val) < 1e-9:
                continue
                
            for i in range(self.size):
                # Calculate numerical gradient of C(G) w.r.t node i
                grad_node = [0.0] * self.dim
                is_connected = False
                for d in range(self.dim):
                    self.G[i][d] += eps
                    val_plus = c_fn(self.G)
                    self.G[i][d] -= 2 * eps
                    val_minus = c_fn(self.G)
                    self.G[i][d] += eps
                    
                    dC_dG = (val_plus - val_minus) / (2 * eps)
                    if abs(dC_dG) > 1e-5:
                        is_connected = True
                    grad_node[d] = dC_dG
                
                if is_connected:
                    # force proposal: -2 * lambda * C(G) * dC_dG
                    force = [-2.0 * l_val * base_val * grad_node[d] for d in range(self.dim)]
                    proposals[i].append((force, name))
                    
                    # Cascade trigger: wake up node if force is non-negligible
                    force_mag = math.sqrt(sum(f**2 for f in force))
                    if force_mag > 1e-4:
                        if not self.node_active[i]:
                            self.node_active[i] = True
                            self.node_sleep_ticks[i] = 0

        # 2. Node-level local consensus/brokerage and coordinate updates
        displacement = 0.0
        active_violation = max([abs(c_fn(self.G)) for c_fn in self.constraints.values()] + [0.0])
        adaptive_eta = self.eta * (1.0 + min(2.0, active_violation))

        for i in range(self.size):
            if not self.node_active[i]:
                self.node_sleep_ticks[i] += 1
                continue
                
            # Local brokerage: sum proposed forces from all active constraints touching this node
            net_force = [0.0] * self.dim
            for force, name in proposals[i]:
                for d in range(self.dim):
                    net_force[d] += force[d]
                    
            node_displacement = 0.0
            for d in range(self.dim):
                step = adaptive_eta * net_force[d]
                self.delta_G[i][d] = step / dt
                self.G[i][d] += step
                node_displacement += step ** 2
                
            displacement += node_displacement
            self.update_count += 1
            
            # 3. Information-driven sleep gating
            if math.sqrt(node_displacement) < threshold:
                self.node_sleep_ticks[i] += 1
                if self.node_sleep_ticks[i] >= 5:
                    self.node_active[i] = False
            else:
                self.node_sleep_ticks[i] = 0

        # 4. KKT Dual Ascent: update multipliers based on constraint violation
        for name, c_fn in self.constraints.items():
            violation = c_fn(self.G)
            self.lambdas[name] = max(0.01, self.lambdas[name] + self.alpha_dual * (violation ** 2))

        # 5. Intrinsic event-driven local time step advancement
        self.local_time += math.sqrt(displacement) + 0.01

        # 6. Save trajectory history and step the ACE Engine
        import copy
        self.trajectory_history.append(copy.deepcopy(self.G))
        if len(self.trajectory_history) > 100:
            self.trajectory_history.pop(0)
        self.ace.evolve(self, self.trajectory_history)


    def compose(self, other: 'RelationalEngine'):
        """Compose: concatenates systems and combines constraints."""
        new_size = self.size + other.size
        composed = RelationalEngine(size=new_size, eta=self.eta, alpha_dual=self.alpha_dual)
        composed.G = self.G + other.G
        
        # Self's constraints point to 0..self.size-1, which are intact in composed.G
        composed.constraints.update(self.constraints)
        
        # Shift other's constraints by self.size offset using default parameters to bind variables
        offset = self.size
        for name, fn in other.constraints.items():
            shifted_fn = lambda G, c_fn=fn, o=offset: c_fn(G[o:])
            composed.constraints[name] = shifted_fn
            
        composed.lambdas.update(self.lambdas)
        composed.lambdas.update(other.lambdas)
        return composed

    def propagate(self, threshold: float = 0.05, dt: float = 0.1):
        """Propagate: runs Resolve only on constraints whose pressure exceeds threshold."""
        active = any(l_val > threshold for l_val in self.lambdas.values())
        if active:
            self.resolve(dt)

    def explain(self, trajectory: List[List[List[float]]] = None, threshold: float = 0.01) -> List[Tuple[str, Callable[[List[List[float]]], float]]]:
        """
        Explain: The Constraint Compiler operator.
        Discovers simple algebraic invariants:
        1. Pairwise distance invariants over a trajectory history.
        2. Radial conservation invariant (if no trajectory provided).
        """
        # First check if the ACE engine has active constraints
        active_discovered = []
        for name, h in self.ace.hypotheses.items():
            if h.status == "active":
                active_discovered.append((name, h.violation_fn))
        if active_discovered:
            return active_discovered

        discovered = []

        
        if trajectory is not None and len(trajectory) > 2:
            # Trajectory shape: T (steps) x N (nodes) x dim
            T_len = len(trajectory)
            
            # Check all pairs of nodes (i, j) for constant distance over time
            for i in range(self.size):
                for j in range(i + 1, self.size):
                    dists = []
                    for t in range(T_len):
                        g_t = trajectory[t]
                        d = math.sqrt((g_t[i][0] - g_t[j][0])**2 + (g_t[i][1] - g_t[j][1])**2)
                        dists.append(d)
                        
                    mean_d = sum(dists) / T_len
                    var_d = sum((d - mean_d)**2 for d in dists) / T_len
                    
                    if var_d < threshold:
                        # Found invariant pairwise distance!
                        # Capture the values using default arguments in lambda to avoid closures binding late
                        def make_pairwise_constraint(node_a=i, node_b=j, target_d=mean_d):
                            return lambda G: (math.sqrt((G[node_a][0] - G[node_b][0])**2 + (G[node_a][1] - G[node_b][1])**2) - target_d)
                            
                        discovered.append((f"distance_{i}_{j}", make_pairwise_constraint(i, j, mean_d)))
        else:
            # Check radial conservation: x^2 + y^2 = R^2 in current state G
            radii_sq = [c[0]**2 + c[1]**2 for c in self.G]
            mean_r_sq = sum(radii_sq) / len(radii_sq)
            var_r_sq = sum((r - mean_r_sq)**2 for r in radii_sq) / len(radii_sq)
            
            if var_r_sq < threshold:
                r_target = math.sqrt(mean_r_sq)
                
                def radius_constraint(G):
                    total_viol = 0.0
                    for coords in G:
                        dist = math.sqrt(coords[0]**2 + coords[1]**2)
                        total_viol += (dist - r_target)
                    return total_viol / len(G)
                    
                discovered.append(("radial_invariant", radius_constraint))
                
        return discovered

