"""
field_kernel.py — The Aether Self-Organizing Field Kernel.
Simulates discrete wave propagation, Laplace potential relaxation, and thermodynamic metrics.
"""

import copy
import math
from typing import Dict, List, Tuple, Any

class AetherFieldKernel:
    """
    Law 2 & 4: Continuous Field Propagation and Relaxation.
    Represents computation as a self-organizing field under conservation rules.
    """
    def __init__(self, size: int = 50):
        self.size = size
        
        # Grid allocations: current potential (u) and previous potential (v)
        self.u = [[0.0] * size for _ in range(size)]
        self.v = [[0.0] * size for _ in range(size)]
        
        # Conductance map: 1.0 = vacuum space, 0.0 = solid boundary/obstacle
        self.conductance = [[1.0] * size for _ in range(size)]
        
        # Sources: fixed potential boundary conditions (e.g. source/sink)
        self.sources: Dict[Tuple[int, int], float] = {}
        
        # Computational thermodynamics statistics
        self.local_time = 0.0
        self.info_gain = 0.0
        self.compute_spent = 0.0

    def observe(self, x: int, y: int, value: float):
        """
        1. observe() — Inject localized energy disturbance at (x,y).
        """
        if 0 <= x < self.size and 0 <= y < self.size:
            if self.conductance[x][y] == 0.0:
                return # Can't perturb inside solid obstacles
                
            old_val = self.u[x][y]
            self.u[x][y] = value
            
            # Shannon information gain: logarithmic displacement
            diff = (value - old_val) ** 2
            self.info_gain += math.log(1.0 + diff)

    def set_source(self, x: int, y: int, value: float):
        """Register a fixed constant boundary potential (source or sink)."""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.sources[(x, y)] = value
            self.u[x][y] = value
            self.v[x][y] = value

    def clear_sources(self):
        """Clear all registered boundary sources."""
        self.sources.clear()

    def add_obstacle(self, x: int, y: int, radius: float = 3.0):
        """Paint a circular obstacle (zero conductance) in the field."""
        for i in range(self.size):
            for j in range(self.size):
                dist = math.sqrt((i - x) ** 2 + (j - y) ** 2)
                if dist <= radius:
                    self.conductance[i][j] = 0.0
                    self.u[i][j] = 0.0
                    self.v[i][j] = 0.0

    def clear_obstacles(self):
        """Reset all conductance values to vacuum (1.0)."""
        self.conductance = [[1.0] * self.size for _ in range(self.size)]

    def diffuse(self, dt: float = 0.1, damping: float = 0.02, wave_speed: float = 0.4):
        """
        2. diffuse() — Evolve the 2D discrete wave equation.
           Propagates disturbances under energy damping constraints.
        """
        size = self.size
        u_new = [[0.0] * size for _ in range(size)]
        c2 = wave_speed ** 2
        
        for x in range(size):
            for y in range(size):
                # Solid boundaries and fixed sources stay static
                if self.conductance[x][y] == 0.0:
                    u_new[x][y] = 0.0
                    continue
                if (x, y) in self.sources:
                    u_new[x][y] = self.sources[(x, y)]
                    continue
                
                # Five-point discrete Laplacian (clamped boundary conditions at edges)
                u_left = self.u[x - 1][y] if x > 0 else 0.0
                u_right = self.u[x + 1][y] if x < size - 1 else 0.0
                u_up = self.u[x][y - 1] if y > 0 else 0.0
                u_down = self.u[x][y + 1] if y < size - 1 else 0.0
                
                laplacian = u_left + u_right + u_up + u_down - 4.0 * self.u[x][y]
                
                # Discrete wave step: u_new = 2u - v + c^2 * L - gamma * (u - v)
                step = 2.0 * self.u[x][y] - self.v[x][y] + c2 * laplacian - damping * (self.u[x][y] - self.v[x][y])
                
                # Clamp value to prevent floating point divergence
                u_new[x][y] = max(-5.0, min(5.0, step))
                
        # Advance temporal states
        self.v = copy.deepcopy(self.u)
        self.u = u_new
        
        self.local_time += dt
        self.compute_spent += float(size * size * 6)

    def stabilize(self, iterations: int = 15):
        """
        4. stabilize() — Run Jacobi relaxation towards Laplace equilibrium.
           Computes flow planning by relaxing potentials.
        """
        size = self.size
        
        for _ in range(iterations):
            u_new = [[0.0] * size for _ in range(size)]
            for x in range(size):
                for y in range(size):
                    if self.conductance[x][y] == 0.0:
                        u_new[x][y] = 0.0
                        continue
                    if (x, y) in self.sources:
                        u_new[x][y] = self.sources[(x, y)]
                        continue
                    
                    u_left = self.u[x - 1][y] if x > 0 else 0.0
                    u_right = self.u[x + 1][y] if x < size - 1 else 0.0
                    u_up = self.u[x][y - 1] if y > 0 else 0.0
                    u_down = self.u[x][y + 1] if y < size - 1 else 0.0
                    
                    # Laplace update: average of neighbors
                    u_new[x][y] = 0.25 * (u_left + u_right + u_up + u_down)
                    
            self.u = u_new
            self.compute_spent += float(size * size * 4)
            
        # Reset velocities so motion does not carry over on stabilize
        self.v = copy.deepcopy(self.u)

    def decay(self, rate: float = 0.01):
        """Decay energy in the field over time."""
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in self.sources:
                    self.u[x][y] *= (1.0 - rate)
                    self.v[x][y] *= (1.0 - rate)

    def measure(self) -> Dict[str, float]:
        """
        Evaluate Potential Energy, System Entropy, and Metabolic Efficiency.
        """
        size = self.size
        potential_energy = 0.0
        sum_abs = 0.0
        pressures = []
        
        for x in range(size - 1):
            for y in range(size - 1):
                # Local gradient: dPhi/dx, dPhi/dy
                grad_x = self.u[x + 1][y] - self.u[x][y]
                grad_y = self.u[x][y + 1] - self.u[x][y]
                
                # Potential energy = sum of gradient magnitudes squared (Dirichlet energy)
                potential_energy += (grad_x ** 2 + grad_y ** 2) * self.conductance[x][y]
                
                val = abs(self.u[x][y])
                sum_abs += val
                pressures.append(val)
                
        # Calculate Information Entropy (Shannon entropy of absolute field distribution)
        entropy = 0.0
        if sum_abs > 1e-9:
            for p_val in pressures:
                p_norm = p_val / sum_abs
                if p_norm > 1e-9:
                    entropy -= p_norm * math.log(p_norm)
                    
        # Metabolic efficiency: Information gained per floating point spent
        metabolism = self.info_gain / (self.compute_spent + 1e-10)
        
        return {
            "potential_energy": potential_energy,
            "entropy": entropy,
            "information_gain": self.info_gain,
            "energy_spent": self.compute_spent,
            "metabolism": metabolism
        }
