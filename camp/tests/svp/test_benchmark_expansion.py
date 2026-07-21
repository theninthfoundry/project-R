"""
test_benchmark_expansion.py — Level 2 Benchmark Expansion.
Verifies RealityOS behavior in complex simulated domains: robotic SLAM loop closure
and swarm robotics coordination (cohesion and collision avoidance).
"""

import unittest
import math
from RealityOS.kernel.aether_kernel import AetherUniverse

class TestBenchmarkExpansion(unittest.TestCase):
    def test_robotic_slam_loop_closure(self):
        """
        Simulate a robot mapping a circle and performing loop closure.
        Verify that adding a constraint between the final node and the start node
        correctly bends the entire historical coordinate path to eliminate drift.
        """
        # Trajectory with 5 steps forming a square path with drift:
        # Node 0: [0, 0]
        # Node 1: [2, 0]
        # Node 2: [2, 2]
        # Node 3: [0, 2]
        # Node 4: [0.3, 0.1] (should close loop back to [0,0], but has drift)
        size = 5
        u = AetherUniverse(size=size, dim=2, eta=0.08, alpha_dual=0.1)
        
        u.observe(0, [0.0, 0.0])
        u.observe(1, [2.0, 0.0])
        u.observe(2, [2.0, 2.0])
        u.observe(3, [0.0, 2.0])
        u.observe(4, [0.3, 0.1])  # drifted end point
        
        # Link adjacent trajectory steps via distance constraints of 2.0
        def make_dist_constraint(a, b, target):
            return lambda G: math.sqrt((G[a][0]-G[b][0])**2 + (G[a][1]-G[b][1])**2) - target
            
        u.constrain("d01", make_dist_constraint(0, 1, 2.0), "d01 = 2.0")
        u.constrain("d12", make_dist_constraint(1, 2, 2.0), "d12 = 2.0")
        u.constrain("d23", make_dist_constraint(2, 3, 2.0), "d23 = 2.0")
        u.constrain("d34", make_dist_constraint(3, 4, 2.0), "d34 = 2.0")
        
        # 1. Establish Loop Closure: Node 4 is topologically linked to Node 0 (must be identical)
        u.relate(4, 0)
        u.constrain("loop_closure", make_dist_constraint(4, 0, 0.0), "d40 = 0.0")
        
        # 2. Run stabilization. The KKT solver must distribute the correction force
        # across all nodes in the loop, bending the entire trajectory path.
        for _ in range(30):
            u.stabilize(dt=0.1)
            
        G = u.get_G()
        
        # Node 4 coordinates should have been pulled close to Node 0 coordinates
        loop_error = math.sqrt((G[4][0] - G[0][0])**2 + (G[4][1] - G[0][1])**2)
        self.assertTrue(loop_error < 0.15, f"Loop closure failed to bend coordinates. Error: {loop_error:.4f}")

    def test_swarm_robotics_cohesion_and_collision(self):
        """
        Verify that swarm agent coordinates satisfy both cohesion (stay close to centroid)
        and collision avoidance (maintain minimum distance) constraints.
        """
        size = 3
        u = AetherUniverse(size=size, dim=2, eta=0.05, alpha_dual=0.1)
        
        # Initialize agents extremely close (colliding)
        u.observe(0, [0.0, 0.0])
        u.observe(1, [0.05, 0.02])
        u.observe(2, [-0.02, 0.04])
        
        # Constraints:
        # 1. Cohesion: keep distance of each agent to centroid <= 1.5
        def cohesion(G):
            cx = sum(c[0] for c in G) / len(G)
            cy = sum(c[1] for c in G) / len(G)
            total_viol = 0.0
            for c in G:
                dist = math.sqrt((c[0]-cx)**2 + (c[1]-cy)**2)
                if dist > 1.5:
                    total_viol += (dist - 1.5)
            return total_viol

        # 2. Collision avoidance: pairwise distance must be >= 0.5
        def collision_01(G):
            d = math.sqrt((G[0][0]-G[1][0])**2 + (G[0][1]-G[1][1])**2)
            return 0.5 - d if d < 0.5 else 0.0
            
        def collision_02(G):
            d = math.sqrt((G[0][0]-G[2][0])**2 + (G[0][1]-G[2][1])**2)
            return 0.5 - d if d < 0.5 else 0.0
            
        def collision_12(G):
            d = math.sqrt((G[1][0]-G[2][0])**2 + (G[1][1]-G[2][1])**2)
            return 0.5 - d if d < 0.5 else 0.0

        u.constrain("cohesion", cohesion, "cohesion = 0")
        u.constrain("col01", collision_01, "col01 = 0")
        u.constrain("col02", collision_02, "col02 = 0")
        u.constrain("col12", collision_12, "col12 = 0")
        
        # Run stabilization loop to resolve conflicting swarm behaviors
        for _ in range(40):
            u.stabilize(dt=0.1)
            
        G = u.get_G()
        
        # Verify no collisions: pairwise distances must be close to or greater than 0.5
        d01 = math.sqrt((G[0][0]-G[1][0])**2 + (G[0][1]-G[1][1])**2)
        d02 = math.sqrt((G[0][0]-G[2][0])**2 + (G[0][1]-G[2][1])**2)
        d12 = math.sqrt((G[1][0]-G[2][0])**2 + (G[1][1]-G[2][1])**2)
        
        self.assertTrue(d01 > 0.4, f"Collision detected: d01 = {d01:.4f}")
        self.assertTrue(d02 > 0.4, f"Collision detected: d02 = {d02:.4f}")
        self.assertTrue(d12 > 0.4, f"Collision detected: d12 = {d12:.4f}")
        
        # Verify cohesion: centroid distance must be within limits
        cx = sum(c[0] for c in G) / len(G)
        cy = sum(c[1] for c in G) / len(G)
        for c in G:
            dist = math.sqrt((c[0]-cx)**2 + (c[1]-cy)**2)
            self.assertTrue(dist < 1.6, f"Cohesion limit violated. Centroid distance: {dist:.4f}")

if __name__ == "__main__":
    unittest.main()
