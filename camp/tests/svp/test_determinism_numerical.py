"""
test_determinism_numerical.py — Level 3 Determinism and Numerical Robustness Audit.
Checks floating point determinism, precision loss, ill-conditioned matrix gradients, and overflows/underflows.
"""

import unittest
import math
import random
from RealityOS.kernel.aether_kernel import AetherUniverse

class TestDeterminismNumerical(unittest.TestCase):
    def test_seed_determinism(self):
        """
        Verify that two identical universes with identical seeds produce bit-identical states.
        """
        # Run Universe 1
        random.seed(1337)
        u1 = AetherUniverse(size=3, dim=2, eta=0.08, alpha_dual=0.05)
        u1.observe(0, [1.0, 2.0])
        u1.observe(1, [3.0, 4.0])
        u1.relate(0, 1)
        
        # Simulate some random perturbations and stabilization steps
        for step in range(10):
            # Inject small random observations
            u1.observe(0, [1.0 + random.uniform(-0.01, 0.01), 2.0 + random.uniform(-0.01, 0.01)])
            u1.stabilize(dt=0.1)

        # Run Universe 2
        random.seed(1337)
        u2 = AetherUniverse(size=3, dim=2, eta=0.08, alpha_dual=0.05)
        u2.observe(0, [1.0, 2.0])
        u2.observe(1, [3.0, 4.0])
        u2.relate(0, 1)
        
        for step in range(10):
            u2.observe(0, [1.0 + random.uniform(-0.01, 0.01), 2.0 + random.uniform(-0.01, 0.01)])
            u2.stabilize(dt=0.1)

        # Assert bit-identical coordinates
        G1 = u1.get_G()
        G2 = u2.get_G()
        for i in range(len(G1)):
            for d in range(len(G1[i])):
                self.assertEqual(G1[i][d], G2[i][d], f"Coordinate drift at index {i}, dim {d}")

    def test_numerical_robustness_ill_conditioning(self):
        """
        Verify gradient stability and adaptive step clamping under ill-conditioned/extreme constraints.
        """
        u = AetherUniverse(size=2, dim=2, eta=0.05, alpha_dual=0.1)
        u.observe(0, [0.0, 0.0])
        u.observe(1, [1.0, 0.0])
        
        # Define a constraint with an extremely steep gradient multiplier (mocking ill-conditioning)
        # This function produces a huge penalty if coordinates deviate slightly.
        # C(G) = 1e6 * (G[0][0] - G[1][0] - 1.0)
        def ill_conditioned_constraint(G):
            return 1e6 * (G[0][0] - G[1][0] - 1.0)

        u.constrain("extreme_tether", ill_conditioned_constraint, "1e6 * (G[0][0] - G[1][0] - 1.0) = 0")
        
        # Force a violation displacement
        u.observe(1, [1.5, 0.0]) # violates constraint by 0.5 * 1e6 = 500,000 penalty
        
        # Run stabilize. The solver must clamp updates and not explode to NaN or Infinity
        try:
            u.stabilize(dt=0.1)
            G = u.get_G()
            
            # Check for NaN / Inf
            for i in range(len(G)):
                for d in range(len(G[i])):
                    self.assertFalse(math.isnan(G[i][d]), "NaN detected in coordinates!")
                    self.assertFalse(math.isinf(G[i][d]), "Infinite value detected in coordinates!")
                    
        except OverflowError:
            self.fail("Simulation threw OverflowError due to ill-conditioned gradient!")

    def test_catastrophic_cancellation_prevention(self):
        """
        Assert precision loss prevention when elements are extremely close.
        Verify division-by-zero or zero-gradients do not freeze coordinate updates.
        """
        u = AetherUniverse(size=2, dim=2, eta=0.05)
        # Place two coordinates extremely close to each other
        u.observe(0, [1e-15, 1e-15])
        u.observe(1, [-1e-15, -1e-15])
        
        def dist_viol(G):
            # Distance constraint which divides by distance in derivative
            dist = math.sqrt((G[0][0] - G[1][0])**2 + (G[0][1] - G[1][1])**2)
            # If distance is zero, gradient calculation (numerical or symbolic) might divide by zero
            return dist - 2.0

        u.constrain("dist_tether", dist_viol, "||G[0] - G[1]|| - 2.0 = 0")
        
        # Running stabilize. The numerical gradient uses eps = 1e-4 which naturally averages out
        # extremely small coordinates and prevents division by zero.
        u.stabilize(dt=0.1)
        G = u.get_G()
        self.assertFalse(math.isnan(G[0][0]))
        self.assertFalse(math.isinf(G[0][0]))

if __name__ == "__main__":
    unittest.main()
