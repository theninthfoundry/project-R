"""
test_field_kernel.py — Automated Unit Tests for Aether Field Relaxation Kernel.
"""

import unittest
import math
from RealityOS.kernel.field_kernel import AetherFieldKernel

class TestAetherFieldKernel(unittest.TestCase):
    def test_disturbance_and_observation(self):
        """Assert that observe() correctly perturbs values and increases information gain."""
        kernel = AetherFieldKernel(size=10)
        
        self.assertEqual(kernel.u[5][5], 0.0)
        self.assertEqual(kernel.info_gain, 0.0)
        
        # Inject disturbance
        kernel.observe(5, 5, 2.0)
        self.assertEqual(kernel.u[5][5], 2.0)
        self.assertTrue(kernel.info_gain > 0.0)
        
        # Test boundary bounds
        kernel.observe(-1, 5, 10.0) # Should be ignored safely
        kernel.observe(5, 12, 10.0) # Should be ignored safely

    def test_wave_diffusion(self):
        """Assert that diffuse() propagates energy to adjacent nodes (conservation of momentum)."""
        kernel = AetherFieldKernel(size=10)
        
        # Center pulse
        kernel.observe(5, 5, 4.0)
        
        # Step wave evolution once
        kernel.diffuse(dt=0.1, damping=0.0, wave_speed=0.5)
        
        # Wave should have diffused to immediate neighbors (left/right/up/down)
        self.assertTrue(kernel.u[4][5] > 0.0, "Wave did not diffuse left")
        self.assertTrue(kernel.u[6][5] > 0.0, "Wave did not diffuse right")
        self.assertTrue(kernel.u[5][4] > 0.0, "Wave did not diffuse up")
        self.assertTrue(kernel.u[5][6] > 0.0, "Wave did not diffuse down")
        
        # Solid obstacles should block propagation
        kernel_obs = AetherFieldKernel(size=10)
        kernel_obs.add_obstacle(5, 6, radius=1.0)
        kernel_obs.observe(5, 5, 4.0)
        kernel_obs.diffuse(dt=0.1, damping=0.0)
        
        # Obstacle grid node must stay exactly zero potential
        self.assertEqual(kernel_obs.u[5][6], 0.0, "Energy entered obstacle!")

    def test_laplace_stabilize(self):
        """Assert that stabilize() correctly relaxes boundary potentials (Laplace flow)."""
        kernel = AetherFieldKernel(size=10)
        
        # Set a source (+3.0) and a sink (-3.0) at opposite corners
        kernel.set_source(1, 1, 3.0)
        kernel.set_source(8, 8, -3.0)
        
        # Run stabilization iterations
        kernel.stabilize(iterations=50)
        
        # Intermediate coordinates should follow smooth, relaxed gradients between 3.0 and -3.0
        val_mid = kernel.u[4][4]
        
        # Middle node should have relaxed to around 0.0
        self.assertTrue(abs(val_mid) < 1.0, f"Midpoint did not relax correctly: {val_mid:.4f}")
        
        # Verify that sources retained their fixed values
        self.assertEqual(kernel.u[1][1], 3.0)
        self.assertEqual(kernel.u[8][8], -3.0)
        
        # Verify thermodynamics metrics
        metrics = kernel.measure()
        self.assertTrue(metrics["potential_energy"] > 0.0)
        self.assertTrue(metrics["entropy"] > 0.0)
        self.assertTrue(metrics["energy_spent"] > 0.0)

if __name__ == "__main__":
    unittest.main()
