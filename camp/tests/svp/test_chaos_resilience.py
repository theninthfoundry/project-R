"""
test_chaos_resilience.py — Level 5 Chaos Engineering and Resilience Audit.
Simulates fault injection (clock jumps, dropped/duplicated events, coordinate corruption)
and verifies system recovery without silent divergence.
"""

import unittest
import math
import random
from RealityOS.kernel.aether_kernel import AetherUniverse

class TestChaosResilience(unittest.TestCase):
    def test_coordinate_corruption_recovery(self):
        """
        Inject random coordinate corruption (e.g., NaN or huge values) and verify that
        the KKT solver self-repairs coordinates under subsequent observations.
        """
        u = AetherUniverse(size=3, dim=2, eta=0.08, alpha_dual=0.05)
        u.observe(0, [0.0, 0.0])
        u.observe(1, [2.0, 0.0])
        u.observe(2, [0.0, 2.0])
        
        # Rigidity constraints between nodes
        def d01(G): return math.sqrt((G[0][0]-G[1][0])**2 + (G[0][1]-G[1][1])**2) - 2.0
        def d02(G): return math.sqrt((G[0][0]-G[2][0])**2 + (G[0][1]-G[2][1])**2) - 2.0
        
        u.constrain("d01", d01, "d01 = 0")
        u.constrain("d02", d02, "d02 = 0")
        
        u.stabilize(dt=0.1)
        
        # Inject Chaos: Corrupt coordinates of Node 1 to a huge noisy value
        u.states[1].coords = [9999.0, -9999.0]
        
        # Assert that the system is currently corrupted
        self.assertTrue(abs(d01(u.get_G())) > 1000.0)
        
        # Next ticks: observations are ingested for Node 0 and Node 2. Node 1 is occluded.
        # Stabilize should pull Node 1 back into alignment based on constraints!
        for _ in range(30):
            u.observe(0, [0.0, 0.0])
            u.observe(2, [0.0, 2.0])
            u.stabilize(dt=0.1)
            
        final_dist_01 = math.sqrt((u.states[0].coords[0]-u.states[1].coords[0])**2 + (u.states[0].coords[1]-u.states[1].coords[1])**2)
        
        # The system recovered back to target distance (2.0) despite massive coordinate corruption
        self.assertTrue(abs(final_dist_01 - 2.0) < 0.2, f"Failed to recover from corruption. Dist: {final_dist_01:.4f}")

    def test_clock_jump_resilience(self):
        """
        Inject sudden jumps in local event time and check that scheduling and velocity
        updates do not overflow or trigger division-by-zero.
        """
        u = AetherUniverse(size=2, dim=2, eta=0.05)
        u.observe(0, [0.0, 0.0])
        u.observe(1, [1.0, 0.0])
        
        # Make a displacement and stabilize
        u.observe(1, [1.1, 0.0])
        u.stabilize(dt=0.1)
        
        # Jump the local time forward by a massive step (e.g. simulation of clock drift)
        u.local_time += 1000.0
        
        # Subsequent updates must execute normally
        u.observe(1, [1.2, 0.0])
        try:
            u.stabilize(dt=0.1)
            # Velocity calculation: step / dt. Since dt is constant, it shouldn't divide by time jump
            self.assertFalse(math.isnan(u.states[1].velocity[0]))
        except Exception as e:
            self.fail(f"Clock jump triggered exception: {e}")

    def test_event_duplication_and_dropping(self):
        """
        Inject event log faults (drop events and duplicate events) and verify that
        the core stabilization logic operates normally.
        """
        u = AetherUniverse(size=2, dim=2, eta=0.05)
        u.observe(0, [0.0, 0.0])
        u.observe(1, [1.0, 0.0])
        
        # 1. Duplicate events
        orig_events_count = len(u.events)
        self.assertTrue(orig_events_count > 0)
        
        # Duplicate the last event
        duplicated_event = u.events[-1]
        u.events.append(duplicated_event)
        u.events.append(duplicated_event)
        
        # Verify execution does not break
        u.stabilize(dt=0.1)
        
        # 2. Drop all events (corrupted memory log)
        u.events.clear()
        
        # Verify core solver can still stabilize coordinates since it depends on current G
        u.observe(1, [1.2, 0.0])
        u.stabilize(dt=0.1)
        self.assertEqual(u.states[1].coords[0], 1.2)

if __name__ == "__main__":
    unittest.main()
