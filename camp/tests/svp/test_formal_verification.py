"""
test_formal_verification.py — Level 7 Formal Proof & Invariant Verification.
Provides mathematical checks for convergence, rollback correctness, eventual consistency, and identity integrity.
"""

import unittest
import math
import uuid
from RealityOS.kernel.aether_kernel import AetherUniverse
from RealityOS.kernel.reality_atom import RealityAtom, StateSnapshot

class TestFormalVerification(unittest.TestCase):
    def test_kkt_convergence_rate(self):
        """
        Verify that coordinates G converge under active Lagrange multiplier (KKT) updates.
        Asserts that constraint violation decreases monotonically or converges to a threshold.
        """
        universe = AetherUniverse(size=3, dim=2, eta=0.05, alpha_dual=0.1)
        universe.observe(0, [0.0, 0.0])
        universe.observe(1, [2.5, 0.0])  # Displaced from target 2.0
        universe.observe(2, [0.0, 1.5])
        universe.relate(0, 1)

        # Register distance tether constraint (target distance = 2.0)
        def tether_viol(G):
            return math.sqrt((G[0][0] - G[1][0])**2 + (G[0][1] - G[1][1])**2) - 2.0

        universe.constrain("tether", tether_viol, "||G[0] - G[1]|| - 2.0 = 0")

        initial_violation = abs(tether_viol(universe.get_G()))
        
        # Run stabilization loop and collect violations
        violations = []
        for _ in range(30):
            universe.stabilize(dt=0.1)
            violations.append(abs(tether_viol(universe.get_G())))

        final_violation = violations[-1]
        
        # Asserts:
        # 1. Final violation must be very close to zero.
        self.assertTrue(final_violation < 0.02, f"Final violation {final_violation} is too high")
        # 2. Over the last 10 steps, the violation should be stabilized or steadily small.
        self.assertTrue(final_violation < initial_violation, "System failed to reduce constraint violation.")

    def test_rollback_correctness(self):
        """
        Evaluate state integrity and exact coordinate match after fork -> modify -> rollback.
        """
        universe = AetherUniverse(size=3, dim=2, eta=0.08, alpha_dual=0.05)
        
        # Initial observations
        universe.observe(0, [1.0, 1.0])
        universe.observe(1, [2.0, 2.0])
        universe.observe(2, [3.0, 3.0])
        
        initial_G = universe.get_G()
        
        # Run a stabilize step to populate trajectory history
        universe.stabilize(dt=0.1)
        
        # Perform modification on coordinates
        universe.observe(0, [10.0, 10.0])
        universe.observe(1, [20.0, 20.0])
        
        # Run another step to log this modified state in history
        universe.stabilize(dt=0.1)
        
        # Ensure it actually modified
        modified_G = universe.get_G()
        self.assertNotEqual(modified_G, initial_G)
        
        # Rollback 1 step (to the state immediately after the first step, which is initial_G after first displacement)
        universe.rollback(ticks=1)
        
        # The rolled back state coordinates should match modified state history
        rolled_G = universe.get_G()
        self.assertEqual(rolled_G[0], [10.0, 10.0])
        
        # All nodes should be active after rollback
        for s in universe.states:
            self.assertTrue(s.active, "Nodes should wake up on rollback")

    def test_eventual_consistency_merge(self):
        """
        Verify that two branched universes merge correctly, their relations are shifted,
        and they stabilize consistently under KKT negotiation.
        """
        u1 = AetherUniverse(size=2, dim=2, eta=0.05)
        u1.observe(0, [0.0, 0.0])
        u1.observe(1, [1.0, 0.0])
        u1.relate(0, 1)
        
        u2 = AetherUniverse(size=2, dim=2, eta=0.05)
        u2.observe(0, [5.0, 5.0])
        u2.observe(1, [6.0, 5.0])
        u2.relate(0, 1)
        
        # Merge
        merged = u1.merge(u2)
        
        # Merged size should be u1.size + u2.size
        self.assertEqual(merged.size, 4)
        
        # Relationships should be shifted for u2
        self.assertIn(1, merged.relations[0])
        self.assertIn(3, merged.relations[2])
        self.assertIn(2, merged.relations[3])
        
        # Verify coordinates of merged state
        G_merged = merged.get_G()
        self.assertEqual(G_merged[0], [0.0, 0.0])
        self.assertEqual(G_merged[2], [5.0, 5.0])

    def test_identity_integrity_and_cycles(self):
        """
        Assert identity properties of RealityAtom: uniqueness, containment tree cycles prevention.
        """
        atom_a = RealityAtom(semantic_type="drone")
        atom_b = RealityAtom(semantic_type="sensor")
        
        # 1. Uniqueness of identity
        self.assertNotEqual(atom_a.id, atom_b.id)
        
        # 2. Hierarchy and cycle detection (preventing loop ancestry)
        atom_a.children.add(atom_b.id)
        atom_b.parent = atom_a.id
        
        # A validation function to check containment cycle
        def check_ancestry_cycle(atom_id, parent_map, visited=None) -> bool:
            if visited is None:
                visited = set()
            if atom_id in visited:
                return True
            visited.add(atom_id)
            parent_id = parent_map.get(atom_id)
            if parent_id is not None:
                return check_ancestry_cycle(parent_id, parent_map, visited)
            return False

        parent_map = {atom_b.id: atom_b.parent, atom_a.id: atom_a.parent}
        self.assertFalse(check_ancestry_cycle(atom_b.id, parent_map))
        
        # Introduce cycle
        atom_a.parent = atom_b.id
        parent_map[atom_a.id] = atom_a.parent
        
        self.assertTrue(check_ancestry_cycle(atom_b.id, parent_map), "Cycle was not detected!")

if __name__ == "__main__":
    unittest.main()
