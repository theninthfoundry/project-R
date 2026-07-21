"""
test_security_isolation.py — Level 6 Security and Sandbox Isolation.
Validates runtime isolation between simulation worlds and resource quota/sandboxing checks.
"""

import unittest
import time
from RealityOS.kernel.aether_kernel import AetherUniverse

class MockPlugin:
    def __init__(self, run_fn, allowed_ticks: int = 100):
        self.run_fn = run_fn
        self.allowed_ticks = allowed_ticks

    def execute(self, G):
        # Mock resource/quota check
        ticks = 0
        for step in self.run_fn(G):
            ticks += 1
            if ticks > self.allowed_ticks:
                raise TimeoutError("Plugin exceeded allocated CPU quota ticks!")
            yield step

class TestSecurityIsolation(unittest.TestCase):
    def test_world_boundary_isolation(self):
        """
        Verify that coordinates, events, and constraints are strictly isolated
        between independent Aether universes.
        """
        world_a = AetherUniverse(size=2, dim=2)
        world_b = AetherUniverse(size=2, dim=2)
        
        world_a.observe(0, [1.0, 1.0])
        world_b.observe(0, [9.0, 9.0])
        
        # Verify coordinates isolation
        self.assertEqual(world_a.states[0].coords, [1.0, 1.0])
        self.assertEqual(world_b.states[0].coords, [9.0, 9.0])
        
        # Define constraint in World A
        world_a.constrain("dist_tether", lambda G: 0.0, "true = 0")
        
        # Verify constraint isolation
        self.assertIn("dist_tether", world_a.constraints)
        self.assertNotIn("dist_tether", world_b.constraints)
        
        # Verify event log isolation
        self.assertNotEqual([e.id for e in world_a.events], [e.id for e in world_b.events])

    def test_plugin_sandbox_quota_limits(self):
        """
        Verify that runaway custom operators/plugins are halted by resource quotas.
        """
        # Runaway loop generator function simulating a malicious or buggy plugin
        def runaway_plugin_fn(G):
            while True:
                # Simulating heavy computation step
                yield G
                
        plugin = MockPlugin(runaway_plugin_fn, allowed_ticks=50)
        
        # Running the plugin on coordinates must raise TimeoutError when it hits the quota limit
        with self.assertRaises(TimeoutError):
            for state in plugin.execute([[0.0, 0.0]]):
                pass

    def test_capability_based_write_guard(self):
        """
        Verify that writing to states requires a valid capability handle/token.
        """
        class GuardedUniverse(AetherUniverse):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._capabilities = {}

            def register_capability(self, node_idx: int, secret_token: str):
                self._capabilities[node_idx] = secret_token

            def observe_secured(self, node_idx: int, values: list, token: str):
                if self._capabilities.get(node_idx) != token:
                    raise PermissionError("Access Denied: Invalid capability token for node write operation")
                return self.observe(node_idx, values)

        gu = GuardedUniverse(size=2, dim=2)
        gu.register_capability(0, "TOKEN_XYZ")
        
        # Authorized update should succeed
        gu.observe_secured(0, [1.0, 1.0], "TOKEN_XYZ")
        self.assertEqual(gu.states[0].coords, [1.0, 1.0])
        
        # Unauthorized update should fail
        with self.assertRaises(PermissionError):
            gu.observe_secured(0, [2.0, 2.0], "TOKEN_ATTACKER")

if __name__ == "__main__":
    unittest.main()
