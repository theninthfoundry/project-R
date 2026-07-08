"""
test_agent_watcher.py — Unit tests for the CAMP monitoring engine.
"""

import unittest
from camp.core.agent_watcher import AgentWatcher
from camp.core.alert_engine import AlertEngine


class TestAgentWatcher(unittest.TestCase):
    def setUp(self):
        self.watcher = AgentWatcher(alpha=0.2)
        self.alert_engine = AlertEngine(alert_budget_per_tick=5)

    def test_agent_registration(self):
        s = self.watcher.register_agent("test_agent", value=0.8, cost_limit=0.5)
        self.assertEqual(s.uid, "test_agent")
        self.assertEqual(s.value, 0.8)
        self.assertEqual(len(s.constraints), 4)
        self.assertEqual(s.constraints[0].upper, 0.5)

    def test_belief_momentum(self):
        """Verify belief momentum smooths out a sudden raw spike."""
        self.watcher.register_agent("test_agent", value=0.5)
        
        # Ingest a healthy observation
        self.watcher.observe("test_agent", cost=0.01, latency=100.0, error=0.0, tokens=1.0)
        
        # Now a sudden extreme spike in latency (e.g. 9000ms)
        telemetry = self.watcher.observe("test_agent", cost=0.01, latency=9000.0, error=0.0, tokens=1.0)
        
        # Verify raw value is indeed 9000
        self.assertEqual(telemetry["raw"][1], 9000.0)
        
        # Verify smoothed belief did NOT snap immediately to 9000 (damped by alpha=0.2)
        # Expected belief = prior (100) * 0.8 + 9000 * 0.2 = 80 + 1800 = 1880
        belief_latency = telemetry["belief"][1]
        self.assertTrue(belief_latency < 2500.0)
        self.assertTrue(belief_latency > 1500.0)

    def test_surprise_trust_decay(self):
        """Verify that high surprise results in trust (precision) decay."""
        s = self.watcher.register_agent("test_agent", value=0.5)
        
        # Initialize trust (precision) at 1.0
        self.assertEqual(s.tau[0], 1.0)
        
        # Trigger an observation that deviates heavily from the prediction, causing surprise
        # Prediction was initialized to 0.0, let's observe a high value
        telemetry = self.watcher.observe("test_agent", cost=8.0, latency=10.0, error=0.0, tokens=1.0)
        
        # Surprise should be positive
        self.assertTrue(telemetry["surprise"] > 5.0)
        
        # Trust (tau) for the cost metric should have decayed from 1.0
        self.assertTrue(s.tau[0] < 1.0)

    def test_alert_engine_priority(self):
        """Verify alerts are prioritized by (value * surprise)."""
        # Register a high value agent (payment_gateway) and a low value agent (logger)
        gw = self.watcher.register_agent("payment_gateway", value=1.0, latency_limit=100.0)
        log = self.watcher.register_agent("logger", value=0.1, latency_limit=100.0)
        
        # Trigger severe violation on both
        self.watcher.observe("payment_gateway", cost=0.0, latency=500.0, error=0.0, tokens=0.0)
        self.watcher.observe("logger", cost=0.0, latency=500.0, error=0.0, tokens=0.0)
        
        # Pull active alerts from engine
        alerts = self.alert_engine.process_tick_alerts([gw, log])
        
        # Both should be in alerts, but high value agent should be first
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0].agent_id, "payment_gateway")
        self.assertEqual(alerts[1].agent_id, "logger")


if __name__ == "__main__":
    unittest.main()
