"""
test_scheduler_ecology.py — Level 1 and 8 Scheduler Correctness & Constraint Ecology.
Validates starvation and priority inversion in the priority scheduler,
and constraint conflicts/oscillations in constraint ecology.
"""

import unittest
import math
from RealityOS.fabric.priority_scheduler import PriorityScheduler
from RealityOS.kernel.event_engine import Event, EventType
from RealityOS.kernel.aether_kernel import AetherUniverse
from RealityOS.kernel.ace_engine import ACEEngine

class TestSchedulerEcology(unittest.TestCase):
    def test_scheduler_starvation(self):
        """
        Verify that in the default PriorityScheduler, low priority events are starved
        under a continuous stream of high priority events.
        """
        scheduler = PriorityScheduler()
        
        # 1. Enqueue a low priority event
        low_event = Event(event_type=EventType.SENSOR_OBSERVATION, priority=1.0)
        scheduler.schedule(low_event)
        
        # 2. Enqueue multiple high priority events
        for _ in range(5):
            high_event = Event(event_type=EventType.EXTERNAL_ACTION, priority=100.0)
            scheduler.schedule(high_event)
            
        # 3. Pop next 5 items. They must all be the high priority events, leaving the low one starved.
        popped_priorities = []
        for _ in range(5):
            evt = scheduler.pop_next()
            popped_priorities.append(evt.priority)
            
        self.assertTrue(all(p == 100.0 for p in popped_priorities))
        self.assertEqual(scheduler.size(), 1)
        self.assertEqual(scheduler.pop_next().priority, 1.0)

    def test_scheduler_aging_mitigation(self):
        """
        Test that an aging mechanism (simulated) boosts low priority events to prevent starvation.
        """
        scheduler = PriorityScheduler()
        
        # Enqueue low priority event
        low_event = Event(event_type=EventType.SENSOR_OBSERVATION, priority=1.0)
        scheduler.schedule(low_event)
        
        # Simulating aging: boost priority of all events in queue that have waited
        # In a real scheduler, this happens periodically before popping
        def age_events(queue, age_rate=50.0):
            aged_queue = []
            for priority_neg, counter, event in queue:
                # priority is stored negative in heapq
                event.priority += age_rate
                # Re-push negative priority
                aged_queue.append((-event.priority, counter, event))
            queue.clear()
            for item in aged_queue:
                import heapq
                heapq.heappush(queue, item)

        # Enqueue high priority events
        for _ in range(2):
            high_event = Event(event_type=EventType.EXTERNAL_ACTION, priority=20.0)
            scheduler.schedule(high_event)
            
        # Age the queue. Low event priority rises from 1.0 to 51.0, exceeding high event priority (20.0)
        age_events(scheduler._queue, age_rate=50.0)
        
        # Pop next event. It should be the low priority one since it was aged/boosted!
        next_event = scheduler.pop_next()
        self.assertEqual(next_event.id, low_event.id)

    def test_constraint_ecology_conflict(self):
        """
        Verify that conflicting constraints lead to infinite dual variable oscillation/increase,
        and that the ACEEngine correctly retires/prunes the conflicting constraint.
        """
        # Create a 2-node universe with conflicting spatial constraints:
        # C1: G[0] - G[1] = 1.0
        # C2: G[0] - G[1] = 2.0
        universe = AetherUniverse(size=2, dim=2, eta=0.08, alpha_dual=0.1)
        universe.observe(0, [0.0, 0.0])
        universe.observe(1, [1.0, 0.0])
        
        def c1_fn(G):
            return math.sqrt((G[0][0]-G[1][0])**2 + (G[0][1]-G[1][1])**2) - 1.0
            
        def c2_fn(G):
            return math.sqrt((G[0][0]-G[1][0])**2 + (G[0][1]-G[1][1])**2) - 2.0
            
        universe.constrain("c1", c1_fn, "c1 = 0")
        universe.constrain("c2", c2_fn, "c2 = 0")
        
        # 1. Verify that running stabilization increases multipliers (shadow prices)
        initial_lambda_1 = universe.constraints["c1"].trust
        initial_lambda_2 = universe.constraints["c2"].trust
        
        for _ in range(5):
            universe.stabilize(dt=0.1)
            
        final_lambda_1 = universe.constraints["c1"].trust
        final_lambda_2 = universe.constraints["c2"].trust
        
        # Multipliers must have grown due to continuous constraint violations
        self.assertTrue(final_lambda_1 > initial_lambda_1)
        self.assertTrue(final_lambda_2 > initial_lambda_2)
        
        # 2. Verify that the evolution pruner (evolve()) retires constraints that are refuted or unfit
        # Create an ACEEngine instance to evolve constraints
        ace = ACEEngine(size=2, promotion_threshold=0.8, refutation_threshold=1.5, max_idle_ticks=5)
        
        # Register them as hypotheses
        from RealityOS.kernel.ace_engine import ConstraintHypothesis
        h1 = ConstraintHypothesis("c1", c1_fn, "c1 = 0", initial_status="active")
        h2 = ConstraintHypothesis("c2", c2_fn, "c2 = 0", initial_status="active")
        ace.hypotheses["c1"] = h1
        ace.hypotheses["c2"] = h2
        
        # Evolve over multiple ticks. Since they conflict, one of them will accumulate high violation error.
        # Once it exceeds the refutation threshold (1.5), it will be retired.
        for _ in range(20):
            universe.stabilize(dt=0.1)
            ace.evolve(universe)
            
        # One or both of the conflicting constraints should be retired to restore sanity
        retired_names = [h.name for h in ace.hypotheses.values() if h.status == "retired"]
        self.assertTrue(len(retired_names) > 0, "No conflicting constraint was retired/pruned!")

if __name__ == "__main__":
    unittest.main()
