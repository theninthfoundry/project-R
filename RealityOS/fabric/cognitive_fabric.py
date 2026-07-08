import uuid
import time
from typing import Dict, List, Set, Optional, Any
from RealityOS.kernel.reality_atom import RealityAtom
from RealityOS.kernel.event_engine import EventEngine, Event, EventType
from RealityOS.kernel.state_transition import StateTransitionNetwork
from RealityOS.kernel.locality_field import FieldManager
from RealityOS.kernel.relation_engine import RelationEngine
from RealityOS.kernel.causal_engine import CausalEngine
from RealityOS.kernel.temporal_engine import TemporalEngine
from RealityOS.fabric.energy_manager import EnergyManager, EnergyState
from RealityOS.fabric.priority_scheduler import PriorityScheduler
from RealityOS.kernel.cognitive_state import CognitiveState
from RealityOS.kernel.evolution import AnalyticEvolution

class CognitiveFabric:
    """
    The orchestrator / central nervous system of RealityOS.
    Binds the kernel engines, managers, and schedulers together.
    """
    def __init__(self):
        self.event_engine = EventEngine()
        self.stn = StateTransitionNetwork()
        self.field_manager = FieldManager()
        self.relation_engine = RelationEngine()
        self.causal_engine = CausalEngine()
        self.temporal_engine = TemporalEngine()
        self.energy_manager = EnergyManager()
        self.scheduler = PriorityScheduler()
        self.evolution_engine = AnalyticEvolution()
        
        # Registry of runtimes
        self.runtimes: List[Any] = []
        
        # Registry of all registered atoms and cognitive states
        self.atoms: Dict[uuid.UUID, RealityAtom] = {}
        self.cognitive_states: Dict[str, CognitiveState] = {}
        
        # Subscribe scheduler to the Event Bus
        self.event_engine.bus.subscribe(self.on_event_published)
        
        # Keep track of last decay step time
        self.last_decay_time = time.time()

    def register_runtime(self, runtime: Any):
        self.runtimes.append(runtime)

    def register_cognitive_state(self, s: CognitiveState):
        self.cognitive_states[s.uid] = s

    def unregister_cognitive_state(self, uid: str):
        if uid in self.cognitive_states:
            del self.cognitive_states[uid]

    def register_atom(self, atom: RealityAtom):
        self.atoms[atom.id] = atom
        self.relation_engine.register_atom(atom)
        self.energy_manager.register_atom(atom)

    def unregister_atom(self, atom_id: uuid.UUID):
        if atom_id in self.atoms:
            del self.atoms[atom_id]
        self.relation_engine.unregister_atom(atom_id)
        self.energy_manager.unregister_atom(atom_id)

    def on_event_published(self, event: Event):
        """Callback when an event enters the bus: route to scheduler."""
        self.scheduler.schedule(event)

    def process_tick(self):
        """Processes a single step in the scheduler queue."""
        # 1. Decay metabolism for active atoms
        now = time.time()
        elapsed = now - self.last_decay_time
        self.last_decay_time = now
        
        for atom in self.atoms.values():
            self.energy_manager.decay_metabolism(atom, elapsed)

        # Optimize overall energy bounds
        self.energy_manager.optimize_budget(self.atoms)

        # 1b. Step all registered Cognitive States
        for cs in list(self.cognitive_states.values()):
            # By default, we observe its current state value
            self.evolution_engine.step(cs, cs.x)

        # 2. Pop next priority event and process
        if not self.scheduler.has_events():
            return

        event = self.scheduler.pop_next()
        
        # Find target atom
        target_atom = self.atoms.get(event.target)
        if not target_atom:
            return # Target atom doesn't exist or is unloaded
            
        # If target atom is deep sleeping, wake it to IDLE
        target_state = self.energy_manager.get_state(target_atom.id)
        if target_state == EnergyState.DEEP_SLEEP:
            target_atom.energy = 2.0 # Force wake to dormant/idle
            
        # Boost target atom energy because of activity
        if event.delta:
            self.energy_manager.record_activity(target_atom, event.delta.magnitude, event.timestamp)

        # 3. Process STN transition
        cascade_events = self.stn.process_transition(target_atom, event)
        
        # 4. Route to Runtimes
        for rt in self.runtimes:
            if rt.accepts(event):
                # Process event inside the runtime
                rt_cascades = rt.process(event, self.atoms)
                cascade_events.extend(rt_cascades)

        # Record causal links and emit cascades
        for child_ev in cascade_events:
            self.causal_engine.record_transition(event, child_ev, "stn_propagation")
            self.event_engine.emit(child_ev)

        # 5. Route field update propagations
        field_cascade = self.field_manager.route_event(event, list(target_atom.fields))
        for field_ev in field_cascade:
            self.causal_engine.record_transition(event, field_ev, "field_propagation")
            self.event_engine.emit(field_ev)
            
    def run_until_idle(self, timeout_steps: int = 100):
        """Runs the event loop until no more events are scheduled."""
        steps = 0
        while self.scheduler.has_events() and steps < timeout_steps:
            self.process_tick()
            steps += 1
