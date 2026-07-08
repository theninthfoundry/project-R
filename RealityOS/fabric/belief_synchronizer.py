"""
Belief Synchronizer — Cross-Agent Belief Merging

When two agents interact (e.g., a robot hands an object to another robot),
their divergent beliefs about that object need to be reconciled.

This is the "communication protocol" of the Cognitive Universe.

Strategies:
    1. AUTHORITY: One agent's belief replaces the other's
    2. CONSENSUS: Weighted average based on confidence
    3. NEGOTIATION: Keep both and flag disagreement for resolution
    4. SENSOR_FUSION: Treat beliefs as independent observations and Kalman-fuse
"""

import uuid
from typing import Dict, List, Optional, Tuple
from enum import Enum
from RealityOS.universe.belief_layer import BeliefLayer, AgentBeliefModel, BeliefState
from RealityOS.kernel.reality_atom import StateVector


class SyncStrategy(Enum):
    AUTHORITY = "authority"       # Trust the more confident agent
    CONSENSUS = "consensus"       # Weighted merge
    SENSOR_FUSION = "sensor_fusion"  # Kalman filter fusion
    NEGOTIATE = "negotiate"       # Flag for external resolution


class SyncResult:
    """Result of a belief synchronization operation."""
    def __init__(self, atom_id: uuid.UUID, strategy: SyncStrategy,
                 merged_state: Optional[StateVector] = None,
                 disagreement: float = 0.0,
                 needs_resolution: bool = False):
        self.atom_id = atom_id
        self.strategy = strategy
        self.merged_state = merged_state
        self.disagreement = disagreement
        self.needs_resolution = needs_resolution


class BeliefSynchronizer:
    """Reconciles divergent beliefs between agents.
    
    This is genuinely novel — no existing AI system has a formal
    protocol for merging subjective world models between agents.
    """
    def __init__(self, belief_layer: BeliefLayer):
        self.belief_layer = belief_layer
        self.sync_history: List[SyncResult] = []
        
        # Threshold for auto-resolution vs. flagging
        self.auto_resolve_threshold = 2.0
        self.conflict_log: List[Tuple[str, str, uuid.UUID, float]] = []
    
    def sync_agents(self, agent_a_id: uuid.UUID, agent_b_id: uuid.UUID,
                     atom_id: uuid.UUID,
                     strategy: SyncStrategy = SyncStrategy.SENSOR_FUSION
                     ) -> SyncResult:
        """Synchronize two agents' beliefs about a specific atom."""
        
        model_a = self.belief_layer.get_agent_model(agent_a_id)
        model_b = self.belief_layer.get_agent_model(agent_b_id)
        
        if not model_a or not model_b:
            return SyncResult(atom_id, strategy, needs_resolution=True)
        
        belief_a = model_a.get_belief(atom_id)
        belief_b = model_b.get_belief(atom_id)
        
        # Handle cases where one agent has no belief
        if not belief_a and not belief_b:
            return SyncResult(atom_id, strategy)
        if not belief_a:
            # Agent A learns from Agent B
            model_a.observe(atom_id, belief_b.mean, belief_b.variance)
            return SyncResult(atom_id, strategy, merged_state=belief_b.mean)
        if not belief_b:
            model_b.observe(atom_id, belief_a.mean, belief_a.variance)
            return SyncResult(atom_id, strategy, merged_state=belief_a.mean)
        
        # Both agents have beliefs — measure disagreement
        divergence = model_a.divergence_from(model_b, atom_id) or 0.0
        
        if strategy == SyncStrategy.AUTHORITY:
            return self._sync_authority(model_a, model_b, belief_a, belief_b, 
                                         atom_id, divergence)
        elif strategy == SyncStrategy.CONSENSUS:
            return self._sync_consensus(model_a, model_b, belief_a, belief_b,
                                         atom_id, divergence)
        elif strategy == SyncStrategy.SENSOR_FUSION:
            return self._sync_kalman(model_a, model_b, belief_a, belief_b,
                                      atom_id, divergence)
        else:
            return self._sync_negotiate(model_a, model_b, belief_a, belief_b,
                                         atom_id, divergence)
    
    def _sync_authority(self, model_a, model_b, 
                         belief_a: BeliefState, belief_b: BeliefState,
                         atom_id: uuid.UUID, div: float) -> SyncResult:
        """More confident agent's belief wins."""
        if belief_a.confidence >= belief_b.confidence:
            winner, loser = belief_a, model_b
            print(f"  [BeliefSync/AUTHORITY] {model_a.agent_name} wins (conf={belief_a.confidence:.3f})")
        else:
            winner, loser = belief_b, model_a
            print(f"  [BeliefSync/AUTHORITY] {model_b.agent_name} wins (conf={belief_b.confidence:.3f})")
        
        loser.observe(atom_id, winner.mean, winner.variance)
        return SyncResult(atom_id, SyncStrategy.AUTHORITY, 
                          merged_state=winner.mean, disagreement=div)
    
    def _sync_consensus(self, model_a, model_b,
                         belief_a: BeliefState, belief_b: BeliefState,
                         atom_id: uuid.UUID, div: float) -> SyncResult:
        """Weighted average based on confidence."""
        w_a = belief_a.confidence
        w_b = belief_b.confidence
        total = w_a + w_b + 1e-10
        
        merged = []
        for i in range(max(len(belief_a.mean), len(belief_b.mean))):
            va = belief_a.mean[i] if i < len(belief_a.mean) else 0.0
            vb = belief_b.mean[i] if i < len(belief_b.mean) else 0.0
            merged.append((va * w_a + vb * w_b) / total)
        
        # Update both agents to the consensus
        merged_var = [0.5] * len(merged)  # Reset uncertainty
        model_a.observe(atom_id, merged, merged_var)
        model_b.observe(atom_id, merged, merged_var)
        
        print(f"  [BeliefSync/CONSENSUS] Merged belief for atom {atom_id}: {[round(v,3) for v in merged]}")
        return SyncResult(atom_id, SyncStrategy.CONSENSUS, 
                          merged_state=merged, disagreement=div)
    
    def _sync_kalman(self, model_a, model_b,
                      belief_a: BeliefState, belief_b: BeliefState,
                      atom_id: uuid.UUID, div: float) -> SyncResult:
        """Treat the other agent's belief as an observation and Kalman-fuse.
        
        Each agent fuses the other's belief into their own using the
        Bayesian update mechanism already built into BeliefState.
        """
        # Agent A incorporates Agent B's belief as a noisy observation
        model_a.observe(atom_id, belief_b.mean, belief_b.variance)
        # Agent B incorporates Agent A's belief as a noisy observation
        model_b.observe(atom_id, belief_a.mean, belief_a.variance)
        
        # After fusion, both should converge
        fused = model_a.get_believed_state(atom_id)
        
        print(f"  [BeliefSync/KALMAN] Fused state for atom {atom_id}: {[round(v,3) for v in fused]}")
        return SyncResult(atom_id, SyncStrategy.SENSOR_FUSION,
                          merged_state=fused, disagreement=div)
    
    def _sync_negotiate(self, model_a, model_b,
                         belief_a: BeliefState, belief_b: BeliefState,
                         atom_id: uuid.UUID, div: float) -> SyncResult:
        """Flag disagreement for external resolution."""
        self.conflict_log.append((
            model_a.agent_name, model_b.agent_name, atom_id, div
        ))
        print(f"  [BeliefSync/NEGOTIATE] CONFLICT flagged between "
              f"{model_a.agent_name} and {model_b.agent_name} "
              f"on atom {atom_id} (divergence={div:.3f})")
        return SyncResult(atom_id, SyncStrategy.NEGOTIATE,
                          disagreement=div, needs_resolution=True)
    
    def sync_all_shared_atoms(self, agent_a_id: uuid.UUID, agent_b_id: uuid.UUID,
                                strategy: SyncStrategy = SyncStrategy.SENSOR_FUSION
                                ) -> List[SyncResult]:
        """Synchronize all atoms both agents share beliefs about."""
        model_a = self.belief_layer.get_agent_model(agent_a_id)
        model_b = self.belief_layer.get_agent_model(agent_b_id)
        
        if not model_a or not model_b:
            return []
        
        shared_atoms = model_a.known_atoms & model_b.known_atoms
        results = []
        for atom_id in shared_atoms:
            result = self.sync_agents(agent_a_id, agent_b_id, atom_id, strategy)
            results.append(result)
        
        return results
