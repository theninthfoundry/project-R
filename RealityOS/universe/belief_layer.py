"""
Belief Layer — Per-Agent Divergent World Models

The deepest innovation in RealityOS's Cognitive Universe.

Current AI: one model, one reality.
RealityOS: every agent maintains its own probabilistic belief about
every atom's state. Two robots in the same room can disagree about
where an object is, what it contains, or whether it even exists.

Architecture:
    Ground Truth (kernel)
        ↓
    Agent A's Belief Layer  ← partial observations + prior knowledge
        ↓
    Agent B's Belief Layer  ← different observations + different priors
        ↓
    Merged Belief (on demand, via BeliefSynchronizer)

This directly addresses Flaw #5 from the architecture plan:
"Reality isn't one world. Every person has different knowledge."
"""

import uuid
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from RealityOS.kernel.reality_atom import RealityAtom, StateVector, Timestamp


@dataclass
class BeliefState:
    """An agent's belief about a single atom's state.
    
    Instead of storing a point estimate, we store a full
    probability distribution (Gaussian approximation for now).
    """
    atom_id: uuid.UUID
    
    # Estimated state (agent's best guess)
    mean: StateVector
    
    # Uncertainty per dimension (diagonal covariance)
    variance: StateVector
    
    # How many observations have contributed
    observation_count: int = 0
    
    # When this belief was last updated
    last_updated: Timestamp = field(default_factory=time.time)
    
    # Source: which sensor/inference produced this belief
    source: str = "prior"
    
    # Confidence decays over time — stale beliefs become uncertain
    staleness_rate: float = 0.01
    
    @property
    def confidence(self) -> float:
        """Overall confidence [0, 1] combining variance and staleness."""
        avg_var = sum(self.variance) / max(len(self.variance), 1)
        uncertainty = 1.0 / (1.0 + avg_var)
        
        age = time.time() - self.last_updated
        freshness = math.exp(-self.staleness_rate * age)
        
        return uncertainty * freshness
    
    def update_with_observation(self, observed_state: StateVector, 
                                 observation_variance: StateVector):
        """Bayesian update: fuse a new observation into this belief.
        
        Uses a simplified Kalman filter update:
            posterior_mean = prior_mean + K * (observation - prior_mean)
            posterior_var  = (1 - K) * prior_var
            K = prior_var / (prior_var + obs_var)
        """
        new_mean = []
        new_var = []
        
        for i in range(len(self.mean)):
            prior_var = self.variance[i]
            obs_var = observation_variance[i] if i < len(observation_variance) else 1.0
            obs_val = observed_state[i] if i < len(observed_state) else self.mean[i]
            
            # Kalman gain
            K = prior_var / (prior_var + obs_var + 1e-10)
            
            # Posterior
            new_mean.append(self.mean[i] + K * (obs_val - self.mean[i]))
            new_var.append((1.0 - K) * prior_var)
        
        self.mean = new_mean
        self.variance = new_var
        self.observation_count += 1
        self.last_updated = time.time()
        self.source = "observation"


class AgentBeliefModel:
    """One agent's complete model of reality.
    
    This is that agent's 'subjective universe' — it may differ
    from ground truth and from every other agent's model.
    """
    def __init__(self, agent_id: uuid.UUID, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        
        # atom_id -> this agent's belief about that atom
        self.beliefs: Dict[uuid.UUID, BeliefState] = {}
        
        # Atoms this agent knows exist (may be a subset of reality)
        self.known_atoms: Set[uuid.UUID] = set()
        
        # Atoms this agent has never observed (blind spots)
        self.unknown_atoms: Set[uuid.UUID] = set()
    
    def observe(self, atom_id: uuid.UUID, 
                observed_state: StateVector,
                observation_variance: StateVector):
        """Agent makes an observation about an atom."""
        self.known_atoms.add(atom_id)
        self.unknown_atoms.discard(atom_id)
        
        if atom_id in self.beliefs:
            self.beliefs[atom_id].update_with_observation(
                observed_state, observation_variance
            )
        else:
            self.beliefs[atom_id] = BeliefState(
                atom_id=atom_id,
                mean=list(observed_state),
                variance=list(observation_variance),
                observation_count=1,
                source="first_observation"
            )
    
    def get_belief(self, atom_id: uuid.UUID) -> Optional[BeliefState]:
        """What does this agent think this atom's state is?"""
        return self.beliefs.get(atom_id)
    
    def get_believed_state(self, atom_id: uuid.UUID) -> Optional[StateVector]:
        """Convenience: return the mean state estimate."""
        belief = self.beliefs.get(atom_id)
        return belief.mean if belief else None
    
    def divergence_from(self, other: 'AgentBeliefModel', 
                         atom_id: uuid.UUID) -> Optional[float]:
        """How much do two agents disagree about a specific atom?
        
        Returns KL-divergence-inspired metric between the two beliefs.
        """
        my_belief = self.beliefs.get(atom_id)
        their_belief = other.beliefs.get(atom_id)
        
        if not my_belief or not their_belief:
            return None
        
        # Simplified divergence: sum of squared differences in means
        # weighted by inverse variance (more certain = bigger disagreement matters more)
        total = 0.0
        for i in range(min(len(my_belief.mean), len(their_belief.mean))):
            diff = my_belief.mean[i] - their_belief.mean[i]
            weight = 1.0 / (my_belief.variance[i] + their_belief.variance[i] + 1e-10)
            total += diff * diff * weight
        
        return total


class BeliefLayer:
    """Manages all agent belief models in the Cognitive Universe.
    
    The central registry for the 'Multi-Reality' system.
    Ground truth lives in the kernel; beliefs live here.
    """
    def __init__(self):
        self.agent_models: Dict[uuid.UUID, AgentBeliefModel] = {}
    
    def register_agent(self, agent_id: uuid.UUID, name: str) -> AgentBeliefModel:
        model = AgentBeliefModel(agent_id, name)
        self.agent_models[agent_id] = model
        return model
    
    def get_agent_model(self, agent_id: uuid.UUID) -> Optional[AgentBeliefModel]:
        return self.agent_models.get(agent_id)
    
    def find_disagreements(self, atom_id: uuid.UUID, 
                            threshold: float = 1.0) -> List[Tuple[str, str, float]]:
        """Find all pairs of agents that disagree about an atom.
        
        Returns list of (agent_a_name, agent_b_name, divergence).
        """
        agents = list(self.agent_models.values())
        disagreements = []
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                div = agents[i].divergence_from(agents[j], atom_id)
                if div is not None and div > threshold:
                    disagreements.append((
                        agents[i].agent_name,
                        agents[j].agent_name,
                        div
                    ))
        
        return disagreements
    
    def consensus_state(self, atom_id: uuid.UUID) -> Optional[StateVector]:
        """Compute the consensus (weighted average) across all agents.
        
        Weights by confidence — more certain agents contribute more.
        """
        total_weight = 0.0
        weighted_sum = None
        
        for model in self.agent_models.values():
            belief = model.get_belief(atom_id)
            if not belief:
                continue
            
            w = belief.confidence
            if weighted_sum is None:
                weighted_sum = [v * w for v in belief.mean]
            else:
                for i in range(len(weighted_sum)):
                    if i < len(belief.mean):
                        weighted_sum[i] += belief.mean[i] * w
            total_weight += w
        
        if weighted_sum and total_weight > 0:
            return [v / total_weight for v in weighted_sum]
        return None
