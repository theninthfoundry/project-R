import uuid
from typing import Dict, List, Set, Optional, Any
from enum import Enum
from RealityOS.kernel.reality_atom import RealityAtom

class PartitionState(Enum):
    COMPUTING = "computing"   # Full state, all relations, active transitions
    IDLE = "idle"             # State cached, relations indexed, no active transitions
    DORMANT = "dormant"       # Summary state only, loadable on demand
    UNLOADED = "unloaded"     # Exists in persistent storage, not in memory

class WorldPartition:
    """
    A chunk of the RealityOS universe (e.g., a Room, a Building, a City).
    Manages loading/unloading of atoms to keep memory usage bounded.
    """
    def __init__(self, name: str, parent_id: Optional[uuid.UUID] = None):
        self.id = uuid.uuid4()
        self.name = name
        self.parent_id = parent_id
        
        self.state = PartitionState.DORMANT
        
        # Atoms that belong to this partition
        self.atoms: Dict[uuid.UUID, RealityAtom] = {}
        
        # Child partitions (e.g., Rooms inside a Building)
        self.children: Set[uuid.UUID] = set()
        
    def add_atom(self, atom: RealityAtom):
        self.atoms[atom.id] = atom
        
    # Mock persistent storage representation
    _DISK_STORAGE: Dict[uuid.UUID, Dict[str, Any]] = {}

    def transition_state(self, new_state: PartitionState):
        """
        Transitions the partition to a new state, loading or unloading 
        data from persistent storage as needed.
        """
        if self.state == new_state:
            return
            
        old_state = self.state
        self.state = new_state
        
        # Unloading to disk
        if new_state == PartitionState.UNLOADED:
            print(f"  [WorldPartition] Unloading '{self.name}' to persistent storage...")
            serialized = {}
            for atom_id, atom in self.atoms.items():
                serialized[atom_id] = {
                    "semantic_type": atom.semantic_type,
                    "state": atom.state.copy(),
                    "energy": atom.energy,
                    "confidence": atom.confidence,
                    "fields": atom.fields.copy()
                }
            WorldPartition._DISK_STORAGE[self.id] = serialized
            self.atoms.clear()
            
        # Loading from disk
        elif old_state == PartitionState.UNLOADED:
            print(f"  [WorldPartition] Loading '{self.name}' from persistent storage...")
            serialized = WorldPartition._DISK_STORAGE.get(self.id, {})
            for atom_id, data in serialized.items():
                atom = RealityAtom(
                    id=atom_id,
                    semantic_type=data["semantic_type"],
                    state=data["state"],
                    energy=data["energy"],
                    confidence=data["confidence"],
                    fields=data["fields"]
                )
                self.atoms[atom_id] = atom
                
        print(f"Partition '{self.name}' transitioned from {old_state.value} to {new_state.value}")


class WorldManager:
    """Manages all world partitions."""
    def __init__(self):
        self.partitions: Dict[uuid.UUID, WorldPartition] = {}
        self.root_id = None
        
    def create_partition(self, name: str, parent_id: Optional[uuid.UUID] = None) -> WorldPartition:
        p = WorldPartition(name, parent_id)
        self.partitions[p.id] = p
        
        if parent_id and parent_id in self.partitions:
            self.partitions[parent_id].children.add(p.id)
            
        if not self.root_id:
            self.root_id = p.id
            
        return p
        
    def get_partition_for_atom(self, atom_id: uuid.UUID) -> Optional[WorldPartition]:
        """Find which partition contains this atom."""
        for p in self.partitions.values():
            if atom_id in p.atoms:
                return p
        return None

    def delete_partition(self, partition_id: uuid.UUID):
        """Safely delete partition and prune global static storage cache to prevent leaks."""
        if partition_id in self.partitions:
            self.partitions.pop(partition_id)
        if partition_id in WorldPartition._DISK_STORAGE:
            WorldPartition._DISK_STORAGE.pop(partition_id)
