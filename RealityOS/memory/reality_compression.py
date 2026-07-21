from typing import List
from RealityOS.kernel.reality_atom import StateSnapshot, RingBuffer

class RealityCompressor:
    """
    Compresses history vectors of Reality Atoms to prevent memory growth.
    Filters redundant states (e.g. constant state values) and retains
    only inflection points or averages.
    """
    def __init__(self, deviation_threshold: float = 0.01):
        self.deviation_threshold = deviation_threshold

    def compress_history(self, history: RingBuffer) -> List[StateSnapshot]:
        """
        Simplifies a list of snapshots by removing values that lie
        within deviation_threshold of their preceding neighbors.
        """
        # Reconstruct chronological order from RingBuffer
        if len(history.buffer) < history.capacity:
            snapshots = list(history.buffer)
        else:
            snapshots = history.buffer[history.head:] + history.buffer[:history.head]

        if len(snapshots) <= 2:
            return list(snapshots)

        compressed = [snapshots[0]]
        
        for i in range(1, len(snapshots) - 1):
            prev = compressed[-1].state
            curr = snapshots[i].state
            nxt = snapshots[i+1].state
            
            # Simple heuristic: if difference between prev, curr, and next is negligible,
            # we skip the current state (it's a linear segment / redundant)
            if self._is_redundant(prev, curr, nxt):
                continue
            compressed.append(snapshots[i])

        compressed.append(snapshots[-1])
        return compressed

    def _is_redundant(self, prev: List[float], curr: List[float], nxt: List[float]) -> bool:
        if len(prev) != len(curr) or len(curr) != len(nxt):
            return False
            
        # Check if current is just a linear interpolation between prev and next
        for p, c, n in zip(prev, curr, nxt):
            expected = (p + n) / 2.0
            if abs(c - expected) > self.deviation_threshold:
                return False
        return True
