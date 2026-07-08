"""
Benchmark: Adversarial Scaling and Complex Transition Workloads

This benchmark tests two specific, falsifiable claims about event-driven architectures:
1. Adversarial scaling: When the rate of change in the world is high (e.g., a "mostly-changing"
   world where 90% of atoms change), event-driven local propagation degrades to perform WORSE
   than naive recompute due to queue bookkeeping and redundant cascade checks.
2. Workload complexity: When the transition function is complex (e.g., simulated neural model
   inference) instead of a scalar toy, the wall-clock speedup of the event-driven engine
   converges toward the theoretical touch ratio because scheduling overhead becomes negligible.

Setup:
- N atoms, sparse random-neighbor connectivity (avg_degree = 6).
- Two transition functions:
    - Toy: simple scalar addition (extremely cheap, dominated by scheduling overhead).
    - Heavy: matrix multiplication / workload simulation (dwarfs scheduling overhead).
"""

import time
import random
import math
from collections import deque
from dataclasses import dataclass

@dataclass
class Atom:
    id: int
    state: float
    neighbors: list
    threshold: float = 0.05

def build_world(n_atoms: int, avg_degree: int, seed: int = 0):
    rng = random.Random(seed)
    atoms = {}
    for i in range(n_atoms):
        atoms[i] = Atom(id=i, state=0.0, neighbors=[])

    deg = min(avg_degree, n_atoms - 1) if n_atoms > 1 else 0
    for i in range(n_atoms):
        neighbors = set()
        while len(neighbors) < deg:
            j = rng.randrange(n_atoms)
            if j != i:
                neighbors.add(j)
        atoms[i].neighbors = list(neighbors)
    return atoms

def heavy_transition_workload(value: float, iterations: int = 200) -> float:
    """Simulates a non-trivial transition function (like evaluating a tiny neural model)."""
    # A small matrix-like multiply-accumulate loop to burn CPU cycles
    acc = value
    for i in range(iterations):
        acc = math.sin(acc) * 1.01 + math.cos(acc) * 0.01
    return acc

def run_naive_tick(atoms, tick_events, heavy_workload: bool):
    touches = 0
    for i, atom in atoms.items():
        delta = tick_events.get(i, 0.0)
        if heavy_workload:
            # Recompute model on every atom
            atom.state = heavy_transition_workload(atom.state + delta * 0.5)
        else:
            atom.state += delta * 0.5
        touches += 1
    return touches

def run_event_driven_tick(atoms, tick_events, heavy_workload: bool, max_cascade_depth=6):
    touches = 0
    queue = deque()
    for i, delta in tick_events.items():
        queue.append((i, delta, 0))

    while queue:
        i, delta, depth = queue.popleft()
        atom = atoms[i]
        
        if heavy_workload:
            atom.state = heavy_transition_workload(atom.state + delta)
        else:
            atom.state += delta
            
        touches += 1

        if depth >= max_cascade_depth:
            continue

        propagated = delta * 0.3  # attenuation per hop
        if abs(propagated) < atom.threshold:
            continue

        for n_id in atom.neighbors:
            queue.append((n_id, propagated, depth + 1))

    return touches

def run_adversarial_experiment(n_atoms=10000, avg_degree=6, ticks=5, seed=42):
    """
    Tests how Naive vs. Event-driven scale when the fraction of active atoms
    grows from 1% to 90% (the "everything is on fire" adversarial case).
    """
    print("\n=== EXPERIMENT 1: Adversarial Change-Rate Scaling (N = 10,000) ===")
    print(f"{'Active %':<10} | {'Active Count':<12} | {'Naive Touches':<15} | {'Event Touches':<15} | {'Touch Ratio':<12} | {'Speedup':<10}")
    print("-" * 88)
    
    fractions = [0.002, 0.01, 0.05, 0.10, 0.30, 0.60, 0.90]
    
    for f in fractions:
        n_events = max(1, int(n_atoms * f))
        
        # Naive Run
        atoms_naive = build_world(n_atoms, avg_degree, seed=seed)
        rng_events = random.Random(seed + 1)
        t0 = time.perf_counter()
        touches_naive = 0
        for _ in range(ticks):
            event_ids = rng_events.sample(list(atoms_naive.keys()), n_events)
            tick_events = {i: rng_events.uniform(-1.0, 1.0) for i in event_ids}
            touches_naive += run_naive_tick(atoms_naive, tick_events, heavy_workload=False)
        t_naive = time.perf_counter() - t0
        
        # Event-driven Run
        atoms_evt = build_world(n_atoms, avg_degree, seed=seed)
        rng_events2 = random.Random(seed + 1)
        t0 = time.perf_counter()
        touches_evt = 0
        for _ in range(ticks):
            event_ids = rng_events2.sample(list(atoms_evt.keys()), n_events)
            tick_events = {i: rng_events2.uniform(-1.0, 1.0) for i in event_ids}
            touches_evt += run_event_driven_tick(atoms_evt, tick_events, heavy_workload=False)
        t_evt = time.perf_counter() - t0
        
        ratio = touches_naive / touches_evt if touches_evt > 0 else 0
        speedup = t_naive / t_evt if t_evt > 0 else 0
        
        print(f"{f*100:>8.1f}% | {n_events:>12} | {touches_naive/ticks:>15.1f} | {touches_evt/ticks:>15.1f} | {ratio:>11.2f}x | {speedup:>8.2f}x")

def run_workload_complexity_experiment(n_atoms=5000, avg_degree=6, ticks=5, seed=42):
    """
    Tests how transition workload complexity (scalar add vs. math loop)
    influences the realization of theoretical touch speedups on wall-clock.
    """
    print("\n=== EXPERIMENT 2: Transition Function Workload Complexity (N = 5,000) ===")
    print("Fixed change rate: 20 events per tick.")
    print(f"{'Workload':<10} | {'Naive ms/tick':<15} | {'Event ms/tick':<15} | {'Touch Ratio':<12} | {'Wall-clock Speedup':<20}")
    print("-" * 88)
    
    n_events = 20
    
    for workload_name, heavy_flag in [("Scalar (Toy)", False), ("Math (Heavy)", True)]:
        # Naive Run
        atoms_naive = build_world(n_atoms, avg_degree, seed=seed)
        rng_events = random.Random(seed + 1)
        t0 = time.perf_counter()
        touches_naive = 0
        for _ in range(ticks):
            event_ids = rng_events.sample(list(atoms_naive.keys()), n_events)
            tick_events = {i: rng_events.uniform(-1.0, 1.0) for i in event_ids}
            touches_naive += run_naive_tick(atoms_naive, tick_events, heavy_workload=heavy_flag)
        t_naive = (time.perf_counter() - t0) * 1000.0 / ticks  # ms per tick
        
        # Event-driven Run
        atoms_evt = build_world(n_atoms, avg_degree, seed=seed)
        rng_events2 = random.Random(seed + 1)
        t0 = time.perf_counter()
        touches_evt = 0
        for _ in range(ticks):
            event_ids = rng_events2.sample(list(atoms_evt.keys()), n_events)
            tick_events = {i: rng_events2.uniform(-1.0, 1.0) for i in event_ids}
            touches_evt += run_event_driven_tick(atoms_evt, tick_events, heavy_workload=heavy_flag)
        t_evt = (time.perf_counter() - t0) * 1000.0 / ticks  # ms per tick
        
        ratio = touches_naive / touches_evt if touches_evt > 0 else 0
        speedup = t_naive / t_evt if t_evt > 0 else 0
        
        print(f"{workload_name:<10} | {t_naive:>13.3f} ms | {t_evt:>13.3f} ms | {ratio:>11.2f}x | {speedup:>18.2f}x")

if __name__ == "__main__":
    run_adversarial_experiment()
    run_workload_complexity_experiment()
