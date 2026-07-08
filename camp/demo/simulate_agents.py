"""
simulate_agents.py — End-to-end agent fleet simulation for CAMP.

Simulates 10 active agents over 30 ticks, generating noisy traffic.
Compares a Naive instant-threshold alert engine against CAMP's
Belief Momentum + Priority-scheduled alert engine.

Outputs the side-by-side comparison to the terminal and logs raw
events to 'agent_logs.jsonl' for streaming to the web dashboard.
"""

import time
import random
import json
from typing import Dict, Any, List
from camp.core.agent_watcher import AgentWatcher
from camp.core.alert_engine import AlertEngine


def run_simulation(ticks: int = 40, log_file: str = "agent_logs.jsonl"):
    print("==========================================================")
    print("      CAMP: COGNITIVE AGENT MONITORING SIMULATOR          ")
    print("==========================================================")
    print(f"Simulating 10 agents over {ticks} ticks...")
    print(f"Writing raw events to: {log_file}")
    
    # Initialize CAMP engines
    watcher = AgentWatcher(alpha=0.15)  # alpha=0.15 belief momentum
    alert_engine = AlertEngine(alert_budget_per_tick=3) # restrict budget to 3 alerts/tick

    # Register Agents in CAMP
    # 7 Healthy agents (low value)
    for i in range(1, 8):
        watcher.register_agent(f"qna_helper_{i}", value=0.3)
        
    # 1 Cost-spiraling agent (medium value)
    watcher.register_agent("translation_v2", value=0.6, cost_limit=0.5)
    
    # 1 Flapping/Noisy agent (low value)
    watcher.register_agent("code_writer", value=0.4, error_limit=0.1)
    
    # 1 Mission-critical gateway agent (high value)
    watcher.register_agent("payment_cortex", value=1.0, latency_limit=4000.0)

    # In-memory track for Naive Alerting counts
    naive_alert_count = 0
    camp_alert_count = 0
    
    # Track flapping alerts specifically
    naive_flapping_alerts = 0
    camp_flapping_alerts = 0

    rng = random.Random(42)

    # Empty log file
    with open(log_file, "w") as f:
        pass

    print("\nStarting Simulation Loop...")
    print(f"{'Tick':<6} | {'Event Src':<16} | {'Metric (Raw)':<15} | {'Naive Alert?':<12} | {'CAMP Belief':<15} | {'CAMP Surprise':<13}")
    print("-" * 88)

    for tick in range(1, ticks + 1):
        # Gather all observations for this tick
        tick_observations = []

        # 1. 7 Healthy agents: standard noisy traffic
        for i in range(1, 8):
            tick_observations.append({
                "agent_id": f"qna_helper_{i}",
                "cost": rng.uniform(0.002, 0.008),
                "latency": rng.uniform(120.0, 350.0),
                "error": 0.0,
                "tokens": rng.uniform(0.8, 1.8)
            })

        # 2. Cost-spiraling agent: cost rises steadily every tick
        cost_trend = 0.02 + (tick * 0.06)  # Exceeds cost_limit (0.5) around tick 9
        tick_observations.append({
            "agent_id": "translation_v2",
            "cost": cost_trend + rng.uniform(-0.01, 0.01),
            "latency": rng.uniform(400.0, 800.0),
            "error": 0.0,
            "tokens": rng.uniform(2.0, 4.0)
        })

        # 3. Flapping agent: error status flips 0.0 and 1.0 randomly
        # Naive engine should alert immediately on every 1.0, causing alert fatigue.
        error_status = 1.0 if (tick % 3 == 0) else 0.0
        tick_observations.append({
            "agent_id": "code_writer",
            "cost": 0.01,
            "latency": 300.0,
            "error": error_status,
            "tokens": 1.0
        })

        # 4. Mission-critical gateway agent: healthy until a major latency spike at tick 25
        latency_val = rng.uniform(1500.0, 2200.0)
        if tick >= 25:
            latency_val = 8500.0 + rng.uniform(-100.0, 100.0) # Exceeds limit (4000)
            
        tick_observations.append({
            "agent_id": "payment_cortex",
            "cost": 0.05,
            "latency": latency_val,
            "error": 0.0,
            "tokens": 5.0
        })

        # Write to log file and evaluate alerts
        with open(log_file, "a") as f:
            for obs in tick_observations:
                f.write(json.dumps(obs) + "\n")

        # Process observations through CAMP
        for obs in tick_observations:
            agent_id = obs["agent_id"]
            
            # --- Evaluate Naive Alerts (Instant thresholds) ---
            agent_config = watcher.agents[agent_id]
            is_naive_alert = False
            
            # Check Cost Limit
            if obs["cost"] > agent_config.constraints[0].upper: is_naive_alert = True
            # Check Latency Limit
            if obs["latency"] > agent_config.constraints[1].upper: is_naive_alert = True
            # Check Error Limit
            if obs["error"] > agent_config.constraints[2].upper: is_naive_alert = True
            # Check Token Limit
            if obs["tokens"] > agent_config.constraints[3].upper: is_naive_alert = True
            
            if is_naive_alert:
                naive_alert_count += 1
                if agent_id == "code_writer":
                    naive_flapping_alerts += 1

            # --- Evaluate CAMP Alerts ---
            telemetry = watcher.observe(
                agent_id=agent_id,
                cost=obs["cost"],
                latency=obs["latency"],
                error=obs["error"],
                tokens=obs["tokens"]
            )
            
            # Log key telemetry for interesting agents (spiraling, flapping, or critical latency spike)
            if agent_id in ["translation_v2", "code_writer", "payment_cortex"] and tick % 4 == 0:
                naive_flag = "ALERT!" if is_naive_alert else "ok"
                
                # Check specific metric belief to display
                if agent_id == "translation_v2":
                    val_str = f"${obs['cost']:.2f} cost"
                    belief_str = f"${telemetry['belief'][0]:.2f}"
                elif agent_id == "code_writer":
                    val_str = f"{obs['error']:.1f} error"
                    belief_str = f"{telemetry['belief'][2]:.2f}"
                else:
                    val_str = f"{int(obs['latency'])}ms lat"
                    belief_str = f"{int(telemetry['belief'][1])}ms"
                    
                print(f"Tick {tick:02d} | {agent_id:<16} | {val_str:<15} | {naive_flag:<12} | {belief_str:<15} | {telemetry['surprise']:.4f}")

        # Tick Alert Engine to process state and register alerts
        active_agents = list(watcher.agents.values())
        tick_camp_alerts = alert_engine.process_tick_alerts(active_agents)
        camp_alert_count += len(tick_camp_alerts)
        
        for alert in tick_camp_alerts:
            if alert.agent_id == "code_writer":
                camp_flapping_alerts += 1

    print("-" * 88)
    print("\n==========================================================")
    print("              SIMULATION ANALYSIS & RESULTS               ")
    print("==========================================================")
    print(f"Total Events Processed       : {watcher.processed_event_count}")
    print()
    print(f"NAIVE ALERTING (Instant Thresholds):")
    print(f" ├─ Total Alerts Fired      : {naive_alert_count}")
    print(f" └─ Noisy Flapping Alerts   : {naive_flapping_alerts} (Alert Fatigue)")
    print()
    print(f"CAMP COGNITIVE ALERTING (Momentum-Damped & Debounced):")
    print(f" ├─ Total Alerts Fired      : {camp_alert_count}")
    print(f" └─ Noisy Flapping Alerts   : {camp_flapping_alerts} (Debounced and filtered)")
    print()
    
    # Calculate noise reduction
    fatigue_reduction = (1 - (camp_flapping_alerts / max(1, naive_flapping_alerts))) * 100
    print(f"CAMP Alert Noise Reduction: {fatigue_reduction:.1f}%")
    print("==========================================================")


if __name__ == "__main__":
    run_simulation()
