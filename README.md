# CAMP — Cognitive Agent Monitoring Platform
> **Observability Infrastructure for AI Agent Fleets — Powered by the Cognitive Dynamics Architecture (CDA).**

CAMP is a self-hosted, lightweight watcher for AI agent runtimes. It tracks costs, latencies, error rates, and token volumes across your agent workflows, detecting anomalies (surprise) and surfacing alerts without crying wolf or burying critical metrics.

CAMP is built on top of the **Project S / Cognitive OS** kernel, utilizing its validated sparse event-driven state updates, priority-scheduled resource allocation, and belief momentum filters.

---

## Key Features

*   **Sparse Event-Driven Updates**: Metrics are evaluated only when active events are received, scaling with the change rate of the world rather than total agent size.
*   **Belief Momentum (Debouncing)**: Exponential moving average belief states smooth out noisy transient metric spikes, eliminating alert fatigue from flapping signals.
*   **Priority-Scheduled Alerts**: Value-weighted priority queues guarantee that critical high-value alerts are processed first during system load spikes or API rate-limit bursts.
*   **VERI Safety Guards**: Built-in boundary verification checks state transitions against safety and budget policies before committing.
*   **Self-Monitoring (Self-Model)**: The monitor keeps a Cognitive State representation of its own health, tracking queue latency, memory usage, and stability.

---

## Directory Structure

```
project S/
├── RealityOS/                     # Cognitive Dynamics Architecture Core
│   ├── kernel/
│   │   ├── cognitive_state.py     # CognitiveState primitive (Eq. 1)
│   │   ├── evidence.py            # Evidence trust provenance definitions
│   │   └── evolution.py           # Analytic evolution loop (predict-observe-compare-update)
│   └── fabric/
│       └── cognitive_fabric.py    # Orchestrates updates and ticks
│
└── camp/                          # Agent Monitoring Product Layer
    ├── core/
    │   ├── agent_watcher.py       # Core metrics monitoring engine
    │   ├── alert_engine.py        # Priority-scheduled debounced alert tracker
    │   ├── ingest.py              # Log normalizer and parser
    │   └── self_model.py          # System self-health monitoring
    ├── api/
    │   ├── server.py              # FastAPI REST + WebSocket Server
    │   └── models.py              # Request/Response schemas
    ├── dashboard/
    │   ├── index.html             # Glassmorphic dark dashboard UI
    │   ├── app.js                 # WebSocket stream listener
    │   └── styles.css             # Theme style definition
    ├── cli/
    │   └── watch.py               # CLI runner tool
    └── demo/
        └── simulate_agents.py     # Side-by-side comparison simulator
```

---

## Quickstart Guide

### 1. Run the Console Simulation
Compare Naive Alerting (instant thresholds) against CAMP's Cognitive Alerting under noisy traffic using the built-in simulator:
```bash
python -m camp.demo.simulate_agents
```
This runs a 40-tick simulation of 10 agents, showing how CAMP reduces false alarms by over 90% while catching genuine cost spirals and latency spikes, and logs raw events to `agent_logs.jsonl`.

### 2. Start the API Server & Dashboard
Run the FastAPI web backend:
```bash
python -m camp.api.server
```
The server will start at [http://127.0.0.1:8000](http://127.0.0.1:8000). 
Open your browser and navigate to the live dashboard:
👉 **[http://127.0.0.1:8000/dashboard/index.html](http://127.0.0.1:8000/dashboard/index.html)**

### 3. Stream Logs via the CLI
Tails logs from `agent_logs.jsonl` and streams them directly to the running API server:
```bash
python -m camp.cli.watch watch-file --file agent_logs.jsonl
```

### 4. Query Alerts via CLI
Query current active alerts directly from the command line:
```bash
python -m camp.cli.watch status
```

---

## Run Unit Tests
Validate the core tracking, belief momentum, and trust decay algorithms:
```bash
python -m unittest camp/tests/test_agent_watcher.py -v
```
