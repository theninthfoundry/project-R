"""
ingest.py — Ingestion parser for CAMP log files and requests.

Handles the parsing and normalization of incoming log records into
structured observations suitable for the AgentWatcher engine.
"""

import json
from typing import Dict, Any, Optional
from camp.core.agent_watcher import AgentWatcher


class IngestManager:
    def __init__(self, watcher: AgentWatcher):
        self.watcher = watcher

    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parses a single JSON log line and updates the watcher.
        Expected format:
        {
            "agent_id": "translator_v1",
            "cost": 0.012,
            "latency": 850.0,
            "error": 0.0,      # 0 or 1
            "tokens": 1.2
        }
        """
        if not line or not line.strip():
            return None
            
        try:
            data = json.loads(line)
            return self.process_event(data)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
        except Exception as e:
            return {"error": f"Failed to process log line: {str(e)}"}

    def process_event(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize fields and dispatch to watcher."""
        agent_id = data.get("agent_id")
        if not agent_id:
            return {"error": "Missing 'agent_id' field"}

        # Extract metrics with fallbacks
        cost = float(data.get("cost", 0.0))
        latency = float(data.get("latency", 0.0))
        error = float(data.get("error", 0.0))
        tokens = float(data.get("tokens", 0.0))

        # Evolve watcher state
        telemetry = self.watcher.observe(
            agent_id=agent_id,
            cost=cost,
            latency=latency,
            error=error,
            tokens=tokens
        )
        return telemetry
