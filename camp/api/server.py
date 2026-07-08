"""
server.py — FastAPI API server for CAMP.

Exposes REST and WebSocket interfaces for agent monitoring, alerts,
and system health metrics. Serves the web dashboard statically.
"""

import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from typing import List, Set

from camp.core.agent_watcher import AgentWatcher
from camp.core.alert_engine import AlertEngine
from camp.core.self_model import SelfModel
from camp.api.models import AgentRegisterRequest, AgentObserveRequest, SystemStatusResponse

app = FastAPI(title="CAMP — Cognitive Agent Monitoring Platform")

# Core Engine Instances
watcher = AgentWatcher()
alert_engine = AlertEngine(alert_budget_per_tick=5)
self_model = SelfModel()

# WebSocket Connection Pool
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# Background task to broadcast system health and updates
async def self_model_broadcaster():
    while True:
        try:
            agents_list = list(watcher.agents.values())
            # Gather active alerts
            alert_engine.process_tick_alerts(agents_list)
            
            # Evolve system self-state
            metrics = self_model.get_system_metrics(
                queue_depth=len(manager.active_connections),
                alerts_count=len(alert_engine.alert_history)
            )
            
            payload = {
                "type": "system_tick",
                "system": metrics,
                "agents": [s.snapshot() for s in agents_list],
                "alerts": alert_engine.get_history(10)
            }
            await manager.broadcast(payload)
        except Exception as e:
            print(f"Error in self_model_broadcaster: {e}")
        await asyncio.sleep(1.0)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(self_model_broadcaster())

# --- API Routes ---

@app.post("/api/register")
async def register_agent(req: AgentRegisterRequest):
    try:
        watcher.register_agent(
            agent_id=req.agent_id,
            value=req.value,
            cost_limit=req.cost_limit,
            latency_limit=req.latency_limit,
            error_limit=req.error_limit,
            token_limit=req.token_limit
        )
        return {"status": "success", "message": f"Agent '{req.agent_id}' registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/observe")
async def observe_agent(req: AgentObserveRequest):
    try:
        telemetry = watcher.observe(
            agent_id=req.agent_id,
            cost=req.cost,
            latency=req.latency,
            error=req.error,
            tokens=req.tokens
        )
        
        # Trigger alert processing tick
        agents_list = list(watcher.agents.values())
        alert_engine.process_tick_alerts(agents_list)
        
        # Broadcast observation details to WebSocket
        await manager.broadcast({
            "type": "observation",
            "agent_id": req.agent_id,
            "telemetry": telemetry,
            "alerts": alert_engine.get_history(10)
        })
        return {"status": "success", "telemetry": telemetry}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/agents")
async def get_agents():
    return watcher.get_all_states()

@app.get("/api/agents/{agent_id}")
async def get_agent_details(agent_id: str):
    agent = watcher.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    # Return snapshot + history details
    return {
        "snapshot": agent.snapshot(),
        "history": list(agent.history.buffer),
        "meta": agent.meta
    }

@app.get("/api/alerts")
async def get_alerts(limit: int = 50):
    return alert_engine.get_history(limit)

@app.get("/api/system", response_model=SystemStatusResponse)
async def get_system_status():
    metrics = self_model.get_system_metrics(
        queue_depth=len(manager.active_connections),
        alerts_count=len(alert_engine.alert_history)
    )
    return metrics

# --- WebSocket ---

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, receive client messages if any
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Serving Dashboard Static Files ---

# Redirect root to dashboard page
@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard/index.html")

# Create dashboard folder dynamically if it doesn't exist
dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")
os.makedirs(dashboard_dir, exist_ok=True)

# Mount static folder
app.mount("/dashboard", StaticFiles(directory=dashboard_dir), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
