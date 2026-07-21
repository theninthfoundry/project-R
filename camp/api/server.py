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
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        async with self._lock:
            targets = list(self.active_connections)
        for connection in targets:
            try:
                await connection.send_json(message)
            except Exception:
                await self.disconnect(connection)

manager = ConnectionManager()

# Background task to broadcast system health and updates
async def self_model_broadcaster():
    while True:
        try:
            agents_list = list(watcher.interactions.values())
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
                "agents": watcher.get_all_states(),
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
        agents_list = list(watcher.interactions.values())
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
    import time
    agent = watcher.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    # Return snapshot + history details mapped from evidence logs
    history_list = []
    for ev in agent.evidence:
        history_list.append({
            "x": ev.data[0],
            "belief": ev.data[1],
            "timestamp": ev.timestamp
        })

    snapshot_data = {
        "uid": agent.uid,
        "x": list(agent.z_a),
        "belief": list(agent.z_b),
        "surprise": agent.meta.get("surprise", 0.0),
        "energy": agent.capacity,
        "tau": list(agent.precision),
        "timestamp": time.time(),
        "version": len(agent.evidence)
    }

    return {
        "snapshot": snapshot_data,
        "history": history_list,
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
        await manager.disconnect(websocket)

# --- Aether Field Kernel Integration ---
from RealityOS.kernel.field_kernel import AetherFieldKernel
from pydantic import BaseModel

field_kernel = AetherFieldKernel(size=50)

class FieldObserveRequest(BaseModel):
    x: int
    y: int
    value: float

class FieldObstacleRequest(BaseModel):
    x: int
    y: int
    radius: float

class FieldSourceRequest(BaseModel):
    x: int
    y: int
    value: float

class DiffuseRequest(BaseModel):
    damping: float = 0.02
    wave_speed: float = 0.4

class StabilizeRequest(BaseModel):
    iterations: int = 15

class DecayRequest(BaseModel):
    rate: float = 0.01

@app.get("/api/field/state")
async def get_field_state():
    return {
        "size": field_kernel.size,
        "u": field_kernel.u,
        "conductance": field_kernel.conductance,
        "sources": [{"x": k[0], "y": k[1], "value": v} for k, v in field_kernel.sources.items()]
    }

@app.post("/api/field/observe")
async def post_field_observe(req: FieldObserveRequest):
    field_kernel.observe(req.x, req.y, req.value)
    return {"status": "success"}

@app.post("/api/field/obstacle")
async def post_field_obstacle(req: FieldObstacleRequest):
    field_kernel.add_obstacle(req.x, req.y, req.radius)
    return {"status": "success"}

@app.post("/api/field/source")
async def post_field_source(req: FieldSourceRequest):
    field_kernel.set_source(req.x, req.y, req.value)
    return {"status": "success"}

@app.post("/api/field/diffuse")
async def post_field_diffuse(req: DiffuseRequest):
    field_kernel.diffuse(damping=req.damping, wave_speed=req.wave_speed)
    return {"status": "success"}

@app.post("/api/field/stabilize")
async def post_field_stabilize(req: StabilizeRequest):
    field_kernel.stabilize(iterations=req.iterations)
    return {"status": "success"}

@app.post("/api/field/decay")
async def post_field_decay(req: DecayRequest):
    field_kernel.decay(rate=req.rate)
    return {"status": "success"}

@app.get("/api/field/measure")
async def get_field_measure():
    return field_kernel.measure()

@app.post("/api/field/reset")
async def post_field_reset():
    field_kernel.u = [[0.0] * field_kernel.size for _ in range(field_kernel.size)]
    field_kernel.v = [[0.0] * field_kernel.size for _ in range(field_kernel.size)]
    field_kernel.clear_sources()
    field_kernel.clear_obstacles()
    field_kernel.info_gain = 0.0
    field_kernel.compute_spent = 0.0
    return {"status": "success"}

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
