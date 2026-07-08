"""
models.py — Pydantic serialization models for CAMP API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class AgentRegisterRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the agent")
    value: float = Field(0.5, ge=0.0, le=1.0, description="Priority / value of the agent")
    cost_limit: float = Field(1.0, description="Max allowed cost per call ($)")
    latency_limit: float = Field(5000.0, description="Max allowed latency (ms)")
    error_limit: float = Field(0.2, description="Max allowed error rate (0.0 to 1.0)")
    token_limit: float = Field(20.0, description="Max allowed tokens per call (thousands)")


class AgentObserveRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the agent")
    cost: float = Field(0.0, ge=0.0, description="Cost of the API call ($)")
    latency: float = Field(0.0, ge=0.0, description="Latency of the call (ms)")
    error: float = Field(0.0, ge=0.0, le=1.0, description="Error status (0.0=success, 1.0=error)")
    tokens: float = Field(0.0, ge=0.0, description="Tokens consumed (in thousands)")


class SystemStatusResponse(BaseModel):
    uptime: float
    memory_mb: float
    queue_depth: int
    ticks_per_sec: float
    alerts_raised: int
    system_state: Dict[str, Any]
