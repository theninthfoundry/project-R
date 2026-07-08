// app.js — Frontend WebSocket & Render Controller for CAMP

let selectedAgentId = null;
let currentAgentsData = {};
let currentSystemData = {};

// 1. WebSocket Setup
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUri = `${protocol}//${window.location.host}/api/ws`;
let socket = new WebSocket(wsUri);

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === "system_tick") {
        updateSystemStats(data.system);
        updateAgentsList(data.agents);
        updateAlertsFeed(data.alerts);
        
        // Refresh details for selected agent
        if (selectedAgentId && currentAgentsData[selectedAgentId]) {
            renderAgentDetails(selectedAgentId);
        }
    } else if (data.type === "observation") {
        // Direct event observation update
        const id = data.agent_id;
        if (selectedAgentId === id) {
            fetchSelectedAgentHistory(id);
        }
    }
};

socket.onclose = () => {
    console.log("WebSocket connection closed. Reconnecting...");
    setTimeout(() => {
        socket = new WebSocket(wsUri);
    }, 2000);
};

// 2. Render Functions

function updateSystemStats(system) {
    currentSystemData = system;
    document.getElementById("sys-mem").textContent = `${system.memory_mb.toFixed(1)} MB`;
    
    // Format uptime into readable minutes/seconds
    const seconds = Math.floor(system.uptime);
    const min = Math.floor(seconds / 60);
    const sec = seconds % 60;
    document.getElementById("sys-uptime").textContent = min > 0 ? `${min}m ${sec}s` : `${sec}s`;
    
    const healthEl = document.getElementById("sys-health");
    if (system.system_state.meta && system.system_state.meta.stable === false) {
        healthEl.textContent = "HEAVY LOAD";
        healthEl.style.color = "var(--warning)";
    } else {
        healthEl.textContent = "STABLE";
        healthEl.style.color = "var(--success)";
    }
}

function updateAgentsList(agents) {
    const listEl = document.getElementById("agent-list");
    document.getElementById("agent-count").textContent = `${agents.length} Active`;
    
    if (agents.length === 0) {
        listEl.innerHTML = `<div class="empty-state">No monitored agents active. Start the simulation to feed logs.</div>`;
        return;
    }
    
    let html = "";
    agents.forEach(agent => {
        // Cache data locally
        currentAgentsData[agent.uid] = agent;
        
        // Determine status class based on surprise & values
        let statusClass = "success";
        const surprise = agent.surprise || 0;
        if (surprise > 2.0) {
            statusClass = "error";
        } else if (surprise > 0.8) {
            statusClass = "warning";
        }
        
        const activeClass = selectedAgentId === agent.uid ? "active" : "";
        
        const cost = agent.x && agent.x[0] ? agent.x[0] : 0.0;
        const latency = agent.x && agent.x[1] ? agent.x[1] : 0.0;
        
        html += `
            <div class="agent-card ${activeClass}" onclick="selectAgent('${agent.uid}')">
                <div class="agent-card-header">
                    <span class="agent-name">${agent.uid}</span>
                    <span class="status-badge ${statusClass}"></span>
                </div>
                <div class="agent-card-metrics">
                    <div class="metric-mini">
                        <span>Cost:</span>
                        <span class="metric-mini-value">$${cost.toFixed(3)}</span>
                    </div>
                    <div class="metric-mini">
                        <span>Latency:</span>
                        <span class="metric-mini-value">${Math.round(latency)}ms</span>
                    </div>
                </div>
            </div>
        `;
    });
    listEl.innerHTML = html;
}

function updateAlertsFeed(alerts) {
    const feedEl = document.getElementById("alert-feed");
    document.getElementById("alert-count").textContent = `${alerts.length} Alerts`;
    
    if (alerts.length === 0) {
        feedEl.innerHTML = `<div class="empty-state">No alerts triggered. System healthy.</div>`;
        return;
    }
    
    let html = "";
    alerts.forEach(alert => {
        const resolvedClass = alert.resolved ? "resolved" : "";
        const priorityLabel = alert.resolved ? "RESOLVED" : `Surprise Score: ${alert.priority.toFixed(1)}`;
        
        html += `
            <div class="alert-item ${resolvedClass}">
                <div class="alert-header">
                    <span class="alert-priority">${priorityLabel}</span>
                    <span style="font-weight: 500; font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">${alert.agent_id}</span>
                </div>
                <div class="alert-msg">${alert.message}</div>
                <div class="alert-time">${new Date(alert.timestamp * 1000).toLocaleTimeString()}</div>
            </div>
        `;
    });
    feedEl.innerHTML = html;
}

function selectAgent(agentId) {
    selectedAgentId = agentId;
    
    // Toggle active classes in list manually for instant feedback
    const cards = document.querySelectorAll('.agent-card');
    cards.forEach(card => card.classList.remove('active'));
    
    // Find active card and add class
    const activeCard = Array.from(cards).find(c => c.querySelector('.agent-name').textContent === agentId);
    if (activeCard) activeCard.classList.add('active');

    fetchSelectedAgentHistory(agentId);
    renderAgentDetails(agentId);
}

function renderAgentDetails(agentId) {
    const agent = currentAgentsData[agentId];
    if (!agent) return;
    
    document.getElementById("active-agent-title").textContent = `Agent Telemetry: ${agent.uid}`;
    
    // Energy budget status
    const energy = agent.energy !== undefined ? agent.energy : 1.0;
    const energyEl = document.getElementById("active-agent-energy");
    energyEl.textContent = `Energy: ${Math.round(energy * 100)}%`;
    energyEl.style.color = energy > 0.4 ? "var(--success)" : "var(--error)";

    // Raw metrics (x)
    const cost = agent.x && agent.x[0] !== undefined ? agent.x[0] : 0.0;
    const latency = agent.x && agent.x[1] !== undefined ? agent.x[1] : 0.0;
    const error = agent.x && agent.x[2] !== undefined ? agent.x[2] : 0.0;
    const tokens = agent.x && agent.x[3] !== undefined ? agent.x[3] : 0.0;
    
    document.getElementById("metric-cost").textContent = `$${cost.toFixed(3)}`;
    document.getElementById("metric-latency").textContent = `${Math.round(latency)} ms`;
    document.getElementById("metric-error").textContent = error.toFixed(2);
    document.getElementById("metric-tokens").textContent = `${tokens.toFixed(1)}k`;

    // Belief states
    const b_cost = agent.belief && agent.belief[0] !== undefined ? agent.belief[0] : 0.0;
    const b_latency = agent.belief && agent.belief[1] !== undefined ? agent.belief[1] : 0.0;
    const b_error = agent.belief && agent.belief[2] !== undefined ? agent.belief[2] : 0.0;
    const b_tokens = agent.belief && agent.belief[3] !== undefined ? agent.belief[3] : 0.0;

    document.getElementById("belief-cost").textContent = `Belief: $${b_cost.toFixed(3)}`;
    document.getElementById("belief-latency").textContent = `Belief: ${Math.round(b_latency)} ms`;
    document.getElementById("belief-error").textContent = `Belief: ${b_error.toFixed(2)}`;
    document.getElementById("belief-tokens").textContent = `Belief: ${b_tokens.toFixed(1)}k`;

    // Surprise & Confusion
    const surprise = agent.surprise !== undefined ? agent.surprise : 0.0;
    document.getElementById("surprise-value").textContent = surprise.toFixed(4);
    
    const confEl = document.getElementById("confusion-flag");
    // We check custom metadata if available, else derive from surprise
    if (surprise > 2.0) {
        confEl.textContent = "SURPRISE SPIKE";
        confEl.style.background = "rgba(239, 68, 68, 0.15)";
        confEl.style.color = "var(--error)";
    } else if (surprise > 0.8) {
        confEl.textContent = "UNSTABLE PROFILE";
        confEl.style.background = "rgba(245, 158, 11, 0.15)";
        confEl.style.color = "var(--warning)";
    } else {
        confEl.textContent = "STABLE PROFILE";
        confEl.style.background = "rgba(16, 185, 129, 0.15)";
        confEl.style.color = "var(--success)";
    }
}

async function fetchSelectedAgentHistory(agentId) {
    try {
        const res = await fetch(`/api/agents/${agentId}`);
        const data = await res.json();
        
        // Render policy constraints
        renderConstraints(data.meta.constraints || []);
        
        // Render history chart
        renderHistoryChart(data.history);
    } catch (e) {
        console.error("Failed to load agent history details:", e);
    }
}

function renderConstraints(constraints) {
    const listEl = document.getElementById("constraint-list");
    if (!constraints || constraints.length === 0) {
        listEl.innerHTML = "<li>No policies defined for this agent.</li>";
        return;
    }
    
    let html = "";
    constraints.forEach(c => {
        const metricName = c.name.replace("_limit", "");
        html += `<li><strong>${metricName}</strong> Limit: &le; ${c.upper.toFixed(2)}</li>`;
    });
    listEl.innerHTML = html;
}

function renderHistoryChart(history) {
    const chartEl = document.getElementById("history-chart");
    if (!history || history.length === 0) {
        chartEl.innerHTML = `<div class="empty-state" style="position: absolute;">No historical timeline data. Select an active agent.</div>`;
        return;
    }
    
    // Render the last 30 historical steps
    const maxSteps = 30;
    const steps = history.slice(-maxSteps);
    
    // Find max value in history to normalize heights
    let maxCost = 0.001;
    steps.forEach(step => {
        const rawCost = step.x && step.x[0] ? step.x[0] : 0.0;
        const beliefCost = step.belief && step.belief[0] ? step.belief[0] : 0.0;
        maxCost = Math.max(maxCost, rawCost, beliefCost);
    });

    let html = "";
    steps.forEach(step => {
        const rawVal = step.x && step.x[0] ? step.x[0] : 0.0;
        const beliefVal = step.belief && step.belief[0] ? step.belief[0] : 0.0;
        
        // Calculate heights in percentages
        const rawHeight = Math.max(5, (rawVal / maxCost) * 100);
        const beliefHeight = Math.max(5, (beliefVal / maxCost) * 100);
        
        html += `
            <div class="chart-bar-container">
                <div class="chart-bar-raw" style="height: ${rawHeight}%;"></div>
                <div class="chart-bar-belief" style="height: ${beliefHeight}%;"></div>
            </div>
        `;
    });
    chartEl.innerHTML = html;
}
