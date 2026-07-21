// field_sandbox.js — Frontend script for the Aether Field Relaxation Sandbox

const canvas = document.getElementById('field-canvas');
const ctx = canvas.getContext('2d');
const size = 50; // Grid size
const cellPixel = canvas.width / size; // Pixel size per grid cell

// App State
let interactionMode = 'observe'; // 'observe' | 'source' | 'sink' | 'obstacle'
let solveMode = 'diffuse';      // 'diffuse' | 'stabilize'
let isMouseDown = false;
let terminalLogs = document.getElementById('terminal-logs');

// Controls
const dampingSlider = document.getElementById('param-damping');
const dampingVal = document.getElementById('val-damping');
const speedSlider = document.getElementById('param-speed');
const speedVal = document.getElementById('val-speed');

// Buttons
const btnObserve = document.getElementById('mode-observe');
const btnSource = document.getElementById('mode-source');
const btnSink = document.getElementById('mode-sink');
const btnObstacle = document.getElementById('mode-obstacle');
const btnDiffuse = document.getElementById('btn-diffuse');
const btnStabilize = document.getElementById('btn-stabilize');
const btnReset = document.getElementById('btn-reset');

// Thermodynamic UI Elements
const statPotential = document.getElementById('stat-potential');
const statEntropy = document.getElementById('stat-entropy');
const statMetabolism = document.getElementById('stat-metabolism');
const statCompute = document.getElementById('stat-compute');

// --- Helper: Terminal Logging ---
function logAPI(command, type = 'command') {
    const div = document.createElement('div');
    div.className = `log-entry ${type}`;
    div.textContent = command;
    terminalLogs.appendChild(div);
    terminalLogs.scrollTop = terminalLogs.scrollHeight;
    
    // Cap logs at 100 entries
    if (terminalLogs.children.length > 100) {
        terminalLogs.removeChild(terminalLogs.firstChild);
    }
}

// --- Interaction Mode Updates ---
function setInteractionMode(mode, btn) {
    interactionMode = mode;
    [btnObserve, btnSource, btnSink, btnObstacle].forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    logAPI(`aether.set_interaction_mode("${mode}")`, 'system');
}

btnObserve.addEventListener('click', () => setInteractionMode('observe', btnObserve));
btnSource.addEventListener('click', () => setInteractionMode('source', btnSource));
btnSink.addEventListener('click', () => setInteractionMode('sink', btnSink));
btnObstacle.addEventListener('click', () => setInteractionMode('obstacle', btnObstacle));

// --- Solver Mode Updates ---
btnDiffuse.addEventListener('click', () => {
    solveMode = 'diffuse';
    btnDiffuse.classList.add('active');
    btnStabilize.classList.remove('active');
    logAPI('aether.set_dynamics("wave_diffusion")', 'system');
});

btnStabilize.addEventListener('click', () => {
    solveMode = 'stabilize';
    btnStabilize.classList.add('active');
    btnDiffuse.classList.remove('active');
    logAPI('aether.set_dynamics("laplace_relaxation")', 'system');
});

// --- Parameter updates ---
dampingSlider.addEventListener('input', (e) => {
    dampingVal.textContent = parseFloat(e.target.value).toFixed(3);
});
speedSlider.addEventListener('input', (e) => {
    speedVal.textContent = parseFloat(e.target.value).toFixed(3);
});

// --- API Calls ---
async function fetchState() {
    try {
        const res = await fetch('/api/field/state');
        return await res.json();
    } catch (e) {
        console.error("Error fetching state: ", e);
        return null;
    }
}

async function sendObserve(x, y, value) {
    await fetch('/api/field/observe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x, y, value })
    });
    logAPI(`aether.observe(${x}, ${y}, ${value.toFixed(2)})`);
}

async function sendObstacle(x, y, radius) {
    await fetch('/api/field/obstacle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x, y, radius })
    });
    logAPI(`aether.add_obstacle(${x}, ${y}, ${radius.toFixed(1)})`);
}

async function sendSource(x, y, value) {
    await fetch('/api/field/source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x, y, value })
    });
    logAPI(`aether.set_source(${x}, ${y}, ${value.toFixed(1)})`);
}

async function stepSimulation() {
    if (solveMode === 'diffuse') {
        const damping = parseFloat(dampingSlider.value);
        const waveSpeed = parseFloat(speedSlider.value);
        await fetch('/api/field/diffuse', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ damping, wave_speed: waveSpeed })
        });
    } else {
        await fetch('/api/field/stabilize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ iterations: 15 })
        });
    }
    
    // Post-decay step
    await fetch('/api/field/decay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rate: 0.015 })
    });
}

async function fetchMetrics() {
    try {
        const res = await fetch('/api/field/measure');
        const metrics = await res.json();
        
        statPotential.textContent = metrics.potential_energy.toFixed(4);
        statEntropy.textContent = metrics.entropy.toFixed(4);
        statMetabolism.textContent = (metrics.metabolism * 1000).toFixed(4);
        statCompute.textContent = `${(metrics.energy_spent / 1000).toFixed(1)}k FLOPs`;
    } catch (e) {
        console.error("Error fetching metrics: ", e);
    }
}

// Reset Field
btnReset.addEventListener('click', async () => {
    await fetch('/api/field/reset', { method: 'POST' });
    logAPI('aether.clear_disturbance_field()', 'danger');
});

// --- Mouse Canvas Event Handlers ---
function getGridCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / cellPixel);
    const y = Math.floor((e.clientY - rect.top) / cellPixel);
    return { x: Math.max(0, Math.min(size - 1, x)), y: Math.max(0, Math.min(size - 1, y)) };
}

canvas.addEventListener('mousedown', (e) => {
    isMouseDown = true;
    handleInteraction(e);
});

canvas.addEventListener('mousemove', (e) => {
    if (isMouseDown) {
        handleInteraction(e);
    }
});

window.addEventListener('mouseup', () => {
    isMouseDown = false;
});

function handleInteraction(e) {
    const { x, y } = getGridCoords(e);
    if (interactionMode === 'observe') {
        sendObserve(x, y, 4.0);
    } else if (interactionMode === 'obstacle') {
        sendObstacle(x, y, 2.0);
    } else if (interactionMode === 'source') {
        sendSource(x, y, 3.0);
    } else if (interactionMode === 'sink') {
        sendSource(x, y, -3.0);
    }
}

// --- Canvas Rendering Loop ---
function drawField(state) {
    if (!state) return;
    const { u, conductance, sources } = state;
    
    // Clear canvas
    ctx.fillStyle = '#090d16'; // --bg-deep
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    for (let x = 0; x < size; x++) {
        for (let y = 0; y < size; y++) {
            const val = u[x][y];
            const cond = conductance[x][y];
            
            let r = 9, g = 13, b = 22; // default
            
            if (cond === 0.0) {
                // Obstacle - Solid slate gray
                r = 30; g = 41; b = 59;
            } else {
                if (val > 0.0) {
                    // Cyan/Blue glow for positive pressure waves
                    const intensity = Math.min(1.0, val);
                    r = Math.floor(9 + intensity * 50);
                    g = Math.floor(13 + intensity * 180);
                    b = Math.floor(22 + intensity * 240);
                } else if (val < 0.0) {
                    // Magenta/Rose glow for negative pressure sinks
                    const intensity = Math.min(1.0, -val);
                    r = Math.floor(9 + intensity * 220);
                    g = Math.floor(13 + intensity * 50);
                    b = Math.floor(22 + intensity * 180);
                }
            }
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(x * cellPixel, y * cellPixel, cellPixel - 0.5, cellPixel - 0.5);
        }
    }
    
    // Draw source/sink locator rings
    sources.forEach(src => {
        ctx.strokeStyle = src.value > 0 ? '#06b6d4' : '#d946ef';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc((src.x + 0.5) * cellPixel, (src.y + 0.5) * cellPixel, cellPixel * 1.5, 0, 2 * Math.PI);
        ctx.stroke();
    });
}

// --- Main Loop (30 FPS) ---
async function runLoop() {
    // 1. Step the simulation
    await stepSimulation();
    
    // 2. Fetch the state and draw
    const state = await fetchState();
    drawField(state);
    
    // 3. Fetch metrics
    await fetchMetrics();
    
    setTimeout(runLoop, 33); // Run at ~30 FPS
}

// Start
runLoop();
