# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pandas as pd

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="DroneSafe", layout="wide")

# ------------------------- CSS / DARK THEME -------------------------
st.markdown("""
<style>
/* General App Background */
[data-testid="stAppViewContainer"] {
    background-color: #0b0b0b !important;
    color: #ffffff !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1f2937 !important;
    color: #ffffff !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 700;
}

/* Markdown / KPI / Labels */
div[data-testid="stMarkdownContainer"] {
    color: #ffffff !important;
    font-weight: 600;
}

/* Sliders and Number Inputs */
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stCheckbox"] label {
    color: #ffffff !important;
    font-weight: 600;
}
div[data-testid="stSlider"] input[type="range"] {
    accent-color: #2563eb !important;
}
input[type="number"] {
    color: #000 !important;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #1e40af !important;
}

/* Alerts */
.stWarning, .stWarning div {
    background-color: #ff4444 !important;
    color: #ffffff !important;
    border-left: 5px solid #ff0000 !important;
}
.stSuccess, .stSuccess div {
    background-color: #22c55e !important;
    color: #ffffff !important;
    border-left: 5px solid #16a34a !important;
}
.stInfo, .stInfo div {
    background-color: #3b82f6 !important;
    color: #ffffff !important;
    border-left: 5px solid #1d4ed8 !important;
}

/* Hero Section */
.hero { display: grid; grid-template-columns: 1.3fr 1fr; align-items: center; padding: 60px 20px; gap: 40px; }
.tag { background: #2563eb; color: white; font-size: 14px; padding: 6px 14px; border-radius: 20px; display: inline-block; margin-bottom: 12px; font-weight: 500; }
.title { font-size: 48px; font-weight: 800; color: white; margin-bottom: 12px; }
.subtitle { font-size: 18px; color: #e5e7eb; margin-bottom: 24px; line-height: 1.5; }
.panel { background: #065f46; padding: 28px; border-radius: 18px; margin-bottom: 20px; font-size: 20px; font-weight: 600; color: #ffffff; }
.panel-large { background: #1e3a8a; padding: 32px; border-radius: 18px; font-size: 26px; font-weight: 700; color: #ffffff; text-align: center; }

/* Stats Section */
.stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; padding: 40px 20px; }
.stat-card { background: #1f2937; border-radius: 18px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); text-align: center; color: #ffffff; }
.stat-value { font-size: 32px; font-weight: 700; color: #2563eb; margin-bottom: 8px; }
.stat-label { font-size: 16px; color: #d1d5db; }

/* Features Section */
.feature-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; padding: 40px 20px; }
.feature-card { background: #1f2937; border-radius: 18px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); text-align: center; font-size: 16px; color: #ffffff; font-weight: 500; transition: transform 0.2s ease; }
.feature-card:hover { transform: translateY(-4px); background-color: #2563eb; }

/* Footer */
.footer { display: grid; grid-template-columns: repeat(3, 1fr); gap: 40px; padding: 40px 20px; background: #111827; color: #d1d5db; margin-top: 40px; }
.footer h4 { font-size: 18px; font-weight: 600; margin-bottom: 12px; color: #ffffff; }
.footer a { display: block; margin-bottom: 8px; font-size: 14px; color: #9ca3af; text-decoration: none; }
.footer a:hover { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ------------------------- HERO -------------------------
st.markdown("""
<div class="hero">
    <div>
        <div class="tag">AI-Powered Collision Prediction</div>
        <div class="title">DroneSafe</div>
        <div class="subtitle">
            Revolutionizing aerial operations with real-time collision detection 
            and AI-powered predictive safety technology.
        </div>
    </div>
    <div>
        <div class="panel">✔ Real-Time Processing</div>
        <div class="panel-large">AI — Powered Intelligence</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------- STATS -------------------------
st.markdown("""
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">99.9%</div>
        <div class="stat-label">Safety Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">+200</div>
        <div class="stat-label">Successful Flights</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------- FEATURES -------------------------
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">Collision Prediction</div>
    <div class="feature-card">Real-Time Alerts</div>
    <div class="feature-card">Flight Data Logs</div>
    <div class="feature-card">AI Route Optimization</div>
</div>
""", unsafe_allow_html=True)

# ------------------------- SESSION STATE INIT -------------------------
if "pos" not in st.session_state: st.session_state["pos"] = None
if "vel" not in st.session_state: st.session_state["vel"] = None
if "running" not in st.session_state: st.session_state["running"] = False

# ------------------------- DRONE SIM FUNCTIONS -------------------------
def init_drones(n, area_size, speed_mean, speed_std, seed=None):
    rnd = np.random.default_rng(seed)
    pos = rnd.uniform(0, area_size, size=(n, 2))
    angles = rnd.uniform(0, 2 * math.pi, size=n)
    speeds = np.clip(rnd.normal(speed_mean, speed_std, size=n), 0.0, None)
    vel = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1)
    return pos, vel

def predict_positions(pos, vel, horizon_s, dt):
    steps = int(math.ceil(horizon_s / dt))
    preds = np.zeros((steps + 1, pos.shape[0], 2))
    for t in range(steps + 1):
        preds[t] = pos + vel * (t * dt)
    times = np.arange(0, (steps + 1) * dt, dt)
    return preds, times

def detect_collisions(preds, times, safe_dist):
    collisions = []
    T, n, _ = preds.shape
    for s in range(T):
        positions = preds[s]
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < safe_dist:
                    collisions.append({"pair": (i, j), "time": float(times[s]), "step": s, "dist": float(d)})
    earliest = {}
    for c in collisions:
        p = c["pair"]
        if p not in earliest or c["time"] < earliest[p]["time"]:
            earliest[p] = c
    return list(earliest.values())

def step_simulation(pos, vel, dt, area_size, bounce=True):
    pos = pos + vel * dt
    if bounce:
        for i in range(pos.shape[0]):
            for k in (0,1):
                if pos[i,k] < 0:
                    pos[i,k] = -pos[i,k]
                    vel[i,k] = -vel[i,k]
                elif pos[i,k] > area_size:
                    pos[i,k] = 2*area_size - pos[i,k]
                    vel[i,k] = -vel[i,k]
    return pos, vel

# ------------------------- SIMULATION CONTROLS -------------------------
st.markdown("##  Live Drone Collision Simulation")

col1, col2 = st.columns([1,1])
with col1:
    n = st.slider("Number of drones", 2, 30, 8)
    area = st.slider("Area size (m)", 50, 2000, 300)
    speed_mean = st.slider("Avg speed (m/s)", 0.1, 40.0, 8.0)
    speed_std = st.slider("Speed std dev", 0.0, 10.0, 2.0)
    safe_dist = st.slider("Safe distance threshold (m)", 1.0, 200.0, 20.0)
    horizon = st.slider("Prediction horizon (s)", 1.0, 30.0, 6.0)
    dt = st.slider("Prediction timestep dt (s)", 0.1, 2.0, 0.5)
    seed_val = st.number_input("Random seed (0 for random)", min_value=0, value=0)
    bounce = st.checkbox("Bounce at boundary", True)
    if st.button("Initialize / Reset"):
        seed = None if seed_val == 0 else int(seed_val)
        st.session_state["pos"], st.session_state["vel"] = init_drones(n, area, speed_mean, speed_std, seed)
        st.session_state["running"] = False
    if st.session_state["pos"] is None:
        st.session_state["pos"], st.session_state["vel"] = init_drones(n, area, speed_mean, speed_std)

with col2:
    start = st.button("Start Simulation")
    stop = st.button("Pause Simulation")
    if start: st.session_state["running"] = True
    if stop: st.session_state["running"] = False

# ------------------------- SIMULATION LOGIC -------------------------
pos, vel = st.session_state["pos"], st.session_state["vel"]

if st.session_state["running"]:
    frames = 10
    for _ in range(frames):
        pos, vel = step_simulation(pos, vel, dt, area, bounce)
        st.session_state["pos"], st.session_state["vel"] = pos, vel
        time.sleep(0.05)

preds, times = predict_positions(pos, vel, horizon, dt)
collisions = detect_collisions(preds, times, safe_dist)

# KPI
st.markdown(f"**Drones:** {n} | **Predicted Collisions:** {len(collisions)} | **Safe Distance:** {safe_dist} m | **Horizon:** {horizon} s")

# Collisions alerts
if collisions:
    for c in collisions:
        i,j = c["pair"]
        st.warning(f"Collision: Drone {i} & Drone {j} in {c['time']:.2f}s (dist {c['dist']:.2f} m)")
else:
    st.success("✅ No collisions predicted")

# ------------------------- PLOT -------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_facecolor("#0b0b0b")
ax.set_xlim(0, area)
ax.set_ylim(0, area)
ax.set_title("Drone Trajectories", color="white")
colors = plt.cm.get_cmap("tab10", n)

for i in range(n):
    traj = preds[:, i, :]
    ax.plot(traj[:,0], traj[:,1], alpha=0.7, color=colors(i))
    ax.scatter(traj[0,0], traj[0,1], s=30, color=colors(i))

for c in collisions:
    i,j = c["pair"]
    s = c["step"]
    ax.scatter(preds[s,i,0], preds[s,i,1], s=80, marker='x', color="red")
    ax.scatter(preds[s,j,0], preds[s,j,1], s=80, marker='x', color="red")

ax.grid(True, alpha=0.3, color="gray")
ax.tick_params(colors='white')
st.pyplot(fig)

# ------------------------- DRONE STATES -------------------------
st.subheader("📋 Drone States (Current)")
rows = [{"Drone": i, "X": float(pos[i,0]), "Y": float(pos[i,1]),
         "Vx": float(vel[i,0]), "Vy": float(vel[i,1]),
         "Speed": math.hypot(vel[i,0], vel[i,1])} for i in range(n)]
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ------------------------- FOOTER -------------------------
st.markdown("""
<div class="footer">
    <div>
        <h4>Product</h4>
        <a href="#">Features</a>
        <a href="#">Pricing</a>
        <a href="#">Demo</a>
    </div>
    <div>
        <h4>Company</h4>
        <a href="#">About</a>
        <a href="#">Careers</a>
        <a href="#">Contact</a>
    </div>
    <div>
        <h4>Resources</h4>
        <a href="#">Docs</a>
        <a href="#">Blog</a>
        <a href="#">Support</a>
    </div>
</div>
""", unsafe_allow_html=True)
