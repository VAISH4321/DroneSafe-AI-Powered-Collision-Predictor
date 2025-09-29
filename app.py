# app.py
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pandas as pd

st.set_page_config(layout="wide", page_title="Drone Collision Predictor")

# -------------------------
# THEME / NAVBAR
# -------------------------
st.markdown(
    """
    <style>
        .stApp {background-color: #0b1e39; color: white;}
        h1, h2, h3, h4 {color: #00c0f3;}
        .stAlert {background-color: #112b4a; border-left: 5px solid #00c0f3;}
    </style>
    """,
    unsafe_allow_html=True
)

selected = option_menu(
    menu_title=None,
    options=["Home", "About", "Team"],
    icons=["house", "info-circle", "people-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0b1e39"},
        "icon": {"color": "white", "font-size": "20px"},
        "nav-link": {"font-size": "18px", "text-align": "center", "margin":"0px", "color": "white"},
        "nav-link-selected": {"background-color": "#00c0f3"},
    },
)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
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

# -------------------------
# HOME PAGE (DRONE SIM)
# -------------------------
if selected == "Home":
    st.title("🚁 Drone Collision Predictor — Flight of the Future")
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.header("Simulation Controls")
        n = st.slider("Number of drones", 2, 30, 8)
        area = st.slider("Area size (m)", 50, 2000, 300)
        speed_mean = st.slider("Avg speed (m/s)", 0.1, 40.0, 8.0)
        speed_std = st.slider("Speed std dev", 0.0, 10.0, 2.0)
        safe_dist = st.slider("Safe distance threshold (m)", 1.0, 200.0, 20.0)
        horizon = st.slider("Prediction horizon (s)", 1.0, 30.0, 6.0)
        dt = st.slider("Prediction timestep dt (s)", 0.1, 2.0, 0.5)
        seed_val = st.number_input("Random seed (0 for random)", min_value=0, value=0)
        bounce = st.checkbox("Bounce at boundary", True)
        st.markdown("---")
        if st.button("Initialize / Reset"):
            seed = None if seed_val == 0 else int(seed_val)
            pos0, vel0 = init_drones(n, area, speed_mean, speed_std, seed)
            st.session_state["pos"] = pos0
            st.session_state["vel"] = vel0
            st.session_state["running"] = False
        if "pos" not in st.session_state:
            st.session_state["pos"], st.session_state["vel"] = init_drones(n, area, speed_mean, speed_std, seed=None)
    
    with col2:
        st.header("Run & Demo")
        if "running" not in st.session_state:
            st.session_state["running"] = False
        run = st.button("Start Simulation") or st.session_state["running"]
        stop = st.button("Pause Simulation")
        if stop:
            st.session_state["running"] = False
        if run:
            st.session_state["running"] = True

    # Simulation
    pos = st.session_state["pos"]
    vel = st.session_state["vel"]
    preds, times = predict_positions(pos, vel, horizon, dt)
    collisions = detect_collisions(preds, times, safe_dist)

    st.subheader("Predicted Collisions")
    if collisions:
        for c in collisions:
            i,j = c["pair"]
            st.warning(f"Collision: Drone {i} & Drone {j} in {c['time']:.2f}s (dist {c['dist']:.2f} m)")
    else:
        st.success("No collisions predicted")

    # Plot
    fig, ax = plt.subplots(figsize=(2,2), facecolor="#0b1e39")
    ax.set_facecolor("#0b1e39")
    ax.set_xlim(0, area)
    ax.set_ylim(0, area)
    ax.set_title("Drone positions & predicted trajectories", color="#00c0f3")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    T, n_d, _ = preds.shape
    for i in range(n_d):
        traj = preds[:, i, :]
        ax.plot(traj[:,0], traj[:,1], linewidth=1, alpha=0.6)
        ax.scatter(traj[0,0], traj[0,1], s=30)
    for c in collisions:
        i,j = c["pair"]
        s = c["step"]
        ax.scatter(preds[s,i,0], preds[s,i,1], s=60, marker='x', color="red")
        ax.scatter(preds[s,j,0], preds[s,j,1], s=60, marker='x', color="red")
    ax.grid(True)
    st.pyplot(fig)

    # Advance simulation if running
    if st.session_state["running"]:
        frames = 10
        for _ in range(frames):
            pos, vel = step_simulation(pos, vel, dt, area, bounce)
            st.session_state["pos"] = pos
            st.session_state["vel"] = vel
            preds, times = predict_positions(pos, vel, horizon, dt)
            collisions = detect_collisions(preds, times, safe_dist)
            time.sleep(0.05)
        st.rerun()
    else:
        st.subheader("Drone States (current)")
        rows = []
        for i in range(pos.shape[0]):
            x,y = pos[i]
            vx,vy = vel[i]
            speed = math.hypot(vx,vy)
            rows.append({"Drone": i, "X": float(x), "Y": float(y), "Vx": float(vx), "Vy": float(vy), "Speed": float(speed)})
        df = pd.DataFrame(rows)
        st.dataframe(df)

    st.markdown("---")
    st.caption("Notes: Uses constant-velocity model. For real-world, integrate sensor noise, Kalman filters, LSTM, 3D, and secure telemetry.")

# -------------------------
# ABOUT PAGE
# -------------------------
elif selected == "About":
    st.title("ℹ️ About This App")
    st.markdown(
        """
        ### What is this?
        This project is part of the **Thales GenTech India Hackathon 2025**.
        It demonstrates an AI-powered **Drone Collision Predictor** for **Flight of the Future**.

        ### Features
        - Simulates drone flight paths in 2D space
        - Detects potential mid-air collisions
        - Visualizes drone trajectories
        - Aerospace-themed UI

        ### Impact
        - Helps air traffic management for drones & UAVs
        - Improves safety & efficiency
        - Can integrate with AI-powered control towers in the future
        """
    )

# -------------------------
# TEAM PAGE
# -------------------------
elif selected == "Team":
    st.title("👨‍💻 Meet the Team / How it Works")
    st.subheader("💡 How it Works")
    st.markdown(
        """
        1. Input drone flight coordinates / paths
        2. Predict possible intersections
        3. Flag drones within danger zone
        4. Visualize safe vs risky trajectories
        """
    )
    st.subheader("👥 Our Team")
    st.markdown(
        """
        - **Vaishnavi M** – AI/ML Engineer (Algorithm & UI)  
        - Roles: Backend, Data, Presentation  

        Built using Python, Streamlit, and AI to align with **Thales domains: Aerospace & Cybersecurity**.
        """
    )
    st.success("🚀 Ready for safe drone skies!")
