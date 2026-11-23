"""Qualitative check of a trained SIMPLE network on one random trajectory.

Loads Simple.onnx, picks a random row of the dataset, starts from that trajectory's start point and lets the network drive
closed-loop (state -> network -> velocity -> next state), then compares with the solver's trajectory: saves a rollout GIF
('Simple_rollout.gif' next to the model) and shows the path and velocity-field plots.

Run:  python predict_simple.py   (from inside the Neural Networks folder)
"""
import os
import numpy as np
import onnxruntime as ort

import config
from utils import parse, upload_csv, plot_trajectory_and_field, animate_rollout

if __name__ == "__main__":
    ONNX_PATH = os.path.join(config.NETWORK_PATH, "Simple.onnx")
    GIF_PATH = os.path.join(config.NETWORK_PATH, "Simple_rollout.gif")
    # Load the exported network with ONNX Runtime and read a random trajectory (with its start / goal / obstacles / DT) from the dataset.
    session = ort.InferenceSession(ONNX_PATH)
    df = upload_csv("Simple Model Result.csv")
    row = df.iloc[np.random.randint(len(df))]

    traj = parse(row["DATA POINTS"])
    start = parse(row["START"])
    goal = parse(row["GOAL"])
    obstacles = parse(row["OBSTACLES"])
    DT = row["DT"]

    # Ground truth from the solver: positions and the velocities applied (None at the last point is replaced by 0).
    true_pos, true_vel = [], []
    for p in traj:
        true_pos.append([float(p["x"]), float(p["y"])])
        true_vel.append([float(0 if p["vx"] is None else p["vx"]), float(0 if p["vy"] is None else p["vy"])])

    true_pos, true_vel = np.array(true_pos), np.array(true_vel)
    # Closed-loop rollout from the true start: ask the network for the velocity, advance x' = x + v * dt, repeat once per solver step.
    curr = true_pos[0]
    pred_pos = [curr]
    pred_vel = []

    for i in range(len(traj) - 1):
        raw_output = session.run(['output'], {'input': np.array([curr], dtype=np.float32)})
        vel = raw_output[0][0]
        pred_vel.append(vel)
        curr = curr + vel * DT
        pred_pos.append(curr)

    # Pad the predicted velocities so they have as many entries as positions (the last point has no control), then animate and plot.
    pred_vel.append(true_vel[-1])
    animate_rollout(np.array(pred_pos), start, goal, obstacles, GIF_PATH, model_type='simple')
    print(f"[Rollout GIF] Saved -> {GIF_PATH}")
    plot_trajectory_and_field(true_pos, np.array(pred_pos), true_vel, np.array(pred_vel), start, goal, obstacles)
