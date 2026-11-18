import os
import numpy as np
import onnxruntime as ort

import config
from utils import parse, upload_csv, plot_trajectory_and_field, animate_rollout, wrap_angle

if __name__ == "__main__":
    ONNX_PATH = os.path.join(config.NETWORK_PATH, "Unicycle.onnx")
    GIF_PATH = os.path.join(config.NETWORK_PATH, "Unicycle_rollout.gif")
    session = ort.InferenceSession(ONNX_PATH)
    df = upload_csv("Unicycle Model Result.csv")
    row = df.iloc[np.random.randint(len(df))]

    traj = parse(row["DATA POINTS"])
    start = parse(row["START"])
    goal = parse(row["GOAL"])
    obstacles = parse(row["OBSTACLES"])
    DT = row["DT"]

    true_pos, true_vel = [], []
    for p in traj:
        true_pos.append([float(p["x"]), float(p["y"]), float(p["theta"])])
        true_vel.append([float(0 if p["v"] is None else p["v"]), float(0 if p["omega"] is None else p["omega"])])

    true_pos, true_vel = np.array(true_pos), np.array(true_vel)
    curr = true_pos[0]
    pred_pos = [curr]
    pred_vel = []

    for i in range(len(traj) - 1):
        raw_output = session.run(['output'], {'input': np.array([curr], dtype=np.float32)})
        vel = raw_output[0][0]
        v, omega = vel
        x, y, theta = curr

        x = x + v * np.cos(theta) * DT
        y = y + v * np.sin(theta) * DT
        theta = wrap_angle(theta + omega * DT)

        curr = np.array([x, y, theta])
        pred_pos.append(curr)
        pred_vel.append(vel)

    pred_vel = np.vstack([pred_vel, pred_vel[-1]])
    animate_rollout(np.array(pred_pos), start, goal, obstacles, GIF_PATH, model_type='unicycle')
    print(f"[Rollout GIF] Saved -> {GIF_PATH}")
    plot_trajectory_and_field(true_pos, np.array(pred_pos), true_vel, pred_vel, start, goal, obstacles, model_type='unicycle')
