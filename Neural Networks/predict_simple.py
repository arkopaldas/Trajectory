import torch
import numpy as np
from Models.config import NETWORK_PATH
from Models.utils import parse, upload_csv, plot_trajectory_and_field

if __name__ == "__main__":
    model = torch.load(NETWORK_PATH + "Simple.pth", weights_only=False)
    model.eval()
    df = upload_csv("Simple Model Result.csv")
    row = df.iloc[np.random.randint(len(df))]
    traj = parse(row["DATA POINTS"])
    start = parse(row["START"])
    goal = parse(row["GOAL"])
    obstacles = parse(row["OBSTACLES"])
    DT = row["DT"]
    true_pos = []
    true_vel = []
    for p in traj:
        x = float(p["x"])
        y = float(p["y"])
        vx = float(0 if p["vx"] is None else p["vx"])
        vy = float(0 if p["vy"] is None else p["vy"])
        true_pos.append([x,y])
        true_vel.append([vx,vy])
    true_pos = np.array(true_pos)
    true_vel = np.array(true_vel)
    curr = true_pos[0]
    pred_pos = [curr]
    pred_vel = []
    with torch.no_grad():
        for i in range(len(traj)-1):
            inp = torch.tensor(curr).float().unsqueeze(0)
            vel = model(inp).cpu().numpy()[0]
            pred_vel.append(vel)
            curr = curr + vel*DT
            pred_pos.append(curr)
    pred_vel.append(true_vel[-1])
    pred_pos = np.array(pred_pos)
    pred_vel = np.array(pred_vel)

    plot_trajectory_and_field(true_pos, pred_pos, true_vel, pred_vel, start, goal, obstacles)