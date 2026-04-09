import torch
import numpy as np
from Models.config import NETWORK_PATH
from Models.utils import parse, upload_csv, plot_trajectory_and_field

if __name__ == "__main__":
    model = torch.load(NETWORK_PATH + "Unicycle.pth", weights_only=False)
    model.eval()
    df = upload_csv("Unicycle Model Result.csv")
    row = df.iloc[np.random.randint(len(df))]
    traj = parse(row["DATA POINTS"])
    start = parse(row["START"])
    goal = parse(row["GOAL"])
    obstacles = parse(row["OBSTACLES"])
    DT = row["DT"]
    true_pos=[]
    true_vel=[]
    for p in traj:
        x = float(p["x"])
        y = float(p["y"])
        theta = float(p["theta"])
        v = float(0 if p["v"] is None else p["v"])
        omega = float(0 if p["omega"] is None else p["omega"])
        true_pos.append([x,y,theta])
        true_vel.append([v,omega])
    true_pos=np.array(true_pos)
    true_vel=np.array(true_vel)
    curr = true_pos[0]
    pred_pos=[curr]
    pred_vel=[]
    with torch.no_grad():
        for i in range(len(traj)-1):
            inp=torch.tensor(curr).float().unsqueeze(0)
            vel=model(inp).cpu().numpy()[0]
            v,omega=vel
            x,y,theta=curr
            x = x + v*np.cos(theta)*DT
            y = y + v*np.sin(theta)*DT
            theta = theta + omega*DT
            curr=np.array([x,y,theta])
            pred_pos.append(curr)
            pred_vel.append(vel)
    pred_pos=np.array(pred_pos)
    pred_vel=np.array(pred_vel)
    pred_vel=np.vstack([pred_vel,pred_vel[-1]])

    plot_trajectory_and_field(true_pos, pred_pos, true_vel, pred_vel, start, goal, obstacles, model_type='unicycle')