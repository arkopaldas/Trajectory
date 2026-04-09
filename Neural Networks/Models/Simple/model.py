import torch.nn as nn
import numpy as np
from Models.utils import parse

class SimpleTrajectoryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        return self.net(x)

def build_simple_dataset(df):
    X = []
    Y = []
    N = []
    for _, row in df.iterrows():
        traj = parse(row["DATA POINTS"])
        for i in range(len(traj)-1):
            p = traj[i]
            p_next = traj[i+1]
            x = float(p["x"])
            y = float(p["y"])
            x_next = float(p_next["x"])
            y_next = float(p_next["y"])
            vx = float(0 if p["vx"] is None else p["vx"])
            vy = float(0 if p["vy"] is None else p["vy"])
            X.append([x,y])
            Y.append([vx,vy])
            N.append([x_next, y_next])
    return np.array(X,dtype=np.float32), np.array(Y,dtype=np.float32), np.array(N,dtype=np.float32)