import torch.nn as nn
import numpy as np
from Models.utils import parse

class UnicycleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,256),
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

def build_unicycle_dataset(df):
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
            theta = float(p["theta"])
            x_next = float(p_next["x"])
            y_next = float(p_next["y"])
            theta_next = float(p_next["theta"])
            v = float(0 if p["v"] is None else p["v"])
            omega = float(0 if p["omega"] is None else p["omega"])
            X.append([x,y,theta])
            Y.append([v,omega])
            N.append([x_next, y_next, theta_next])
    return np.array(X,dtype=np.float32), np.array(Y,dtype=np.float32), np.array(N,dtype=np.float32)