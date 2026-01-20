import ast
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from google.colab import files
from torch.utils.data import Dataset, DataLoader

MAX_OBSTACLES = 5

uploaded = files.upload()

csv_name = list(uploaded.keys())[0]
df = pd.read_csv(csv_name)

# Helper to parse stringified Python objects
def parse(obj):
    return ast.literal_eval(obj)

def encode_region(region):
    return np.array(region, dtype=np.float32)

def encode_obstacles(obstacles):
    vec = []
    for obs in obstacles[:MAX_OBSTACLES]:
        vec.extend(obs)
    while len(vec) < 4 * MAX_OBSTACLES:
        vec.extend([0.0, 0.0, 0.0, 0.0])
    return np.array(vec, dtype=np.float32)

X, Y = [], []

for _, row in df.iterrows():
    start = encode_region(parse(row["START"]))
    goal = encode_region(parse(row["GOAL"]))
    obstacles = encode_obstacles(parse(row["OBSTACLES"]))

    traj = parse(row["DATA POINTS"])

    for i in range(len(traj) - 1):
        p  = traj[i]
        p2 = traj[i + 1]

        t  = float(p["t"])
        x  = float(p["x"])
        y  = float(p["y"])
        vx = float(0.0 if p["vx"] is None else p["vx"])
        vy = float(0.0 if p["vy"] is None else p["vy"])

        x2  = float(p2["x"])
        y2  = float(p2["y"])
        vx2 = float(0.0 if p2["vx"] is None else p2["vx"])
        vy2 = float(0.0 if p2["vy"] is None else p2["vy"])

        inp = np.concatenate([
            start,
            goal,
            obstacles,
            np.array([t, x, y, vx, vy], dtype=np.float32)
        ])

        out = np.array([x2, y2, vx2, vy2], dtype=np.float32)

        X.append(inp)
        Y.append(out)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

print("Dataset shape:", X.shape, Y.shape)

class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class TrajectoryNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # x,y,vx,vy
        )

    def forward(self, x):
        return self.net(x)

dataset = TrajectoryDataset(X, Y)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

model = TrajectoryNet(X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

EPOCHS = 30

for epoch in range(EPOCHS):
    total_loss = 0.0
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.6f}")

torch.save(model.state_dict(), "trajectory_net.pth")
print("Model saved as trajectory_net.pth")
