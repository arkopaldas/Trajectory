import os
import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from Models.config import MAX_OBSTACLES, TRAIN_RATIO, BATCH_SIZE, EPOCHS, LR, DATASET_PATH

class TrajectoryDataset(Dataset):
    def __init__(self, X, Y, N):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.N = torch.tensor(N, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.N[idx]

def parse(obj):
    return ast.literal_eval(obj)

def encode_obstacles(obstacles):
    vec = []
    for obs in obstacles[:MAX_OBSTACLES]:
        vec.extend(obs)
    while len(vec) < 4 * MAX_OBSTACLES:
        vec.extend([0,0,0,0])
    return np.array(vec, dtype=np.float32)

def upload_csv(name=None):
    if name is None:
        name = input("Enter CSV filename: ")
    filepath = os.path.join(DATASET_PATH, name)
    file_size = os.path.getsize(filepath)
    chunks = []
    chunk_size = 50_000
    with open(filepath, "rb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Loading {name}", leave=True) as pbar:
            last_pos = 0
            for chunk in pd.read_csv(f, chunksize=chunk_size):
                chunks.append(chunk)
                current_pos = f.tell()
                pbar.update(current_pos - last_pos)
                last_pos = current_pos
    return pd.concat(chunks, ignore_index=True)

def create_dataloaders(X, Y, N):
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]
    N = N[idx]
    split = int(TRAIN_RATIO * len(X))
    X_train = X[:split]
    Y_train = Y[:split]
    N_train = N[:split]
    X_test = X[split:]
    Y_test = Y[split:]
    N_test = N[split:]
    train_ds = TrajectoryDataset(X_train, Y_train, N_train)
    test_ds  = TrajectoryDataset(X_test, Y_test, N_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def compute_next_state(state, control, model_type, dt):
    if model_type == 'simple':
        return state + control * dt
    elif model_type == 'unicycle':
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        v = control[:, 0]
        omega = control[:, 1]
        next_x = x + v * torch.cos(theta) * dt
        next_y = y + v * torch.sin(theta) * dt
        next_theta = theta + omega * dt
        return torch.stack([next_x, next_y, next_theta], dim=1)
    else:
        raise ValueError("Unknown model_type")

def train_model_bar(model, train_loader, model_type, dt, scale):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    epoch_bar = tqdm(range(EPOCHS), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0
        for xb, yb, nb in train_loader:
            pred_control = model(xb)
            pred_next_state = compute_next_state(xb, pred_control, model_type, dt)
            # loss = loss_fn(pred_next_state, nb)
            if model_type == 'unicycle':
                loss = loss_fn(pred_next_state[:, :2], nb[:, :2])
            else:
                loss = loss_fn(pred_next_state, nb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_mse = total_loss / len(train_loader)
        norm_rmse = np.sqrt(avg_mse) / scale
        epoch_bar.set_postfix({"Norm RMSE": f"{norm_rmse:.4f}", "Percentage(%)": f"{norm_rmse*100:.2f}"})
    return model

def train_model_print(model, train_loader, model_type, dt, scale):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb, nb in train_loader:
            pred_control = model(xb)
            pred_next_state = compute_next_state(xb, pred_control, model_type, dt)
            # loss = loss_fn(pred_next_state, nb)
            if model_type == 'unicycle':
                loss = loss_fn(pred_next_state[:, :2], nb[:, :2])
            else:
                loss = loss_fn(pred_next_state, nb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_mse = total_loss / len(train_loader)
        norm_rmse = np.sqrt(avg_mse) / scale
        print(f"Epoch {epoch+1}/{EPOCHS}  Normalized RMSE = {norm_rmse:.4f}  ({norm_rmse*100:.2f}%)")
    return model

def train_model(model, train_loader, model_type, dt, scale, disp="bar"):
    if disp == "bar":
        return train_model_bar(model, train_loader, model_type, dt, scale)
    elif disp == "print":
        return train_model_print(model, train_loader, model_type, dt, scale)
    else:
        raise ValueError("The display value was invalid")
        return None

def evaluate_model(model, test_loader, model_type, dt, scale, theta_scale=None):
    loss_fn = torch.nn.MSELoss()
    model.eval()
    total_loss = 0
    preds = []
    true = []
    with torch.no_grad():
        for xb, yb, nb in test_loader:
            pred_control = model(xb)
            pred_next_state = compute_next_state(xb, pred_control, model_type, dt)
            loss = loss_fn(pred_next_state, nb)
            total_loss += loss.item()
            preds.append(pred_next_state.cpu().numpy())
            true.append(nb.cpu().numpy())
    preds = np.vstack(preds)
    true = np.vstack(true)
    per_axis_rmse = np.sqrt(np.mean((preds - true)**2, axis=0))
    avg_mse = total_loss / len(test_loader)
    norm_rmse = np.sqrt(avg_mse) / scale
    print(f"\nTest Normalized RMSE = {norm_rmse:.4f}  ({norm_rmse*100:.2f}%)")
    if model_type == 'simple':
        print(f"X error: {per_axis_rmse[0]/scale:.4f}  ({per_axis_rmse[0]/scale*100:.2f}%)")
        print(f"Y error: {per_axis_rmse[1]/scale:.4f}  ({per_axis_rmse[1]/scale*100:.2f}%)")
    elif model_type == 'unicycle':
        print(f"X error: {per_axis_rmse[0]/scale:.4f}  ({per_axis_rmse[0]/scale*100:.2f}%)")
        print(f"Y error: {per_axis_rmse[1]/scale:.4f}  ({per_axis_rmse[1]/scale*100:.2f}%)")
        print(f"θ error: {per_axis_rmse[2]/theta_scale:.4f}  ({per_axis_rmse[2]/theta_scale*100:.2f}%)")
    return preds, true

def plot_environment(start, goal, obstacles):
    plt.gca().add_patch(plt.Rectangle((start[0], start[1]), start[2]-start[0], start[3]-start[1], color="green", alpha=0.5))
    plt.gca().add_patch(plt.Rectangle((goal[0], goal[1]), goal[2]-goal[0], goal[3]-goal[1], color="blue", alpha=0.5))

    for obs in obstacles:
        x1,y1,x2,y2 = obs
        plt.gca().add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, color="gray", alpha=0.5))

def plot_trajectory_and_field(true_pos, pred_pos, true_vel, pred_vel, start, goal, obstacles, model_type='simple'):
    if model_type == 'unicycle':
        true_theta = true_pos[:, 2]
        pred_theta = pred_pos[:, 2]
        true_vel_xy  = np.stack([true_vel[:, 0] * np.cos(true_theta), true_vel[:, 0] * np.sin(true_theta)], axis=1)
        pred_vel_xy  = np.stack([pred_vel[:, 0] * np.cos(pred_theta), pred_vel[:, 0] * np.sin(pred_theta)], axis=1)
    else:
        true_vel_xy = true_vel
        pred_vel_xy = pred_vel

    plt.figure(figsize=(7, 7))
    plt.plot(true_pos[:, 0], true_pos[:, 1], 'k-', label="True Trajectory")
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', label="Predicted Trajectory")
    plot_environment(start, goal, obstacles)
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()

    plt.figure(figsize=(7, 7))
    plt.quiver(true_pos[:, 0], true_pos[:, 1], true_vel_xy[:, 0], true_vel_xy[:, 1], color='black', alpha=0.25, label="True Velocity")
    plt.quiver(pred_pos[:, 0], pred_pos[:, 1], pred_vel_xy[:, 0], pred_vel_xy[:, 1], color='red', alpha=0.25, label="Pred Velocity")
    plot_environment(start, goal, obstacles)
    plt.title("Velocity Field Comparison")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()