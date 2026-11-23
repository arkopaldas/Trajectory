"""Turns the generated trajectory CSVs into PyTorch training data.

A trajectory is a list of states with the control that was applied at each step. The data is cut into WINDOWS:
    X[i] = a start state (d,), Y[i] = the K controls applied from it (K, 2), N[i] = the K states that follow (K, d),
with K = config.ROLLOUT_HORIZON (K = 1: plain single-step prediction). d = 2 for the simple model (x, y), d = 3 for
the unicycle (x, y, theta). create_dataloaders then shuffles, splits into train / test and computes the z-score statistics.
"""
# Third-party imports.
import torch
import numpy as np
import multiprocessing
from torch.utils.data import Dataset, DataLoader

# Local modules (this folder): settings and helper functions.
import config
from utils import parse, compute_normalization

# Minimal torch Dataset that returns one window (X, Y, N) as float32 tensors.
class TrajectoryDataset(Dataset):
    """
    One sample = a window of the trajectory:
      X[i]  start state                      (d,)
      Y[i]  the K controls applied from it   (K, 2)
      N[i]  the K states that follow         (K, d)
    K = config.ROLLOUT_HORIZON (K = 1 is plain single-step prediction).
    """
    def __init__(self, X, Y, N):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.N = torch.tensor(N, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.N[idx]

# Cuts every trajectory of the dataframe into sliding windows of `horizon` steps. The control of the final state is stored
# as None in the CSV and replaced by 0 (it is never used, because windows end before it). Returns numpy arrays (X, Y, N).
def _build_windows(df, state_keys, control_keys, horizon):
    if horizon is None:
        horizon = config.ROLLOUT_HORIZON
    if horizon < 1:
        raise ValueError("ROLLOUT_HORIZON must be >= 1")
    X, Y, N = [], [], []
    for _, row in df.iterrows():
        # The trajectory of this row: a list of dicts {t, x, y, ...}.
        traj = parse(row["DATA POINTS"])
        # states[i] = the state at step i; controls[i] = the control applied at step i (0 for the last point, which has none).
        states = [[float(p[k]) for k in state_keys] for p in traj]
        controls = [[float(0 if p[k] is None else p[k]) for k in control_keys] for p in traj]
        # One window per start index i: X = state i, Y = controls i..i+K-1, N = states i+1..i+K.
        for i in range(len(traj) - horizon):
            X.append(states[i])
            Y.append(controls[i:i + horizon])
            N.append(states[i + 1:i + 1 + horizon])
    if not X:
        raise ValueError(f"No trajectory has more than {horizon} step(s); lower config.ROLLOUT_HORIZON")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(N, dtype=np.float32)

# Windows for the simple model: state (x, y), control (vx, vy).
def build_simple_dataset(df, horizon=None):
    return _build_windows(df, ("x", "y"), ("vx", "vy"), horizon)

# Windows for the unicycle: state (x, y, theta), control (v, omega).
def build_unicycle_dataset(df, horizon=None):
    return _build_windows(df, ("x", "y", "theta"), ("v", "omega"), horizon)

# Whole trajectories as arrays (x, y); trajectories shorter than config.ROLLOUT_STEPS are skipped. (Not used at the moment.)
def build_simple_sequences(df):
    sequences = []
    for _, row in df.iterrows():
        traj = parse(row["DATA POINTS"])
        if len(traj) < config.ROLLOUT_STEPS:
            continue
        sequences.append(np.array([[float(p["x"]), float(p["y"])] for p in traj], dtype=np.float32))
    return sequences

# Whole trajectories as arrays (x, y, theta); trajectories shorter than config.ROLLOUT_STEPS are skipped. (Not used at the moment.)
def build_unicycle_sequences(df):
    sequences = []
    for _, row in df.iterrows():
        traj = parse(row["DATA POINTS"])
        if len(traj) < config.ROLLOUT_STEPS:
            continue
        sequences.append(np.array([[float(p["x"]), float(p["y"]), float(p["theta"])] for p in traj], dtype=np.float32))
    return sequences

def create_dataloaders(X, Y, N, batch_size=None):
    """Returns (train_loader, test_loader, stats). stats = z-score statistics of the TRAINING split only."""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    num_workers = min(4, multiprocessing.cpu_count())
    # Shuffle all windows, then split: the first TRAIN_RATIO of them train the network, the rest test it. The normalisation
    # statistics are computed on the training part only (no information leaks from the test split).
    idx = np.random.permutation(len(X))
    X, Y, N = X[idx], Y[idx], N[idx]

    split = int(config.TRAIN_RATIO * len(X))
    stats = compute_normalization(X[:split], Y[:split], N[:split])
    train_ds = TrajectoryDataset(X[:split], Y[:split], N[:split])
    test_ds = TrajectoryDataset(X[split:], Y[split:], N[split:])

    # Data loaders: the training loader reshuffles every epoch, the test loader keeps the order.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    return train_loader, test_loader, stats
