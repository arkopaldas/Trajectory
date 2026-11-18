import os
import ast
import warnings
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

import config
from progress import progress

torch.multiprocessing.set_sharing_strategy('file_system')

def parse(obj):
    return ast.literal_eval(obj)

def encode_obstacles(obstacles):
    vec = []
    for obs in obstacles[:config.MAX_OBSTACLES]:
        vec.extend(obs)
    while len(vec) < 4 * config.MAX_OBSTACLES:
        vec.extend([0, 0, 0, 0])
    return np.array(vec, dtype=np.float32)

def upload_csv(name=None):
    if name is None:
        name = input("Enter CSV filename: ")
    filepath = os.path.join(config.DATASET_PATH, name)
    file_size = os.path.getsize(filepath)
    chunks = []
    chunk_size = 50000
    with open(filepath, "rb") as f:
        with progress(total=file_size, unit="B", unit_scale=True, desc=f"Loading {name}") as pbar:
            last_pos = 0
            for chunk in pd.read_csv(f, chunksize=chunk_size):
                chunks.append(chunk)
                current_pos = f.tell()
                pbar.update(current_pos - last_pos)
                last_pos = current_pos
    return pd.concat(chunks, ignore_index=True)

# --------------------------------------------------------------------------- angles
def wrap_angle(theta):
    """Wrap an angle (radians, numpy or torch) to [-pi, pi]. Used for training, evaluation and rollouts."""
    if torch.is_tensor(theta):
        return torch.atan2(torch.sin(theta), torch.cos(theta))
    return np.arctan2(np.sin(theta), np.cos(theta))

# --------------------------------------------------------------------------- normalisation
def compute_normalization(X, Y, N):
    """
    Z-score statistics from TRAINING data only.
    X (n, d) start states, Y (n, K, 2) controls, N (n, K, d) following states.
    Input statistics cover every state the network sees (start states and the states it is fed back in multi-step mode).
    """
    d = X.shape[1]
    if not config.NORMALIZE:
        stats = dict(in_mean=np.zeros(d), in_std=np.ones(d), out_mean=np.zeros(Y.shape[-1]), out_std=np.ones(Y.shape[-1]))
    else:
        states = np.concatenate([X, N.reshape(-1, d)], axis=0)
        controls = Y.reshape(-1, Y.shape[-1])
        stats = dict(in_mean=states.mean(axis=0), in_std=np.maximum(states.std(axis=0), 1e-6),
                     out_mean=controls.mean(axis=0), out_std=np.maximum(controls.std(axis=0), 1e-6))
    return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}

# --------------------------------------------------------------------------- kinematics and loss
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
        next_theta = wrap_angle(theta + omega * dt)
        return torch.stack([next_x, next_y, next_theta], dim=1)
    else:
        raise ValueError(f"Unknown Kinematic Definition: {model_type}")

def rollout_states(model, x0, model_type, dt, steps):
    """Feed the network its own predicted state `steps` times. Returns (batch, steps, d). steps = 1 is single-step."""
    states = []
    s = x0
    for _ in range(steps):
        s = compute_next_state(s, model(s), model_type, dt)
        states.append(s)
    return torch.stack(states, dim=1)

def state_error(pred, true, model_type):
    """pred - true per axis in physical units; the heading difference is wrapped to [-pi, pi]."""
    err = pred - true
    if model_type == 'unicycle':
        err = torch.cat([err[..., :2], wrap_angle(err[..., 2:3])], dim=-1)
    return err

def normalized_loss(err, state_std, model_type):
    """Mean squared z-score error per axis (error / state std), the heading axis weighted by config.LOSS_RATIO."""
    per_axis = torch.mean((err / state_std) ** 2, dim=(0, 1))
    weights = torch.ones_like(per_axis)
    if model_type == 'unicycle':
        weights[2] = config.LOSS_RATIO
    elif model_type != 'simple':
        raise ValueError(f"Invalid Kinematic Definition: {model_type}")
    return torch.sum(per_axis * weights) / torch.sum(weights)

# --------------------------------------------------------------------------- training / evaluation
def train_model(model, train_loader, model_type, dt, scale, lr=None, epochs=None, disp="bar"):
    if lr is None:
        lr = config.LR
    if epochs is None:
        epochs = config.EPOCHS
    if disp not in ("bar", "print"):
        raise ValueError("disp must be 'bar' or 'print'")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batches = len(train_loader)
    bar = progress(total=epochs * batches, desc="Training", unit="batch") if disp == "bar" else None

    try:
        for epoch in range(epochs):
            model.train()
            total_loss, total_pos = 0.0, 0.0
            if bar is not None:
                bar.set_description(f"Training epoch {epoch + 1}/{epochs}", refresh=False)
            for xb, yb, nb in train_loader:
                pred = rollout_states(model, xb, model_type, dt, nb.shape[1])
                err = state_error(pred, nb, model_type)
                loss = normalized_loss(err, model.in_std, model_type)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()

                total_loss += loss.item()
                total_pos += torch.mean(torch.sum(err[..., :2] ** 2, dim=-1)).item()
                if bar is not None:
                    bar.update(1)

            avg_loss = total_loss / batches
            norm_rmse = np.sqrt(total_pos / batches) / scale
            if bar is not None:
                bar.set_postfix({"Loss(z)": f"{avg_loss:.4f}", "Pos RMSE": f"{norm_rmse*100:.2f}%"}, refresh=False)
            else:
                print(f"Epoch {epoch+1}/{epochs} -> Loss(z) = {avg_loss:.4f} | Normalized position RMSE = {norm_rmse:.4f} ({norm_rmse*100:.2f}%)")
    finally:
        if bar is not None:
            bar.close()
    return model

def evaluate_model(model, test_loader, model_type, dt, scale, theta_scale=None):
    model.eval()
    total_loss = 0.0
    preds, true = [], []

    with torch.no_grad():
        for xb, yb, nb in test_loader:
            pred = rollout_states(model, xb, model_type, dt, nb.shape[1])
            err = state_error(pred, nb, model_type)
            total_loss += normalized_loss(err, model.in_std, model_type).item()
            preds.append(pred.cpu().numpy())
            true.append(nb.cpu().numpy())

    preds, true = np.concatenate(preds), np.concatenate(true)          # (n, K, d)
    err = preds - true
    if model_type == 'unicycle':
        err[..., 2] = wrap_angle(err[..., 2])
    per_axis_rmse = np.sqrt(np.mean(err ** 2, axis=(0, 1)))
    pos_rmse = np.sqrt(np.mean(np.sum(err[..., :2] ** 2, axis=-1)))
    avg_loss = total_loss / len(test_loader)

    print(f"\nTest Loss (z-score space) = {avg_loss:.4f}")
    print(f"Test Normalized RMSE = {pos_rmse/scale:.4f} ({pos_rmse/scale*100:.2f}%)")
    print(f"X Spatial Axis Deviation: {per_axis_rmse[0]/scale:.4f} ({per_axis_rmse[0]/scale*100:.2f}%)")
    print(f"Y Spatial Axis Deviation: {per_axis_rmse[1]/scale:.4f} ({per_axis_rmse[1]/scale*100:.2f}%)")
    if model_type == 'unicycle' and theta_scale is not None:
        print(f"θ Orientation Deviation:  {per_axis_rmse[2]/theta_scale:.4f} ({per_axis_rmse[2]/theta_scale*100:.2f}%)")
    if err.shape[1] > 1:
        per_step = np.sqrt(np.mean(np.sum(err[..., :2] ** 2, axis=-1), axis=0))
        print("Position RMSE per rollout step: " + "  ".join(f"{i+1}:{v:.3f}" for i, v in enumerate(per_step)))
    return preds.reshape(-1, preds.shape[-1]), true.reshape(-1, true.shape[-1])

def export_onnx(model, input_dim, path):
    """Export with the normalisation folded into the first/last layer: raw state in, raw control out."""
    model.eval()
    folded = model.folded_network()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, input_dim, device=device)
    common = dict(export_params=True, do_constant_folding=True, input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    try:   # legacy exporter: a single self-contained .onnx file and it honours the requested opset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(folded, dummy_input, path, opset_version=config.ONNX_OPSET, dynamo=False, **common)
    except (TypeError, RuntimeError, ValueError) as err:   # a future PyTorch without the legacy exporter
        print(f"[ONNX Generation] legacy exporter unavailable ({type(err).__name__}); using the default exporter, opset 18")
        torch.onnx.export(folded, dummy_input, path, opset_version=18, **common)
    try:   # sanity check: the ONNX file must reproduce the trained model on raw inputs
        import onnxruntime as ort
        probe = (model.in_mean + model.in_std * torch.randn(16, input_dim, device=device)).detach()
        with torch.no_grad():
            expected = model(probe).cpu().numpy()
        got = ort.InferenceSession(path).run(['output'], {'input': probe.cpu().numpy().astype(np.float32)})[0]
        print(f"[ONNX Check] max |ONNX - PyTorch| on raw inputs = {np.max(np.abs(got - expected)):.2e}")
    except ImportError:
        pass

# --------------------------------------------------------------------------- plots
def plot_environment(start, goal, obstacles):
    plt.gca().add_patch(plt.Rectangle((start[0], start[1]), start[2]-start[0], start[3]-start[1], color="green", alpha=0.5))
    plt.gca().add_patch(plt.Rectangle((goal[0], goal[1]), goal[2]-goal[0], goal[3]-goal[1], color="blue", alpha=0.5))
    for obs in obstacles:
        x1, y1, x2, y2 = obs
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color="gray", alpha=0.5))

def plot_trajectory_and_field(true_pos, pred_pos, true_vel, pred_vel, start, goal, obstacles, model_type='simple'):
    if model_type == 'unicycle':
        true_theta, pred_theta = true_pos[:, 2], pred_pos[:, 2]
        true_vel_xy = np.stack([true_vel[:, 0] * np.cos(true_theta), true_vel[:, 0] * np.sin(true_theta)], axis=1)
        pred_vel_xy = np.stack([pred_vel[:, 0] * np.cos(pred_theta), pred_vel[:, 0] * np.sin(pred_theta)], axis=1)
    else:
        true_vel_xy, pred_vel_xy = true_vel, pred_vel

    plt.figure(figsize=(5, 5))
    plt.plot(true_pos[:, 0], true_pos[:, 1], 'k-', label="True Trajectory")
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', label="Predicted Trajectory")
    plot_environment(start, goal, obstacles)
    plt.title("Spatial Trajectory Evaluation Profile")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.quiver(true_pos[:, 0], true_pos[:, 1], true_vel_xy[:, 0], true_vel_xy[:, 1], color='black', alpha=0.25, label="True Field Vector")
    plt.quiver(pred_pos[:, 0], pred_pos[:, 1], pred_vel_xy[:, 0], pred_vel_xy[:, 1], color='red', alpha=0.25, label="Pred Field Vector")
    plot_environment(start, goal, obstacles)
    plt.title("Vector Resolution Velocity Field")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def _arrow_headings(pos, model_type):
    """Heading of the arrow at every path point. Unicycle: its theta. Simple: the direction of travel."""
    if model_type == 'unicycle':
        return np.asarray(pos[:, 2], dtype=float)
    delta = np.diff(pos[:, :2], axis=0)
    heads = np.arctan2(delta[:, 1], delta[:, 0])
    moved = np.hypot(delta[:, 0], delta[:, 1]) > 1e-9
    last = next((h for h, m in zip(heads, moved) if m), 0.0)   # standing still keeps the previous heading
    for i in range(len(heads)):
        if moved[i]:
            last = heads[i]
        heads[i] = last
    return np.append(heads, heads[-1])

def animate_rollout(pred_pos, start, goal, obstacles, path, model_type='simple'):
    """
    Save a GIF of an arrow driving along the predicted path (pred_pos: (T+1, 2 or 3) array).
    The arrow follows theta for the unicycle and the direction of travel for the simple model.
    """
    pred_pos = np.asarray(pred_pos, dtype=float)
    heads = _arrow_headings(pred_pos, model_type)
    frames_per_step = max(1, int(config.GIF_FRAMES_PER_STEP))

    pts, ths = [], []
    for i in range(len(pred_pos) - 1):
        for f in range(frames_per_step):
            a = f / frames_per_step
            pts.append(pred_pos[i, :2] * (1 - a) + pred_pos[i + 1, :2] * a)
            ths.append(heads[i] + a * wrap_angle(heads[i + 1] - heads[i]))
    pts.append(pred_pos[-1, :2]); ths.append(heads[-1])
    pts, ths = np.array(pts), np.array(ths)
    steps = np.minimum(np.arange(len(pts)) / frames_per_step, len(pred_pos) - 1)
    hold = int(round(config.GIF_HOLD_SECONDS * config.GIF_FPS))
    frame_ids = list(range(len(pts))) + [len(pts) - 1] * hold

    corners = [(start[0], start[1]), (start[2], start[3]), (goal[0], goal[1]), (goal[2], goal[3])]
    corners += [(o[0], o[1]) for o in obstacles] + [(o[2], o[3]) for o in obstacles]
    allpts = np.vstack([np.array(corners, dtype=float), pred_pos[:, :2]])
    pad = max(config.GIF_ARROW_LENGTH, 0.05 * float(np.ptp(allpts, axis=0).max()))
    lo, hi = allpts.min(axis=0) - pad, allpts.max(axis=0) + pad

    L, W = config.GIF_ARROW_LENGTH, config.GIF_ARROW_WIDTH
    local = np.array([[L * 0.65, 0.0], [-L * 0.35, W * 0.5], [-L * 0.35, -W * 0.5]])

    def arrow_xy(p, th):
        c, s = np.cos(th), np.sin(th)
        return (np.array([[c, -s], [s, c]]) @ local.T).T + p

    fig, ax = plt.subplots(figsize=config.GIF_FIGSIZE)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.add_patch(plt.Rectangle((start[0], start[1]), start[2]-start[0], start[3]-start[1], color="green", alpha=0.5))
    ax.add_patch(plt.Rectangle((goal[0], goal[1]), goal[2]-goal[0], goal[3]-goal[1], color="blue", alpha=0.5))
    for x1, y1, x2, y2 in obstacles:
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color="gray", alpha=0.5))
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], "r--", alpha=0.25, linewidth=1)
    trail, = ax.plot([], [], "r-", linewidth=1.8)
    robot = plt.Polygon(arrow_xy(pts[0], ths[0]), closed=True, color="crimson", zorder=5)
    ax.add_patch(robot)
    label = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_title("Predicted rollout")

    def update(k):
        trail.set_data(pts[:k + 1, 0], pts[:k + 1, 1])
        robot.set_xy(arrow_xy(pts[k], ths[k]))
        extra = f"  heading={np.degrees(ths[k]):.0f}°" if model_type == 'unicycle' else ""
        label.set_text(f"step {steps[k]:.1f}/{len(pred_pos) - 1}\npos=({pts[k, 0]:.1f}, {pts[k, 1]:.1f}){extra}")
        return trail, robot, label

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    anim = animation.FuncAnimation(fig, lambda i: update(frame_ids[i]), frames=len(frame_ids), blit=False)
    bar = progress(total=len(frame_ids), desc="Rendering GIF", unit="frame")
    try:
        anim.save(path, writer=animation.PillowWriter(fps=config.GIF_FPS), dpi=config.GIF_DPI,
                  progress_callback=lambda i, n: bar.update(1))
    finally:
        bar.close()
        plt.close(fig)
    return path
