"""Helpers of the verification module: reading the problem from the dataset, loading the ONNX network, simulating the exact
closed loop, splitting the start region into cells, and reading / writing counterexamples and reports.

Counterexample files use the SAME columns as the expert dataset (see Datasets/Counterexamples/README.md), so they can be
inspected and plotted with the same tools. Reports are written to Verification/Results/<N> Trained/.
"""
import os
import ast
import math
import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------- environment (read from the dataset CSV)
class Environment:
    """Everything the safety constraints and the dynamics need. Values come from the first row of the dataset CSV."""

    def __init__(self, model):
        # Read the problem from the FIRST row of the dataset CSV: the constants are identical in every row.
        info = config.MODELS[model]
        path = os.path.join(config.DATASET_PATH, info["dataset"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset not found: {path}\nCheck config.DATASET_SIZE")
        self.model = model
        # Column order of the dataset (reused so that counterexample files have exactly the same columns).
        self.columns = list(pd.read_csv(path, nrows=0).columns)
        self.row = pd.read_csv(path, nrows=1).iloc[0].to_dict()
        row = self.row

        self.state_names, self.control_names = info["state"], info["control"]
        self.state_dim = len(self.state_names)
        self.dt = float(row["DT"])
        # Number of closed-loop steps = horizon / dt, rounded up (the same as TOTAL STEPS in the dataset).
        self.T = int(math.ceil(float(row["BOUND"]) / self.dt))          # number of closed-loop steps (= TOTAL STEPS)
        self.start = tuple(float(v) for v in ast.literal_eval(row["START"]))
        self.goal = tuple(float(v) for v in ast.literal_eval(row["GOAL"]))
        self.obstacles = [tuple(float(v) for v in o) for o in ast.literal_eval(row["OBSTACLES"])]
        self.margin = float(row["MARGIN"])
        # Workspace box: from config.WORKSPACE if given, otherwise (0, 0, goal x2, goal y2) like in the solver models.
        self.workspace = tuple(config.WORKSPACE) if config.WORKSPACE is not None else (0.0, 0.0, self.goal[2], self.goal[3])

        # Control bounds and (unicycle only) heading bounds from the dataset columns VMIN / VMAX / WMIN / WMAX / THMIN / THMAX.
        if model == "simple":
            self.ctrl_lo = [float(row["VMIN"])] * 2
            self.ctrl_hi = [float(row["VMAX"])] * 2
            self.theta_bounds = None
            self.start_theta = None
        else:
            self.ctrl_lo = [float(row["VMIN"]), float(row["WMIN"])]
            self.ctrl_hi = [float(row["VMAX"]), float(row["WMAX"])]
            self.theta_bounds = (float(row["THMIN"]), float(row["THMAX"]))
            self.start_theta = tuple(config.START_THETA_RANGE) if config.START_THETA_RANGE is not None else self.theta_bounds


def wrap_angle(theta):
    """Same wrapping as 'Neural Networks/utils.py'."""
    return np.arctan2(np.sin(theta), np.cos(theta))


# --------------------------------------------------------------------------- networks
# Path of the network to verify: '<Model>.onnx', or '<Model>_CEGIS.onnx' when config.USE_CEGIS_NETWORK is True.
def network_file(model):
    suffix = config.CEGIS_SUFFIX if config.USE_CEGIS_NETWORK else ""
    return os.path.join(config.NETWORK_PATH, config.MODELS[model]["onnx"] + suffix + ".onnx")


# Path of the fine-tuned network that the CEGIS loop writes (the original network is never overwritten).
def cegis_network_file(model):
    return os.path.join(config.NETWORK_PATH, config.MODELS[model]["onnx"] + config.CEGIS_SUFFIX + ".onnx")


def load_onnx_layers(path):
    """
    Reads the Linear/ReLU stack out of an ONNX file. Returns [(W (out, in), b (out,), has_relu), ...] in float64.
    Supports the two layouts PyTorch exports: Gemm, or MatMul followed by Add.
    """
    import onnx
    from onnx import numpy_helper
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Network not found: {path}")
    model = onnx.load(path)   # picks up a neighbouring ".onnx.data" file automatically
    # All weights of the ONNX graph by name; the nodes are then walked in order and turned into (W, b, relu) layers.
    init = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in model.graph.initializer}
    layers = []
    for node in model.graph.node:
        attrs = {a.name: a for a in node.attribute}
        # Gemm = y = alpha * A @ B(^T) + beta * C, the usual way PyTorch exports a Linear layer.
        if node.op_type == "Gemm":
            alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
            beta = attrs["beta"].f if "beta" in attrs else 1.0
            if "transA" in attrs and attrs["transA"].i:
                raise ValueError("Gemm with transA is not supported")
            W = init[node.input[1]]
            W = W if ("transB" in attrs and attrs["transB"].i) else W.T
            b = init[node.input[2]] if len(node.input) > 2 else np.zeros(W.shape[0])
            layers.append([alpha * W, beta * b, False])
        # Older export style: MatMul (weights) followed by a separate Add (bias).
        elif node.op_type == "MatMul":
            W = init[node.input[1]]
            layers.append([W.T, np.zeros(W.shape[1]), False])
        elif node.op_type == "Add":
            layers[-1][1] = layers[-1][1] + init[node.input[1]]
        # A ReLU belongs to the layer before it. Any other operator is rejected (the verifier only handles Linear / ReLU networks).
        elif node.op_type == "Relu":
            layers[-1][2] = True
        else:
            raise ValueError(f"Unsupported ONNX node: {node.op_type}")
    return [(W, b, relu) for W, b, relu in layers]


def forward(layers, x):
    """float64 forward pass of the extracted Linear/ReLU stack. x: (n, d)."""
    a = np.asarray(x, dtype=np.float64)
    for W, b, relu in layers:
        # One layer: affine map, then ReLU if the layer has one.
        a = a @ W.T + b
        if relu:
            a = np.maximum(a, 0.0)
    return a


# ONNX Runtime session (CPU) for the network; used for fast exact rollouts.
def make_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------- exact closed-loop rollout (true dynamics)
def step_dynamics(env, state, control):
    """The true dynamics, exactly as in 'predict_simple.py' / 'predict_unicycle.py'. No approximation."""
    if env.model == "simple":
        return state + control * env.dt
    # Unicycle: exact kinematics with the heading wrapped to [-pi, pi] (same as the training and prediction code).
    x, y, theta = state
    v, omega = control
    return np.array([x + v * np.cos(theta) * env.dt, y + v * np.sin(theta) * env.dt, wrap_angle(theta + omega * env.dt)])


def exact_rollout(env, session, x0):
    """Rolls the ONNX policy out in closed loop for env.T steps. Returns states (T+1, d) and controls (T, c)."""
    states = [np.asarray(x0, dtype=np.float64)]
    controls = []
    for _ in range(env.T):
        # Ask the network (float32, like in deployment) for the control, then advance the true dynamics (float64).
        u = session.run(["output"], {"input": np.array([states[-1]], dtype=np.float32)})[0][0].astype(np.float64)
        controls.append(u)
        states.append(step_dynamics(env, states[-1], u))
    return np.array(states), np.array(controls)


# --------------------------------------------------------------------------- start-region cells
def make_cells(env):
    """Splits the start region (and the start heading range for the unicycle) into boxes. Returns [(lo, hi), ...]."""
    x1, y1, x2, y2 = env.start
    nx, ny = config.START_CELLS_XY
    # Equally spaced cell borders in x, y (and heading for the unicycle); every combination of intervals is one cell (lower, upper corner).
    xs, ys = np.linspace(x1, x2, nx + 1), np.linspace(y1, y2, ny + 1)
    ts = np.linspace(env.start_theta[0], env.start_theta[1], config.START_CELLS_THETA + 1) if env.state_dim == 3 else [None]
    cells = []
    for ix in range(nx):
        for iy in range(ny):
            for it in range(len(ts) - 1 if env.state_dim == 3 else 1):
                lo, hi = [xs[ix], ys[iy]], [xs[ix + 1], ys[iy + 1]]
                if env.state_dim == 3:
                    lo.append(ts[it])
                    hi.append(ts[it + 1])
                cells.append((np.array(lo), np.array(hi)))
    return cells


# --------------------------------------------------------------------------- counterexample files (dataset schema)
# Path of the counterexample CSV: Datasets/Counterexamples/<N> Trajectories/<Model> Model Counterexamples.csv
def counterexample_file(model):
    return os.path.join(config.COUNTEREXAMPLE_PATH, f"{config.MODELS[model]['name']} Model Counterexamples.csv")


# Converts a trajectory into the list of per-step dicts stored in the DATA POINTS column (t, state names, control names;
# the control of the last state is None), the same layout as in the expert dataset.
def data_points(env, states, controls):
    points = []
    for t in range(len(states)):
        p = {"t": t}
        for name, value in zip(env.state_names, states[t]):
            p[name] = float(value)
        for j, name in enumerate(env.control_names):
            p[name] = float(controls[t][j]) if t < len(controls) else None
        points.append(p)
    return points


def counterexample_row(env, ce):
    """One dataset row (same columns as the expert dataset) holding the counterexample trajectory."""
    # Start from the first dataset row (so all problem constants are copied) and overwrite what is specific to this counterexample.
    row = dict(env.row)
    states, controls = ce["states"], ce["controls"]
    row.update({
        # STATUS marks the row as a counterexample; LENGTH is the Manhattan path length of the trajectory; the times are the
        # Marabou build / solve times in ms; DATA POINTS holds the replayed closed-loop trajectory.
        "STATUS": "COUNTEREXAMPLE",
        "TOTAL STEPS": env.T,
        "LENGTH": float(np.sum(np.abs(np.diff(states[:, :2], axis=0)))),
        "BUILD TIME (ms)": ce.get("build_ms"),
        "SOLVE TIME (ms)": ce.get("solve_ms"),
        "TOTAL TIME (ms)": (ce.get("build_ms") or 0) + (ce.get("solve_ms") or 0),
        "DATA POINTS": str(data_points(env, states, controls)),
    })
    if "SOLVER" in row:
        row["SOLVER"] = "Marabou"
    return row


# Appends the counterexamples to the counterexample CSV (the header is written only when the file is new). Existing rows are kept.
def save_counterexamples(env, ces):
    if not ces:
        return
    path = counterexample_file(env.model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame = pd.DataFrame([counterexample_row(env, ce) for ce in ces], columns=env.columns)
    frame.to_csv(path, mode="a", header=not os.path.isfile(path), index=False)


def load_counterexample_starts(env):
    """Start states of every stored counterexample, (n, d). Empty (0, d) if there is none."""
    path = counterexample_file(env.model)
    if not os.path.isfile(path):
        return np.zeros((0, env.state_dim))
    starts = []
    for text in pd.read_csv(path)["DATA POINTS"]:
        first = ast.literal_eval(text)[0]
        # The start state is the first data point of each stored counterexample trajectory.
        starts.append([float(first[k]) for k in env.state_names])
    return np.array(starts).reshape(-1, env.state_dim)


# --------------------------------------------------------------------------- reports
# Writes the per-cell verification table to Verification/Results/<N> Trained/<Model> Model <kind>.csv and returns the path.
def save_report(model, rows, kind):
    os.makedirs(config.RESULT_PATH, exist_ok=True)
    path = os.path.join(config.RESULT_PATH, f"{config.MODELS[model]['name']} Model {kind}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# Seconds -> 'h:mm:ss'.
def format_seconds(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
