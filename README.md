# Trajectory

A research notebook for trajectory planning using formal methods (SMT/MILP) and neural network imitation learning, applied to two robot motion models: a **Simple (double integrator)** model and a **Unicycle** model.

---

## Project Structure

```
Trajectory/
├── Controllers/
│   ├── milp_unicycle.py          # MILP controller for the unicycle model
│   └── smt_simple.py             # SMT (Z3) controller for the simple model
│
├── Experiments/
│   ├── milp_simple.py            # MILP planner for the simple model (with circle/polygon obstacles)
│   ├── milp_fixed_unicycle.py    # MILP unicycle planner with randomised start points
│   └── smt_unicycle.py           # SMT (Z3) planner for the unicycle model
│
├── Generators/
│   ├── simple.py                 # Batch dataset generator for the simple model (Z3 + multiprocessing)
│   └── unicycle.py               # Batch dataset generator for the unicycle model (MILP + multiprocessing)
│
├── Datasets/
│   ├── Schemas/
│   │   ├── Simple Model Dataset Schema.csv
│   │   └── Unicycle Model Dataset Schema.csv
│   └── Results/
│       ├── Simple Model Result.csv
│       └── Unicycle Model Result.csv
│
└── Neural Networks/
    ├── train_simple.py           # Training script for the simple model network
    ├── train_unicycle.py         # Training script for the unicycle model network
    ├── predict_simple.py         # Inference + trajectory visualisation (simple)
    ├── predict_unicycle.py       # Inference + trajectory visualisation (unicycle)
    ├── Networks/                 # Saved model weights (.pth files, git-ignored)
    └── Models/
        ├── config.py             # Shared hyperparameters and path configuration
        ├── utils.py              # Dataset class, training loop, evaluation, plotting
        ├── Simple/
        │   ├── model.py          # SimpleTrajectoryNet architecture + dataset builder
        │   └── __init__.py
        └── Unicycle/
            ├── model.py          # UnicycleNet architecture + dataset builder
            └── __init__.py
```

---

## Models

### Simple Model
A double-integrator (point mass) that moves freely in 2D. Controls are `(vx, vy)` velocities. The neural network takes `(x, y)` as input and predicts the control `(vx, vy)` to apply.

### Unicycle Model
A unicycle that moves forward with speed `v` and turns with angular velocity `ω`. Heading `θ` evolves over time. The neural network takes `(x, y, θ)` as input and predicts the control `(v, ω)` to apply.

---

## Pipeline

```
Generators  →   Datasets/Results    →   Neural Networks (train) →   Networks/*.pth  →   Neural Networks (predict)
    ↑
Controllers / Experiments (single-instance solvers used for verification and experimentation)
```

1. **Controllers** — standalone single-run solvers for quick verification and tuning.
2. **Experiments** — extended single-run variants (randomised starts, mixed obstacle shapes).
3. **Generators** — parallelised batch solvers that write thousands of trajectories to CSV.
4. **Neural Networks** — imitation learning: trains on the generated CSV data, then predicts trajectories at inference time.

---

## Dependencies

Install all dependencies via pip:

```bash
pip install z3-solver pulp torch numpy pandas matplotlib tqdm highspy
```

| Package | Purpose |
|---|---|
| `z3-solver` | SMT solver (Z3) used in simple/unicycle SMT planners |
| `pulp` | MILP modelling; uses HiGHS (preferred) or CBC (fallback) |
| `torch` | Neural network training and inference (PyTorch) |
| `numpy` | Numerical arrays |
| `pandas` | CSV I/O for datasets |
| `matplotlib` | Trajectory and velocity field plotting |
| `tqdm` | Progress bars during batch generation and result writing |
| `highspy` | MILP solver dependency for HiGHS (optional to install) |

> **Note:** PuLP will attempt to use the HiGHS solver automatically. If HiGHS is unavailable it falls back to CBC, which ships bundled with PuLP.

---

## Usage

### 1. Run a single controller (quick test)
```bash
python Controllers/milp_unicycle.py
python Controllers/smt_simple.py
```

### 2. Generate a dataset
```bash
python Generators/simple.py
python Generators/unicycle.py
```
Results are written to `Datasets/Results/`. The number of instances is controlled by `max_instances` at the top of each generator file.

### 3. Train a neural network
Run from inside the `Neural Networks/` directory so that the relative `Models.*` imports resolve correctly:
```bash
cd "Neural Networks"
python train_simple.py
python train_unicycle.py
```
Trained weights are saved to `Neural Networks/Networks/`.

### 4. Run inference
```bash
cd "Neural Networks"
python predict_simple.py
python predict_unicycle.py
```
A random trajectory is sampled from the result CSV and the model's predicted trajectory is plotted alongside the ground truth.

---

## Dataset Schema

### Simple Model
| Column | Description |
|---|---|
| `BOUND` | Time horizon |
| `DT` | Time step |
| `VMIN` / `VMAX` | Velocity bounds |
| `MARGIN` | Obstacle clearance margin |
| `START` / `GOAL` | Bounding box regions `(x1, y1, x2, y2)` |
| `OBSTACLES` | List of obstacle rectangles |
| `OPTIMIZE` | Whether path length was minimised |
| `STATUS` | `SAT` or `UNSAT` |
| `TOTAL STEPS` | Number of time steps `T` |
| `LENGTH` | Total path length |
| `BUILD TIME (ms)` / `SOLVE TIME (ms)` / `TOTAL TIME (ms)` | Timing breakdown |
| `DATA POINTS` | Per-step trajectory data `[{t, x, y, vx, vy}, ...]` |

### Unicycle Model
Same as above, plus:

| Column | Description |
|---|---|
| `WMIN` / `WMAX` | Angular velocity bounds |
| `THMIN` / `THMAX` | Heading angle bounds |
| `DTHETA` | Angle discretisation step |
| `SOLVER` | Which solver was used (`HiGHS` or `CBC`) |
| `DATA POINTS` | Per-step data `[{t, x, y, v, omega, theta}, ...]` |

---

## Notes

- The `Datasets/Results/` CSV files are large (20–26 MB each) and are tracked in this repo. Consider adding them to `.gitignore` and hosting them separately (e.g. via Git LFS or a data registry) if the repo grows.
- Trained model weights (`Networks/*.pth`) are not tracked by git. Re-generate them by running the training scripts.
- All scripts use `if __name__ == "__main__":` guards, making them safe to import and safe for multiprocessing on Windows.
- File paths in `Generators/` and `Neural Networks/Models/config.py` are constructed with `os.path` for cross-platform compatibility.