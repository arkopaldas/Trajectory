# Trajectory

A research notebook for trajectory planning using formal methods (SMT/MILP) and neural network imitation learning, applied to two robot motion models: a **Simple (double integrator)** model and a **Unicycle** model.

---

## Project Structure

```text
Codebase/                                 # Main Workspace Root
│
├── Controllers/                         # Core Optimisation Modules
│   ├── config.py                        # Constants for individual instance generation
│   ├── milp_unicycle.py                 # Single-instance MILP path solver
│   └── smt_simple.py                    # Single-instance SMT path solver
│
├── Generators/                          # Batch Dataset Synthesizers
│   ├── config.py                        # Mass synthesis paths and parallel worker constraints
│   ├── simple.py                        # Synthesizes bulk trajectories for Simple Models
│   └── unicycle.py                      # Synthesizes bulk trajectories for Unicycle Models
│
├── Experiments/                         # Advanced Research Frameworks
│   ├── config.py                        # Experimental presets (RL bins, Halfspace models)
│   ├── milp_fixed_unicycle.py           # MILP path solver with randomized start regions
│   ├── milp_simple.py                   # MILP shape-agnostic halfspace solver with circle shapes
│   ├── smt_unicycle.py                  # Piecewise-angle nonholonomic SMT model
│   └── rl_unicycle.py                   # Reinforcement Learning (Q-learning) with dynamic trail rendering
│
├── Networks/                            # Saved Weights Root (Outside Neural Networks folder!)
│   └── N Trained/                       # Segmented by DATASET_SIZE configuration
│       ├── Simple.onnx                  # Dependency-free serialized simple inference graph
│       └── Unicycle.onnx                # Dependency-free serialized unicycle inference graph
│
├── Datasets/                            # Input schemas and parsed database records
│   ├── Schemas/                         # Input entry-point coordinate schemas (.csv)
│   └── Results/                         # Synthesized multi-instance solving outputs (.csv)
│
├── Verifier/                            # Safety verification of trained networks (Marabou + simulation)
│   ├── config.py                        # Safety switches, sampling size, Marabou settings, paths
│   ├── utils.py                         # Shared engine: ONNX loader, classifier, Marabou encoding, CSV writer
│   ├── verify_simple.py                 # Verifies Simple.onnx networks
│   ├── verify_unicycle.py               # Verifies Unicycle.onnx networks
│   └── selftest.py                      # Marabou-free consistency test of the query encoding
│
├── Verification/                        # Verifier outputs: unsafe-trajectory CSVs, reports, summaries (created on run)
│   └── N Trained/
│
└── Neural Networks/                     # Deep Learning Compilation Pipeline
    ├── config.py                        # Master training config (Learning rates, Adaptive Presets)
    ├── dataset.py                       # High-performance PyTorch Datasets & Dataloaders
    ├── utils.py                         # Reusable mathematical transitions, loaders, & plotting
    ├── train_all.py                     # Concurrent multiprocessing training executor
    ├── train_simple.py                  # PyTorch trainer and ONNX exporter for SimpleNet
    ├── train_unicycle.py                # PyTorch trainer and ONNX exporter for UnicycleNet
    ├── predict_simple.py                # ONNX-driven rollout simulator & vector field comparison
    ├── predict_unicycle.py              # ONNX-driven rollout simulator & vector field comparison
    └── Arch/                            # Cleaned structural neural topologies
        ├── __init__.py                  
        ├── simple_net.py                # SimpleNet model layer definitions
        └── unicycle_net.py              # UnicycleNet model layer definitions
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
4. **Neural Networks** — imitation learning, which trains on the generated CSV data, then predicts trajectories at inference time.

---

## Dependencies

Install all dependencies via Python PIP:

```bash
pip install z3-solver pulp torch numpy pandas matplotlib tqdm highspy
```

| Package      | Purpose                                                  |
| ------------ | -------------------------------------------------------- |
| `z3-solver`  | SMT solver (Z3) used in simple/unicycle SMT planners     |
| `pulp`       | MILP modelling; uses HiGHS (preferred) or CBC (fallback) |
| `torch`      | Neural network training and inference (PyTorch)          |
| `numpy`      | Numerical arrays                                         |
| `pandas`     | CSV I/O for datasets                                     |
| `matplotlib` | Trajectory and velocity field plotting                   |
| `tqdm`       | Progress bars during batch generation and result writing |
| `highspy`    | MILP solver dependency for HiGHS (optional to install)   |

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
Results are written to `Datasets/Results/* Trajectories`. The number of instances is controlled by `max_instances` at the top of each generator file.

### 3. Train a neural network
Run from inside the `Neural Networks/` directory so that the relative `Models.*` imports resolve correctly:
```bash
cd "Neural Networks"
python train_simple.py
python train_unicycle.py
```
Trained weights are saved to `Neural Networks/Networks/`.

#### Training Configuration
| Dataset | Batch   | Epochs | Notes             | Time Taken |
| ------- | ------- | ------ | ----------------- | ---------- |
| 10      | 4–8     | 1000   | High Overfit Risk | 3–4 mins   |
| 100     | 16–32   | 500    | Overfit Risk      |            |
| 1000    | 32–64   | 250    | Stable            |            |
| 10000   | 64–128  | 100    | Efficient         | 8–15 mins  |
| 100000  | 128–256 | 50     | Fast              |            |

### 4. Run inference
```bash
cd "Neural Networks"
python predict_simple.py
python predict_unicycle.py
```
A random trajectory is sampled from the result CSV and the model's predicted trajectory is plotted alongside the ground truth.

---

<!--## Verifier

`Verifier/` checks whether a trained network produces SAFE trajectories from the start region. `verify_simple.py` and `verify_unicycle.py` share the engine in `utils.py`; every switch and tolerance is in `Verifier/config.py`.

```bash
cd Verifier
pip install maraboupy onnx          # maraboupy is only needed for the formal stage
python verify_simple.py                                # all "<N> Trained" folders under ../Networks + summary table
python verify_unicycle.py "../Networks/1000 Trained"   # one folder
python verify_simple.py --samples 5000 --formal-steps 2 --timeout 60
python verify_simple.py --no-formal                    # simulation stage only
python verify_simple.py --marabou-only [--replay]      # no simulation: full-horizon Marabou only
python selftest.py                                     # checks the Marabou encoding without Marabou
```

**Classification.** A trajectory is UNSAFE if it enters an obstacle, leaves the workspace, never reaches the goal (`GOAL_MODE`: `any` step or `final` step) or exceeds the control limits. Entering the margin is allowed unless `CHECK_MARGIN = True`. Each check can be switched off in `config.py`.

**Verdict.** UNSAFE as soon as one violating trajectory exists (simulation or a Marabou SAT counterexample that is reproduced by simulation). SAFE only when every Marabou query is UNSAT over the whole start region and covers all T steps; otherwise `SAFE (partial proof)` or `INCONCLUSIVE`.

**Marabou-only mode** (`--marabou-only` or `MARABOU_ONLY = True`). No simulation. Two variants (`--marabou-mode`):
- `points` (default): N start states are sampled (`MARABOU_ONLY_SAMPLES`, or `--samples`). Each is pinned in its own full-horizon Marabou query. SAT is a violating trajectory, read from the assignment (exact here, since the start is fixed). UNSAT on every property is safe. The percentage is unsafe starts / decided starts, as in the simulation stage. This mode cannot claim SAFE for the whole region.
- `region`: the start region is split into cells; UNSAT cells are proved safe and SAT cells give counterexamples. The percentage is the share of the start region (area / volume) in counterexample cells, plus the proved and undecided shares. This is the mode that can return SAFE.

**Time limits.** `MARABOU_TIMEOUT` caps every query, and `MAX_RUNTIME_SECONDS` caps the whole Marabou stage per network. When the budget ends, the script stops cleanly and reports the remaining starts / cells as not run or undecided. `--workers`, `--samples` and `--max-runtime` trade coverage against time. `INTERVAL_PREFILTER` is off in point mode, so every verdict comes from Marabou.

**Outputs** (`Verification/<N> Trained/`): `<Model> Model Unsafe Trajectories.csv` (dataset schema, `STATUS = UNSAFE`), `<Model> Model Report.csv` (flags for every sampled start), `<Model> Model Formal Cells.csv`, plus `Verification/<Model> Model Summary.csv`.-->

## Dataset Schema

### Simple Model
| Column                                                    | Description                                         |
| --------------------------------------------------------- | --------------------------------------------------- |
| `BOUND`                                                   | Time horizon                                        |
| `DT`                                                      | Time step                                           |
| `VMIN` / `VMAX`                                           | Velocity bounds                                     |
| `MARGIN`                                                  | Obstacle clearance margin                           |
| `START` / `GOAL`                                          | Bounding box regions `(x1, y1, x2, y2)`             |
| `OBSTACLES`                                               | List of obstacle rectangles                         |
| `OPTIMIZE`                                                | Whether path length was minimised                   |
| `STATUS`                                                  | `SAT` or `UNSAT`                                    |
| `TOTAL STEPS`                                             | Number of time steps `T`                            |
| `LENGTH`                                                  | Total path length                                   |
| `BUILD TIME (ms)` / `SOLVE TIME (ms)` / `TOTAL TIME (ms)` | Timing breakdown                                    |
| `DATA POINTS`                                             | Per-step trajectory data `[{t, x, y, vx, vy}, ...]` |

### Unicycle Model
Same as above, plus:

| Column            | Description                                       |
| ----------------- | ------------------------------------------------- |
| `WMIN` / `WMAX`   | Angular velocity bounds                           |
| `THMIN` / `THMAX` | Heading angle bounds                              |
| `DTHETA`          | Angle discretisation step                         |
| `SOLVER`          | Which solver was used (`HiGHS` or `CBC`)          |
| `DATA POINTS`     | Per-step data `[{t, x, y, v, omega, theta}, ...]` |

---

## Notes

- The `Datasets/Results/` CSV files are large (20–26 MB each) and are tracked in this repo. Consider adding them to `.gitignore` and hosting them separately (e.g. via Git LFS or a data registry) if the repo grows.
- Trained model weights (`Networks/*.pth`) are not tracked by git. Re-generate them by running the training scripts.
- All scripts use `if __name__ == "__main__":` guards, making them safe to import and safe for multiprocessing on Windows.
- File paths in `Generators/` and `Neural Networks/Models/config.py` are constructed with `os.path` for cross-platform compatibility.
