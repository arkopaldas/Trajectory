# Trajectory

A research codebase for **trajectory planning with formal methods (SMT / MILP)**, **neural-network imitation learning** and
**formal verification of the learned controllers**. Two robot motion models are studied:

- a **Simple** model: a point that is steered directly by its velocity (kinematic single integrator), and
- a **Unicycle** model: a robot that drives forward with speed `v` and turns with angular velocity `ω`.

The idea in one sentence: an optimiser (the *expert*) plans collision-free trajectories from many start points, a neural
network learns to imitate that expert (`state → control`), and the learned network is then **verified** in closed loop with
Marabou and, if it is unsafe, **fine-tuned with counterexamples (CEGIS)**.

---

## Project Structure

```text
Codebase/                                 # repository root
│
├── Controllers/                          # Single-instance solvers (quick checks, formulation tuning)
│   ├── config.py                         #   constants for the two solvers below
│   ├── smt_simple.py                     #   Z3 (SMT) planner, Simple model
│   └── milp_unicycle.py                  #   PuLP (MILP) planner, Unicycle model
│
├── Generators/                           # Batch dataset generators (parallel, many start points)
│   ├── config.py                         #   problem constants, number of instances, workers, file paths
│   ├── simple.py                         #   writes "Simple Model Result.csv"
│   ├── unicycle.py                       #   writes "Unicycle Model Result.csv"
│   └── progress.py                       #   live progress bar used by the generators
│
├── Datasets/                             # Data (see Datasets/README.md)
│   ├── Schemas/                          #   header-only CSVs that define the column layout of the datasets
│   ├── Results/<N> Trajectories/         #   generated expert datasets (created by Generators, not stored in git)
│   └── Counterexamples/<N> Trajectories/ #   unsafe trajectories found by the verifier
│
├── Neural Networks/                      # Imitation learning
│   ├── config.py                         #   training settings and per-dataset-size presets
│   ├── dataset.py                        #   CSV -> training windows -> PyTorch data loaders
│   ├── utils.py                          #   loss, physics, training / evaluation loops, ONNX export, plots, GIFs
│   ├── progress.py                       #   live progress bar (same file as in Generators/)
│   ├── train_simple.py                   #   trains SimpleNet and exports Simple.onnx
│   ├── train_unicycle.py                 #   trains UnicycleNet and exports Unicycle.onnx
│   ├── train_all.py                      #   runs both trainings in parallel
│   ├── predict_simple.py                 #   closed-loop rollout of Simple.onnx vs. the expert trajectory
│   ├── predict_unicycle.py               #   closed-loop rollout of Unicycle.onnx vs. the expert trajectory
│   └── Arch/                             #   network architectures
│       ├── base.py                       #     NormalizedNet: z-score normalisation + ONNX folding
│       ├── simple_net.py                 #     SimpleNet   (2 -> ... -> 2)
│       └── unicycle_net.py               #     UnicycleNet (3 -> ... -> 2)
│
├── Networks/<N> Trained/                 # Trained networks (ONNX), one folder per dataset size
│   ├── Simple.onnx   (+ Simple.onnx.data in the older folders)
│   └── Unicycle.onnx
│
├── Verification/                         # Formal verification (Marabou) and CEGIS fine-tuning (see its own README)
│   ├── config.py                         #   constraint switches, tolerances, start-cell grid, Marabou and CEGIS settings
│   ├── constraints.py                    #   the safety constraints, defined once
│   ├── utils.py                          #   environment from the CSV, ONNX reader, exact rollout, counterexample I/O
│   ├── marabou_encoder.py                #   closed-loop encoding for Marabou
│   ├── verifier.py                       #   cell-by-cell verification, SAFE / UNSAFE / UNKNOWN
│   ├── cegis.py                          #   counterexample-guided fine-tuning loop
│   ├── verify_simple.py / verify_unicycle.py   # entry points: verification
│   ├── cegis_simple.py  / cegis_unicycle.py    # entry points: CEGIS
│   ├── selftest.py                       #   Marabou-free consistency test of the encoding
│   └── Results/<N> Trained/              #   per-cell verification tables (created on run)
│
├── Experiments/                          # Stand-alone side studies (not part of the main pipeline)
│   ├── config.py                         #   constants of the experiments
│   ├── milp_simple.py                    #   MILP with half-space obstacles (rectangles, circles, polygons)
│   ├── milp_fixed_unicycle.py            #   unicycle MILP from one random start point
│   ├── smt_unicycle.py                   #   unicycle SMT model with heading bins (angles in degrees)
│   ├── rl_unicycle.py                    #   tabular Q-learning unicycle with an animated rollout
│   └── unicycle_trajectory.gif           #   example output of rl_unicycle.py
│
├── Instructions.md / Prompt.md           # Design brief / prompt that specified the Verification module
└── README.md
```

> Every script imports its neighbours with plain `from config import ...`, so **run each script from inside its own
> folder** (e.g. `cd "Neural Networks"` and then `python train_simple.py`).

---

## Models

### Simple model
A point that moves freely in 2D and is controlled by its velocity `(vx, vy)`:

```
x' = x + vx · dt        y' = y + vy · dt        |vx|, |vy| ≤ VMAX
```
The network takes the state `(x, y)` and predicts the control `(vx, vy)`. The expert encoding is exact (linear).

### Unicycle model
A robot with speed `v ∈ [0, VMAX]` and turn rate `ω ∈ [WMIN, WMAX]`:

```
x' = x + v · cos(θ) · dt      y' = y + v · sin(θ) · dt      θ' = θ + ω · dt
```
The network takes the state `(x, y, θ)` and predicts the control `(v, ω)`.

> **Approximations in the expert.** `cos` and `sin` cannot be used in an MILP, so the unicycle expert (`Generators/unicycle.py`,
> `Controllers/milp_unicycle.py`) cuts the heading into bins of width `DTHETA` (45°) and uses the cos / sin of the bin centre.
> The generated trajectories are therefore only approximately consistent with the exact kinematics. The neural-network
> training, the prediction scripts and the verifier all use the **exact** kinematics.

### Planning problem (all models)
A 100 × 100 workspace. The robot starts anywhere in the start region `(0, 0, 10, 10)`, must end in the goal region
`(90, 90, 100, 100)` and must stay at least `MARGIN` away from eight rectangular obstacles. Regions are always written
`(x_min, y_min, x_max, y_max)`. The path length that is minimised is the *Manhattan* length `Σ(|Δx| + |Δy|)`.

---

## Pipeline

```
Controllers / Experiments            (single-instance solvers for testing and tuning)
        │
        ▼
Generators ──► Datasets/Results/<N> Trajectories/*.csv ──► Neural Networks (train) ──► Networks/<N> Trained/*.onnx
                                                                                          │            │
                                                                       Neural Networks (predict)   Verification (Marabou)
                                                                                                       │
                                                                          SAFE ◄── UNSAFE ──► counterexamples ──► CEGIS
                                                                                                                    │
                                                                                      Networks/<N> Trained/<Model>_CEGIS.onnx
```

1. **Controllers** – standalone single-run solvers for quick verification and tuning of the expert formulation.
2. **Experiments** – extended single-run variants (half-space obstacles, fixed start point, SMT unicycle, reinforcement learning).
3. **Generators** – parallelised batch solvers that write thousands of trajectories to CSV (one row per start point).
4. **Neural Networks** – physics-informed behavioural cloning: the network predicts the control, the known dynamics turn it into
   the next state, and the loss compares that state with the next state of the expert trajectory.
5. **Verification** – closed-loop formal verification of the trained network with Marabou, and CEGIS fine-tuning if it is unsafe.

---

## Dependencies

Python 3.11 is recommended (the verification module was developed with it).

```bash
# expert solvers, data and training
pip install z3-solver pulp highspy numpy pandas matplotlib tqdm torch onnx onnxruntime pillow
# optional: colours of the progress bar in older Windows consoles
pip install colorama
# verification / CEGIS only
pip install maraboupy
```

| Package               | Purpose                                                                       |
| --------------------- | ----------------------------------------------------------------------------- |
| `z3-solver`           | SMT solver (Z3) for the Simple-model planner and the SMT unicycle experiment  |
| `pulp`, `highspy`     | MILP modelling; HiGHS is tried first, CBC (bundled with PuLP) is the fallback |
| `torch`               | Training and fine-tuning of the neural networks                               |
| `onnx`, `onnxruntime` | Export, reading and executing the trained networks                            |
| `numpy`, `pandas`     | Arrays and CSV I/O                                                            |
| `matplotlib`, `pillow`| Plots and rollout GIFs (Pillow writes the GIFs)                               |
| `tqdm`                | Base of the live progress bar in `progress.py`                                |
| `maraboupy`           | Marabou verifier (only needed in `Verification/`, not for `selftest.py`)      |

---

## Usage

### 1. Run a single controller (quick test)
```bash
cd Controllers
python smt_simple.py
python milp_unicycle.py
```
Each script solves one problem, shows a plot and prints a timing summary. Settings are in `Controllers/config.py`.

### 2. Generate a dataset
```bash
cd Generators
python simple.py
python unicycle.py
```
The number of start points is `MAX_INSTANCES` in `Generators/config.py` (it also names the output folder).
Results are written to `Datasets/Results/<MAX_INSTANCES> Trajectories/<Model> Model Result.csv`.
`NUM_WORKERS` (default: all CPU cores) controls the parallelism; lower it on a weak machine.

### 3. Train a neural network
```bash
cd "Neural Networks"
python train_simple.py
python train_unicycle.py
python train_all.py          # both at once, in parallel
```
Set `DATASET_SIZE` in `Neural Networks/config.py` to the size of the dataset you generated. It selects the dataset folder,
the output folder (`Networks/<N> Trained/`) and the hyper-parameter preset:

| Dataset size | Batch | Learning rate | Epochs |
| ------------ | ----- | ------------- | ------ |
| 10           | 32    | 1e-4          | 350    |
| 100          | 64    | 3e-4          | 250    |
| 1000         | 128   | 5e-4          | 150    |
| 10000        | 256   | 1e-3          | 100    |
| 100000       | 512   | 2e-3          | 60     |

(`BATCH_SIZE`, `LR` and `EPOCHS` of `TRAINING_PRESETS` are used. The `PATIENCE` and `SCHEDULER` entries are listed in the
config but are not implemented by the training loop, which uses plain Adam.)

### 4. Run inference
```bash
cd "Neural Networks"
python predict_simple.py
python predict_unicycle.py
```
A random trajectory is taken from the dataset; the network drives the robot in closed loop from the same start state. The
predicted path is compared with the expert path (plots) and an animation `<Model>_rollout.gif` is saved next to the model.

### 5. Verify a network and fine-tune it
```bash
cd Verification
pip install maraboupy
python selftest.py            # checks the encoding without Marabou
python verify_simple.py       # SAFE / UNSAFE / UNKNOWN
python verify_unicycle.py
python cegis_simple.py        # fine-tune with counterexamples until SAFE
python cegis_unicycle.py
```
No script takes arguments; everything is set in `Verification/config.py`. See [`Verification/README.md`](Verification/README.md)
for how the verification works, the safety constraints, the outputs and the CEGIS loss.

### 6. Experiments
```bash
cd Experiments
python milp_simple.py
python milp_fixed_unicycle.py
python smt_unicycle.py
python rl_unicycle.py [--save-gif]
```

---

## Dataset Schema

The generators write one **row per start point**. The column layout is defined by the header-only files in
`Datasets/Schemas/`; each column is explained in [`Datasets/Schemas/README.md`](Datasets/Schemas/README.md) and the
counterexample files in [`Datasets/Counterexamples/README.md`](Datasets/Counterexamples/README.md). In short:

### Simple model
| Column                                                    | Description                                                                                 |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `BOUND`                                                   | Time horizon in seconds                                                                     |
| `DT`                                                      | Time step                                                                                   |
| `VMIN` / `VMAX`                                           | Velocity bounds (per axis)                                                                  |
| `MARGIN`                                                  | Obstacle clearance margin                                                                   |
| `START` / `GOAL`                                          | Start and goal regions `(x1, y1, x2, y2)`                                                   |
| `OBSTACLES`                                               | List of obstacle rectangles                                                                 |
| `OPTIMIZE`                                                | Whether the path length was minimised                                                       |
| `STATUS`                                                  | `SAT` or `UNSAT`                                                                            |
| `TOTAL STEPS`                                             | Number of time steps `T`                                                                    |
| `LENGTH`                                                  | Total (Manhattan) path length                                                               |
| `BUILD TIME (ms)` / `SOLVE TIME (ms)` / `TOTAL TIME (ms)` | Timing breakdown                                                                            |
| `DATA POINTS`                                             | Per-step data `[{t, x, y, vx, vy}, ...]` (stored as text; `None` controls at the last step) |

### Unicycle model
Same as above, plus:

| Column            | Description                                                                        |
| ----------------- | ---------------------------------------------------------------------------------- |
| `WMIN` / `WMAX`   | Angular velocity bounds (rad/s)                                                    |
| `THMIN` / `THMAX` | Heading bounds (rad)                                                               |
| `DTHETA`          | Width of the heading bins used by the MILP approximation (rad)                     |
| `SOLVER`          | Solver that produced the row (`HiGHS` or `CBC`; `Marabou` in counterexample files) |
| `STATUS`          | PuLP status (`Optimal`, `Infeasible`, ...) instead of `SAT` / `UNSAT`              |
| `DATA POINTS`     | Per-step data `[{t, x, y, v, omega, theta}, ...]`                                  |

---

## Notes

- **Datasets are not stored here.** The generated `Datasets/Results/` CSVs are large (up to 100 000 trajectories);
  generate them with the scripts in `Generators/`, or host them separately (Git LFS, a release asset, a data registry).
  GitHub rejects single files above 100 MB.
- **ONNX external data.** `Simple.onnx` in the folders for 10 – 10000 trajectories stores its weights in `Simple.onnx.data`
  (older exporter); keep the two files next to each other. The newer `Unicycle.onnx` files and
  `100000 Trained/Simple.onnx` are self-contained (the current export in `Neural Networks/utils.py`); the matching
  `Unicycle.onnx.data` files in the older folders are left-overs and are not read by the `.onnx` file.
- **Matching sizes.** `DATASET_SIZE` in `Neural Networks/config.py` and in `Verification/config.py`, and `MAX_INSTANCES` in
  `Generators/config.py`, must agree so that the scripts find the right dataset and network folders.
- The batch / training / verification entry scripts (`Generators/`, `Neural Networks/train_*`, `predict_*`, `Verification/verify_*`, `cegis_*`) use `if __name__ == "__main__":` guards, which makes them safe for `multiprocessing` on Windows. The single-run scripts in `Controllers/` and `Experiments/` (except `rl_unicycle.py`) execute immediately when started and are meant to be run directly, not imported.
- Paths are built with `os.path`, so the code runs on Windows, Linux and macOS.
- The constants of the planning problem exist in three places that are **not linked**: `Controllers/config.py`,
  `Generators/config.py` and `Experiments/config.py`. The verifier reads none of them; it reads the problem from the
  first row of the dataset CSV, so it always checks what the network was trained for.

### Known limitations
- The unicycle expert data uses the binned (approximate) `cos` / `sin`; see *Models*.
- `Experiments/smt_unicycle.py`: the helper `angle_based_expr` is given `x[t]` as base for both the `x` and the `y` update, so the
  `y` constraint reads `y[t+1] == x[t] + v·sin(θ)·dt`. Check this before relying on that experiment.
- `Experiments/milp_simple.py`: for circles and polygons the half-space normals are not normalised, so `MARGIN` is not an exact
  Euclidean distance there (it is for rectangles). Circles are approximated by an inscribed 8-gon.
- Several configuration entries and helpers are kept for reference but are currently unused: `PATIENCE` / `SCHEDULER`,
  `MAX_OBSTACLES` / `encode_obstacles`, `ROLLOUT_STEPS` / `build_*_sequences` (all in `Neural Networks`).
