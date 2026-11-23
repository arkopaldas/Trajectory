# Datasets

Data used and produced by the codebase. This folder is the `Datasets/` directory of the repository (the code refers to it as
`../Datasets` from `Generators/`, `Neural Networks/` and `Verification/`).

```text
Datasets/
├── Schemas/                              # column layout of the datasets (header-only CSVs)   -> Schemas/README.md
│   ├── Simple Model Dataset Schema.csv
│   └── Unicycle Model Dataset Schema.csv
├── Counterexamples/                      # unsafe trajectories found by the verifier           -> Counterexamples/README.md
│   └── <N> Trajectories/
│       └── <Model> Model Counterexamples.csv
└── Results/                              # generated expert datasets (NOT included, see below)
    └── <N> Trajectories/
        ├── Simple Model Result.csv
        └── Unicycle Model Result.csv
```

`<N>` is the number of trajectories (start points) in the dataset: 10, 100, 1000, 10000 or 100000.

## Who reads and writes what

| Folder            | Written by                                             | Read by                                                                                      |
| ----------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `Schemas/`        | by hand (fixed)                                        | `Generators/simple.py`, `Generators/unicycle.py` (column layout of the result files)         |
| `Results/`        | `Generators/simple.py`, `Generators/unicycle.py`       | `Neural Networks/train_*.py`, `predict_*.py`, `Verification/` (problem constants), `cegis.py` |
| `Counterexamples/`| `Verification/verifier.py` (`utils.save_counterexamples`) | `Verification/cegis.py` (start states for fine-tuning)                                   |

## The `Results` folder is not included

The generated datasets are large (the 100 000-trajectory files are far above GitHub's 100 MB file limit). Create them with

```bash
cd Generators
python simple.py
python unicycle.py
```

after setting `MAX_INSTANCES` in `Generators/config.py`; the files appear in `Datasets/Results/<MAX_INSTANCES> Trajectories/`.
Or keep them outside git (Git LFS, a release asset, a data registry).

## Common data format: one row = one trajectory

Every dataset (expert results and counterexamples) has **one row per trajectory**. The columns
describe the problem (time step, bounds, regions, obstacles), the solver outcome (status, length, timings) and the whole
trajectory in the last column, `DATA POINTS`.

`DATA POINTS` is a Python list of dicts stored **as text** in the CSV. Each dict is one time step `t = 0 .. TOTAL STEPS`:

| Model    | Keys of each data point                                                           |
| -------- | --------------------------------------------------------------------------------- |
| Simple   | `t`, `x`, `y`, `vx`, `vy`                                                         |
| Unicycle | `t`, `x`, `y`, `v`, `omega`, `theta`                                              |

- `(x, y[, theta])` is the **state** at step `t`; the other keys are the **control** that is applied at step `t`.
- A trajectory has `TOTAL STEPS + 1` data points: the last state has no control, so its control entries are `None`.
- The state at `t = 0` is the start state of the trajectory (every row of a dataset has a different one).

Reading a file:

```python
import ast
import pandas as pd

df = pd.read_csv("Datasets/Results/1000 Trajectories/Simple Model Result.csv")
row = df[df["STATUS"] == "SAT"].iloc[0]
points = ast.literal_eval(row["DATA POINTS"])     # list of dicts
start = (points[0]["x"], points[0]["y"])
```

The columns `START`, `GOAL` and `OBSTACLES` are also stored as text (`"(0, 0, 10, 10)"`, `"[(20.0, 20.0, 30.0, 30.0), ...]"`);
parse them with `ast.literal_eval` as well. Boxes are always `(x_min, y_min, x_max, y_max)`.
