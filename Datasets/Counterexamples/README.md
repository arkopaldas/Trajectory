# Counterexamples

Unsafe trajectories that the formal verifier (`Verification/`) found in trained neural-network controllers.

```text
Counterexamples/
└── <N> Trajectories/                       # <N> = size of the dataset the network was trained on
    └── <Model> Model Counterexamples.csv   # Simple / Unicycle
```

The file shipped here, `1000 Trajectories/Simple Model Counterexamples.csv`, contains **2 counterexamples** for the Simple
network trained on 1000 trajectories (`Networks/1000 Trained/Simple.onnx`).

## What a counterexample is

The verifier rolls the trained network out **in closed loop** (`state -> network -> control -> dynamics -> next state ...`)
from start states in the start region. A *counterexample* is a start state for which this closed-loop trajectory
**violates an active safety constraint**. The constraints are switched on or off in `Verification/config.py`
(`CHECK_OBSTACLES`, `CHECK_CONTROL_BOUNDS`, `CHECK_WORKSPACE`, `CHECK_THETA_BOUNDS`, `GOAL_MODE`).
Marabou proposes a start state; the verifier then replays it with the real network (onnxruntime) and the exact dynamics and
stores it only if the replayed trajectory really violates a constraint.

For the two stored examples only the **obstacle** constraint was active: a state must stay out of every obstacle inflated by `MARGIN`.

## File format

The file uses **exactly the same columns as the expert dataset** (`Datasets/Schemas/Simple Model Dataset Schema.csv`,
described in `../Schemas/README.md`), so counterexamples can be loaded, inspected and plotted with the same code as normal
trajectories. Rows are **appended** every time the verifier finds a counterexample; the file is never overwritten, so it can
contain counterexamples of several runs (and a start state may appear more than once).

Compared with an expert row, these columns have a different meaning:

| Column | In a counterexample row |
| --- | --- |
| `STATUS` | Always `COUNTEREXAMPLE`. |
| `SOLVER` (Unicycle files only) | `Marabou`. |
| `TOTAL STEPS` | Number of closed-loop steps `T` that were simulated (the horizon of the problem, `ceil(BOUND / DT)`). |
| `LENGTH` | Manhattan path length `sum(abs(dx) + abs(dy))` of the simulated trajectory. |
| `BUILD TIME (ms)` | Time spent building the Marabou query that produced the counterexample. |
| `SOLVE TIME (ms)` | Time Marabou needed to solve it. |
| `TOTAL TIME (ms)` | Build + solve time. |
| `DATA POINTS` | The **replayed closed-loop trajectory of the network** (not an expert trajectory): `[{t, x, y, vx, vy}, ...]` with `T + 1` entries. The control at step `t` is the network's output for the state at step `t`; the last entry has `vx = vy = None`. The **first** entry is the counterexample start state. |

All other columns (`BOUND`, `DT`, `VMIN`, `VMAX`, `MARGIN`, `START`, `GOAL`, `OBSTACLES`, `OPTIMIZE`) are copied from the first row of the
expert dataset, so each row is self-contained: it carries the problem it was found for. As in the expert data, `START` is the
whole start region, not the start point.

## The two stored counterexamples

Both rows: `BOUND = 60`, `DT = 2.5`, `VMIN/VMAX = -2.5/2.5`, `MARGIN = 2.5`, `TOTAL STEPS = 24`, goal `(90, 90, 100, 100)`, 8 obstacles.

| Row | Start state `(x, y)` | `LENGTH` | Violation (obstacle constraint) | Marabou solve time |
| --- | --- | --- | --- | --- |
| 0 | `(1.0160, 0.6644)` | 174.26 | The path first enters the inflated zone of obstacle `(20, 20, 30, 30)` at step 3. The deepest intrusion is 2.35 units into the inflated zone of obstacle `(50, 40, 60, 50)` (step 12, at `(58.27, 39.85)`); the obstacle itself is not touched. | about 23 s |
| 1 | `(0.3493, 0.9249)` | 174.70 | The path first enters the inflated zone of obstacle `(20, 20, 30, 30)` at step 3 and goes **into the obstacle itself** (step 6, at `(29.39, 23.27)`). The deepest intrusion is 3.11 units, which is what `Verification/Results/1000 Trained/Simple Model Verification.csv` reports for start cell 0 (`VIOLATIONS = {'obstacle': 3.1115}`). | about 1121 s |

Intrusion depth is measured as in `Verification/constraints.py` (`obstacle_depths`): the smaller of the distances to the
nearest x-side and y-side of the inflated rectangle; a value above 0 means unsafe.

Other things visible in these rows (not violations, because the corresponding checks were off in that run):
- both trajectories end near `(88.2, 87.7)`, just outside the goal region (`GOAL_MODE = "off"`), and
- the network's velocities slightly exceed the bound `VMAX = 2.5` (largest absolute values 2.503 and 2.509), because the network
  output is not clipped (`CHECK_CONTROL_BOUNDS = False`).

## How the files are used

- `Verification/cegis.py` takes the **first data point** (the start state) of every row and fine-tunes the network so that
  trajectories from those start states become safe. Counterexamples of all rounds are reused in every round.
- They are also useful for debugging and plotting: reading a row works exactly like reading an expert row.

```python
import ast
import pandas as pd

df = pd.read_csv("Datasets/Counterexamples/1000 Trajectories/Simple Model Counterexamples.csv")
row = df.iloc[1]
points = ast.literal_eval(row["DATA POINTS"])        # [{'t': 0, 'x': ..., 'y': ..., 'vx': ..., 'vy': ...}, ...]
start_state = (points[0]["x"], points[0]["y"])       # the counterexample
obstacles = ast.literal_eval(row["OBSTACLES"])
```
