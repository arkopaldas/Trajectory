# Schemas

Header-only CSV files that define the **column layout** of the expert datasets. They contain no data rows.

| File                               | Model    | Columns |
| ---------------------------------- | -------- | ------- |
| `Simple Model Dataset Schema.csv`  | Simple   | 16      |
| `Unicycle Model Dataset Schema.csv`| Unicycle | 22      |

## How they are used

`Generators/simple.py` and `Generators/unicycle.py` read the schema with `pandas.read_csv`, add one row per generated
trajectory and write the result to `Datasets/Results/<N> Trajectories/<Model> Model Result.csv`. The schema therefore fixes
the **name and the order of the columns** of every result file (and of the counterexample files, which reuse the same columns).
Do not rename or reorder the columns without changing the generators, `Neural Networks/` and `Verification/`, which access them by name.

> **Why the CSV files have no comments.** CSV has no comment syntax, and the generators load these files with a plain
> `pd.read_csv`; a comment line would be read as the header and break the pipeline. The documentation for the columns lives here instead.

## Simple model (`Simple Model Dataset Schema.csv`)

Column order:
`BOUND, DT, VMIN, VMAX, MARGIN, START, GOAL, OBSTACLES, OPTIMIZE, STATUS, TOTAL STEPS, LENGTH, BUILD TIME (ms), SOLVE TIME (ms), TOTAL TIME (ms), DATA POINTS`

| Column | Type | Meaning | Default (`Generators/config.py`) |
| --- | --- | --- | --- |
| `BOUND` | float | Planning horizon in seconds. The number of steps is `ceil(BOUND / DT)`. | `60.0` |
| `DT` | float | Length of one time step in seconds. | `2.5` |
| `VMIN`, `VMAX` | float | Lower / upper bound of the velocity of **each axis** (`vx`, `vy`). | `-2.5`, `2.5` |
| `MARGIN` | float | Clearance kept around every obstacle: the obstacle rectangle is inflated by this distance on every side. | `2.5` |
| `START` | text, box | The **start region** `(x_min, y_min, x_max, y_max)` from which the start points were drawn. It is the same in every row; the actual start point of a row is the first entry of `DATA POINTS`. | `(0, 0, 10, 10)` |
| `GOAL` | text, box | The goal region the trajectory has to end in. | `(90, 90, 100, 100)` |
| `OBSTACLES` | text, list of boxes | Rectangular obstacles `(x_min, y_min, x_max, y_max)`; there are 8. | see `Generators/config.py` |
| `OPTIMIZE` | bool | `True`: the path length was minimised; `False`: any feasible trajectory. | `True` |
| `STATUS` | text | Solver outcome: `SAT` (trajectory found) or `UNSAT` (none). `COUNTEREXAMPLE` in the counterexample files. | `SAT` / `UNSAT` |
| `TOTAL STEPS` | int | Number of time steps `T` of the trajectory. | `24` |
| `LENGTH` | float | Total path length `sum(abs(dx) + abs(dy))` over all steps (Manhattan length, not Euclidean). Empty for `UNSAT`. | |
| `BUILD TIME (ms)` | float | Time to build the solver model. | |
| `SOLVE TIME (ms)` | float | Time spent in the solver. | |
| `TOTAL TIME (ms)` | float | Build + solve time. | |
| `DATA POINTS` | text, list of dicts | The trajectory: `[{t, x, y, vx, vy}, ...]` with `T + 1` entries. `vx`, `vy` of the last entry are `None`. Empty list for `UNSAT`. | |

Dynamics of the Simple model: `x' = x + vx * DT`, `y' = y + vy * DT`.

## Unicycle model (`Unicycle Model Dataset Schema.csv`)

Column order:
`BOUND, DT, VMIN, VMAX, WMIN, WMAX, THMIN, THMAX, DTHETA, MARGIN, START, GOAL, OBSTACLES, OPTIMIZE, STATUS, SOLVER, TOTAL STEPS, LENGTH, BUILD TIME (ms), SOLVE TIME (ms), TOTAL TIME (ms), DATA POINTS`

The columns `BOUND`, `DT`, `VMIN`, `VMAX`, `MARGIN`, `START`, `GOAL`, `OBSTACLES`, `OPTIMIZE`, `TOTAL STEPS`, `LENGTH`, the three timing
columns and `DATA POINTS` mean the same as above, except for the following.

| Column | Type | Meaning | Default (`Generators/config.py`) |
| --- | --- | --- | --- |
| `BOUND` | float | Planning horizon in seconds (longer than for the Simple model). | `70.0` |
| `DT` | float | Time step in seconds. | `3.0` |
| `VMIN`, `VMAX` | float | Bounds of the forward speed `v` (no reversing). | `0.0`, `3.0` |
| `WMIN`, `WMAX` | float | Bounds of the turn rate `omega` in **rad/s**. | `-pi/2`, `pi/2` (-90 / +90 deg/s) |
| `THMIN`, `THMAX` | float | Range of the heading `theta` in **rad**; also the range of start headings used by the verifier. | `-0.3927`, `1.9635` (-22.5 / 112.5 deg) |
| `DTHETA` | float | Width of one heading bin in **rad**. The MILP replaces `cos(theta)` / `sin(theta)` by the value at the centre of the bin that contains `theta`, so the expert trajectories are an **approximation** of the exact unicycle. | `0.7854` (45 deg, i.e. 3 bins) |
| `STATUS` | text | The PuLP status string: `Optimal`, `Infeasible`, `Not Solved`, ... (`Optimal` / `Feasible` rows contain a trajectory). `COUNTEREXAMPLE` in counterexample files. | |
| `SOLVER` | text | Solver that produced the row: `HiGHS`, or `CBC` if HiGHS failed. `Marabou` in counterexample files. | |
| `DATA POINTS` | text, list of dicts | `[{t, x, y, v, omega, theta}, ...]` with `T + 1` entries; `v`, `omega` of the last entry are `None`. | |

Dynamics of the Unicycle model (exact, as used by the network training and the verifier):
`x' = x + v * cos(theta) * DT`, `y' = y + v * sin(theta) * DT`, `theta' = theta + omega * DT` (heading wrapped to `[-pi, pi]`
in training and verification).

## Differences between the two schemas

- Unicycle adds `WMIN`, `WMAX`, `THMIN`, `THMAX`, `DTHETA` (turn rate, heading range, heading bins) and `SOLVER`.
- The `STATUS` values differ because the two experts use different solvers: Z3 (`SAT` / `UNSAT`) and PuLP (`Optimal`, ...).
- The control and state keys inside `DATA POINTS` differ (`vx`, `vy` versus `v`, `omega`, `theta`).
