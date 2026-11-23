"""Single-instance MILP planner for the UNICYCLE model (forward speed v and turn rate omega).

True model:   x' = x + v cos(theta) dt,   y' = y + v sin(theta) dt,   theta' = theta + omega dt
MILP model:   cos(theta) and sin(theta) cannot be used in a linear program, so the heading is cut into bins of width
              DTHETA. In each step exactly one bin is selected (binary s) and the bin-centre cos/sin is used.
              The step is therefore only an APPROXIMATION of the true dynamics, which is why the datasets produced
              from this formulation have approximate headings (the neural-network training and the verifier use the
              true dynamics instead).
Problem:      start in START_REGION, end in GOAL_REGION, keep MARGIN away from every obstacle, respect the control
              and heading bounds, minimise the (Manhattan) path length if OPTIMIZE.
Solver:       PuLP with HiGHS (preferred) and CBC as fallback.

Run:          python milp_unicycle.py      (shows a matplotlib figure and prints a timing summary)
Related:      Generators/unicycle.py is the same formulation wrapped in a function for many start points.
"""
import math
import time
import pulp
import numpy as np
import matplotlib.pyplot as plt
from pulp import PulpSolverError

# Problem constants come from the local config.py.
from config import (
    OPTIMIZE, MARGIN, TIME_BOUND, DEG2RAD, START_REGION, GOAL_REGION, OBSTACLES,
    MILP_DT as DT, MILP_T as T, MILP_DTHETA as DTHETA,
    MILP_VMIN as VMIN, MILP_VMAX as VMAX,
    MILP_WMIN as WMIN, MILP_WMAX as WMAX,
    MILP_THMIN as THMIN, MILP_THMAX as THMAX
)

# Wall-clock start, used for the timing summary at the bottom of the script.
start_time = time.time()

# Heading bins: the centre of every bin of width DTHETA between THMIN and THMAX, and the cos/sin evaluated once at
# every centre. K is the number of bins (3 with the default settings: 0, 45 and 90 degrees).
theta_mids = [THMIN + DTHETA/2.0 + i*DTHETA for i in range(int((THMAX - THMIN) / DTHETA))]
cos_table = {mid: math.cos(mid) for mid in theta_mids}
sin_table = {mid: math.sin(mid) for mid in theta_mids}
K = len(theta_mids)

# Big-M constants for the 'either-or' (disjunctive) constraints below. They only have to be larger than any value
# the constrained expression can take: the workspace size for x and y, one full turn for theta.
Mx = max(GOAL_REGION[2], START_REGION[2])
My = max(GOAL_REGION[3], START_REGION[3])
Mtheta = 2 * math.pi

# ---- Decision variables ----
prob = pulp.LpProblem("MILP_Path_Planner", pulp.LpMinimize)

# x_t, y_t: positions (t = 0..T) inside the workspace [0, goal x2] x [0, goal y2].
# v_t, omega_t: controls (t = 0..T-1). theta_t: heading (t = 0..T), not wrapped here.
x = {t: pulp.LpVariable(f"x_{t}", lowBound=0.0, upBound=GOAL_REGION[2]) for t in range(T+1)}
y = {t: pulp.LpVariable(f"y_{t}", lowBound=0.0, upBound=GOAL_REGION[3]) for t in range(T+1)}
v = {t: pulp.LpVariable(f"v_{t}", lowBound=VMIN, upBound=VMAX) for t in range(T)}
omega = {t: pulp.LpVariable(f"omega_{t}", lowBound=WMIN, upBound=WMAX) for t in range(T)}
theta = {t: pulp.LpVariable(f"theta_{t}", lowBound=THMIN, upBound=THMAX) for t in range(T+1)}

# s[t,k] = 1 when the heading at step t lies in bin k (exactly one bin per step).
# u[t,k] = v[t] when bin k is selected and 0 otherwise. This is how the product v * cos(theta_bin) stays linear.
s = {(t,k): pulp.LpVariable(f"s_{t}_{k}", cat="Binary") for t in range(T) for k in range(K)}
u = {(t,k): pulp.LpVariable(f"u_{t}_{k}", lowBound=VMIN, upBound=VMAX) for t in range(T) for k in range(K)}

# adx_t, ady_t >= |x_(t+1) - x_t| and |y_(t+1) - y_t|; they are only used for the objective.
adx = {t: pulp.LpVariable(f"adx_{t}", lowBound=0.0) for t in range(T)}
ady = {t: pulp.LpVariable(f"ady_{t}", lowBound=0.0) for t in range(T)}

# ---- Boundary conditions ----
sx1, sy1, sx2, sy2 = START_REGION
gx1, gy1, gx2, gy2 = GOAL_REGION
prob += x[0] >= sx1
prob += x[0] <= sx2
prob += y[0] >= sy1
prob += y[0] <= sy2

# The last state must lie in the goal region.
prob += x[T] >= gx1
prob += x[T] <= gx2
prob += y[T] >= gy1
prob += y[T] <= gy2

# ---- Per-step constraints ----
for t in range(T+1):
    # Workspace box and heading limits for every state.
    prob += x[t] >= START_REGION[0]
    prob += x[t] <= GOAL_REGION[2]
    prob += y[t] >= START_REGION[1]
    prob += y[t] <= GOAL_REGION[3]
    prob += theta[t] >= THMIN
    prob += theta[t] <= THMAX

    # Obstacle avoidance (big-M). For each obstacle and each step four binaries b1..b4 are created and exactly one of them
    # must be 0 (their sum is 3). A binary equal to 0 switches ON the constraint it belongs to, a binary equal to 1
    # relaxes it by the big-M. So the point must be (b1) left of, (b2) right of, (b3) below or (b4) above the obstacle,
    # in each case by at least MARGIN.
    for (x1,y1,x2,y2) in OBSTACLES:
        b1 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b1", cat="Binary")
        b2 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b2", cat="Binary")
        b3 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b3", cat="Binary")
        b4 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b4", cat="Binary")
        # Exactly one of the four 'sides' is enforced.
        prob += b1 + b2 + b3 + b4 == 3
        prob += x[t] <= (x1 - MARGIN) + Mx * (b1)
        prob += x[t] >= (x2 + MARGIN) - Mx * (b2)
        prob += y[t] <= (y1 - MARGIN) + My * (b3)
        prob += y[t] >= (y2 + MARGIN) - My * (b4)

    # Controls exist for t = 0..T-1 only.
    if t < T:
        # Control bounds.
        prob += v[t] >= VMIN
        prob += v[t] <= VMAX
        prob += omega[t] >= WMIN
        prob += omega[t] <= WMAX
        # Heading-bin selection: exactly one bin is active, and theta[t] has to lie inside the selected bin's interval
        # (the two theta constraints are relaxed by Mtheta for all bins that are not selected).
        prob += pulp.lpSum([s[(t,k)] for k in range(K)]) == 1
        for k, mid in enumerate(theta_mids):
            lower = mid - DTHETA/2.0
            upper = mid + DTHETA/2.0
            prob += theta[t] >= lower - Mtheta*(1 - s[(t,k)])
            prob += theta[t] <= upper + Mtheta*(1 - s[(t,k)])
            # u[t,k] is forced to 0 unless bin k is selected, in which case it equals v[t] (four linear 'if s then u = v' bounds).
            prob += u[(t,k)] <= VMAX * s[(t,k)]
            prob += u[(t,k)] >= VMIN * s[(t,k)]
            prob += u[(t,k)] <= v[t] - VMIN * (1 - s[(t,k)])
            prob += u[(t,k)] >= v[t] - VMAX * (1 - s[(t,k)])
        # Position update using the cos/sin of the SELECTED bin centre (only one u[t,k] is non-zero). Approximate!
        prob += x[t+1] == x[t] + pulp.lpSum([u[(t,k)] * cos_table[theta_mids[k]] * DT for k in range(K)])
        prob += y[t+1] == y[t] + pulp.lpSum([u[(t,k)] * sin_table[theta_mids[k]] * DT for k in range(K)])
        # Heading update (exact and linear in omega).
        prob += theta[t+1] == theta[t] + omega[t] * DT
        # Absolute step lengths for the objective (same trick as the 'linearizer' in smt_simple.py).
        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        prob += adx[t] >= dx
        prob += adx[t] >= -dx
        prob += adx[t] >= 0
        prob += ady[t] >= dy
        prob += ady[t] >= -dy
        prob += ady[t] >= 0

# ---- Objective ----
# Minimise the summed Manhattan path length, or (OPTIMIZE False) just look for any feasible solution.
if OPTIMIZE:
    prob += pulp.lpSum([adx[t] + ady[t] for t in range(T)])
else:
    prob += 0

# ---- Solve: try HiGHS first, fall back to the CBC solver that ships with PuLP if HiGHS is not available ----
solve_start = time.time()
used_solver = None
try:
    prob.solve(pulp.HiGHS_CMD(msg=False))
    used_solver = "HiGHS"
except (PulpSolverError, Exception) as e:
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    used_solver = "CBC (fallback)"
solve_end = time.time()

# ---- Report ----
status = pulp.LpStatus[prob.status]
print("Solver used:", used_solver)
print("Solver status:", status)

# A solution exists: print the path length and plot the path with the environment.
if status in ("Optimal", "Feasible"):
    trajectory = [(pulp.value(x[t]), pulp.value(y[t])) for t in range(T+1)]
    total_length = sum((pulp.value(adx[t]) + pulp.value(ady[t])) for t in range(T))
    print(f"Trajectory found with length ≈ {total_length:.6f}")
    print(f"Total steps (T) = {T}, total time = {T*DT}")

    fig, ax = plt.subplots(figsize=(6,6))
    # Obstacles (hatched = inflated by MARGIN), start region (green), goal region (blue), path (black).
    for (x1, y1, x2, y2) in OBSTACLES:
        ax.add_patch(plt.Rectangle((x1-MARGIN, y1-MARGIN), (x2-x1)+2*MARGIN, (y2-y1)+2*MARGIN, color='red', alpha=0.3, hatch="///"))
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color='red', alpha=0.5))

    ax.add_patch(plt.Rectangle((sx1, sy1), sx2-sx1, sy2-sy1, color='green', alpha=0.5))
    ax.add_patch(plt.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1, color='blue', alpha=0.5))

    xs = [pt[0] for pt in trajectory]
    ys = [pt[1] for pt in trajectory]
    ax.plot(xs, ys, marker='o', color='black', linewidth=1, markersize=2)
    ax.set_xlim(0, GOAL_REGION[2])
    ax.set_ylim(0, GOAL_REGION[3])
    ax.set_aspect('equal')
    ax.set_title(f"MILP Trajectory (solver={used_solver}, dt={DT}, T={T}, margin={MARGIN})")
    plt.show()
else:
    print("No valid trajectory found (solver status:", status, ")")

# ---- Timing summary (build / solve / total) ----
end_time = time.time()

build_time_ms = (solve_start - start_time) * 1000
solve_time_ms = (solve_end - solve_start) * 1000
total_time_ms = (end_time - start_time) * 1000

print(f"\n--- Timing Summary ---")
print(f"Model build time: {build_time_ms:.3f} ms")
print(f"Solving time:     {solve_time_ms:.3f} ms")
print(f"Total runtime:    {total_time_ms:.3f} ms\n")