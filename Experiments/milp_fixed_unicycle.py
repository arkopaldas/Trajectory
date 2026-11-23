"""Unicycle MILP started from ONE random start point (instead of a whole start region).

Same formulation as Controllers/milp_unicycle.py / Generators/unicycle.py (heading bins + big-M obstacle avoidance;
cos / sin are evaluated at the bin centre, so the dynamics are approximate), but the start state is pinned to a single
randomly generated point, with its own horizon (70 s) and margin (2.0). Useful for debugging one instance.

Run:    python milp_fixed_unicycle.py      (prints a summary, shows a plot and a timing table)
"""
import math
import time
import pulp
import random
import matplotlib.pyplot as plt
from pulp import PulpSolverError

from config import (
    OPTIMIZE, START_REGION, GOAL_REGION, OBSTACLES,
    MILP_FIXED_DT as DT, MILP_FIXED_T as T, MILP_FIXED_MARGIN as MARGIN,
    MILP_FIXED_DTHETA as DTHETA, MILP_FIXED_VMIN as VMIN, MILP_FIXED_VMAX as VMAX,
    MILP_FIXED_WMIN as WMIN, MILP_FIXED_WMAX as WMAX,
    MILP_FIXED_THMIN as THMIN, MILP_FIXED_THMAX as THMAX
)

# Wall-clock start for the timing summary.
start_time = time.time()

# Rejection sampling: random points in `region` that are at least `offset` apart (here: 10 points, 1 unit apart).
def generate_points(region=START_REGION, n=10, offset=1, attempts=10000):
    xmin, ymin, xmax, ymax = region
    points = []
    check = 0
    while len(points) < n and check <= attempts:
        x_val = random.uniform(xmin, xmax)
        y_val = random.uniform(ymin, ymax)
        if all(math.hypot(x_val - px, y_val - py) >= offset for px, py in points):
            points.append((x_val, y_val))
        check = check + 1
    return points

# Heading bins: centre of every bin of width DTHETA and the cos / sin at the centres; K = number of bins.
theta_mids = [THMIN + DTHETA/2.0 + i*DTHETA for i in range(int((THMAX - THMIN) / DTHETA))]
cos_table = {mid: math.cos(mid) for mid in theta_mids}
sin_table = {mid: math.sin(mid) for mid in theta_mids}
K = len(theta_mids)

# Big-M constants (workspace size, one full turn).
Mx = max(GOAL_REGION[2], START_REGION[2])
My = max(GOAL_REGION[3], START_REGION[3])
Mtheta = 2 * math.pi

# ---- Variables ----
# x, y, theta: states (t = 0..T). v, omega: controls (t = 0..T-1). s[t,k]: heading bin k is active at step t.
# u[t,k]: equals v[t] when bin k is active, otherwise 0. adx, ady: |dx|, |dy| for the objective.
prob = pulp.LpProblem("MILP_Path_Planner", pulp.LpMinimize)

x = {t: pulp.LpVariable(f"x_{t}", lowBound=0.0, upBound=GOAL_REGION[2]) for t in range(T+1)}
y = {t: pulp.LpVariable(f"y_{t}", lowBound=0.0, upBound=GOAL_REGION[3]) for t in range(T+1)}
v = {t: pulp.LpVariable(f"v_{t}", lowBound=VMIN, upBound=VMAX) for t in range(T)}
omega = {t: pulp.LpVariable(f"omega_{t}", lowBound=WMIN, upBound=WMAX) for t in range(T)}
theta = {t: pulp.LpVariable(f"theta_{t}", lowBound=THMIN, upBound=THMAX) for t in range(T+1)}

s = {(t,k): pulp.LpVariable(f"s_{t}_{k}", cat="Binary") for t in range(T) for k in range(K)}
u = {(t,k): pulp.LpVariable(f"u_{t}_{k}", lowBound=VMIN, upBound=VMAX) for t in range(T) for k in range(K)}

adx = {t: pulp.LpVariable(f"adx_{t}", lowBound=0.0) for t in range(T)}
ady = {t: pulp.LpVariable(f"ady_{t}", lowBound=0.0) for t in range(T)}

# Pick the start: the first of 10 random points of the start region becomes the fixed initial position.
starts = generate_points(START_REGION, 10, 1)

sx1, sy1, sx2, sy2 = START_REGION
gx1, gy1, gx2, gy2 = GOAL_REGION

# Initial position pinned to that single point (>= and <= together mean equality).
prob += x[0] >= starts[0][0]
prob += x[0] <= starts[0][0]
prob += y[0] >= starts[0][1]
prob += y[0] <= starts[0][1]

# Final position inside the goal region.
prob += x[T] >= gx1
prob += x[T] <= gx2
prob += y[T] >= gy1
prob += y[T] <= gy2

# ---- Per-step constraints: workspace box, heading limits, obstacle avoidance, dynamics ----
for t in range(T+1):
    prob += x[t] >= START_REGION[0]
    prob += x[t] <= GOAL_REGION[2]
    prob += y[t] >= START_REGION[1]
    prob += y[t] <= GOAL_REGION[3]
    prob += theta[t] >= THMIN
    prob += theta[t] <= THMAX

    # Obstacle avoidance (big-M): exactly one of the four binaries is 0 (sum == 3); the side whose binary is 0 is enforced
    # (left / right / below / above the obstacle by at least MARGIN), the other three are relaxed by the big-M.
    for (x1,y1,x2,y2) in OBSTACLES:
        b1 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b1", cat="Binary")
        b2 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b2", cat="Binary")
        b3 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b3", cat="Binary")
        b4 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b4", cat="Binary")

        prob += b1 + b2 + b3 + b4 == 3
        prob += x[t] <= (x1 - MARGIN) + Mx * b1
        prob += x[t] >= (x2 + MARGIN) - Mx * b2
        prob += y[t] <= (y1 - MARGIN) + My * b3
        prob += y[t] >= (y2 + MARGIN) - My * b4

    # Controls and dynamics (t = 0..T-1). Exactly one heading bin is active; theta must lie in that bin; u ties v to the active bin.
    if t < T:
        prob += v[t] >= VMIN
        prob += v[t] <= VMAX
        prob += omega[t] >= WMIN
        prob += omega[t] <= WMAX
        prob += pulp.lpSum([s[(t,k)] for k in range(K)]) == 1
        for k, mid in enumerate(theta_mids):
            lower = mid - DTHETA/2.0
            upper = mid + DTHETA/2.0
            prob += theta[t] >= lower - Mtheta*(1 - s[(t,k)])
            prob += theta[t] <= upper + Mtheta*(1 - s[(t,k)])
            prob += u[(t,k)] <= VMAX * s[(t,k)]
            prob += u[(t,k)] >= VMIN * s[(t,k)]
            prob += u[(t,k)] <= v[t] - VMIN * (1 - s[(t,k)])
            prob += u[(t,k)] >= v[t] - VMAX * (1 - s[(t,k)])
        # Position update with the cos / sin of the active bin centre (approximate) and the exact heading update.
        prob += x[t+1] == x[t] + pulp.lpSum([u[(t,k)] * cos_table[theta_mids[k]] * DT for k in range(K)])
        prob += y[t+1] == y[t] + pulp.lpSum([u[(t,k)] * sin_table[theta_mids[k]] * DT for k in range(K)])
        prob += theta[t+1] == theta[t] + omega[t] * DT
        # Absolute step lengths (for the objective).
        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        prob += adx[t] >= dx
        prob += adx[t] >= -dx
        prob += adx[t] >= 0
        prob += ady[t] >= dy
        prob += ady[t] >= -dy
        prob += ady[t] >= 0

# Objective: minimise the Manhattan path length, or a constant 0 objective if only feasibility is wanted.
if OPTIMIZE:
    prob += pulp.lpSum([adx[t] + ady[t] for t in range(T)])
else:
    prob += 0

# ---- Solve: HiGHS first, CBC as fallback ----
solve_start = time.time()
used_solver = None
try:
    prob.solve(pulp.HiGHS_CMD(msg=False))
    used_solver = "HiGHS"
except (PulpSolverError, Exception):
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    used_solver = "CBC (fallback)"
solve_end = time.time()

# ---- Report and plot ----
status = pulp.LpStatus[prob.status]
print("Solver used:", used_solver)
print("Solver status:", status)

if status in ("Optimal", "Feasible"):
    trajectory = [(pulp.value(x[t]), pulp.value(y[t])) for t in range(T+1)]
    total_length = sum((pulp.value(adx[t]) + pulp.value(ady[t])) for t in range(T))
    print(f"Trajectory found with length ≈ {total_length:.6f}")
    print(f"Total steps (T) = {T}, total time = {T*DT}")

    fig, ax = plt.subplots(figsize=(6,6))
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

# ---- Timing summary ----
end_time = time.time()
build_time_ms = (solve_start - start_time) * 1000
solve_time_ms = (solve_end - solve_start) * 1000
total_time_ms = (end_time - start_time) * 1000

print(f"\n--- Timing Summary ---")
print(f"Model build time: {build_time_ms:.3f} ms")
print(f"Solving time:     {solve_time_ms:.3f} ms")
print(f"Total runtime:    {total_time_ms:.3f} ms\n")