"""Single-instance SMT planner for the SIMPLE model (a point that is steered directly by its velocity).

Model:   x[t+1] = x[t] + vx[t] * DT,   y[t+1] = y[t] + vy[t] * DT,   |vx|, |vy| <= DT
Problem: find a trajectory of T steps that starts in START_REGION, ends in GOAL_REGION, stays at least MARGIN away
         from every obstacle and inside the workspace box, and (if OPTIMIZE) has the shortest path length.
Solver:  Z3 (z3-solver). The dynamics are linear, so the encoding is exact (no approximation).

Run:     python smt_simple.py      (shows a matplotlib figure and prints a timing summary)
Related: Generators/simple.py is the same model wrapped in a function and run for many start points.
"""
import math
import time
from z3 import *
import matplotlib.pyplot as plt

# Problem constants come from the local config.py (see the comments there).
from config import (
    OPTIMIZE, MARGIN, TIME_BOUND, START_REGION, GOAL_REGION, OBSTACLES,
    SMT_DT as DT, SMT_T as T, SMT_VMIN as VMIN, SMT_VMAX as VMAX
)

# Wall-clock start, used for the timing summary at the bottom of the script.
start_time = time.time()

# Z3 condition: the point (x, y) lies inside the closed rectangle rect = (x1, y1, x2, y2).
def inside_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return And(x >= x1, x <= x2, y >= y1, y <= y2)

# Z3 condition: (x, y) is outside the rectangle inflated by `margin` on every side. Outside means ONE of four
# half-planes holds (left of / right of / below / above the inflated box), hence the Or(...).
def outside_rect(x, y, rect, margin=0.0):
    x1, y1, x2, y2 = rect
    return Or(x < x1 - margin, x > x2 + margin, y < y1 - margin, y > y2 + margin)

# Reads the value of a Z3 variable from a model as a plain Python number (Z3 stores exact rationals).
def get_val(model, var):
    val = model.evaluate(var, model_completion=True)
    if val is None:
        return None
    if val.is_int():
        return val.as_long()
    return float(val.as_fraction())

# Encodes |expr| for the minimisation: creates a fresh real variable `var` with var >= expr, var >= -expr and
# var >= 0. Minimising var then drives it to |expr|. The path length therefore is the sum of |dx| + |dy| per step
# (Manhattan length), not the Euclidean length.
def linearizer(opt, expr, prefix, t):
    var = Real(f"{prefix}_{t}")
    opt.add(var >= expr)
    opt.add(var >= -expr)
    opt.add(var >= 0)
    return var

# ---- Build the problem ----
# Z3 Optimize object: holds the hard constraints and the objective.
opt = Optimize()

# Decision variables: positions x_0..x_T, y_0..y_T and the control velocities vx_0..vx_(T-1), vy_0..vy_(T-1).
x = [Real(f"x_{t}") for t in range(T+1)]
y = [Real(f"y_{t}") for t in range(T+1)]
vx = [Real(f"vx_{t}") for t in range(T)]
vy = [Real(f"vy_{t}") for t in range(T)]

# Boundary conditions: start inside the start region, end inside the goal region.
opt.add(inside_rect(x[0], y[0], START_REGION))
opt.add(inside_rect(x[T], y[T], GOAL_REGION))

step_lengths = []
# Per-step constraints. Every state (t = 0..T) must stay in the workspace box and outside all inflated obstacles;
# every control (t = 0..T-1) must respect the velocity limits and obey the dynamics.
for t in range(T+1):
    # Workspace box: from the lower-left corner of the start region to the upper-right corner of the goal region.
    opt.add(x[t] >= START_REGION[0], x[t] <= GOAL_REGION[2])
    opt.add(y[t] >= START_REGION[1], y[t] <= GOAL_REGION[3])
    for rect in OBSTACLES:
        opt.add(outside_rect(x[t], y[t], rect, MARGIN))
    if t < T:
        opt.add(vx[t] >= VMIN, vx[t] <= VMAX)
        opt.add(vy[t] >= VMIN, vy[t] <= VMAX)
        # The (exact, linear) dynamics of the simple model.
        opt.add(x[t+1] == x[t] + vx[t] * DT)
        opt.add(y[t+1] == y[t] + vy[t] * DT)
        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        # Collect |dx| + |dy| of this step for the objective.
        step_lengths.append(linearizer(opt, dx, "adx", t) + linearizer(opt, dy, "ady", t))

# Objective: total (Manhattan) path length. Only minimised when OPTIMIZE is True.
total_path_length = Sum(step_lengths)
if OPTIMIZE:
    opt.minimize(total_path_length)

# ---- Solve ----
solve_start = time.time()
result = opt.check()
solve_end = time.time()

# ---- Report: print the trajectory length and draw the environment and the path ----
if result == sat:
    model = opt.model()
    trajectory = [(get_val(model, x[t]), get_val(model, y[t])) for t in range(T+1)]
    print(f"Trajectory found with length ≈ {get_val(model, total_path_length)}")
    print(f"Total steps (T) = {T}, total time = {T*DT}")

    fig, ax = plt.subplots(figsize=(6,6))

    # Obstacles: the light hatched box is the inflated (margin) zone, the darker box is the obstacle itself.
    for (x1, y1, x2, y2) in OBSTACLES:
        ax.add_patch(plt.Rectangle((x1-MARGIN, y1-MARGIN), (x2-x1)+2*MARGIN, (y2-y1)+2*MARGIN, color='red', alpha=0.3, hatch="///", label="inflated"))
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color='red', alpha=0.5))

    # Start region (green) and goal region (blue).
    sx1, sy1, sx2, sy2 = START_REGION
    gx1, gy1, gx2, gy2 = GOAL_REGION
    ax.add_patch(plt.Rectangle((sx1, sy1), sx2-sx1, sy2-sy1, color='green', alpha=0.5))
    ax.add_patch(plt.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1, color='blue', alpha=0.5))

    # The planned path as black dots joined by a line.
    xs, ys = zip(*trajectory)
    ax.plot(xs, ys, marker='o', color='black', markersize=2, linewidth=1)
    ax.set_xlim(0, GOAL_REGION[2])
    ax.set_ylim(0, GOAL_REGION[3])
    ax.set_title(f"Optimal Trajectory (dt={DT}, time_bound={TIME_BOUND}, steps={T}, margin={MARGIN})\n")
    ax.set_aspect('equal')
    plt.show()
else:
    print("No valid trajectory found")

# ---- Timing: build = creating the constraints, solve = Z3 check(), total = whole script ----
end_time = time.time()

build_time_ms = (solve_start - start_time) * 1000
solve_time_ms = (solve_end - solve_start) * 1000
total_time_ms = (end_time - start_time) * 1000

print(f"\n--- Timing Summary ---")
print(f"Model build time: {build_time_ms:.3f} ms")
print(f"Solving time:     {solve_time_ms:.3f} ms")
print(f"Total runtime:    {total_time_ms:.3f} ms\n")