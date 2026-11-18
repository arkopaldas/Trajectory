import math
import time
from z3 import *
import matplotlib.pyplot as plt

from config import (
    OPTIMIZE, START_REGION,
    SMT_TIME_BOUND as TIME_BOUND, SMT_DT as DT, SMT_T as T, SMT_MARGIN as MARGIN,
    SMT_DTHETA as DTHETA, SMT_VMIN as VMIN, SMT_VMAX as VMAX,
    SMT_WMIN as WMIN, SMT_WMAX as WMAX, SMT_THMIN as THMIN, SMT_THMAX as THMAX,
    SMT_GOAL_REGION as GOAL_REGION, SMT_OBSTACLES as OBSTACLES
)

start_time = time.time()
degtorad = math.pi / 180

theta_mids = [THMIN + DTHETA/2 + i*DTHETA for i in range(int((THMAX - THMIN) / DTHETA))]
cos_table = {theta: math.cos(theta*degtorad) for theta in theta_mids}
sin_table = {theta: math.sin(theta*degtorad) for theta in theta_mids}

def inside_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return And(x >= x1, x <= x2, y >= y1, y <= y2)

def outside_rect(x, y, rect, margin=0.0):
    x1, y1, x2, y2 = rect
    return Or(x < x1 - margin, x > x2 + margin, y < y1 - margin, y > y2 + margin)

def get_val(model, var):
    val = model.evaluate(var, model_completion=True)
    if val is None:
        return None
    if val.is_int():
        return val.as_long()
    return float(val.as_fraction())

def linearizer(opt, expr, prefix, t):
    var = Real(f"{prefix}_{t}")
    opt.add(var >= expr)
    opt.add(var >= -expr)
    opt.add(var >= 0)
    return var

def angle_based_expr(base, v, theta, dt, dtheta, theta_mids, cos_table, sin_table):
    expr_x = expr_y = None
    for i, mid in enumerate(reversed(theta_mids)):
        cond = And(theta >= mid - dtheta/2, theta < mid + dtheta/2)
        expr_x = If(cond, base + v * cos_table[mid] * dt, expr_x) if expr_x is not None else base + v * cos_table[mid] * dt
        expr_y = If(cond, base + v * sin_table[mid] * dt, expr_y) if expr_y is not None else base + v * sin_table[mid] * dt
    return expr_x, expr_y

opt = Optimize()

v = [Real(f"v_{t}") for t in range(T)]
x = [Real(f"x_{t}") for t in range(T+1)]
y = [Real(f"y_{t}") for t in range(T+1)]
omega = [Real(f"omega_{t}") for t in range(T)]
theta = [Real(f"theta_{t}") for t in range(T+1)]

opt.add(inside_rect(x[0], y[0], START_REGION))
opt.add(inside_rect(x[T], y[T], GOAL_REGION))

step_lengths = []
for t in range(T+1):
    opt.add(x[t] >= START_REGION[0], x[t] <= GOAL_REGION[2])
    opt.add(y[t] >= START_REGION[1], y[t] <= GOAL_REGION[3])
    for rect in OBSTACLES:
        opt.add(outside_rect(x[t], y[t], rect, MARGIN))
    if t < T:
        opt.add(v[t] >= VMIN, v[t] <= VMAX)
        opt.add(omega[t] >= WMIN, omega[t] <= WMAX)
        opt.add(theta[t] >= THMIN, theta[t] <= THMAX)
        x_expr, y_expr = angle_based_expr(x[t], v[t], theta[t], DT, DTHETA, theta_mids, cos_table, sin_table)
        opt.add(x[t+1] == x_expr)
        opt.add(y[t+1] == y_expr)
        opt.add(theta[t+1] == theta[t] + omega[t]*DT)
        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        step_lengths.append(linearizer(opt, dx, "adx", t) + linearizer(opt, dy, "ady", t))

total_path_length = Sum(step_lengths)
if OPTIMIZE:
    opt.minimize(total_path_length)

solve_start = time.time()
result = opt.check()
solve_end = time.time()

if result == sat:
    model = opt.model()
    trajectory = [(get_val(model, x[t]), get_val(model, y[t])) for t in range(T+1)]
    print(f"Trajectory found with length ≈ {get_val(model, total_path_length)}")
    print(f"Total steps (T) = {T}, total time = {T*DT}")

    fig, ax = plt.subplots(figsize=(5,5))
    for (x1, y1, x2, y2) in OBSTACLES:
        ax.add_patch(plt.Rectangle((x1-MARGIN, y1-MARGIN), (x2-x1)+2*MARGIN, (y2-y1)+2*MARGIN, color='red', alpha=0.3, hatch="///", label="inflated"))
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color='red', alpha=0.5))

    sx1, sy1, sx2, sy2 = START_REGION
    gx1, gy1, gx2, gy2 = GOAL_REGION
    ax.add_patch(plt.Rectangle((sx1, sy1), sx2-sx1, sy2-sy1, color='green', alpha=0.5))
    ax.add_patch(plt.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1, color='blue', alpha=0.5))

    xs, ys = zip(*trajectory)
    ax.plot(xs, ys, marker='o', color='black', markersize=2, linewidth=1)
    ax.set_xlim(0, GOAL_REGION[2])
    ax.set_ylim(0, GOAL_REGION[3])
    ax.set_aspect('equal')
    ax.set_title(f"Optimal Trajectory (dt={DT}, time_bound={TIME_BOUND}, steps={T}, margin={MARGIN})\n")
    plt.show()
else:
    print("No valid trajectory found.")

end_time = time.time()
build_time_ms = (solve_start - start_time) * 1000
solve_time_ms = (solve_end - solve_start) * 1000
total_time_ms = (end_time - start_time) * 1000

print(f"\n--- Timing Summary ---")
print(f"Model build time: {build_time_ms:.3f} ms")
print(f"Solving time:     {solve_time_ms:.3f} ms")
print(f"Total runtime:    {total_time_ms:.3f} ms\n")