import math
import time
from z3 import *
import matplotlib.pyplot as plt

# Start Timer
start_time = time.time()

# Parameters
time_bound = 11.0
dt = 1.0
T = math.ceil(time_bound / dt)
margin = 0.75
dtheta = 10
optimize = True
degtorad = math.pi / 180
Vmin, Vmax = 0, dt
Wmin, Wmax = -90, 90
THmin, THmax = 0, 90

# Regions - Start, Goal, Obstacle.
start_region = (0, 0, 10, 10)
goal_region = (20, 20, 30, 30)
obstacles = [
    # (20.0, 20.0, 30.0, 30.0),
    # (40.0, 20.0, 50.0, 30.0),
    # (20.0, 40.0, 30.0, 50.0),
    # (20.0, 70.0, 30.0, 80.0),
    # (40.0, 60.0, 50.0, 70.0),
    # (50.0, 40.0, 60.0, 50.0),
    # (70.0, 30.0, 80.0, 40.0),
    # (70.0, 10.0, 80.0, 20.0),

    # (20.0, 20.0, 25.0, 25.0),
    # (20.0, 30.0, 25.0, 35.0),
    # (30.0, 20.0, 35.0, 25.0),
    # (30.0, 30.0, 35.0, 35.0),

    (12.5, 15.0, 15.0, 17.5),
    # (15.0, 12.5, 17.5, 15.0),
    # (12.5, 12.5, 15.0, 15.0),
]

# Precompute Midvalues of Intervals table
theta_mids = [THmin + dtheta/2 + i*dtheta for i in range(int((THmax - THmin) / dtheta))]

# Cosine and Sine tables for the Midpoints
cos_table = {theta: math.cos(theta*degtorad) for theta in theta_mids}
sin_table = {theta: math.sin(theta*degtorad) for theta in theta_mids}

def inside_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return And(x >= x1, x <= x2, y >= y1, y <= y2)

def outside_rect(x, y, rect, margin=0.0):
    x1, y1, x2, y2 = rect
    return Or(x < x1 - margin, x > x2 + margin, y < y1 - margin, y > y2 + margin)

def get_val(model, var):
    val = model.evaluate(var,  model_completion=True)
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
    x_expr = RealVal(0)
    y_expr = RealVal(0)
    expr_x = expr_y = None
    for i, mid in enumerate(reversed(theta_mids)):
        cond = And(theta >= mid - dtheta/2, theta < mid + dtheta/2)
        expr_x = If(cond, base + v * cos_table[mid] * dt, expr_x) if expr_x is not None else base + v * cos_table[mid] * dt
        expr_y = If(cond, base + v * sin_table[mid] * dt, expr_y) if expr_y is not None else base + v * sin_table[mid] * dt
    return expr_x, expr_y

# Optimizer
opt = Optimize()

# Variables for positions, velocity and angle
v = [Real(f"v_{t}") for t in range(T)]
x = [Real(f"x_{t}") for t in range(T+1)]
y = [Real(f"y_{t}") for t in range(T+1)]
omega = [Real(f"omega_{t}") for t in range(T)]
theta = [Real(f"theta_{t}") for t in range(T+1)]

# Initial and Final State Constraints
opt.add(inside_rect(x[0], y[0], start_region))
opt.add(inside_rect(x[T], y[T], goal_region))

# Board Bounds, Avoiding obstacles with margin, Velocity bounds, Motion equations and Objective function
step_lengths = []
for t in range(T+1):
    opt.add(x[t] >= start_region[0], x[t] <= goal_region[2])
    opt.add(y[t] >= start_region[1], y[t] <= goal_region[3])
    for rect in obstacles:
        opt.add(outside_rect(x[t], y[t], rect, margin))
    if t < T:
        opt.add(v[t] >= Vmin, v[t] <= Vmax)
        opt.add(omega[t] >= Wmin, omega[t] <= Wmax)
        opt.add(theta[t] >= THmin, theta[t] <= THmax)
        x_expr, y_expr = angle_based_expr(x[t], v[t], theta[t], dt, dtheta, theta_mids, cos_table, sin_table)
        opt.add(x[t+1] == x_expr)
        opt.add(y[t+1] == y_expr)
        opt.add(theta[t+1] == theta[t] + omega[t]*dt)
        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        # step_lengths.append((dx*dx + dy*dy)**0.5)
        step_lengths.append(linearizer(opt, dx, "adx", t) + linearizer(opt, dy, "ady", t))

total_path_length = Sum(step_lengths)
if optimize:
    opt.minimize(total_path_length)

solve_start = time.time()
result = opt.check()
solve_end = time.time()

# Solving and visualizing
if opt.check() == sat:
    model = opt.model()
    trajectory = [(get_val(model, x[t]), get_val(model, y[t])) for t in range(T+1)]
    print(f"Trajectory found with length = {get_val(model, total_path_length)}\n")
    print(f"Total steps (T) = {T}, total time = {T*dt}\n")

    # Visualization
    fig, ax = plt.subplots(figsize=(5,5))

    # Plot inflated obstacles (for visualization)
    for (x1, y1, x2, y2) in obstacles:
        ax.add_patch(plt.Rectangle((x1-margin, y1-margin), (x2-x1)+2*margin, (y2-y1)+2*margin, color='red', alpha=0.3, hatch="///", label="inflated"))
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color='red', alpha=0.5))

    # Plot start and goal regions
    sx1, sy1, sx2, sy2 = start_region
    gx1, gy1, gx2, gy2 = goal_region
    ax.add_patch(plt.Rectangle((sx1, sy1), sx2-sx1, sy2-sy1, color='green', alpha=0.5))
    ax.add_patch(plt.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1, color='blue', alpha=0.5))

    # Plot trajectory
    xs, ys = zip(*trajectory)
    ax.plot(xs, ys, marker='o', color='black', markersize=2, linewidth=1)
    ax.set_xlim(0, goal_region[2])
    ax.set_ylim(0, goal_region[3])
    ax.set_aspect('equal')
    ax.set_title(f"Optimal Trajectory (dt={dt}, time_bound={time_bound}, steps={T}, margin={margin})\n")
    plt.show()

else:
    print("No valid trajectory found.")

# End timing
end_time = time.time()

# Report times in milliseconds
build_time_ms = (solve_start - start_time) * 1000
solve_time_ms = (solve_end - solve_start) * 1000
total_time_ms = (end_time - start_time) * 1000

print(f"\n--- Timing Summary ---")
print(f"Model build time: {build_time_ms:.3f} ms  ({build_time_ms/1000:.6f} s)")
print(f"Solving time:     {solve_time_ms:.3f} ms  ({solve_time_ms/1000:.6f} s)")
print(f"Total runtime:    {total_time_ms:.3f} ms  ({total_time_ms/1000:.6f} s)\n")
