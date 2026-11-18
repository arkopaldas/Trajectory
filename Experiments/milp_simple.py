import math
import time
import pulp
import matplotlib.pyplot as plt
from pulp import PulpSolverError

from config import (
    OPTIMIZE, START_REGION, GOAL_REGION,
    MILP_SIMPLE_DT as DT, MILP_SIMPLE_T as T, MILP_SIMPLE_VMIN as VMIN, MILP_SIMPLE_VMAX as VMAX,
    MILP_SIMPLE_MARGIN as MARGIN, MILP_SIMPLE_BIG_M as BIG_M, MILP_SIMPLE_OBSTACLES as OBSTACLES
)

start_time = time.time()

def rectangle_to_halfspaces(rect):
    x1, y1, x2, y2 = rect
    return [(1, 0, x2), (-1, 0, -x1), (0, 1, y2), (0, -1, -y1)]

def polygon_to_halfspaces(vertices):
    halfspaces = []
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1)%n]
        a = y2 - y1
        b = -(x2 - x1)
        c = a*x1 + b*y1
        halfspaces.append((a, b, c))
    return halfspaces

def circle_to_halfspaces(cx, cy, r, approx_sides=8):
    vertices = []
    for i in range(approx_sides):
        angle = 2 * math.pi * i / approx_sides
        vertices.append((cx + r*math.cos(angle), cy + r*math.sin(angle)))
    return polygon_to_halfspaces(vertices)

def get_halfspaces(obstacle):
    if obstacle["type"] == "rectangle":
        return rectangle_to_halfspaces(obstacle["params"])
    elif obstacle["type"] == "circle":
        return circle_to_halfspaces(*obstacle["params"])
    elif obstacle["type"] == "polygon":
        return polygon_to_halfspaces(obstacle["params"])

prob = pulp.LpProblem("Trajectory_Planning_MILP", pulp.LpMinimize)

x = [pulp.LpVariable(f"x_{t}", lowBound=START_REGION[0], upBound=GOAL_REGION[2]) for t in range(T+1)]
y = [pulp.LpVariable(f"y_{t}", lowBound=START_REGION[1], upBound=GOAL_REGION[3]) for t in range(T+1)]
vx = [pulp.LpVariable(f"vx_{t}", lowBound=VMIN, upBound=VMAX) for t in range(T)]
vy = [pulp.LpVariable(f"vy_{t}", lowBound=VMIN, upBound=VMAX) for t in range(T)]

prob += x[0] >= START_REGION[0]
prob += x[0] <= START_REGION[2]
prob += y[0] >= START_REGION[1]
prob += y[0] <= START_REGION[3]

prob += x[T] >= GOAL_REGION[0]
prob += x[T] <= GOAL_REGION[2]
prob += y[T] >= GOAL_REGION[1]
prob += y[T] <= GOAL_REGION[3]

for t in range(T):
    prob += x[t+1] == x[t] + vx[t] * DT
    prob += y[t+1] == y[t] + vy[t] * DT

for t in range(T+1):
    for r, obstacle in enumerate(OBSTACLES):
        halfspaces = get_halfspaces(obstacle)
        binaries = []
        for i, (a, b, c) in enumerate(halfspaces):
            bi = pulp.LpVariable(f"b_{t}_{r}_{i}", cat='Binary')
            binaries.append(bi)
            prob += a*x[t] + b*y[t] >= (c + MARGIN) - BIG_M*(1 - bi)
        prob += pulp.lpSum(binaries) >= 1

step_lengths = []
for t in range(T):
    dx = x[t+1] - x[t]
    dy = y[t+1] - y[t]
    adx = pulp.LpVariable(f"adx_{t}", lowBound=0)
    ady = pulp.LpVariable(f"ady_{t}", lowBound=0)
    prob += adx >= dx
    prob += adx >= -dx
    prob += ady >= dy
    prob += ady >= -dy
    step_lengths.append(adx + ady)
total_path_length = pulp.lpSum(step_lengths)

if OPTIMIZE:
    prob += total_path_length

solve_start = time.time()
used_solver = None
try:
    prob.solve(pulp.HiGHS_CMD(msg=False))
    used_solver = "HiGHS"
except (PulpSolverError, Exception):
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    used_solver = "CBC (fallback)"
solve_end = time.time()

if pulp.LpStatus[prob.status] == "Optimal":
    trajectory = [(pulp.value(x[t]), pulp.value(y[t])) for t in range(T+1)]
    print(f"Trajectory found with length ≈ {pulp.value(total_path_length)}")
    print(f"Total steps (T) = {T}, total time = {T*DT}")
    print(f"Solver used: {used_solver}")

    fig, ax = plt.subplots(figsize=(6,6))
    for obs in OBSTACLES:
        if obs["type"] == "rectangle":
            x1, y1, x2, y2 = obs["params"]
            ax.add_patch(plt.Rectangle((x1-MARGIN, y1-MARGIN), (x2-x1)+2*MARGIN, (y2-y1)+2*MARGIN, color='red', alpha=0.3, hatch="///"))
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color='red', alpha=0.5))
        elif obs["type"] == "circle":
            cx, cy, r = obs["params"]
            ax.add_patch(plt.Circle((cx, cy), r+MARGIN, color='red', alpha=0.3, hatch="///"))
            ax.add_patch(plt.Circle((cx, cy), r, color='red', alpha=0.5))
        elif obs["type"] == "polygon":
            verts = obs["params"]
            ax.add_patch(plt.Polygon(verts, color='red', alpha=0.5))

    sx1, sy1, sx2, sy2 = START_REGION
    gx1, gy1, gx2, gy2 = GOAL_REGION

    ax.add_patch(plt.Rectangle((sx1, sy1), sx2-sx1, sy2-sy1, color='green', alpha=0.5))
    ax.add_patch(plt.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1, color='blue', alpha=0.5))

    xs, ys = zip(*trajectory)
    ax.plot(xs, ys, marker='o', color='black', markersize=2, linewidth=1)
    ax.set_xlim(0, GOAL_REGION[2])
    ax.set_ylim(0, GOAL_REGION[3])
    ax.set_aspect('equal')
    plt.show()
else:
    print("No valid trajectory found")

end_time = time.time()
build_time_ms = (solve_start - start_time) * 1000
solve_time_ms = (solve_end - solve_start) * 1000
total_time_ms = (end_time - start_time) * 1000

print(f"\n--- Timing Summary ---")
print(f"Model build time: {build_time_ms:.3f} ms")
print(f"Solving time:     {solve_time_ms:.3f} ms")
print(f"Total runtime:    {total_time_ms:.3f} ms\n")