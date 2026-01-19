import math
import time
import pulp
import random
import matplotlib.pyplot as plt
from pulp import PulpSolverError

# Start Timer
start_time = time.time()

# Parameters
time_bound = 210.0
dt = 1.0
T = math.ceil(time_bound / dt)
margin = 1.5
dtheta = 10
optimize = True
degtorad = math.pi / 180.0
Vmin, Vmax = 0.0, dt
Wmin, Wmax = -90.0, 90.0
THmin, THmax = 0.0, 90.0

start_region = (0, 0, 10, 10)
goal_region = (90, 90, 100, 100)
obstacles = [
    (20.0, 20.0, 30.0, 30.0),
    (40.0, 20.0, 50.0, 30.0),
    (20.0, 40.0, 30.0, 50.0),
    (20.0, 70.0, 30.0, 80.0),
    (40.0, 60.0, 50.0, 70.0),
    (50.0, 40.0, 60.0, 50.0),
    (70.0, 30.0, 80.0, 40.0),
    (70.0, 10.0, 80.0, 20.0),
]

# Random initial point generation
def generate_points(xmin=0, xmax=10, ymin=0, ymax=10, n=10, offset=1, attempts=10000):
    points = []
    check = 0
    while len(points) < n and check <= attempts:
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)

        if all(math.hypot(x - px, y - py) >= offset for px, py in points):
            points.append((x, y))
        check = check + 1
    return points

# theta midpoints and trig tables (constants)
theta_mids = [THmin + dtheta/2.0 + i*dtheta for i in range(int((THmax - THmin) / dtheta))]
cos_table = {mid: math.cos(mid*degtorad) for mid in theta_mids}
sin_table = {mid: math.sin(mid*degtorad) for mid in theta_mids}
K = len(theta_mids)

# Big-M values (tighten based on domain extents)
max_x = max(goal_region[2], start_region[2])
max_y = max(goal_region[3], start_region[3])
Mx = max(max_x, max_y) + margin + 10.0
Mtheta = 360.0
Mv = Vmax

# Build MILP with PuLP
prob = pulp.LpProblem("MILP_Path_Planner", pulp.LpMinimize)

# continuous variables
x = {t: pulp.LpVariable(f"x_{t}", lowBound=0.0, upBound=goal_region[2]) for t in range(T+1)}
y = {t: pulp.LpVariable(f"y_{t}", lowBound=0.0, upBound=goal_region[3]) for t in range(T+1)}
v = {t: pulp.LpVariable(f"v_{t}", lowBound=Vmin, upBound=Vmax) for t in range(T)}
omega = {t: pulp.LpVariable(f"omega_{t}", lowBound=Wmin, upBound=Wmax) for t in range(T)}
theta = {t: pulp.LpVariable(f"theta_{t}", lowBound=THmin, upBound=THmax) for t in range(T+1)}

# binaries to pick angle bin at each time step t
s = {(t,k): pulp.LpVariable(f"s_{t}_{k}", cat="Binary") for t in range(T) for k in range(K)}

# u_{t,k} = v_t * s_{t,k} (linearized using big-M)
u = {(t,k): pulp.LpVariable(f"u_{t}_{k}", lowBound=Vmin, upBound=Vmax) for t in range(T) for k in range(K)}

# absolute differences (linearizer for L1 length)
adx = {t: pulp.LpVariable(f"adx_{t}", lowBound=0.0) for t in range(T)}
ady = {t: pulp.LpVariable(f"ady_{t}", lowBound=0.0) for t in range(T)}

# 1) Initial and goal region constraints (inside rectangles)
sx1, sy1, sx2, sy2 = start_region
gx1, gy1, gx2, gy2 = goal_region
starts = generate_points(sx1, sx2, sy1, sy2, 1)
# goals = random_unique_points(gx1, gy1, gx2, gy2, 1)

prob += x[0] == starts[0][0]
prob += y[0] == starts[0][1]

prob += x[T] >= gx1
prob += x[T] <= gx2
prob += y[T] >= gy1
prob += y[T] <= gy2

# 2) Board bounds for all t (keeps variables well-bounded)
for t in range(T+1):
    prob += x[t] >= 0.0
    prob += x[t] <= goal_region[2]
    prob += y[t] >= 0.0
    prob += y[t] <= goal_region[3]
    prob += theta[t] >= THmin
    prob += theta[t] <= THmax

# 3) Obstacle avoidance using OR->Big-M via 4 binaries per obstacle/time (correct polarity)
for t in range(T+1):
    for (x1,y1,x2,y2) in obstacles:
        b1 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b1", cat="Binary")
        b2 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b2", cat="Binary")
        b3 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b3", cat="Binary")
        b4 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b4", cat="Binary")

        # at least one "outside" condition must hold
        # prob += b1 + b2 + b3 + b4 <= 3
        # The condition above is taking a lot of time
        prob += b1 + b2 + b3 + b4 == 3

        # If b1 == 0 then x <= x1 - margin (tight)
        prob += x[t] <= (x1 - margin) + Mx * (b1)

        # If b2 == 0 then x >= x2 + margin (tight)
        prob += x[t] >= (x2 + margin) - Mx * (b2)

        # If b3 == 0 then y <= y1 - margin (tight)
        prob += y[t] <= (y1 - margin) + Mx * (b3)

        # If b4 == 0 then y >= y2 + margin (tight)
        prob += y[t] >= (y2 + margin) - Mx * (b4)

# 4) Motion equations for t in [0..T-1]
for t in range(T):
    prob += v[t] >= Vmin
    prob += v[t] <= Vmax
    prob += omega[t] >= Wmin
    prob += omega[t] <= Wmax

    # 4.a) exactly one angle bin selected
    prob += pulp.lpSum([s[(t,k)] for k in range(K)]) == 1

    # 4.b) relate theta to selected bins -> if s[t,k]==1 then theta in bin range
    for k, mid in enumerate(theta_mids):
        lower = mid - dtheta/2.0
        upper = mid + dtheta/2.0
        prob += theta[t] >= lower - Mtheta*(1 - s[(t,k)])
        prob += theta[t] <= upper + Mtheta*(1 - s[(t,k)])

    # 4.c) linearize u_{t,k} = v_t * s_{t,k} (0 <= u <= Vmax)
        prob += u[(t,k)] <= Vmax * s[(t,k)]
        prob += u[(t,k)] >= Vmin * s[(t,k)]
        prob += u[(t,k)] <= v[t] - Vmin * (1 - s[(t,k)])
        prob += u[(t,k)] >= v[t] - Vmax * (1 - s[(t,k)])

    # 4.d) x_{t+1} = x_t + sum_k u_{t,k} * cos_k * dt
    prob += x[t+1] == x[t] + pulp.lpSum([u[(t,k)] * cos_table[theta_mids[k]] * dt for k in range(K)])
    # 4.e) y_{t+1} = y_t + sum_k u_{t,k] * sin_k * dt
    prob += y[t+1] == y[t] + pulp.lpSum([u[(t,k)] * sin_table[theta_mids[k]] * dt for k in range(K)])

    # 4.f) theta dynamics: theta_{t+1} = theta_t + omega_t * dt
    prob += theta[t+1] == theta[t] + omega[t] * dt

    # 4.g) L1 linearization for distances
    dx = x[t+1] - x[t]
    dy = y[t+1] - y[t]
    prob += adx[t] >= dx
    prob += adx[t] >= -dx
    prob += adx[t] >= 0
    prob += ady[t] >= dy
    prob += ady[t] >= -dy
    prob += ady[t] >= 0

# Objective: minimize L1 path length (sum of adx + ady)
if optimize:
    prob += pulp.lpSum([adx[t] + ady[t] for t in range(T)])
else:
    prob += 0

# Solve and time (try Gurobi, fallback to CBC)
solve_start = time.time()
used_solver = None
try:
    # try Gurobi (requires gurobipy and license)
    gurobi_solver = pulp.GUROBI_CMD(msg=False, timeLimit=600)
    prob.solve(gurobi_solver)
    used_solver = "Gurobi"
except (PulpSolverError, Exception) as e:
    # fallback to CBC
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    used_solver = "CBC (fallback)"
solve_end = time.time()

status = pulp.LpStatus[prob.status]
print("Solver used:", used_solver)
print("Solver status:", status)

# Extract solution and diagnostic checks
def point_inside_inflated(px, py, rect, margin):
    x1,y1,x2,y2 = rect
    return (px > x1 - margin + 1e-8 and px < x2 + margin - 1e-8 and py > y1 - margin + 1e-8 and py < y2 + margin - 1e-8)

if status in ("Optimal", "Feasible"):
    trajectory = [(pulp.value(x[t]), pulp.value(y[t])) for t in range(T+1)]
    total_length = sum((pulp.value(adx[t]) + pulp.value(ady[t])) for t in range(T))
    print(f"Trajectory found with length ≈ {total_length:.6f}")
    print(f"Total steps (T) = {T}, total time = {T*dt}")
    # print([pulp.value(v[t]) for t in range(T)].count(0.0))

    # diagnostic: check for obstacle violations
    violations = []
    for t,(px,py) in enumerate(trajectory):
        if px is None or py is None:
            continue
        for ridx,rect in enumerate(obstacles):
            if point_inside_inflated(px,py,rect,margin):
                violations.append((t, ridx, px, py))
    if violations:
        print(f"Amount of viloations - {len(violations)}")

    # Visualization
    fig, ax = plt.subplots(figsize=(6,6))
    for (x1, y1, x2, y2) in obstacles:
        ax.add_patch(plt.Rectangle((x1-margin, y1-margin), (x2-x1)+2*margin, (y2-y1)+2*margin, color='red', alpha=0.3, hatch="///"))
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color='red', alpha=0.5))

    ax.add_patch(plt.Rectangle((sx1, sy1), sx2-sx1, sy2-sy1, color='green', alpha=0.5))
    ax.add_patch(plt.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1, color='blue', alpha=0.5))

    xs = [pt[0] for pt in trajectory]
    ys = [pt[1] for pt in trajectory]
    ax.plot(xs, ys, marker='o', color='black', linewidth=1, markersize=2)
    ax.set_xlim(0, goal_region[2])
    ax.set_ylim(0, goal_region[3])
    ax.set_aspect('equal')
    ax.set_title(f"MILP Trajectory (solver={used_solver}, dt={dt}, T={T}, margin={margin})")
    plt.show()

else:
    print("No valid trajectory found (solver status:", status, ")")

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
