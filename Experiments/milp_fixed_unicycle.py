import math
import time
import pulp
import random
import matplotlib.pyplot as plt
from pulp import PulpSolverError

start_time = time.time()

time_bound = 70.0
dt = 3.0
T = math.ceil(time_bound / dt)
degtorad = math.pi / 180.0
margin = 2.0
dtheta = 45 * degtorad
optimize = True
Vmin, Vmax = 0.0, dt
Wmin, Wmax = -90 * degtorad, 90 * degtorad
THmin, THmax = -22.5 * degtorad, 112.5 * degtorad

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

def generate_points(region=(0,0,10,10), n=10, offset=1, attempts=10000):
    xmin, ymin, xmax, ymax = region
    points = []
    check = 0
    while len(points) < n and check <= attempts:
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)

        if all(math.hypot(x - px, y - py) >= offset for px, py in points):
            points.append((x, y))
        check = check + 1
    return points

theta_mids = [THmin + dtheta/2.0 + i*dtheta for i in range(int((THmax - THmin) / dtheta))]
cos_table = {mid: math.cos(mid) for mid in theta_mids}
sin_table = {mid: math.sin(mid) for mid in theta_mids}
K = len(theta_mids)

Mx = max(goal_region[2], start_region[2])
My = max(goal_region[3], start_region[3])
Mtheta = 2 * math.pi

prob = pulp.LpProblem("MILP_Path_Planner", pulp.LpMinimize)

x = {t: pulp.LpVariable(f"x_{t}", lowBound=0.0, upBound=goal_region[2]) for t in range(T+1)}
y = {t: pulp.LpVariable(f"y_{t}", lowBound=0.0, upBound=goal_region[3]) for t in range(T+1)}
v = {t: pulp.LpVariable(f"v_{t}", lowBound=Vmin, upBound=Vmax) for t in range(T)}
omega = {t: pulp.LpVariable(f"omega_{t}", lowBound=Wmin, upBound=Wmax) for t in range(T)}
theta = {t: pulp.LpVariable(f"theta_{t}", lowBound=THmin, upBound=THmax) for t in range(T+1)}

s = {(t,k): pulp.LpVariable(f"s_{t}_{k}", cat="Binary") for t in range(T) for k in range(K)}
u = {(t,k): pulp.LpVariable(f"u_{t}_{k}", lowBound=Vmin, upBound=Vmax) for t in range(T) for k in range(K)}

adx = {t: pulp.LpVariable(f"adx_{t}", lowBound=0.0) for t in range(T)}
ady = {t: pulp.LpVariable(f"ady_{t}", lowBound=0.0) for t in range(T)}

starts = generate_points((0,0,10,10), 10, 1)

sx1, sy1, sx2, sy2 = start_region
gx1, gy1, gx2, gy2 = goal_region

prob += x[0] >= starts[0][0]
prob += x[0] <= starts[0][0]
prob += y[0] >= starts[0][1]
prob += y[0] <= starts[0][1]

prob += x[T] >= gx1
prob += x[T] <= gx2
prob += y[T] >= gy1
prob += y[T] <= gy2

for t in range(T+1):
    prob += x[t] >= start_region[0]
    prob += x[t] <= goal_region[2]
    prob += y[t] >= start_region[1]
    prob += y[t] <= goal_region[3]
    prob += theta[t] >= THmin
    prob += theta[t] <= THmax

    for (x1,y1,x2,y2) in obstacles:
        b1 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b1", cat="Binary")
        b2 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b2", cat="Binary")
        b3 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b3", cat="Binary")
        b4 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b4", cat="Binary")

        prob += b1 + b2 + b3 + b4 == 3
        prob += x[t] <= (x1 - margin) + Mx * (b1)
        prob += x[t] >= (x2 + margin) - Mx * (b2)
        prob += y[t] <= (y1 - margin) + My * (b3)
        prob += y[t] >= (y2 + margin) - My * (b4)

    if t < T:
        prob += v[t] >= Vmin
        prob += v[t] <= Vmax
        prob += omega[t] >= Wmin
        prob += omega[t] <= Wmax
        prob += pulp.lpSum([s[(t,k)] for k in range(K)]) == 1
        for k, mid in enumerate(theta_mids):
            lower = mid - dtheta/2.0
            upper = mid + dtheta/2.0
            prob += theta[t] >= lower - Mtheta*(1 - s[(t,k)])
            prob += theta[t] <= upper + Mtheta*(1 - s[(t,k)])
            prob += u[(t,k)] <= Vmax * s[(t,k)]
            prob += u[(t,k)] >= Vmin * s[(t,k)]
            prob += u[(t,k)] <= v[t] - Vmin * (1 - s[(t,k)])
            prob += u[(t,k)] >= v[t] - Vmax * (1 - s[(t,k)])
        prob += x[t+1] == x[t] + pulp.lpSum([u[(t,k)] * cos_table[theta_mids[k]] * dt for k in range(K)])
        prob += y[t+1] == y[t] + pulp.lpSum([u[(t,k)] * sin_table[theta_mids[k]] * dt for k in range(K)])
        prob += theta[t+1] == theta[t] + omega[t] * dt
        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        prob += adx[t] >= dx
        prob += adx[t] >= -dx
        prob += adx[t] >= 0
        prob += ady[t] >= dy
        prob += ady[t] >= -dy
        prob += ady[t] >= 0

if optimize:
    prob += pulp.lpSum([adx[t] + ady[t] for t in range(T)])
else:
    prob += 0

solve_start = time.time()
used_solver = None
try:
    prob.solve(pulp.HiGHS_CMD(msg=False))
    used_solver = "HiGHS"
except (PulpSolverError, Exception) as e:
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    used_solver = "CBC (fallback)"
solve_end = time.time()

status = pulp.LpStatus[prob.status]
print("Solver used:", used_solver)
print("Solver status:", status)

if status in ("Optimal", "Feasible"):
    trajectory = [(pulp.value(x[t]), pulp.value(y[t])) for t in range(T+1)]
    total_length = sum((pulp.value(adx[t]) + pulp.value(ady[t])) for t in range(T))
    print(f"Trajectory found with length ≈ {total_length:.6f}")
    print(f"Total steps (T) = {T}, total time = {T*dt}")

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

end_time = time.time()

build_time_ms = (solve_start - start_time) * 1000
solve_time_ms = (solve_end - solve_start) * 1000
total_time_ms = (end_time - start_time) * 1000

print(f"\n--- Timing Summary ---")
print(f"Model build time: {build_time_ms:.3f} ms  ({build_time_ms/1000:.6f} s)")
print(f"Solving time:     {solve_time_ms:.3f} ms  ({solve_time_ms/1000:.6f} s)")
print(f"Total runtime:    {total_time_ms:.3f} ms  ({total_time_ms/1000:.6f} s)\n")