import ast
import math
import time
import pulp
import random
import pandas as pd
from pulp import PulpSolverError
from google.colab import files

# Parameters
time_bound = 210.0
dt = 1.0
T = math.ceil(time_bound / dt)
margin = 1.5
quantity = 10
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

print("Please upload your CSV file")
uploaded = files.upload()
input_file = list(uploaded.keys())[0]
output_file = "Unicycle Model Result.csv"

def parser(s):
    if pd.isna(s):
        return None
    try:
        val = ast.literal_eval(s)
        return val
    except (ValueError, SyntaxError):
        return None

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

def point_inside_inflated(px, py, rect, margin):
    x1, y1, x2, y2 = rect
    return (px > x1 - margin + 1e-8 and px < x2 + margin - 1e-8 and py > y1 - margin + 1e-8 and py < y2 + margin - 1e-8)

def milp_unicycle_solver(time_bound, dt, vmin, vmax, wmin, wmax, thmin, thmax, dth, margin, start_region, goal_region, obstacles, optimize=True):
    start_time = time.time()
    T = math.ceil(time_bound / dt)
    degtorad = math.pi / 180.0
    dtheta = dth
    Vmin, Vmax = vmin, vmax
    Wmin, Wmax = wmin, wmax
    THmin, THmax = thmin, thmax

    # Angle discretization and trigonometric tables
    theta_mids = [THmin + dtheta/2.0 + i*dtheta for i in range(int((THmax - THmin) / dtheta))]
    cos_table = {mid: math.cos(mid*degtorad) for mid in theta_mids}
    sin_table = {mid: math.sin(mid*degtorad) for mid in theta_mids}
    K = len(theta_mids)

    # Big-M constants
    max_x = max(goal_region[2], start_region[2])
    max_y = max(goal_region[3], start_region[3])
    Mx = max(max_x, max_y) + margin + 10.0
    Mtheta = 360.0
    Mv = Vmax

    # Build MILP
    prob = pulp.LpProblem("MILP_Path_Planner", pulp.LpMinimize)

    # Variables
    x = {t: pulp.LpVariable(f"x_{t}", lowBound=0.0, upBound=goal_region[2]) for t in range(T+1)}
    y = {t: pulp.LpVariable(f"y_{t}", lowBound=0.0, upBound=goal_region[3]) for t in range(T+1)}
    v = {t: pulp.LpVariable(f"v_{t}", lowBound=Vmin, upBound=Vmax) for t in range(T)}
    omega = {t: pulp.LpVariable(f"omega_{t}", lowBound=Wmin, upBound=Wmax) for t in range(T)}
    theta = {t: pulp.LpVariable(f"theta_{t}", lowBound=THmin, upBound=THmax) for t in range(T+1)}
    s = {(t,k): pulp.LpVariable(f"s_{t}_{k}", cat="Binary") for t in range(T) for k in range(K)}
    u = {(t,k): pulp.LpVariable(f"u_{t}_{k}", lowBound=Vmin, upBound=Vmax) for t in range(T) for k in range(K)}
    adx = {t: pulp.LpVariable(f"adx_{t}", lowBound=0.0) for t in range(T)}
    ady = {t: pulp.LpVariable(f"ady_{t}", lowBound=0.0) for t in range(T)}

    sx1, sy1, sx2, sy2 = start_region
    gx1, gy1, gx2, gy2 = goal_region

    # Start & goal regions
    prob += x[0] >= sx1
    prob += x[0] <= sx2
    prob += y[0] >= sy1
    prob += y[0] <= sy2

    prob += x[T] >= gx1
    prob += x[T] <= gx2
    prob += y[T] >= gy1
    prob += y[T] <= gy2

    # Bounds
    for t in range(T+1):
        prob += x[t] >= 0.0
        prob += x[t] <= goal_region[2]
        prob += y[t] >= 0.0
        prob += y[t] <= goal_region[3]

        # Obstacle avoidance (with +M*b_i formulation)
        for (x1, y1, x2, y2) in obstacles:
            b1 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b1", cat="Binary")
            b2 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b2", cat="Binary")
            b3 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b3", cat="Binary")
            b4 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b4", cat="Binary")

            # At least one side outside (sum ≥ 1)
            prob += b1 + b2 + b3 + b4 == 3

            prob += x[t] <= (x1 - margin) + Mx * b1
            prob += x[t] >= (x2 + margin) - Mx * b2
            prob += y[t] <= (y1 - margin) + Mx * b3
            prob += y[t] >= (y2 + margin) - Mx * b4

    # Motion dynamics
    for t in range(T):
        prob += pulp.lpSum([s[(t,k)] for k in range(K)]) == 1  # one theta bin active

        for k, mid in enumerate(theta_mids):
            lower = mid - dtheta/2.0
            upper = mid + dtheta/2.0
            prob += theta[t] >= lower - Mtheta*(1 - s[(t,k)])
            prob += theta[t] <= upper + Mtheta*(1 - s[(t,k)])
            prob += u[(t,k)] <= Vmax * s[(t,k)]
            prob += u[(t,k)] >= Vmin * s[(t,k)]
            prob += u[(t,k)] <= v[t] - Vmin * (1 - s[(t,k)])
            prob += u[(t,k)] >= v[t] - Vmax * (1 - s[(t,k)])

        # Position updates
        prob += x[t+1] == x[t] + pulp.lpSum([u[(t,k)] * cos_table[theta_mids[k]] * dt for k in range(K)])
        prob += y[t+1] == y[t] + pulp.lpSum([u[(t,k)] * sin_table[theta_mids[k]] * dt for k in range(K)])
        prob += theta[t+1] == theta[t] + omega[t] * dt

        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        prob += adx[t] >= dx
        prob += adx[t] >= -dx
        prob += ady[t] >= dy
        prob += ady[t] >= -dy

    # Objective: minimize L1 path length (sum of adx + ady)
    if optimize:
        prob += pulp.lpSum([adx[t] + ady[t] for t in range(T)])
    else:
        prob += 0

    build_end = time.time()
    build_time_ms = (build_end - start_time) * 1000
    used_solver = "None"

    try:
        solver = pulp.GUROBI_CMD(msg=False, timeLimit=600)
        prob.solve(solver)
        used_solver = "Gurobi"
    except (PulpSolverError, Exception):
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        used_solver = "CBC"

    solve_end = time.time()
    solve_time_ms = (solve_end - build_end) * 1000
    total_time_ms = (solve_end - start_time) * 1000

    status = pulp.LpStatus[prob.status]
    if status in ("Optimal", "Feasible"):
        trajectory = [(pulp.value(x[t]), pulp.value(y[t])) for t in range(T+1)]
        velocities = [pulp.value(v[t]) for t in range(T)]
        omegas = [pulp.value(omega[t]) for t in range(T)]
        thetas = [pulp.value(theta[t]) for t in range(T+1)]
        total_length = sum((pulp.value(adx[t]) + pulp.value(ady[t])) for t in range(T))
        data_points = [{
            "t": t,
            "x": trajectory[t][0],
            "y": trajectory[t][1],
            "v": velocities[t] if t < T else None,
            "omega": omegas[t] if t < T else None,
            "theta": thetas[t] if t < T+1 else None
        } for t in range(T+1)]
        violations = []
        for i,(px,py) in enumerate(trajectory):
            if px is None or py is None:
                continue
            for ridx,rect in enumerate(obstacles):
                if point_inside_inflated(px,py,rect,margin):
                    violations.append((i, ridx, px, py))
    else:
        trajectory = []
        total_length = None
        data_points = []
        violations = []
    return {
        "BOUND": time_bound,
        "DT": dt,
        "VMIN": vmin,
        "VMAX": vmax,
        "WMIN": wmin,
        "WMAX": wmax,
        "THMIN": thmin,
        "THMAX": thmax,
        "DTHETA": dtheta,
        "MARGIN": margin,
        "START": str(start_region),
        "GOAL": str(goal_region),
        "OBSTACLES": str(obstacles),
        "OPTIMIZE": optimize,
        "STATUS": status,
        "SOLVER": used_solver,
        "TOTAL STEPS": T,
        "LENGTH": total_length,
        "VIOLATIONS": len(violations),
        "BUILD TIME (ms)": build_time_ms,
        "SOLVE TIME (ms)": solve_time_ms,
        "TOTAL TIME (ms)": total_time_ms,
        "DATA POINTS": str(data_points),
        "VIOLATION POINTS": str(violations),
    }

df = pd.read_csv(input_file)

# for col in ['BOUND', 'DT', 'VMIN', 'VMAX', 'WMIN', 'WMAX', 'THMIN', 'THMAX', 'DTHETA', 'MARGIN', 'START', 'GOAL', 'OBSTACLES', 'OPTIMIZE']:
#     if col not in df.columns:
#         raise ValueError(f"Missing required column '{col}' in CSV file!")

result_cols = ['STATUS', 'SOLVER', 'TOTAL STEPS', 'LENGTH', 'VIOLATIONS', 'BUILD TIME (ms)', 'SOLVE TIME (ms)', 'TOTAL TIME (ms)', 'DATA POINTS', 'VIOLATION POINTS']
for col in result_cols:
    if col not in df.columns:
        df[col] = None
    df[col] = df[col].astype('object')

initial_points = generate_points(start_region[0], start_region[2], start_region[1], start_region[3], quantity)

# for i, row in df.iterrows():
for i, item in enumerate(initial_points):
    print(f"\nSolving instance {i+1}/{len(initial_points)}")
    result = milp_unicycle_solver(
        # time_bound=float(row['BOUND']),
        # dt=float(row['DT']),
        # vmin=float(row['VMIN']),
        # vmax=float(row['VMAX']),
        # wmin=float(row['WMIN']),
        # wmax=float(row['WMAX']),
        # thmin=float(row['THMIN']),
        # thmax=float(row['THMAX']),
        # dth=float(row['DTHETA']),
        # margin=float(row['MARGIN']),
        # start_region=parser(row['START']),
        # goal_region=parser(row['GOAL']),
        # obstacles=parser(row['OBSTACLES']),
        # optimize=bool(row['OPTIMIZE'])

        time_bound=time_bound,
        dt=dt,
        vmin=Vmin,
        vmax=Vmax,
        wmin=Wmin,
        wmax=Wmax,
        thmin=THmin,
        thmax=THmax,
        dth=dtheta,
        margin=margin,
        start_region=(item[0], item[1], item[0], item[1]),
        goal_region=goal_region,
        obstacles=obstacles,
        optimize=optimize
    )
    for key, val in result.items():
        df.loc[i, key] = val

df.to_csv(output_file, index=False)
print(f"\nAll instances processed and results saved to '{output_file}'")
files.download(output_file)
