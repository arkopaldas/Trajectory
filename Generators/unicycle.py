import os
import ast
import math
import time
import pulp
import random
import pandas as pd
from tqdm import tqdm
from pulp import PulpSolverError
from multiprocessing import Pool, cpu_count


time_bound = 70.0
dt = 3.0
T = math.ceil(time_bound / dt)
degtorad = math.pi / 180.0
margin = 2.0
dtheta = 45 * degtorad
optimize = True
Vmin, Vmax = 0.0, dt
Wmin, Wmax = -90.0 * degtorad, 90.0 * degtorad
THmin, THmax = -22.5 * degtorad, 112.5 * degtorad
max_instances = 10000
num_workers = cpu_count()

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
initial_points = []

theta_mids = [THmin + dtheta/2.0 + i*dtheta for i in range(int((THmax - THmin) / dtheta))]
cos_table = {mid: math.cos(mid) for mid in theta_mids}
sin_table = {mid: math.sin(mid) for mid in theta_mids}
K = len(theta_mids)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "..", "Datasets", "Schemas", "Unicycle Model Dataset Schema.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "Datasets", "Results", "Unicycle Model Result.csv")

def parser(s):
    if pd.isna(s):
        return None
    try:
        val = ast.literal_eval(s)
        return val
    except (ValueError, SyntaxError):
        return None

def generate_points(region=(0,0,10,10), n=10, offset=1, attempts=1000000):
    xmin, ymin, xmax, ymax = region
    points = []
    check = 0
    while len(points) < n and check <= attempts:
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        if all(math.hypot(x - px, y - py) >= offset for px, py in points):
            points.append((x, y))
        check = check + 1
    if len(points) < n:
        print(f"Could only place {len(points)} points. Try reducing n, offset, or region size.")
    return points

def solve_instance(args):
    i, item = args
    result = unicycle_solver(
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
        start=(item[0], item[1], item[0], item[1]),
        goal=goal_region,
        obstacles=obstacles,
        optimize=optimize
    )
    return (i, result)

def unicycle_solver(time_bound, dt, vmin, vmax, wmin, wmax, thmin, thmax, dth, margin, start, goal, obstacles, optimize=True):
    start_time = time.time()
    T = math.ceil(time_bound / dt)
    dtheta = dth
    Vmin, Vmax = vmin, vmax
    Wmin, Wmax = wmin, wmax
    THmin, THmax = thmin, thmax

    Mx = max(goal[2], start[2])
    My = max(goal[3], start[3])
    Mtheta = 2 * math.pi

    prob = pulp.LpProblem("MILP_Path_Planner", pulp.LpMinimize)

    x = {t: pulp.LpVariable(f"x_{t}", lowBound=0.0, upBound=goal[2]) for t in range(T+1)}
    y = {t: pulp.LpVariable(f"y_{t}", lowBound=0.0, upBound=goal[3]) for t in range(T+1)}
    v = {t: pulp.LpVariable(f"v_{t}", lowBound=Vmin, upBound=Vmax) for t in range(T)}
    omega = {t: pulp.LpVariable(f"omega_{t}", lowBound=Wmin, upBound=Wmax) for t in range(T)}
    theta = {t: pulp.LpVariable(f"theta_{t}", lowBound=THmin, upBound=THmax) for t in range(T+1)}
    s = {(t,k): pulp.LpVariable(f"s_{t}_{k}", cat="Binary") for t in range(T) for k in range(K)}
    u = {(t,k): pulp.LpVariable(f"u_{t}_{k}", lowBound=Vmin, upBound=Vmax) for t in range(T) for k in range(K)}
    adx = {t: pulp.LpVariable(f"adx_{t}", lowBound=0.0) for t in range(T)}
    ady = {t: pulp.LpVariable(f"ady_{t}", lowBound=0.0) for t in range(T)}

    sx1, sy1, sx2, sy2 = start
    gx1, gy1, gx2, gy2 = goal

    prob += x[0] >= sx1
    prob += x[0] <= sx2
    prob += y[0] >= sy1
    prob += y[0] <= sy2

    prob += x[T] >= gx1
    prob += x[T] <= gx2
    prob += y[T] >= gy1
    prob += y[T] <= gy2

    for t in range(T+1):
        prob += x[t] >= 0.0
        prob += x[t] <= goal[2]
        prob += y[t] >= 0.0
        prob += y[t] <= goal[3]
        for (x1, y1, x2, y2) in obstacles:
            b1 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b1", cat="Binary")
            b2 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b2", cat="Binary")
            b3 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b3", cat="Binary")
            b4 = pulp.LpVariable(f"obs_{t}_{x1}_{y1}_{x2}_{y2}_b4", cat="Binary")
            prob += b1 + b2 + b3 + b4 == 3
            prob += x[t] <= (x1 - margin) + Mx * b1
            prob += x[t] >= (x2 + margin) - Mx * b2
            prob += y[t] <= (y1 - margin) + My * b3
            prob += y[t] >= (y2 + margin) - My * b4

        if t < T:
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
            prob += ady[t] >= dy
            prob += ady[t] >= -dy

    if optimize:
        prob += pulp.lpSum([adx[t] + ady[t] for t in range(T)])
    else:
        prob += 0

    build_end = time.time()
    build_time_ms = (build_end - start_time) * 1000
    used_solver = "None"

    try:
        prob.solve(pulp.HiGHS_CMD(msg=False))
        used_solver = "HiGHS"
    except (PulpSolverError, Exception):
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
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
    else:
        trajectory = []
        total_length = None
        data_points = []
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
        "BUILD TIME (ms)": build_time_ms,
        "SOLVE TIME (ms)": solve_time_ms,
        "TOTAL TIME (ms)": total_time_ms,
        "DATA POINTS": str(data_points),
    }

if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    result_cols = ['STATUS', 'SOLVER', 'TOTAL STEPS', 'LENGTH', 'BUILD TIME (ms)', 'SOLVE TIME (ms)', 'TOTAL TIME (ms)', 'DATA POINTS']
    for col in result_cols:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype('object')

    initial_points = generate_points(start_region, max_instances, 0.001)
    print(f"Using {num_workers} workers")
    inputs = list(enumerate(initial_points))
    results = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(solve_instance, inputs), total=len(inputs), desc="Solving Instances"):
            results.append(result)
    print(f"\nAll instances are solved\n")
    for i, result in tqdm(results, total=len(results), desc="Writing Results"):
        for key, val in result.items():
            df.loc[i, key] = val

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAll results are saved to '{OUTPUT_FILE}'")
