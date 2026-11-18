import os
import ast
import math
import time
import pulp
import random
import pandas as pd
from progress import progress
from pulp import PulpSolverError
from multiprocessing import Pool

from config import (
    OPTIMIZE, MARGIN, UNICYCLE_TIME_BOUND as TIME_BOUND, MAX_INSTANCES, EPSILON, NUM_WORKERS,
    START_REGION, GOAL_REGION, OBSTACLES,
    UNICYCLE_DT as DT, UNICYCLE_T as T, UNICYCLE_DTHETA as DTHETA,
    UNICYCLE_VMIN as VMIN, UNICYCLE_VMAX as VMAX,
    UNICYCLE_WMIN as WMIN, UNICYCLE_WMAX as WMAX,
    UNICYCLE_THMIN as THMIN, UNICYCLE_THMAX as THMAX,
    INPUT_FILE_UNICYCLE as INPUT_FILE, OUTPUT_FILE_UNICYCLE as OUTPUT_FILE
)

theta_mids = [THMIN + DTHETA/2.0 + i*DTHETA for i in range(int((THMAX - THMIN) / DTHETA))]
cos_table = {mid: math.cos(mid) for mid in theta_mids}
sin_table = {mid: math.sin(mid) for mid in theta_mids}
K = len(theta_mids)

def parser(s):
    if pd.isna(s):
        return None
    try:
        val = ast.literal_eval(s)
        return val
    except (ValueError, SyntaxError):
        return None

def generate_points(region=START_REGION, n=MAX_INSTANCES, offset=EPSILON, attempts=1000000):
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
        time_bound=TIME_BOUND,
        dt=DT,
        vmin=VMIN,
        vmax=VMAX,
        wmin=WMIN,
        wmax=WMAX,
        thmin=THMIN,
        thmax=THMAX,
        dth=DTHETA,
        margin=MARGIN,
        start=(item[0], item[1], item[0], item[1]),
        goal=GOAL_REGION,
        obstacles=OBSTACLES,
        optimize=OPTIMIZE
    )
    return (i, result)

def unicycle_solver(time_bound, dt, vmin, vmax, wmin, wmax, thmin, thmax, dth, margin, start, goal, obstacles, optimize=True):
    start_time = time.time()
    T_local = math.ceil(time_bound / dt)

    Mx = max(goal[2], start[2])
    My = max(goal[3], start[3])
    Mtheta = 2 * math.pi

    prob = pulp.LpProblem("MILP_Path_Planner", pulp.LpMinimize)

    x = {t: pulp.LpVariable(f"x_{t}", lowBound=0.0, upBound=goal[2]) for t in range(T_local+1)}
    y = {t: pulp.LpVariable(f"y_{t}", lowBound=0.0, upBound=goal[3]) for t in range(T_local+1)}
    v = {t: pulp.LpVariable(f"v_{t}", lowBound=vmin, upBound=vmax) for t in range(T_local)}
    omega = {t: pulp.LpVariable(f"omega_{t}", lowBound=wmin, upBound=wmax) for t in range(T_local)}
    theta = {t: pulp.LpVariable(f"theta_{t}", lowBound=thmin, upBound=thmax) for t in range(T_local+1)}
    s = {(t,k): pulp.LpVariable(f"s_{t}_{k}", cat="Binary") for t in range(T_local) for k in range(K)}
    u = {(t,k): pulp.LpVariable(f"u_{t}_{k}", lowBound=vmin, upBound=vmax) for t in range(T_local) for k in range(K)}
    adx = {t: pulp.LpVariable(f"adx_{t}", lowBound=0.0) for t in range(T_local)}
    ady = {t: pulp.LpVariable(f"ady_{t}", lowBound=0.0) for t in range(T_local)}

    sx1, sy1, sx2, sy2 = start
    gx1, gy1, gx2, gy2 = goal

    prob += x[0] >= sx1
    prob += x[0] <= sx2
    prob += y[0] >= sy1
    prob += y[0] <= sy2

    prob += x[T_local] >= gx1
    prob += x[T_local] <= gx2
    prob += y[T_local] >= gy1
    prob += y[T_local] <= gy2

    for t in range(T_local+1):
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

        if t < T_local:
            prob += pulp.lpSum([s[(t,k)] for k in range(K)]) == 1
            for k, mid in enumerate(theta_mids):
                lower = mid - dth/2.0
                upper = mid + dth/2.0
                prob += theta[t] >= lower - Mtheta*(1 - s[(t,k)])
                prob += theta[t] <= upper + Mtheta*(1 - s[(t,k)])
                prob += u[(t,k)] <= vmax * s[(t,k)]
                prob += u[(t,k)] >= vmin * s[(t,k)]
                prob += u[(t,k)] <= v[t] - vmin * (1 - s[(t,k)])
                prob += u[(t,k)] >= v[t] - vmax * (1 - s[(t,k)])

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
        prob += pulp.lpSum([adx[t] + ady[t] for t in range(T_local)])
    else:
        prob += 0

    build_end = time.time()
    build_time_ms = (build_end - start_time) * 1000
    used_solver = "None"

    try:
        prob.solve(pulp.HiGHS_CMD(msg=False, timeLimit=60))
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
        trajectory = [(pulp.value(x[t]), pulp.value(y[t])) for t in range(T_local+1)]
        velocities = [pulp.value(v[t]) for t in range(T_local)]
        omegas = [pulp.value(omega[t]) for t in range(T_local)]
        thetas = [pulp.value(theta[t]) for t in range(T_local+1)]
        total_length = sum((pulp.value(adx[t]) + pulp.value(ady[t])) for t in range(T_local))
        data_points = [{
            "t": t,
            "x": trajectory[t][0],
            "y": trajectory[t][1],
            "v": velocities[t] if t < T_local else None,
            "omega": omegas[t] if t < T_local else None,
            "theta": thetas[t] if t < T_local+1 else None
        } for t in range(T_local+1)]
    else:
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
        "DTHETA": dth,
        "MARGIN": margin,
        "START": str(START_REGION),
        "GOAL": str(GOAL_REGION),
        "OBSTACLES": str(OBSTACLES),
        "OPTIMIZE": optimize,
        "STATUS": status,
        "SOLVER": used_solver,
        "TOTAL STEPS": T_local,
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

    initial_points = generate_points(START_REGION, MAX_INSTANCES, EPSILON)
    print(f"\nUsing {NUM_WORKERS} workers\n")
    inputs = list(enumerate(initial_points))
    results = []
    with Pool(processes=NUM_WORKERS) as pool:
        for result in progress(pool.imap_unordered(solve_instance, inputs, chunksize=1), total=len(inputs), desc="Solving Instances"):
            results.append(result)
    print(f"\nAll instances are solved\n")
    for i, result in progress(results, total=len(results), desc="Writing Results"):
        for key, val in result.items():
            df.loc[i, key] = val

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAll results are saved to successfully ^_^ !")