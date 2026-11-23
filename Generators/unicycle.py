"""Batch dataset generator for the UNICYCLE model (forward speed v, turn rate omega).

What it does
    1. Draws MAX_INSTANCES random start points inside START_REGION (generate_points).
    2. Solves the planning problem from each start point as a MILP (unicycle_solver), in parallel on NUM_WORKERS
       processes. HiGHS is tried first (time limit 60 s per instance); if it fails, CBC is used.
    3. Writes one CSV row per start point (status, solver used, path length, timings and the trajectory).

Model used in the MILP (APPROXIMATE, see Controllers/milp_unicycle.py for the same formulation in stand-alone form):
    x' = x + v*cos(theta_bin)*dt, y' = y + v*sin(theta_bin)*dt, theta' = theta + omega*dt
    where theta_bin is the centre of the heading bin that theta lies in (bins of width DTHETA).
    The trajectories are therefore only approximately consistent with the exact unicycle equations.

Inputs : the schema CSV (INPUT_FILE, header only, from Datasets/Schemas) and the constants in config.py.
Output : Datasets/Results/<MAX_INSTANCES> Trajectories/Unicycle Model Result.csv
         (read later by Neural Networks/train_unicycle.py and Verification/verify_unicycle.py)
Run    : python unicycle.py        (from inside the Generators folder)

Row format: STATUS is the PuLP status string ("Optimal", "Infeasible", "Not Solved", ...). DATA POINTS = str(list of
dicts) with keys t, x, y, v, omega, theta; v and omega of the last point are None. Infeasible rows have no DATA POINTS.
"""
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

# Heading bins, computed once per process: centre of each bin of width DTHETA between THMIN and THMAX, plus the
# cos / sin of every centre. K = number of bins.
theta_mids = [THMIN + DTHETA/2.0 + i*DTHETA for i in range(int((THMAX - THMIN) / DTHETA))]
cos_table = {mid: math.cos(mid) for mid in theta_mids}
sin_table = {mid: math.sin(mid) for mid in theta_mids}
K = len(theta_mids)

# Turns a cell that holds a Python literal stored as text (e.g. the DATA POINTS column) back into an object.
# Returns None for empty or malformed cells. (Helper for analysis code; not used by the generator itself.)
def parser(s):
    if pd.isna(s):
        return None
    try:
        val = ast.literal_eval(s)
        return val
    except (ValueError, SyntaxError):
        return None

# Rejection sampling of start points inside `region`: a point is kept only if it is at least `offset` away from every
# point kept so far. Stops after `attempts` draws (then fewer than `n` points are returned and a message is printed).
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

# Worker entry point for multiprocessing.Pool: unpacks (index, (x, y)), solves from that point (degenerate start
# region x, y, x, y) and returns (index, result_dict) so the results can be re-ordered after imap_unordered.
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

# Builds and solves the MILP for ONE start region and returns a dict with one entry per CSV column.
# Timings are in milliseconds: build = model construction, solve = solver call, total = both.
def unicycle_solver(time_bound, dt, vmin, vmax, wmin, wmax, thmin, thmax, dth, margin, start, goal, obstacles, optimize=True):
    start_time = time.time()
    T_local = math.ceil(time_bound / dt)

    # Big-M constants for the disjunctive obstacle constraints (workspace size) and for the heading-bin constraints (2*pi).
    Mx = max(goal[2], start[2])
    My = max(goal[3], start[3])
    Mtheta = 2 * math.pi

    prob = pulp.LpProblem("MILP_Path_Planner", pulp.LpMinimize)

    # Variables: x, y, theta for t = 0..T; v, omega for t = 0..T-1.
    # s[t,k] (binary): heading at step t lies in bin k. u[t,k]: equals v[t] if bin k is active, else 0 (keeps v*cos linear).
    # adx, ady: |dx|, |dy| of each step, used only by the objective. Variable bounds already encode the v / omega / theta limits.
    x = {t: pulp.LpVariable(f"x_{t}", lowBound=0.0, upBound=goal[2]) for t in range(T_local+1)}
    y = {t: pulp.LpVariable(f"y_{t}", lowBound=0.0, upBound=goal[3]) for t in range(T_local+1)}
    v = {t: pulp.LpVariable(f"v_{t}", lowBound=vmin, upBound=vmax) for t in range(T_local)}
    omega = {t: pulp.LpVariable(f"omega_{t}", lowBound=wmin, upBound=wmax) for t in range(T_local)}
    theta = {t: pulp.LpVariable(f"theta_{t}", lowBound=thmin, upBound=thmax) for t in range(T_local+1)}
    s = {(t,k): pulp.LpVariable(f"s_{t}_{k}", cat="Binary") for t in range(T_local) for k in range(K)}
    u = {(t,k): pulp.LpVariable(f"u_{t}_{k}", lowBound=vmin, upBound=vmax) for t in range(T_local) for k in range(K)}
    adx = {t: pulp.LpVariable(f"adx_{t}", lowBound=0.0) for t in range(T_local)}
    ady = {t: pulp.LpVariable(f"ady_{t}", lowBound=0.0) for t in range(T_local)}

    # Boundary conditions: start inside the start region, end inside the goal region.
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

    # Per-step constraints: workspace box, obstacle avoidance and (for t < T) the dynamics.
    for t in range(T_local+1):
        prob += x[t] >= 0.0
        prob += x[t] <= goal[2]
        prob += y[t] >= 0.0
        prob += y[t] <= goal[3]
        # Obstacle avoidance (big-M, 'either-or'): four binaries per obstacle and step, exactly one of them is 0 (sum == 3).
        # The one that is 0 activates its constraint: left of / right of / below / above the obstacle by at least `margin`.
        # The other three constraints are relaxed by the big-M.
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

        # Dynamics. Exactly one heading bin is selected; theta must lie in that bin's interval (relaxed for the other bins);
        # u[t,k] is tied to v[t] for the selected bin and to 0 otherwise.
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

            # Position update with the cos / sin of the selected bin centre, and the exact heading update.
            prob += x[t+1] == x[t] + pulp.lpSum([u[(t,k)] * cos_table[theta_mids[k]] * dt for k in range(K)])
            prob += y[t+1] == y[t] + pulp.lpSum([u[(t,k)] * sin_table[theta_mids[k]] * dt for k in range(K)])
            prob += theta[t+1] == theta[t] + omega[t] * dt
            # Absolute step lengths for the objective.
            dx = x[t+1] - x[t]
            dy = y[t+1] - y[t]
            prob += adx[t] >= dx
            prob += adx[t] >= -dx
            prob += ady[t] >= dy
            prob += ady[t] >= -dy

    # Objective: minimise the total Manhattan path length, or (optimize False) a constant 0 objective = feasibility only.
    if optimize:
        prob += pulp.lpSum([adx[t] + ady[t] for t in range(T_local)])
    else:
        prob += 0

    # Model construction is finished; everything after this is solver time.
    build_end = time.time()
    build_time_ms = (build_end - start_time) * 1000
    # Solve: HiGHS with a 60 s limit; if it is missing or raises any error, fall back to CBC (which ships with PuLP).
    # The solver that was actually used is stored in the SOLVER column.
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

    # Translate the solver status to text and, if a solution exists, read trajectory, controls and headings out of the model.
    status = pulp.LpStatus[prob.status]
    if status in ("Optimal", "Feasible"):
        trajectory = [(pulp.value(x[t]), pulp.value(y[t])) for t in range(T_local+1)]
        velocities = [pulp.value(v[t]) for t in range(T_local)]
        omegas = [pulp.value(omega[t]) for t in range(T_local)]
        thetas = [pulp.value(theta[t]) for t in range(T_local+1)]
        total_length = sum((pulp.value(adx[t]) + pulp.value(ady[t])) for t in range(T_local))
        # Per-step records (t, x, y, v, omega, theta). v and omega of the last state are None (no control after the end).
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
    # One value per CSV column. NOTE: START / GOAL / OBSTACLES store the global regions from config (START is the whole start
    # region); the actual start point of an instance is the first entry of DATA POINTS.
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

# ---- Main: generate start points, solve in parallel, write the CSV ----
if __name__ == "__main__":
    # Load the (header-only) schema CSV that fixes the column order; add any missing result columns as 'object' columns.
    df = pd.read_csv(INPUT_FILE)
    result_cols = ['STATUS', 'SOLVER', 'TOTAL STEPS', 'LENGTH', 'BUILD TIME (ms)', 'SOLVE TIME (ms)', 'TOTAL TIME (ms)', 'DATA POINTS']
    for col in result_cols:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype('object')

    # One start point per instance; the enumerate index becomes the CSV row number.
    initial_points = generate_points(START_REGION, MAX_INSTANCES, EPSILON)
    print(f"\nUsing {NUM_WORKERS} workers\n")
    inputs = list(enumerate(initial_points))
    results = []
    # Solve all instances in parallel (results arrive in arbitrary order; the index restores the order).
    with Pool(processes=NUM_WORKERS) as pool:
        for result in progress(pool.imap_unordered(solve_instance, inputs, chunksize=1), total=len(inputs), desc="Solving Instances"):
            results.append(result)
    print(f"\nAll instances are solved\n")
    # Write each result into row i of the table.
    for i, result in progress(results, total=len(results), desc="Writing Results"):
        for key, val in result.items():
            df.loc[i, key] = val

    # Create the output folder if needed and save the CSV.
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAll results are saved to successfully ^_^ !")