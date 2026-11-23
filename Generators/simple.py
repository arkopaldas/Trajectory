"""Batch dataset generator for the SIMPLE model (velocity-controlled point).

What it does
    1. Draws MAX_INSTANCES random start points inside START_REGION (generate_points).
    2. Solves the planning problem from each start point with the Z3 SMT optimiser (simple_solver), in parallel on
       NUM_WORKERS processes. The encoding is exact: x' = x + vx*dt, y' = y + vy*dt, |vx|,|vy| <= DT.
    3. Writes one CSV row per start point (status, path length, timings and the whole trajectory) to OUTPUT_FILE.

Inputs : the schema CSV (INPUT_FILE, header only, from Datasets/Schemas) and the constants in config.py.
Output : Datasets/Results/<MAX_INSTANCES> Trajectories/Simple Model Result.csv
         (read later by Neural Networks/train_simple.py and Verification/verify_simple.py)
Run    : python simple.py          (from inside the Generators folder, because config / progress are imported locally)

Row format (see Datasets/Schemas/README.md for the column-by-column description):
    STATUS = "SAT" (trajectory found) or "UNSAT" (none exists / none within the horizon).
    DATA POINTS = str(list of dicts) with keys t, x, y, vx, vy; vx / vy of the last point are None (no control after
    the final state). Use ast.literal_eval to turn the string back into Python objects.
"""
import os
import ast
import math
import time
import random
from z3 import *
import pandas as pd
from progress import progress
from multiprocessing import Pool

from config import (
    OPTIMIZE, MARGIN, TIME_BOUND, MAX_INSTANCES, EPSILON, NUM_WORKERS,
    START_REGION, GOAL_REGION, OBSTACLES,
    SIMPLE_DT as DT, SIMPLE_T as T, SIMPLE_VMIN as VMIN, SIMPLE_VMAX as VMAX,
    INPUT_FILE_SIMPLE as INPUT_FILE, OUTPUT_FILE_SIMPLE as OUTPUT_FILE
)

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

# Rejection sampling of start points: draw uniform random points in `region` and keep a point only if it is at least
# `offset` away from every point kept so far, so that no two instances start (almost) at the same place.
# Gives up after `attempts` draws and then returns fewer than `n` points (a message is printed).
# NOTE: the distance check against all previous points makes this O(n^2); for 100000 points it takes a while.
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

# Worker entry point for multiprocessing.Pool: unpacks (index, (x, y)), solves that start point and returns
# (index, result_dict). The start 'region' passed to the solver is degenerate (x, y, x, y), i.e. a single fixed point.
# The index is returned so results can be put back in order, because imap_unordered finishes in random order.
def solve_instance(args):
    i, item = args
    result = simple_solver(
        time_bound=TIME_BOUND,
        dt=DT,
        vmin=VMIN,
        vmax=VMAX,
        margin=MARGIN,
        start=(item[0], item[1], item[0], item[1]),
        goal=GOAL_REGION,
        obstacles=OBSTACLES,
        optimize=OPTIMIZE
    )
    return (i, result)

# Builds and solves the SMT problem for ONE start region and returns a dict with one entry per CSV column.
# Build/solve/total timings are measured in milliseconds. The encoding is the same as in Controllers/smt_simple.py;
# the helper functions are defined inside so that every worker process is self-contained.
def simple_solver(time_bound, dt, vmin, vmax, margin, start, goal, obstacles, optimize=True):
    start_time = time.time()
    # Number of time steps (horizon / step size, rounded up).
    T_local = math.ceil(time_bound / dt)

    # Z3 condition: point inside the closed rectangle.
    def inside_rect(x_v, y_v, rect):
        x1, y1, x2, y2 = rect
        return And(x_v >= x1, x_v <= x2, y_v >= y1, y_v <= y2)

    # Z3 condition: point outside the rectangle inflated by margin_v (outside one of its four inflated sides).
    def outside_rect(x_v, y_v, rect, margin_v=0.0):
        x1, y1, x2, y2 = rect
        return Or(x_v < x1 - margin_v, x_v > x2 + margin_v, y_v < y1 - margin_v, y_v > y2 + margin_v)

    # Reads a Z3 model value as a plain Python number.
    def get_val(model, var):
        val = model.evaluate(var, model_completion=True)
        if val is None:
            return None
        if val.is_int():
            return val.as_long()
        return float(val.as_fraction())

    # Encodes |expr| through a helper variable (var >= expr, var >= -expr, var >= 0); minimising var gives |expr|.
    def linearizer(opt_v, expr, prefix, step):
        var = Real(f"{prefix}_{step}")
        opt_v.add(var >= expr)
        opt_v.add(var >= -expr)
        opt_v.add(var >= 0)
        return var

    # Variables: positions x_0..x_T, y_0..y_T and velocities vx_0..vx_(T-1), vy_0..vy_(T-1).
    opt = Optimize()

    x = [Real(f"x_{t}") for t in range(T_local+1)]
    y = [Real(f"y_{t}") for t in range(T_local+1)]
    vx = [Real(f"vx_{t}") for t in range(T_local)]
    vy = [Real(f"vy_{t}") for t in range(T_local)]

    # Start inside the (single-point) start region, end inside the goal region.
    opt.add(inside_rect(x[0], y[0], start))
    opt.add(inside_rect(x[T_local], y[T_local], goal))

    step_lengths = []
    # For every state: stay in the workspace box and outside the inflated obstacles. For every control: respect the
    # velocity bounds and the dynamics. Also collect the per-step path length |dx| + |dy| for the objective.
    for t in range(T_local+1):
        opt.add(x[t] >= start[0], x[t] <= goal[2])
        opt.add(y[t] >= start[1], y[t] <= goal[3])
        for rect in obstacles:
            opt.add(outside_rect(x[t], y[t], rect, margin))
        if t < T_local:
            opt.add(vx[t] >= vmin, vx[t] <= vmax)
            opt.add(vy[t] >= vmin, vy[t] <= vmax)
            opt.add(x[t+1] == x[t] + vx[t] * dt)
            opt.add(y[t+1] == y[t] + vy[t] * dt)
            step_lengths.append(linearizer(opt, x[t+1] - x[t], "adx", t) + linearizer(opt, y[t+1] - y[t], "ady", t))

    # Objective: minimise the total (Manhattan) path length when optimize is True.
    total_path_length = Sum(step_lengths)
    if optimize:
        opt.minimize(total_path_length)

    # Solve and time it (the 'build time' is everything before this point).
    solve_start = time.time()
    result = opt.check()
    solve_end = time.time()

    end_time = time.time()
    build_time_ms = (solve_start - start_time) * 1000
    solve_time_ms = (solve_end - solve_start) * 1000
    total_time_ms = (end_time - start_time) * 1000

    # SAT: read the trajectory and controls out of the model and store them as a list of per-step dicts (t, x, y, vx, vy).
    # The last state has no control, hence None for vx / vy.
    if result == sat:
        model = opt.model()
        trajectory = [(get_val(model, x[t]), get_val(model, y[t])) for t in range(T_local+1)]
        velocities = [(get_val(model, vx[t]), get_val(model, vy[t])) for t in range(T_local)]
        trajectory_length = get_val(model, total_path_length)
        data_points = [{
            "t": t,
            "x": trajectory[t][0],
            "y": trajectory[t][1],
            "vx": velocities[t][0] if t < T_local else None,
            "vy": velocities[t][1] if t < T_local else None
        } for t in range(T_local+1)]
        status = "SAT"
    else:
        trajectory_length = None
        data_points = []
        status = "UNSAT"

    # One value per CSV column. NOTE: START / GOAL / OBSTACLES store the global regions from config (START is the whole start
    # region, not the individual start point); the real start point of this instance is the first entry of DATA POINTS.
    return {
        "BOUND": time_bound,
        "DT": dt,
        "VMIN": vmin,
        "VMAX": vmax,
        "MARGIN": margin,
        "START": str(START_REGION),
        "GOAL": str(GOAL_REGION),
        "OBSTACLES": str(OBSTACLES),
        "OPTIMIZE": optimize,
        "STATUS": status,
        "TOTAL STEPS": T_local,
        "LENGTH": trajectory_length,
        "BUILD TIME (ms)": build_time_ms,
        "SOLVE TIME (ms)": solve_time_ms,
        "TOTAL TIME (ms)": total_time_ms,
        "DATA POINTS": str(data_points),
    }

# ---- Main: generate start points, solve in parallel, write the CSV ----
if __name__ == "__main__":
    # Load the (header-only) schema CSV: it fixes the column order of the result file. Missing result columns are added and
    # made 'object' typed so that they can hold lists / strings / None.
    df = pd.read_csv(INPUT_FILE)
    result_cols = ['STATUS', 'TOTAL STEPS', 'LENGTH', 'BUILD TIME (ms)', 'SOLVE TIME (ms)', 'TOTAL TIME (ms)', 'DATA POINTS']
    for col in result_cols:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype('object')

    # One start point per instance. The enumerate index becomes the CSV row number.
    initial_points = generate_points(START_REGION, MAX_INSTANCES, EPSILON)
    print(f"\nUsing {NUM_WORKERS} workers\n")
    inputs = list(enumerate(initial_points))
    results = []
    # Solve all instances in parallel; imap_unordered hands back results as soon as any worker finishes one.
    # chunksize=1 because individual solves can differ a lot in duration.
    with Pool(processes=NUM_WORKERS) as pool:
        for result in progress(pool.imap_unordered(solve_instance, inputs, chunksize=1), total=len(inputs), desc="Solving Instances"):
            results.append(result)
    print(f"\nAll instances are solved\n")
    # Write each result into row i of the table (assigning to a new row label appends the row).
    for i, result in progress(results, total=len(results), desc="Writing Results"):
        for key, val in result.items():
            df.loc[i, key] = val

    # Create the output folder if it does not exist yet and save the CSV.
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAll results are saved to successfully ^_^ !")