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

def simple_solver(time_bound, dt, vmin, vmax, margin, start, goal, obstacles, optimize=True):
    start_time = time.time()
    T_local = math.ceil(time_bound / dt)

    def inside_rect(x_v, y_v, rect):
        x1, y1, x2, y2 = rect
        return And(x_v >= x1, x_v <= x2, y_v >= y1, y_v <= y2)

    def outside_rect(x_v, y_v, rect, margin_v=0.0):
        x1, y1, x2, y2 = rect
        return Or(x_v < x1 - margin_v, x_v > x2 + margin_v, y_v < y1 - margin_v, y_v > y2 + margin_v)

    def get_val(model, var):
        val = model.evaluate(var, model_completion=True)
        if val is None:
            return None
        if val.is_int():
            return val.as_long()
        return float(val.as_fraction())

    def linearizer(opt_v, expr, prefix, step):
        var = Real(f"{prefix}_{step}")
        opt_v.add(var >= expr)
        opt_v.add(var >= -expr)
        opt_v.add(var >= 0)
        return var

    opt = Optimize()

    x = [Real(f"x_{t}") for t in range(T_local+1)]
    y = [Real(f"y_{t}") for t in range(T_local+1)]
    vx = [Real(f"vx_{t}") for t in range(T_local)]
    vy = [Real(f"vy_{t}") for t in range(T_local)]

    opt.add(inside_rect(x[0], y[0], start))
    opt.add(inside_rect(x[T_local], y[T_local], goal))

    step_lengths = []
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

    total_path_length = Sum(step_lengths)
    if optimize:
        opt.minimize(total_path_length)

    solve_start = time.time()
    result = opt.check()
    solve_end = time.time()

    end_time = time.time()
    build_time_ms = (solve_start - start_time) * 1000
    solve_time_ms = (solve_end - solve_start) * 1000
    total_time_ms = (end_time - start_time) * 1000

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

if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    result_cols = ['STATUS', 'TOTAL STEPS', 'LENGTH', 'BUILD TIME (ms)', 'SOLVE TIME (ms)', 'TOTAL TIME (ms)', 'DATA POINTS']
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