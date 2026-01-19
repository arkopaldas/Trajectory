import ast
import math
import time
import random
from z3 import *
import pandas as pd
from google.colab import files

# Parameters
time_bound = 110.0
dt = 1.0
T = math.ceil(time_bound / dt)
Vmin, Vmax = -dt, dt
optimize = True
margin = 1.5
quantity = 50

# Regions - Start, Goal, Obstacle.
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
output_file = "Simplified Model Result.csv"

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

def simple_solver(time_bound, dt, vmin, vmax, margin, start_region, goal_region, obstacles, optimize=True):
    start_time = time.time()
    T = math.ceil(time_bound / dt)
    Vmin, Vmax = vmin, vmax

    def inside_rect(x, y, rect):
        x1, y1, x2, y2 = rect
        return And(x >= x1, x <= x2, y >= y1, y <= y2)

    def outside_rect(x, y, rect, margin=0.0):
        x1, y1, x2, y2 = rect
        return Or(x < x1 - margin, x > x2 + margin, y < y1 - margin, y > y2 + margin)

    def get_val(model, var):
        val = model.evaluate(var, model_completion=True)
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

    opt = Optimize()

    x = [Real(f"x_{t}") for t in range(T+1)]
    y = [Real(f"y_{t}") for t in range(T+1)]
    vx = [Real(f"vx_{t}") for t in range(T)]
    vy = [Real(f"vy_{t}") for t in range(T)]

    opt.add(inside_rect(x[0], y[0], start_region))
    opt.add(inside_rect(x[T], y[T], goal_region))

    step_lengths = []
    for t in range(T+1):
        opt.add(x[t] >= start_region[0], x[t] <= goal_region[2])
        opt.add(y[t] >= start_region[1], y[t] <= goal_region[3])
        for rect in obstacles:
            opt.add(outside_rect(x[t], y[t], rect, margin))
        if t < T:
            opt.add(vx[t] >= Vmin, vx[t] <= Vmax)
            opt.add(vy[t] >= Vmin, vy[t] <= Vmax)
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
        trajectory = [(get_val(model, x[t]), get_val(model, y[t])) for t in range(T+1)]
        velocities = [(get_val(model, vx[t]), get_val(model, vy[t])) for t in range(T)]
        trajectory_length = get_val(model, total_path_length)
        data_points = [{
            "t": t,
            "x": trajectory[t][0],
            "y": trajectory[t][1],
            "vx": velocities[t][0] if t < T else None,
            "vy": velocities[t][1] if t < T else None
        } for t in range(T+1)]
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
        "START": str(start_region),
        "GOAL": str(goal_region),
        "OBSTACLES": str(obstacles),
        "OPTIMIZE": optimize,
        "STATUS": status,
        "TOTAL STEPS": T,
        "LENGTH": trajectory_length,
        "BUILD TIME (ms)": build_time_ms,
        "SOLVE TIME (ms)": solve_time_ms,
        "TOTAL TIME (ms)": total_time_ms,
        "DATA POINTS": str(data_points),
    }

df = pd.read_csv(input_file)

# for col in ['BOUND', 'DT', 'VMIN', 'VMAX', 'MARGIN', 'START', 'GOAL', 'OBSTACLES', 'OPTIMIZE']:
#     if col not in df.columns:
#         raise ValueError(f"Missing required column '{col}' in CSV file!")

result_cols = ['STATUS', 'TOTAL STEPS', 'LENGTH', 'BUILD TIME (ms)', 'SOLVE TIME (ms)', 'TOTAL TIME (ms)', 'DATA POINTS']
for col in result_cols:
    if col not in df.columns:
        df[col] = None
    df[col] = df[col].astype('object')

initial_points = generate_points(start_region[0], start_region[2], start_region[1], start_region[3], quantity)

# for i, row in df.iterrows():
for i, item in enumerate(initial_points):
    print(f"\nSolving instance {i+1}/{len(initial_points)}")
    result = simple_solver(
        # time_bound=float(row['BOUND']),
        # dt=float(row['DT']),
        # vmin=float(row['VMIN']),
        # vmax=float(row['VMAX']),
        # margin=float(row['MARGIN']),
        # start_region=parser(row['START']),
        # goal_region=parser(row['GOAL']),
        # obstacles=parser(row['OBSTACLES']),
        # optimize=bool(row['OPTIMIZE'])

        time_bound=time_bound,
        dt=dt,
        vmin=Vmin,
        vmax=Vmax,
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
