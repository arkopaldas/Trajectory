"""Settings for the BATCH dataset generators (simple.py and unicycle.py).

Everything the generators need is here: the planning problem (regions, obstacles, margin, horizon), the model
constants (time step, control and heading bounds) and the input / output file paths.
The same constants (copied) are stored in every row of the generated CSV, so the dataset describes the problem
it was generated for; the verifier in ../Verification reads them back from the CSV.
"""
# os: path handling; math: pi / ceil; cpu_count: default number of parallel worker processes.
import os
import math
from multiprocessing import cpu_count

# True: every instance minimises the path length. False: any feasible trajectory is accepted (faster).
OPTIMIZE = True
# Clearance kept around every obstacle (the obstacle rectangle is inflated by this much).
MARGIN = 2.5
# Planning horizon in seconds for the simple model. Steps = ceil(TIME_BOUND / DT).
TIME_BOUND = 60.0
# The unicycle gets a longer horizon (it can only move forward, so it needs more steps to turn and reach the goal).
UNICYCLE_TIME_BOUND = 70.0
DEG2RAD = math.pi / 180.0
# Number of start points (= trajectories = CSV rows) to generate. Also names the output folder '<N> Trajectories'.
MAX_INSTANCES = 100000
# Minimum distance between two generated start points (see generate_points). Shrinks as the dataset grows so that
# the points still fit into the 10 x 10 start region.
EPSILON = 10 / MAX_INSTANCES
# Number of parallel solver processes. Lower it on a weak machine (each worker builds its own solver model, so RAM grows with it).
NUM_WORKERS = cpu_count()

# Start / goal regions and obstacles as (x_min, y_min, x_max, y_max) boxes. The start region is where the random start
# points are drawn; every trajectory begins at one of them.
START_REGION = (0, 0, 10, 10)
GOAL_REGION = (90, 90, 100, 100)
OBSTACLES = [
    (20.0, 20.0, 30.0, 30.0),
    (40.0, 20.0, 50.0, 30.0),
    (20.0, 40.0, 30.0, 50.0),
    (20.0, 70.0, 30.0, 80.0),
    (40.0, 60.0, 50.0, 70.0),
    (50.0, 40.0, 60.0, 50.0),
    (70.0, 30.0, 80.0, 40.0),
    (70.0, 10.0, 80.0, 20.0),
]

# ---- Simple model: velocity-controlled point, x' = x + vx dt, y' = y + vy dt. Velocity of each axis in [-DT, DT] ----
SIMPLE_DT = 2.5
SIMPLE_T = math.ceil(TIME_BOUND / SIMPLE_DT)
SIMPLE_VMIN, SIMPLE_VMAX = -SIMPLE_DT, SIMPLE_DT

# ---- Unicycle model ----
# DTHETA: width of one heading bin used by the MILP approximation of cos/sin. v in [0, DT], omega in [-90, 90] deg/s,
# heading kept within [-22.5, 112.5] deg (three 45-degree bins).
UNICYCLE_DT = 3.0
UNICYCLE_T = math.ceil(UNICYCLE_TIME_BOUND / UNICYCLE_DT)
UNICYCLE_DTHETA = 45 * DEG2RAD
UNICYCLE_VMIN, UNICYCLE_VMAX = 0.0, UNICYCLE_DT
UNICYCLE_WMIN, UNICYCLE_WMAX = -90.0 * DEG2RAD, 90.0 * DEG2RAD
UNICYCLE_THMIN, UNICYCLE_THMAX = -22.5 * DEG2RAD, 112.5 * DEG2RAD

# ---- Paths (built with os.path so they work on Windows, Linux and macOS) ----
# INPUT: the schema CSV (header only) from Datasets/Schemas; it defines the column order of the result file.
# OUTPUT: Datasets/Results/<MAX_INSTANCES> Trajectories/<Model> Model Result.csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "Datasets", "Schemas") + os.sep
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "Datasets", "Results", f"{MAX_INSTANCES} Trajectories") + os.sep

INPUT_FILE_SIMPLE = os.path.join(INPUT_PATH, "Simple Model Dataset Schema.csv")
OUTPUT_FILE_SIMPLE = os.path.join(OUTPUT_PATH, "Simple Model Result.csv")

INPUT_FILE_UNICYCLE = os.path.join(INPUT_PATH, "Unicycle Model Dataset Schema.csv")
OUTPUT_FILE_UNICYCLE = os.path.join(OUTPUT_PATH, "Unicycle Model Result.csv")
