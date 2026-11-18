import os
import math
from multiprocessing import cpu_count

OPTIMIZE = True
MARGIN = 2.5
TIME_BOUND = 60.0
UNICYCLE_TIME_BOUND = 70.0
DEG2RAD = math.pi / 180.0
MAX_INSTANCES = 100000
EPSILON = 10 / MAX_INSTANCES
NUM_WORKERS = cpu_count()

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

SIMPLE_DT = 2.5
SIMPLE_T = math.ceil(TIME_BOUND / SIMPLE_DT)
SIMPLE_VMIN, SIMPLE_VMAX = -SIMPLE_DT, SIMPLE_DT

UNICYCLE_DT = 3.0
UNICYCLE_T = math.ceil(UNICYCLE_TIME_BOUND / UNICYCLE_DT)
UNICYCLE_DTHETA = 45 * DEG2RAD
UNICYCLE_VMIN, UNICYCLE_VMAX = 0.0, UNICYCLE_DT
UNICYCLE_WMIN, UNICYCLE_WMAX = -90.0 * DEG2RAD, 90.0 * DEG2RAD
UNICYCLE_THMIN, UNICYCLE_THMAX = -22.5 * DEG2RAD, 112.5 * DEG2RAD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "Datasets", "Schemas") + os.sep
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "Datasets", "Results", f"{MAX_INSTANCES} Trajectories") + os.sep

INPUT_FILE_SIMPLE = os.path.join(INPUT_PATH, "Simple Model Dataset Schema.csv")
OUTPUT_FILE_SIMPLE = os.path.join(OUTPUT_PATH, "Simple Model Result.csv")

INPUT_FILE_UNICYCLE = os.path.join(INPUT_PATH, "Unicycle Model Dataset Schema.csv")
OUTPUT_FILE_UNICYCLE = os.path.join(OUTPUT_PATH, "Unicycle Model Result.csv")
