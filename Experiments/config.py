"""Settings for the stand-alone EXPERIMENT scripts in this folder.

The experiments are small, independent studies of other planning techniques (MILP with polygon obstacles, a MILP
with a fixed start point, an SMT unicycle and a reinforcement-learning unicycle). They do not feed the dataset /
training / verification pipeline; each script just prints a result and shows a plot or animation.
Every script reads only its own block of constants below (the prefix tells which: MILP_FIXED_, MILP_SIMPLE_, SMT_, RL_).
"""
# math: pi / ceil. numpy is imported for completeness (not used by the constants below).
import math
import numpy as np

# Shared by the MILP / SMT scripts: True = minimise the path length, False = any feasible trajectory.
OPTIMIZE = True
# Default obstacle margin and horizon (the experiments below override them with their own values).
MARGIN = 2.5
TIME_BOUND = 60.0
# Multiply degrees by this to get radians.
DEG2RAD = math.pi / 180.0

# Shared start region, goal region and rectangular obstacles as (x_min, y_min, x_max, y_max); same layout as the
# main pipeline (Generators/config.py).
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

# ---- milp_fixed_unicycle.py: unicycle MILP started from ONE random point of the start region ----
# Heading is restricted to three 45-degree bins (-22.5..112.5 degrees), v in [0, DT], omega in [-90, 90] deg/s.
MILP_FIXED_TIME_BOUND = 70.0
MILP_FIXED_DT = 3.0
MILP_FIXED_T = math.ceil(MILP_FIXED_TIME_BOUND / MILP_FIXED_DT)
MILP_FIXED_MARGIN = 2.0
MILP_FIXED_DTHETA = 45 * DEG2RAD
MILP_FIXED_VMIN, MILP_FIXED_VMAX = 0.0, MILP_FIXED_DT
MILP_FIXED_WMIN, MILP_FIXED_WMAX = -90.0 * DEG2RAD, 90.0 * DEG2RAD
MILP_FIXED_THMIN, MILP_FIXED_THMAX = -22.5 * DEG2RAD, 112.5 * DEG2RAD

# ---- milp_simple.py: simple (velocity-controlled) model with a half-space / big-M obstacle encoding ----
# OBSTACLES here are dicts {type, params}: 'rectangle' = (x1, y1, x2, y2), 'circle' = (cx, cy, r) (approximated by an
# 8-sided polygon), 'polygon' = list of vertices in counter-clockwise order. BIG_M must exceed the workspace size.
MILP_SIMPLE_TIME_BOUND = 50.0
MILP_SIMPLE_DT = 3.0
MILP_SIMPLE_T = math.ceil(MILP_SIMPLE_TIME_BOUND / MILP_SIMPLE_DT)
MILP_SIMPLE_VMIN, MILP_SIMPLE_VMAX = -MILP_SIMPLE_DT, MILP_SIMPLE_DT
MILP_SIMPLE_MARGIN = 3.0
MILP_SIMPLE_BIG_M = 1000
MILP_SIMPLE_OBSTACLES = [
    {"type": "rectangle", "params": (20.0, 20.0, 30.0, 30.0)},
    {"type": "rectangle", "params": (40.0, 20.0, 50.0, 30.0)},
    {"type": "rectangle", "params": (20.0, 40.0, 30.0, 50.0)},
    {"type": "rectangle", "params": (20.0, 70.0, 30.0, 80.0)},
    {"type": "rectangle", "params": (40.0, 60.0, 50.0, 70.0)},
    {"type": "rectangle", "params": (50.0, 40.0, 60.0, 50.0)},
    {"type": "rectangle", "params": (70.0, 30.0, 80.0, 40.0)},
    {"type": "rectangle", "params": (70.0, 10.0, 80.0, 20.0)},
    {"type": "circle", "params": (70.0, 70.0, 5.0)},
]

# ---- smt_unicycle.py: small SMT unicycle example with its own goal region and one obstacle ----
# Angles are in DEGREES in this script (DTHETA = 10 degree heading bins, heading in [0, 90], omega in [-90, 90] deg/s).
SMT_TIME_BOUND = 11.0
SMT_DT = 1.0
SMT_T = math.ceil(SMT_TIME_BOUND / SMT_DT)
SMT_MARGIN = 0.75
SMT_DTHETA = 10
SMT_VMIN, SMT_VMAX = 0, SMT_DT
SMT_WMIN, SMT_WMAX = -90, 90
SMT_THMIN, SMT_THMAX = 0, 90
SMT_GOAL_REGION = (20, 20, 30, 30)
SMT_OBSTACLES = [
    (12.5, 15.0, 15.0, 17.5),
]

# ---- rl_unicycle.py: tabular Q-learning on a 10 x 10 board with circular obstacles ----
# START = (x, y, heading in radians); GOAL = (x, y) with a success radius; OBSTACLES = (cx, cy, radius).
# The robot moves at constant speed V; the agent only chooses the turn rate from OMEGA_ACTIONS (rad/s).
RL_BOARD_SIZE = 10.0
RL_START = (1.0, 1.0, 0.0)
RL_GOAL = (9.0, 9.0)
RL_GOAL_RADIUS = 0.4
RL_OBSTACLES = [
    (3.0, 3.0, 0.8),
    (5.0, 6.5, 0.9),
    (7.0, 3.5, 0.7),
    (4.5, 8.0, 0.6),
    (2.0, 6.0, 0.6),
]
RL_V = 0.5
RL_DT = 0.5
RL_OMEGA_ACTIONS = [-0.9, -0.4, 0.0, 0.4, 0.9]
# State discretisation: x, y and heading are cut into this many bins for the Q-table.
RL_N_X_BINS = 26
RL_N_Y_BINS = 26
RL_N_THETA_BINS = 16
# Training: number of episodes, step limit per episode, learning rate (ALPHA), discount (GAMMA) and the epsilon-greedy
# exploration schedule (EPS_START falls linearly to EPS_END over EPS_DECAY_EPISODES episodes).
RL_N_EPISODES = 6000
RL_MAX_STEPS_PER_EPISODE = 250
RL_ALPHA = 0.2
RL_GAMMA = 0.97
RL_EPS_START = 1.0
RL_EPS_END = 0.05
RL_EPS_DECAY_EPISODES = 5000
# Save the final rollout as a GIF (also switched on by the --save-gif command-line flag) and seed for reproducibility.
RL_SAVE_GIF = True
RL_GIF_PATH = "unicycle_trajectory.gif"
RL_RANDOM_SEED = 7