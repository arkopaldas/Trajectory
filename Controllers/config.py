"""Constants for the SINGLE-INSTANCE solvers in this folder (smt_simple.py and milp_unicycle.py).

These scripts solve ONE trajectory-planning problem (one start region, one goal region, a set of rectangular
obstacles) and plot the result. They are meant for quick checks and for tuning the formulation before the
batch generators in ../Generators are run. The bulk dataset generation has its own copy of these constants in
Generators/config.py (the two files are NOT linked, change both if you want the same setting everywhere).

Units: distances are in the abstract units of the 100 x 100 workspace, time is in seconds, angles are in radians.
"""
import math

# True: minimise the path length. False: accept any feasible trajectory (much faster, but not optimal).
OPTIMIZE = True
# Safety margin: every obstacle rectangle is inflated by this distance on every side before it is avoided.
MARGIN = 2.5
# Planning horizon in seconds. The number of steps is T = ceil(TIME_BOUND / DT) (see SMT_T / MILP_T below).
TIME_BOUND = 60.0
# Multiply degrees by this factor to get radians.
DEG2RAD = math.pi / 180.0

# Regions are axis-aligned boxes written as (x_min, y_min, x_max, y_max).
# The robot starts anywhere inside START_REGION and must finish inside GOAL_REGION.
START_REGION = (0, 0, 10, 10)
GOAL_REGION = (90, 90, 100, 100)
# Rectangular obstacles, each one (x_min, y_min, x_max, y_max). Same layout as the generators and the verifier use.
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

# ---- Simple model (solved with the Z3 SMT solver in smt_simple.py) ----
# DT = length of one time step; T = number of steps; the velocity of each axis is limited to [-DT, DT]
# (so one step can move at most DT * DT = 6.25 units per axis).
SMT_DT = 2.5
SMT_T = math.ceil(TIME_BOUND / SMT_DT)
SMT_VMIN, SMT_VMAX = -SMT_DT, SMT_DT

# ---- Unicycle model (solved as a MILP in milp_unicycle.py) ----
# DT = time step, T = number of steps.
# DTHETA = width of one heading bin: the MILP cannot evaluate cos/sin, so the heading range is cut into bins and the
# cos/sin of each bin centre is used (this is the approximation the neural-network training does NOT use).
# v in [0, DT] (forward only), omega in [-90, 90] deg/s, heading theta kept in [-22.5, 112.5] deg = three 45-degree bins.
MILP_DT = 3.0
MILP_T = math.ceil(TIME_BOUND / MILP_DT)
MILP_DTHETA = 45 * DEG2RAD
MILP_VMIN, MILP_VMAX = 0.0, MILP_DT
MILP_WMIN, MILP_WMAX = -90.0 * DEG2RAD, 90.0 * DEG2RAD
MILP_THMIN, MILP_THMAX = -22.5 * DEG2RAD, 112.5 * DEG2RAD