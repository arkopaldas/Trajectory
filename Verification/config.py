import os
import numpy as np

# =====================================================================================================
# Which network is verified / fine-tuned
# =====================================================================================================
DATASET_SIZE = 1000            # selects "../Networks/<N> Trained" and "../Datasets/Results/<N> Trajectories"
USE_CEGIS_NETWORK = False      # verify "<Model>_CEGIS.onnx" (the fine-tuned network) instead of "<Model>.onnx"
CEGIS_SUFFIX = "_CEGIS"        # fine-tuned networks are written next to the original, the original is never touched

# =====================================================================================================
# Safety constraints. A constraint that is switched off has no effect on verification or on CEGIS training.
# =====================================================================================================
CHECK_CONTROL_BOUNDS = False   # every applied control stays inside [VMIN, VMAX] (and [WMIN, WMAX] for the unicycle)
CHECK_OBSTACLES = True         # the state never enters an obstacle inflated by MARGIN
CHECK_WORKSPACE = False        # x and y stay inside the workspace box
CHECK_THETA_BOUNDS = False     # unicycle only: the heading stays inside [THMIN, THMAX]
GOAL_MODE = "off"              # "final": state after the last step is in the goal | "any": some state is | "off"

# Optional overrides of values that are otherwise read from the dataset CSV (None = read from the CSV)
WORKSPACE = None               # (xmin, ymin, xmax, ymax); default (0, 0, goal x2, goal y2) as in the MILP / SMT models
START_THETA_RANGE = None       # (min, max) of the start heading; default (THMIN, THMAX) of the dataset

# A constraint is violated when its violation degree is >= PROPERTY_EPS (distance units / control units).
# A floating point solver cannot tell "touching" from "violating by 1e-12", so SAFE means "no violation >= PROPERTY_EPS".
PROPERTY_EPS = 1e-3
# A Marabou counterexample is accepted only if the exact closed-loop rollout violates a constraint by >= CONFIRM_EPS.
CONFIRM_EPS = 1e-4

# =====================================================================================================
# Verification: how the start region is split and how Marabou is run
# =====================================================================================================
START_CELLS_XY = (4, 4)        # the (x, y) start region is split into this many cells; every cell is verified on its own
START_CELLS_THETA = 4          # unicycle only: the heading range is split into this many cells as well
QUERY_TIMEOUT = 1800           # seconds per Marabou query (a timeout makes the answer UNKNOWN)
TOTAL_TIMEOUT = 24 * 3600      # seconds for one whole verification run
NUM_WORKERS = 3                # Marabou workers; keep 1 on a weak machine (more workers = more RAM)
USE_BOUND_PRUNING = True       # drop property cases that the propagated variable bounds already rule out (sound shortcut)

# Unicycle only. Marabou cannot compute cos / sin / products, so they are enclosed by piecewise-linear
# bands: the true value is always inside the band, and the band half-width is about (step^2) / 8.
# Marabou UNSAT therefore stays a proof, and every Marabou SAT is replayed on the true dynamics before it is trusted.
TRIG_STEP = 0.05               # knot spacing (rad) for cos / sin
SQUARE_STEP = 0.05             # knot spacing for the squares inside v * cos(theta) and v * sin(theta)
MAX_PWL_KNOTS = 400            # upper limit of knots per function (the band gets wider if it is hit, still sound)
MAX_REFINEMENTS = 2            # a SAT that does not replay is solved again with half the spacing, this many times

# =====================================================================================================
# CEGIS (counterexample guided fine-tuning)
# =====================================================================================================
MAX_CEGIS_ITERATIONS = 20
CEGIS_TIME_LIMIT = 48 * 3600   # seconds for the whole CEGIS run
CEGIS_RESUME = False           # True: continue from an existing "<Model>_CEGIS.onnx" instead of the original network
MAX_COUNTEREXAMPLES = 20       # counterexamples collected per verification round (at most one per start cell)
FINETUNE_STEPS = 500           # optimiser steps per CEGIS iteration
FINETUNE_LR = 1e-4
CE_BATCH = 64                  # counterexample trajectories per optimiser step

# L_total = L_ori + LAMBDA_CE * L_ce
LAMBDA_CE = 1.0
CE_WEIGHTS = {"control": 1.0, "obstacle": 1.0, "workspace": 1.0, "theta": 1.0, "goal": 1.0}   # only active constraints count
CE_STATE_BUFFER = 0.5          # the loss pushes states this far (distance units) beyond the unsafe set
CE_CONTROL_BUFFER = 0.05       # same for the control bounds (control units)

# =====================================================================================================
# Paths (same convention as "Neural Networks/config.py")
# =====================================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_PATH = os.path.join(BASE_DIR, "..", "Networks", f"{DATASET_SIZE} Trained") + os.sep
DATASET_PATH = os.path.join(BASE_DIR, "..", "Datasets", "Results", f"{DATASET_SIZE} Trajectories") + os.sep
COUNTEREXAMPLE_PATH = os.path.join(BASE_DIR, "..", "Datasets", "Counterexamples", f"{DATASET_SIZE} Trajectories") + os.sep
RESULT_PATH = os.path.join(BASE_DIR, "Results", f"{DATASET_SIZE} Trained") + os.sep
NEURAL_NETWORKS_PATH = os.path.join(BASE_DIR, "..", "Neural Networks")

MODELS = {
    "simple": {
        "name": "Simple", "onnx": "Simple", "dataset": "Simple Model Result.csv",
        "state": ("x", "y"), "control": ("vx", "vy"), "ok_status": ("SAT",),
    },
    "unicycle": {
        "name": "Unicycle", "onnx": "Unicycle", "dataset": "Unicycle Model Result.csv",
        "state": ("x", "y", "theta"), "control": ("v", "omega"), "ok_status": ("Optimal", "Feasible"),
    },
}
