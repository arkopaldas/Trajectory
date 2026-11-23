"""Settings for training and evaluating the neural-network controllers.

The networks imitate the optimal controllers that were produced by the solvers: given the current state they predict the
control (simple model: vx, vy; unicycle: v, omega). The physics (x' = x + v dt, ...) is applied to the prediction OUTSIDE
the network, and the loss compares the resulting next state with the next state in the dataset.
Everything is read from here by dataset.py, utils.py, train_*.py and predict_*.py. Change DATASET_SIZE to train on
another dataset size (it selects the dataset folder, the output folder and the training preset at once).
"""
import os
import numpy as np

# Maximum number of obstacles encoded by utils.encode_obstacles (currently NOT used by the pipeline: the networks see
# only the state, not the obstacles).
MAX_OBSTACLES = 10
# Fraction of the samples used for training (the rest is the test split). The split is random over windows, not over trajectories.
TRAIN_RATIO = 0.7
# Weight of the heading (theta) term relative to x and y in the loss (unicycle only).
LOSS_RATIO = 0.01   # weight of the heading (theta) error relative to x and y. The loss is computed on z-score
                    # normalised errors, so 1.0 already balances the axes; lower it to de-emphasise heading
# Maximum gradient norm (gradient clipping keeps training stable).
GRADIENT_CLIP = 1.0
# Minimum trajectory length for the build_*_sequences helpers in dataset.py (these helpers are currently not used).
ROLLOUT_STEPS = 3  # minimum trajectory length for the build_*_sequences helpers
# Multiply degrees by this to get radians (not used by the other scripts at the moment).
DEG2RAD = np.pi / 180

# Training horizon: how many steps the network is unrolled through the kinematics during training/testing.
#   1  -> single-step prediction (state_t -> state_t+1)
#   K  -> multi-step: the net is fed its own predicted state K times and every step is compared with the data
# Number of steps the network is unrolled during training / testing (see the explanation above).
ROLLOUT_HORIZON = 1

# Z-score normalisation of the network inputs (x, y, theta) and outputs (vx, vy / v, omega).
# Statistics come from the training split only and are folded into the exported ONNX, so the
# ONNX model still takes raw states and returns raw controls.
# True: inputs / outputs are z-score normalised inside the model; the statistics are folded into the exported ONNX file.
NORMALIZE = True

# Which dataset to use: the number of trajectories in 'Datasets/Results/<DATASET_SIZE> Trajectories'. Also selects the
# output folder 'Networks/<DATASET_SIZE> Trained' and the entry of TRAINING_PRESETS below (must be one of its keys).
DATASET_SIZE = 1000
# Fallback hyper-parameters used by utils.train_model when no value is passed. The train_*.py scripts always pass the
# values of the preset selected by DATASET_SIZE, so these three are only defaults.
BATCH_SIZE = 256
EPOCHS = 500
LR = 1e-3

# ONNX export (legacy exporter: one self-contained .onnx file, honours this opset)
# ONNX export: operator-set version of the exported model. The verifier (Marabou) supports the plain Gemm / Relu operators this produces.
ONNX_OPSET = 11

# Rollout GIF written by the predict_*.py scripts (an arrow driving along the predicted path)
# Rollout GIF written by the predict_*.py scripts (an arrow driving along the predicted path).
GIF_FPS = 20
GIF_FRAMES_PER_STEP = 5      # interpolated frames between two time steps
GIF_HOLD_SECONDS = 1.0       # the last frame is held this long
GIF_FIGSIZE = (6, 6)
GIF_DPI = 80
GIF_ARROW_LENGTH = 6.0       # arrow size in environment units
GIF_ARROW_WIDTH = 3.5

# ---- Paths (relative to this file) ----
# NETWORK_PATH: where trained models (.onnx) and rollout GIFs are saved. DATASET_PATH: where the generated CSV files are read.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_PATH = os.path.join(BASE_DIR, "..", "Networks", f"{DATASET_SIZE} Trained") + os.sep
DATASET_PATH = os.path.join(BASE_DIR, "..", "Datasets", "Results", f"{DATASET_SIZE} Trajectories") + os.sep

# Hyper-parameters per dataset size (more data -> larger batches, higher learning rate, fewer epochs).
# Only BATCH_SIZE, LR and EPOCHS are used by the training scripts. PATIENCE (early stopping) and SCHEDULER (learning-rate
# schedule) are listed for reference but are NOT implemented in utils.train_model; the optimiser is plain Adam.
TRAINING_PRESETS = {
    10: {
        "BATCH_SIZE": 32,
        "LR": 1e-4,
        "EPOCHS": 350,
        "PATIENCE": 20,
        "SCHEDULER": "None",
    },
    100: {
        "BATCH_SIZE": 64,
        "LR": 3e-4,
        "EPOCHS": 250,
        "PATIENCE": 15,
        "SCHEDULER": "CosineAnnealingLR",
    },
    1000: {
        "BATCH_SIZE": 128,
        "LR": 5e-4,
        "EPOCHS": 150,
        "PATIENCE": 12,
        "SCHEDULER": "ReduceLROnPlateau",
    },
    10000: {
        "BATCH_SIZE": 256,
        "LR": 1e-3,
        "EPOCHS": 100,
        "PATIENCE": 8,
        "SCHEDULER": "OneCycleLR",
    },
    100000: {
        "BATCH_SIZE": 512,
        "LR": 2e-3,
        "EPOCHS": 60,
        "PATIENCE": 5,
        "SCHEDULER": "OneCycleLR",
    }
}

# Older preset set that was kept for reference (commented out, not used).
# TRAINING_PRESETS = {
#     10: {
#         "HIDDEN_DIM": 32,
#         "BATCH_SIZE": 32,
#         "LR": 1e-3,
#         "WEIGHT_DECAY": 1e-3,      # Higher regularization to prevent memorizing 10 paths
#         "EPOCHS": 400,
#         "PATIENCE": 40,
#         "SCHEDULER": "CosineAnnealingLR",
#     },
#     100: {
#         "HIDDEN_DIM": 32,
#         "BATCH_SIZE": 64,
#         "LR": 1.5e-3,
#         "WEIGHT_DECAY": 5e-4,
#         "EPOCHS": 300,
#         "PATIENCE": 30,
#         "SCHEDULER": "CosineAnnealingLR",
#     },
#     1000: {
#         "HIDDEN_DIM": 64,
#         "BATCH_SIZE": 256,
#         "LR": 2e-3,
#         "WEIGHT_DECAY": 1e-4,
#         "EPOCHS": 200,
#         "PATIENCE": 20,
#         "SCHEDULER": "CosineAnnealingLR",
#     },
#     10000: {
#         "HIDDEN_DIM": 64,
#         "BATCH_SIZE": 512,
#         "LR": 3e-3,
#         "WEIGHT_DECAY": 1e-5,
#         "EPOCHS": 120,
#         "PATIENCE": 15,
#         "SCHEDULER": "OneCycleLR",
#     },
#     100000: {
#         "HIDDEN_DIM": 64,          # Use 128 for UnicycleNet if complex obstacle avoidance is included
#         "BATCH_SIZE": 2048,
#         "LR": 5e-3,
#         "WEIGHT_DECAY": 0.0,       # Data volume naturally prevents overfitting
#         "EPOCHS": 60,
#         "PATIENCE": 10,
#         "SCHEDULER": "OneCycleLR",
#     }
# }
