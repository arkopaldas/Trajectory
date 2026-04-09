import os
import numpy as np

MAX_OBSTACLES = 10
TRAIN_RATIO = 0.7
BATCH_SIZE = 256
EPOCHS = 2000
LR = 1e-3
DEG2RAD = np.pi / 180

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORK_PATH = os.path.join(BASE_DIR, "Networks") + os.sep
DATASET_PATH = os.path.join(BASE_DIR, "..", "Datasets", "Results") + os.sep