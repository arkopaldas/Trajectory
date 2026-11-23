"""Trains the controller network for the SIMPLE model and exports it as ONNX.

Pipeline: read 'Simple Model Result.csv' (made by Generators/simple.py) -> keep the SAT rows -> cut windows
(dataset.py) -> split / normalise -> train SimpleNet with the state-space loss (utils.train_model) -> print test errors ->
write 'Networks/<DATASET_SIZE> Trained/Simple.onnx' (normalisation folded in; raw state in, raw control out).
Hyper-parameters come from config.TRAINING_PRESETS[config.DATASET_SIZE].

Run:  python train_simple.py     (from inside the Neural Networks folder)
"""
import os
import numpy as np

import config
from dataset import build_simple_dataset, create_dataloaders
from Arch.simple_net import SimpleNet
from utils import train_model, evaluate_model, upload_csv, export_onnx

if __name__ == "__main__":
    # Load the dataset; DT (the time step of the simulation) is read from the first row because it is the same in every row.
    df = upload_csv("Simple Model Result.csv")
    DT = df["DT"].iloc[0]
    # Keep only successfully solved instances (SAT, with a path length); UNSAT rows have no trajectory.
    df = df[df["STATUS"].isin(["SAT"])].reset_index(drop=True)
    df = df[df["LENGTH"].notna()].reset_index(drop=True)

    # Hyper-parameters for this dataset size. (The .get fallback is the entry itself, so DATASET_SIZE has to be a key of
    # TRAINING_PRESETS, otherwise a KeyError is raised.)
    presets = config.TRAINING_PRESETS.get(config.DATASET_SIZE, config.TRAINING_PRESETS[config.DATASET_SIZE])

    # Training windows. SCALE_XY is the diagonal of the bounding box of all following states; it is only used to report the
    # position error as a percentage of the workspace size.
    X, Y, N = build_simple_dataset(df)
    # All 'next' states as a flat list, used only to measure the size of the workspace covered by the data.
    states_next = N.reshape(-1, N.shape[-1])
    SCALE_XY = np.sqrt((states_next[:, 0].max() - states_next[:, 0].min())**2 + (states_next[:, 1].max() - states_next[:, 1].min())**2)
    print(f"\n[Simple Net Setup] Spatial Diagonal Boundary Scale: {SCALE_XY:.4f} units")
    print(f"[Simple Net Setup] Rollout horizon: {N.shape[1]} step(s) | Z-score normalization: {'on' if config.NORMALIZE else 'off'}\n")

    # Split and normalise; stats (training split only) go into the model's normalisation buffers.
    train_loader, test_loader, stats = create_dataloaders(X, Y, N, batch_size=presets["BATCH_SIZE"])
    model = SimpleNet()
    model.set_normalization(**stats)

    # Train, then evaluate on the held-out test split.
    model = train_model(
        model, train_loader, model_type='simple', dt=DT, scale=SCALE_XY,
        lr=presets["LR"], epochs=presets["EPOCHS"]
    )

    evaluate_model(model, test_loader, model_type='simple', dt=DT, scale=SCALE_XY)

    # Export to ONNX. The trained network is later loaded by predict_simple.py and by the verifier in ../Verification.
    os.makedirs(config.NETWORK_PATH, exist_ok=True)
    ONNX_PATH = os.path.join(config.NETWORK_PATH, "Simple.onnx")
    export_onnx(model, X.shape[1], ONNX_PATH)
    print(f"\n[ONNX Generation] Export achieved successfully ^_^ !")