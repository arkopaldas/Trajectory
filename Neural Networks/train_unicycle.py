import os
import numpy as np

import config
from dataset import build_unicycle_dataset, create_dataloaders
from Arch.unicycle_net import UnicycleNet
from utils import train_model, evaluate_model, upload_csv, export_onnx

if __name__ == "__main__":
    df = upload_csv("Unicycle Model Result.csv")
    DT = df["DT"].iloc[0]
    df = df[df["STATUS"].isin(["Optimal", "Feasible"])].reset_index(drop=True)
    df = df[df["LENGTH"].notna()].reset_index(drop=True)

    presets = config.TRAINING_PRESETS.get(config.DATASET_SIZE, config.TRAINING_PRESETS[config.DATASET_SIZE])

    X, Y, N = build_unicycle_dataset(df)
    states_next = N.reshape(-1, N.shape[-1])
    SCALE_XY = np.sqrt((states_next[:, 0].max() - states_next[:, 0].min())**2 + (states_next[:, 1].max() - states_next[:, 1].min())**2)
    SCALE_THETA = states_next[:, 2].max() - states_next[:, 2].min()
    SCALE_COMBINED = np.sqrt(SCALE_XY**2 + SCALE_THETA**2)

    print(f"\n[Unicycle Net Setup] Combined Bound Resolution Scale: {SCALE_COMBINED:.4f}")
    print(f"[Unicycle Net Setup] Spatial Resolution: {SCALE_XY:.4f} units | Angular Resolution: {SCALE_THETA:.4f} rad")
    print(f"[Unicycle Net Setup] Rollout horizon: {N.shape[1]} step(s) | Angle wrapping: on | Z-score normalization: {'on' if config.NORMALIZE else 'off'}\n")

    train_loader, test_loader, stats = create_dataloaders(X, Y, N, batch_size=presets["BATCH_SIZE"])
    model = UnicycleNet()
    model.set_normalization(**stats)

    model = train_model(
        model, train_loader, model_type='unicycle', dt=DT, scale=SCALE_XY,
        lr=presets["LR"], epochs=presets["EPOCHS"]
    )

    evaluate_model(model, test_loader, model_type='unicycle', dt=DT, scale=SCALE_XY, theta_scale=SCALE_THETA)

    os.makedirs(config.NETWORK_PATH, exist_ok=True)
    ONNX_PATH = os.path.join(config.NETWORK_PATH, "Unicycle.onnx")
    export_onnx(model, X.shape[1], ONNX_PATH)
    print(f"\n[ONNX Generation] Export achieved successfully ^_^ !")