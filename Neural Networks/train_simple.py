import os
import numpy as np

import config
from dataset import build_simple_dataset, create_dataloaders
from Arch.simple_net import SimpleNet
from utils import train_model, evaluate_model, upload_csv, export_onnx

if __name__ == "__main__":
    df = upload_csv("Simple Model Result.csv")
    DT = df["DT"].iloc[0]
    df = df[df["STATUS"].isin(["SAT"])].reset_index(drop=True)
    df = df[df["LENGTH"].notna()].reset_index(drop=True)

    presets = config.TRAINING_PRESETS.get(config.DATASET_SIZE, config.TRAINING_PRESETS[config.DATASET_SIZE])

    X, Y, N = build_simple_dataset(df)
    states_next = N.reshape(-1, N.shape[-1])
    SCALE_XY = np.sqrt((states_next[:, 0].max() - states_next[:, 0].min())**2 + (states_next[:, 1].max() - states_next[:, 1].min())**2)
    print(f"\n[Simple Net Setup] Spatial Diagonal Boundary Scale: {SCALE_XY:.4f} units")
    print(f"[Simple Net Setup] Rollout horizon: {N.shape[1]} step(s) | Z-score normalization: {'on' if config.NORMALIZE else 'off'}\n")

    train_loader, test_loader, stats = create_dataloaders(X, Y, N, batch_size=presets["BATCH_SIZE"])
    model = SimpleNet()
    model.set_normalization(**stats)

    model = train_model(
        model, train_loader, model_type='simple', dt=DT, scale=SCALE_XY,
        lr=presets["LR"], epochs=presets["EPOCHS"]
    )

    evaluate_model(model, test_loader, model_type='simple', dt=DT, scale=SCALE_XY)

    os.makedirs(config.NETWORK_PATH, exist_ok=True)
    ONNX_PATH = os.path.join(config.NETWORK_PATH, "Simple.onnx")
    export_onnx(model, X.shape[1], ONNX_PATH)
    print(f"\n[ONNX Generation] Export achieved successfully ^_^ !")