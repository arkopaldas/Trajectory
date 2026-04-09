import os
import torch
import numpy as np
from Models.config import NETWORK_PATH
from Models.Unicycle.model import UnicycleNet, build_unicycle_dataset
from Models.utils import create_dataloaders, train_model, evaluate_model, upload_csv

if __name__ == "__main__":
    df = upload_csv("Unicycle Model Result.csv")
    DT = df["DT"].iloc[0]
    X, Y, N = build_unicycle_dataset(df)
    x_range = N[:, 0].max() - N[:, 0].min()
    y_range = N[:, 1].max() - N[:, 1].min()
    theta_range = N[:, 2].max() - N[:, 2].min()
    SCALE_XY = np.sqrt(x_range**2 + y_range**2)
    SCALE_THETA = theta_range
    # SCALE_COMBINED = np.sqrt(SCALE_XY**2 + SCALE_THETA**2)
    SCALE_COMBINED = SCALE_XY
    print(f"\nEnvironment scale (diagonal):       {SCALE_XY:.4f} units")
    print(f"Environment scale (orientation):    {SCALE_THETA:.4f} radians   ({np.degrees(SCALE_THETA):.2f} degrees)\n")
    train_loader, test_loader = create_dataloaders(X, Y, N)
    model = UnicycleNet()
    model = train_model(model, train_loader, model_type='unicycle', dt=DT, scale=SCALE_COMBINED)
    preds, true = evaluate_model(model, test_loader, model_type='unicycle', dt=DT, scale=SCALE_XY, theta_scale=SCALE_THETA)
    os.makedirs(NETWORK_PATH, exist_ok=True)
    torch.save(model, NETWORK_PATH + "Unicycle.pth")
    print("Saved Model as Unicycle.pth")