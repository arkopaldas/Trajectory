import os
import torch
import numpy as np
from Models.config import NETWORK_PATH
from Models.Simple.model import SimpleTrajectoryNet, build_simple_dataset
from Models.utils import create_dataloaders, train_model, evaluate_model, upload_csv

if __name__ == "__main__":
    df = upload_csv("Simple Model Result.csv")
    DT = df["DT"].iloc[0]
    X, Y, N = build_simple_dataset(df)
    x_range = N[:, 0].max() - N[:, 0].min()
    y_range = N[:, 1].max() - N[:, 1].min()
    SCALE_XY = np.sqrt(x_range**2 + y_range**2)
    print(f"\nEnvironment scale (diagonal):   {SCALE_XY:.4f} units\n")
    train_loader, test_loader = create_dataloaders(X, Y, N)
    model = SimpleTrajectoryNet()
    model = train_model(model, train_loader, model_type='simple', dt=DT, scale=SCALE_XY)
    preds, true = evaluate_model(model, test_loader, model_type='simple', dt=DT, scale=SCALE_XY)
    os.makedirs(NETWORK_PATH, exist_ok=True)
    torch.save(model, NETWORK_PATH + "Simple.pth")
    print("Saved Model as Simple.pth")