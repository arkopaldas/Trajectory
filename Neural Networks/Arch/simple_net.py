"""Network for the SIMPLE model: state (x, y) -> control (vx, vy).
Fully connected, ReLU activations, hidden widths 256-512-256-128-64 (ReLU keeps the network piecewise linear, which is what
the Marabou verifier requires).
"""
import torch.nn as nn
from Arch.base import NormalizedNet

# 2 inputs (x, y), 2 outputs (vx, vy). Normalisation comes from NormalizedNet.
class SimpleNet(NormalizedNet):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Linear(2, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            ),
            input_dim=2, output_dim=2
        )

# Earlier, smaller architecture (two tanh layers) kept for reference; not used.
# class SimpleNet(NormalizedNet):
#     def __init__(self, hidden_dim=32):
#         super().__init__(
#             nn.Sequential(
#                 nn.Linear(2, hidden_dim),
#                 nn.Tanh(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.Tanh(),
#                 nn.Linear(hidden_dim, 2)
#             ),
#             input_dim=2, output_dim=2
#         )