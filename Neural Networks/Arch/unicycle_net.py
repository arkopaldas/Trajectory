import torch.nn as nn
from Arch.base import NormalizedNet

class UnicycleNet(NormalizedNet):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Linear(3, 256),
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
            input_dim=3, output_dim=2
        )

# class UnicycleNet(NormalizedNet):
#     def __init__(self, hidden_dim=64):
#         super().__init__(
#             nn.Sequential(
#                 nn.Linear(3, hidden_dim),
#                 nn.Tanh(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.Tanh(),
#                 nn.Linear(hidden_dim, 2)
#             ),
#             input_dim=3, output_dim=2
#         )