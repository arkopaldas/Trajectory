"""Base class shared by all controller networks.

NormalizedNet wraps a plain Linear / ReLU stack with z-score normalisation of the input (state) and of the output
(control). The statistics are stored as buffers (saved with the model, never trained), so forward() always takes RAW
states and returns RAW controls. For the ONNX export, folded_network() removes the normalisation again by folding the
statistics into the first and last Linear layer, which gives a plain network that is easy for the verifier to encode.
"""
# copy: deep copy of the network when it is folded for export.
import copy
import torch
import torch.nn as nn


class NormalizedNet(nn.Module):
    """
    MLP with z-score normalisation of its input and output.

    forward(raw_state) -> raw_control, so the training loop, the physics and the plots always work in
    physical units. Internally the layers see (x - in_mean) / in_std and their output is mapped back with
    out * out_std + out_mean. The four statistics are buffers (saved with the model, not trained).
    """

    # net: the Linear / ReLU stack. The four buffers start as identity statistics (mean 0, std 1).
    def __init__(self, net, input_dim, output_dim):
        super().__init__()
        self.net = net
        self.register_buffer("in_mean", torch.zeros(input_dim))
        self.register_buffer("in_std", torch.ones(input_dim))
        self.register_buffer("out_mean", torch.zeros(output_dim))
        self.register_buffer("out_std", torch.ones(output_dim))

    # Loads the statistics (from utils.compute_normalization) into the buffers.
    def set_normalization(self, in_mean, in_std, out_mean, out_std):
        for name, value in (("in_mean", in_mean), ("in_std", in_std), ("out_mean", out_mean), ("out_std", out_std)):
            buf = getattr(self, name)
            buf.copy_(torch.as_tensor(value, dtype=buf.dtype, device=buf.device))

    # raw state -> normalise -> network -> de-normalise -> raw control.
    def forward(self, x):
        z = (x - self.in_mean) / self.in_std
        return self.net(z) * self.out_std + self.out_mean

    @torch.no_grad()
    def folded_network(self):
        """
        Plain Linear/ReLU stack equivalent to forward(): the input statistics are folded into the first
        Linear layer and the output statistics into the last one. This is what gets exported to ONNX
        (raw states in, raw controls out, no extra Sub/Div/Mul/Add nodes).
        """
        folded = copy.deepcopy(self.net)
        linears = [m for m in folded if isinstance(m, nn.Linear)]
        first, last = linears[0], linears[-1]
        # W1 @ ((x - mean) / std) + b1  =  (W1 / std) @ x + (b1 - W1 @ (mean / std))
        first.bias.copy_(first.bias - first.weight @ (self.in_mean / self.in_std))
        first.weight.copy_(first.weight / self.in_std.unsqueeze(0))
        # (WL @ a + bL) * out_std + out_mean
        last.weight.copy_(last.weight * self.out_std.unsqueeze(1))
        last.bias.copy_(last.bias * self.out_std + self.out_mean)
        return folded
