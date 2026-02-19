"""Example qINN module used by TorchQINNLayer.

Replace this file with your definitive qINN implementation, keeping the
`QINNModule` class signature compatible with the bridge.
"""

import torch
import torch.nn as nn


class QINNModule(nn.Module):
    def __init__(self, in_features=50, out_features=50, hidden_features=64, use_pennylane=False):
        super().__init__()
        self.use_pennylane = use_pennylane
        self.linear_in = nn.Linear(in_features, hidden_features)
        self.activation = nn.SiLU()
        self.linear_out = nn.Linear(hidden_features, out_features)

        if self.use_pennylane:
            try:
                import pennylane as qml  # noqa: F401
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "use_pennylane=True ma PennyLane non e' installato nell'ambiente"
                ) from exc

    def forward(self, x):
        x = self.linear_in(x)
        x = self.activation(x)
        x = self.linear_out(x)
        return x
