"""qINN module for FastCaloQ BNReLUqINN integration.

This module is intentionally configurable so it can be dropped into the current
Keras->Torch bridge without further code changes in `train.py` / `evaluate.py`.

Input/output contract expected by the bridge:
- input tensor shape:  [batch, in_features]
- output tensor shape: [batch, out_features]
"""

from typing import Optional

import torch
import torch.nn as nn


class QINNModule(nn.Module):
    """Hybrid classical/quantum module compatible with TorchQINNLayer.

    Parameters
    ----------
    in_features:
        Input feature dimension coming from FastCaloQ generator input.
    out_features:
        Output feature dimension consumed by downstream Keras dense layers.
    hidden_features:
        Hidden size for the classical fallback path.
    use_pennylane:
        If True, build a PennyLane quantum core. If False, classical MLP only.
    n_qubits:
        Number of qubits for the quantum block.
    n_q_layers:
        Number of trainable quantum layers.
    dropout:
        Dropout for the classical fallback path.
    q_device:
        PennyLane device backend, e.g. "default.qubit".
    q_diff_method:
        PennyLane differentiation method.
    q_shots:
        Number of shots; None uses analytic mode where supported.
    q_entanglement:
        "linear" or "ring" entanglement for the custom ansatz.
    """

    def __init__(
        self,
        in_features: int = 50,
        out_features: int = 50,
        hidden_features: int = 64,
        use_pennylane: bool = False,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        dropout: float = 0.0,
        q_device: str = "default.qubit",
        q_diff_method: str = "best",
        q_shots: Optional[int] = None,
        q_entanglement: str = "linear",
    ):
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features e out_features devono essere > 0")
        if n_qubits <= 0 or n_q_layers <= 0:
            raise ValueError("n_qubits e n_q_layers devono essere > 0")

        self.use_pennylane = use_pennylane
        self.in_features = in_features
        self.out_features = out_features

        if not self.use_pennylane:
            self.model = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, out_features),
            )
            return

        try:
            import pennylane as qml
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "use_pennylane=True ma PennyLane non e' installato nell'ambiente"
            ) from exc

        if q_entanglement not in {"linear", "ring"}:
            raise ValueError("q_entanglement deve essere 'linear' o 'ring'")

        # Classical pre-processing to match qubit dimension.
        self.pre_net = nn.Sequential(
            nn.Linear(in_features, n_qubits),
            nn.Tanh(),
        )

        dev = qml.device(q_device, wires=n_qubits, shots=q_shots)

        def _entangle_layer():
            if q_entanglement == "linear":
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
            else:  # ring
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

        @qml.qnode(dev, interface="torch", diff_method=q_diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer_idx in range(n_q_layers):
                for wire in range(n_qubits):
                    qml.RY(weights[layer_idx, wire, 0], wires=wire)
                    qml.RZ(weights[layer_idx, wire, 1], wires=wire)
                _entangle_layer()
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        weight_shapes = {"weights": (n_q_layers, n_qubits, 2)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, hidden_features),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)

        if not self.use_pennylane:
            return self.model(x)

        x_proj = self.pre_net(x)
        x_q = self.q_layer(x_proj)
        y = self.post_net(x_q)
        return y
