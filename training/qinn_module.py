"""qINN module aligned with QuantumINN block design.

This module keeps backward compatibility with FastCaloQ bridge kwargs while
implementing a QuantumINN architecture based on shared unitary blocks.
"""

import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------
# 1) Unitary template U(theta)
# ---------------------------
def _entangle_ring(wires: List[int]):
    for i, w in enumerate(wires):
        qml.CNOT(wires=[w, wires[(i + 1) % len(wires)]])


def unitary_template(weights: torch.Tensor, wires: List[int]):
    """Simple trainable ansatz with RZ/RY/RX + ring entanglement.

    weights shape: [depth, n_qubits, 3]
    """
    depth, n_qubits, _ = weights.shape
    if n_qubits != len(wires):
        raise ValueError("weights qubit dimension does not match wires")

    for l in range(depth):
        for i, w in enumerate(wires):
            phi_z, phi_y, phi_x = weights[l, i, 0], weights[l, i, 1], weights[l, i, 2]
            qml.RZ(phi_z, w)
            qml.RY(phi_y, w)
            qml.RX(phi_x, w)
        _entangle_ring(wires)


# ---------------------------
# 2) Quantum invertible block
# ---------------------------
class QuantumINNBlock(nn.Module):
    """Quantum invertible block with shared weights for forward/inverse."""

    def __init__(
        self,
        n_qubits: int,
        depth: int = 2,
        device_name: str = "default.qubit",
        shots: Optional[int] = None,
        wire_permutation: Optional[List[int]] = None,
        interface: str = "torch",
        diff_method: str = "auto",
        block_id: int = 0,
        capture_quantum_state: bool = False,
        capture_state_kind: str = "statevector",
        capture_every_n_calls: int = 1,
    ):
        super().__init__()

        if n_qubits <= 0:
            raise ValueError("n_qubits must be > 0")
        if depth <= 0:
            raise ValueError("depth must be > 0")

        self.n_qubits = n_qubits
        self.depth = depth
        self.wires = list(range(n_qubits))
        self.perm = wire_permutation if wire_permutation is not None else list(range(n_qubits))
        self.block_id = block_id
        self.capture_quantum_state = capture_quantum_state
        self.capture_state_kind = capture_state_kind
        self.capture_every_n_calls = max(1, int(capture_every_n_calls))
        self._capture_call_count = 0
        self._captured_count = 0
        self._last_capture = {}

        init_scale = 0.01
        self.weights = nn.Parameter(init_scale * torch.randn(depth, n_qubits, 3))

        self.dev = qml.device(device_name, wires=n_qubits, shots=shots)

        def circuit_forward(x, weights):
            x_perm = x[..., self.perm]
            qml.AngleEmbedding(x_perm, wires=self.wires, rotation="Y")
            unitary_template(weights, self.wires)
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        def circuit_inverse(z, weights):
            z_perm = z[..., self.perm]
            qml.AngleEmbedding(z_perm, wires=self.wires, rotation="Y")
            qml.adjoint(unitary_template)(weights, self.wires)
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.qnode_fwd = qml.QNode(circuit_forward, self.dev, interface=interface, diff_method=diff_method)
        self.qnode_inv = qml.QNode(circuit_inverse, self.dev, interface=interface, diff_method=diff_method)

        if self.capture_quantum_state:
            if self.capture_state_kind not in {"statevector", "density_matrix"}:
                raise ValueError("capture_state_kind must be 'statevector' or 'density_matrix'")
            self._dev_state = qml.device(device_name, wires=n_qubits, shots=None)

            def circuit_forward_state(x, weights):
                x_perm = x[..., self.perm]
                qml.AngleEmbedding(x_perm, wires=self.wires, rotation="Y")
                unitary_template(weights, self.wires)
                if self.capture_state_kind == "statevector":
                    return qml.state()
                return qml.density_matrix(wires=self.wires)

            def circuit_inverse_state(z, weights):
                z_perm = z[..., self.perm]
                qml.AngleEmbedding(z_perm, wires=self.wires, rotation="Y")
                qml.adjoint(unitary_template)(weights, self.wires)
                if self.capture_state_kind == "statevector":
                    return qml.state()
                return qml.density_matrix(wires=self.wires)

            self.qnode_fwd_state = qml.QNode(circuit_forward_state, self._dev_state, interface=interface, diff_method=diff_method)
            self.qnode_inv_state = qml.QNode(circuit_inverse_state, self._dev_state, interface=interface, diff_method=diff_method)

    @staticmethod
    def _qnode_output_to_tensor(out):
        """Normalize PennyLane QNode outputs to a torch.Tensor.

        With multiple expvals PennyLane can return a Python list/tuple of tensors;
        downstream torch layers expect a single tensor.
        """
        if isinstance(out, (list, tuple)):
            if len(out) == 0:
                raise ValueError("QNode output is empty")
            out = torch.stack(out, dim=-1)

        return out

    def _maybe_capture_state(self, x: torch.Tensor, direction: str):
        if not self.capture_quantum_state:
            return

        self._capture_call_count += 1
        if self._capture_call_count % self.capture_every_n_calls != 0:
            return
        if direction == "forward":
            state = self.qnode_fwd_state(x, self.weights)
        elif direction == "inverse":
            state = self.qnode_inv_state(x, self.weights)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        payload = {
            "block_id": self.block_id,
            "direction": direction,
            "capture_index": self._captured_count,
            "call_index": self._capture_call_count,
            "state_kind": self.capture_state_kind,
            "state": state.detach().cpu(),
        }
        self._last_capture[direction] = payload
        self._captured_count += 1

    def get_latest_capture(self):
        if not self.capture_quantum_state:
            return None
        return {
            "block_id": self.block_id,
            "captures": dict(self._last_capture),
        }

    def forward_block(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qnode_fwd(x, self.weights)
        out = self._qnode_output_to_tensor(out)
        self._maybe_capture_state(x, direction="forward")
        # PennyLane may emit float64 expvals; keep dtype aligned with module weights
        # (typically float32) to avoid matmul dtype mismatch in torch Linear layers.
        return out.to(dtype=self.weights.dtype)

    def inverse_block(self, z: torch.Tensor) -> torch.Tensor:
        out = self.qnode_inv(z, self.weights)
        out = self._qnode_output_to_tensor(out)
        self._maybe_capture_state(z, direction="inverse")
        return out.to(dtype=self.weights.dtype)


# ---------------------------
# 3) Full QuantumINN model
# ---------------------------
class QuantumINN(nn.Module):
    """Stack of quantum invertible blocks with wire permutations."""

    def __init__(
        self,
        n_features_in: int,
        n_features_out: Optional[int] = None,
        n_qubits: Optional[int] = None,
        n_blocks: int = 2,
        depth_per_block: int = 2,
        shots: Optional[int] = None,
        device_name: str = "default.qubit",
        nonneg_output: bool = True,
        q_diff_method: str = "auto",
        capture_quantum_state: bool = False,
        capture_state_kind: str = "statevector",
        capture_every_n_calls: int = 1,
    ):
        super().__init__()

        if n_features_in <= 0:
            raise ValueError("n_features_in must be > 0")
        if n_blocks <= 0:
            raise ValueError("n_blocks must be > 0")

        self.n_features_in = n_features_in
        self.n_features_out = n_features_out if n_features_out is not None else n_features_in
        if n_qubits is None:
            target = min(64, max(4, n_features_in))
            n_qubits = 1 << (math.ceil(math.log2(target)))

        self.n_qubits = n_qubits
        self.nonneg_output = nonneg_output

        self.pre = nn.Linear(self.n_features_in, self.n_qubits)
        self.post = nn.Linear(self.n_qubits, self.n_features_out)

        perms = []
        base = list(range(self.n_qubits))
        step = max(1, self.n_qubits // 3)
        for k in range(n_blocks):
            shift = (step * k) % self.n_qubits
            perm = base[shift:] + base[:shift]
            perms.append(perm)

        self.blocks = nn.ModuleList(
            [
                QuantumINNBlock(
                    n_qubits=self.n_qubits,
                    depth=depth_per_block,
                    device_name=device_name,
                    shots=shots,
                    wire_permutation=perms[i],
                    diff_method=q_diff_method,
                    block_id=i,
                    capture_quantum_state=capture_quantum_state,
                    capture_state_kind=capture_state_kind,
                    capture_every_n_calls=capture_every_n_calls,
                )
                for i in range(n_blocks)
            ]
        )

        self.act_in = nn.Tanh()
        self.act_mid = nn.Tanh()
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act_in(self.pre(x))
        for blk in self.blocks:
            h = blk.forward_block(h)
            h = self.act_mid(h)
        z = self.post(h)
        return z

    def inverse(self, z: torch.Tensor, enforce_nonneg: Optional[bool] = None) -> torch.Tensor:
        if enforce_nonneg is None:
            enforce_nonneg = self.nonneg_output

        h = self.act_in(self.pre(z))
        for blk in reversed(self.blocks):
            h = blk.inverse_block(h)
            h = self.act_mid(h)
        x_rec = self.post(h)
        if enforce_nonneg:
            x_rec = self.softplus(x_rec)
        return x_rec

    def collect_latest_quantum_state(self):
        snapshots = []
        for blk in self.blocks:
            snap = blk.get_latest_capture()
            if snap is not None:
                snapshots.append(snap)
        return snapshots


# ---------------------------
# 4) MMD loss (RBF)
# ---------------------------
def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """Maximum Mean Discrepancy with RBF kernel."""
    with torch.no_grad():
        if sigma is None:
            xy = torch.cat([x, y], dim=0)
            d2 = torch.cdist(xy, xy, p=2.0).pow(2)
            positive = d2[d2 > 0]
            if positive.numel() == 0:
                sigma = 1.0
            else:
                med = torch.median(positive).clamp(min=1e-6)
                sigma = torch.sqrt(med * 0.5).item()

    gamma = 1.0 / (2.0 * (sigma**2 + 1e-12))

    def k(a, b):
        d2 = torch.cdist(a, b, p=2.0).pow(2)
        return torch.exp(-gamma * d2)

    k_xx = k(x, x)
    k_yy = k(y, y)
    k_xy = k(x, y)

    m = x.shape[0]
    n = y.shape[0]
    term_xx = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1) + 1e-12)
    term_yy = (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1) + 1e-12)
    term_xy = (2.0 * k_xy.sum()) / (m * n + 1e-12)
    return term_xx + term_yy - term_xy


class QINNModule(nn.Module):
    """FastCaloQ-compatible wrapper around QuantumINN.

    Compatible kwargs:
    - in_features/out_features (existing bridge contract)
    - use_pennylane: False -> classical fallback MLP
    - n_qubits, n_blocks, depth_per_block
    - n_q_layers: alias for depth_per_block (for backward compatibility)
    - q_device, q_shots, q_diff_method
    - nonneg_output
    """

    def __init__(
        self,
        in_features: int = 50,
        out_features: int = 50,
        hidden_features: int = 64,
        use_pennylane: bool = True,
        n_qubits: Optional[int] = None,
        n_blocks: int = 2,
        depth_per_block: int = 2,
        n_q_layers: Optional[int] = None,
        q_device: str = "default.qubit",
        q_shots: Optional[int] = None,
        q_diff_method: str = "auto",
        nonneg_output: bool = True,
        dropout: float = 0.0,
        q_capture_quantum_state: bool = False,
        q_capture_state_kind: str = "statevector",
        q_capture_every_n_calls: int = 1,
        **_: object,
    ):
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be > 0")

        self.use_pennylane = use_pennylane
        self.nonneg_output = nonneg_output

        if not self.use_pennylane:
            self.model = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, out_features),
            )
            return

        try:
            global qml
            import pennylane as qml
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "use_pennylane=True but PennyLane is not installed in the environment"
            ) from exc

        if n_q_layers is not None:
            depth_per_block = n_q_layers

        self.model = QuantumINN(
            n_features_in=in_features,
            n_features_out=out_features,
            n_qubits=n_qubits,
            n_blocks=n_blocks,
            depth_per_block=depth_per_block,
            shots=q_shots,
            device_name=q_device,
            nonneg_output=nonneg_output,
            q_diff_method=q_diff_method,
            capture_quantum_state=q_capture_quantum_state,
            capture_state_kind=q_capture_state_kind,
            capture_every_n_calls=q_capture_every_n_calls,
        )

    def save_checkpoint_artifacts(self, checkpoint_dir: str, iteration: int):
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), checkpoint_path / f"qinn_module_state-{int(iteration)}.pt")

        if self.use_pennylane:
            snapshots = self.model.collect_latest_quantum_state()
            if snapshots:
                payload = {
                    "iteration": int(iteration),
                    "state_kind": snapshots[0]["captures"].get("forward", snapshots[0]["captures"].get("inverse", {})).get("state_kind", "unknown") if snapshots else "unknown",
                    "blocks": snapshots,
                }
                torch.save(payload, checkpoint_path / f"qinn_quantum_state-{int(iteration)}.pt")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        if not self.use_pennylane:
            return self.model(x)

        # Generator semantics: latent -> data-like features.
        return self.model.inverse(x, enforce_nonneg=self.nonneg_output)


if __name__ == "__main__":
    torch.manual_seed(0)

    d = 64
    model = QINNModule(
        in_features=d,
        out_features=d,
        use_pennylane=False,
        n_qubits=32,
        n_blocks=3,
        depth_per_block=2,
        q_shots=None,
        q_device="default.qubit",
        nonneg_output=True,
    )

    batch = 32
    z = torch.randn(batch, d)
    x_hat = model(z)
    x_real = torch.rand(batch, d) * 2.0

    loss = mmd_rbf(x_hat, x_real)
    print("MMD loss (demo):", float(loss))
