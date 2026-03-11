"""Standalone qINN training/evaluation pipeline (separate from GAN).

This module is intentionally independent from WGANGP so qINN experiments can be
run with their own objective and checkpoint lifecycle.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from common import get_kin
from data import filter_energy
from qinn_module import QINNModule, mmd_rbf


@dataclass
class QINNTrainConfig:
    input_file: str
    output_dir: str
    batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.0
    recon_weight: float = 1.0
    prior_weight: float = 0.1
    checkpoint_every: int = 10
    progress_every: int = 1
    seed: int = 11
    split_energy_position: str = ""
    max_events: int = 0


@dataclass
class QINNEvalConfig:
    input_file: str
    checkpoint_path: str
    output_dir: str
    batch_size: int = 512
    split_energy_position: str = ""
    max_events: int = 0


def _load_showers(input_file: str, split_energy_position: str = "", max_events: int = 0) -> np.ndarray:
    with h5py.File(input_file, "r") as f:
        showers = f["showers"][:]
        energies = f["incident_energies"][:]

    kin, particle = get_kin(input_file)
    showers = filter_energy(particle, energies, split_energy_position, showers)

    if max_events and max_events > 0:
        showers = showers[:max_events]

    return showers.astype(np.float32)


def _make_loader(array: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    tensor = torch.from_numpy(array)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_qinn(config: QINNTrainConfig, module_kwargs: Dict) -> Tuple[QINNModule, Dict]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    x = _load_showers(config.input_file, config.split_energy_position, config.max_events)
    n_features = x.shape[1]

    kwargs = dict(module_kwargs)
    kwargs.setdefault("in_features", n_features)
    kwargs.setdefault("out_features", n_features)

    model = QINNModule(**kwargs)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loader = _make_loader(x, config.batch_size, shuffle=True)

    out_dir = Path(config.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"epoch": [], "loss": [], "recon": [], "prior": []}

    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_prior = 0.0
        nb = 0

        for (xb,) in loader:
            optimizer.zero_grad(set_to_none=True)

            z = model.model.forward(xb) if model.use_pennylane else model(xb)
            x_rec = model.model.inverse(z, enforce_nonneg=model.nonneg_output) if model.use_pennylane else model(xb)

            recon_loss = torch.mean((x_rec - xb) ** 2)
            z_prior = torch.randn_like(z)
            prior_loss = mmd_rbf(z, z_prior)
            loss = config.recon_weight * recon_loss + config.prior_weight * prior_loss

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach())
            epoch_recon += float(recon_loss.detach())
            epoch_prior += float(prior_loss.detach())
            nb += 1

        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss / max(1, nb))
        history["recon"].append(epoch_recon / max(1, nb))
        history["prior"].append(epoch_prior / max(1, nb))

        if epoch % max(1, config.progress_every) == 0 or epoch == 1 or epoch == config.epochs:
            print(
                f"[INFO] qINN Epoch {epoch}/{config.epochs} "
                f"loss={history['loss'][-1]:.6f} recon={history['recon'][-1]:.6f} prior={history['prior'][-1]:.6f}"
            )

        if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
            model.save_checkpoint_artifacts(str(ckpt_dir), epoch)
            torch.save(
                {
                    "epoch": epoch,
                    "model_kwargs": kwargs,
                    "train_config": vars(config),
                    "history": history,
                },
                ckpt_dir / f"qinn_train_meta-{epoch}.pt",
            )

    with open(out_dir / "qinn_history.json", "w") as fp:
        json.dump(history, fp, indent=2)

    fig, ax = plt.subplots()
    ax.plot(history["epoch"], history["loss"], label="total")
    ax.plot(history["epoch"], history["recon"], label="recon")
    ax.plot(history["epoch"], history["prior"], label="prior")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend(fontsize=10)
    fig.savefig(out_dir / "loss.pdf")
    plt.close(fig)

    return model, history


def evaluate_qinn(config: QINNEvalConfig, module_kwargs: Dict) -> Dict:
    x = _load_showers(config.input_file, config.split_energy_position, config.max_events)
    n_features = x.shape[1]

    kwargs = dict(module_kwargs)
    kwargs.setdefault("in_features", n_features)
    kwargs.setdefault("out_features", n_features)

    model = QINNModule(**kwargs)
    state = torch.load(config.checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    loader = _make_loader(x, config.batch_size, shuffle=False)

    rec_losses = []
    generated = []
    truth = []

    with torch.no_grad():
        for (xb,) in loader:
            z = torch.randn(xb.shape[0], kwargs["in_features"], dtype=xb.dtype)
            x_gen = model(z)
            z_data = model.model.forward(xb) if model.use_pennylane else xb
            x_rec = model.model.inverse(z_data, enforce_nonneg=model.nonneg_output) if model.use_pennylane else model(xb)

            rec_losses.append(float(torch.mean((x_rec - xb) ** 2)))
            generated.append(x_gen.cpu().numpy())
            truth.append(xb.cpu().numpy())

    gen = np.concatenate(generated, axis=0)
    tru = np.concatenate(truth, axis=0)

    # simple chi2 on total shower energy histogram
    e_gen = gen.sum(axis=1)
    e_tru = tru.sum(axis=1)
    bins = np.linspace(min(e_tru.min(), e_gen.min()), max(e_tru.max(), e_gen.max()), 51)
    h_gen, _ = np.histogram(e_gen, bins=bins)
    h_tru, _ = np.histogram(e_tru, bins=bins)
    denom = h_tru + 1e-6
    chi2 = float(np.mean(((h_gen - h_tru) ** 2) / denom))

    results = {
        "reconstruction_mse": float(np.mean(rec_losses)),
        "chi2_total_energy": chi2,
        "n_events": int(tru.shape[0]),
    }

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "qinn_eval.json", "w") as fp:
        json.dump(results, fp, indent=2)

    return results
