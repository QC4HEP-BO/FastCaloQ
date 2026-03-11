from argparse import ArgumentParser
import json
from pathlib import Path

from qinn_pipeline import QINNTrainConfig, train_qinn


def _pick_value(args_value, cfg, key, default):
    """Prefer explicit CLI value, then config value, then fallback default."""
    if args_value is not None:
        return args_value
    if key in cfg:
        return cfg[key]
    return default


def main(args):
    cfg = json.load(open(args.config, "r")) if args.config else {}
    module_kwargs = cfg.get("qinn_module_kwargs", {})

    train_cfg = QINNTrainConfig(
        input_file=args.input_file,
        output_dir=args.output_path,
        batch_size=int(_pick_value(args.batch_size, cfg, "qinn_batch_size", 256)),
        epochs=int(_pick_value(args.epochs, cfg, "qinn_epochs", 200)),
        lr=float(_pick_value(args.lr, cfg, "qinn_lr", 1e-4)),
        weight_decay=float(_pick_value(args.weight_decay, cfg, "qinn_weight_decay", 0.0)),
        recon_weight=float(_pick_value(args.recon_weight, cfg, "qinn_recon_weight", 1.0)),
        prior_weight=float(_pick_value(args.prior_weight, cfg, "qinn_prior_weight", 0.1)),
        checkpoint_every=int(_pick_value(args.checkpoint_every, cfg, "qinn_checkpoint_every", 10)),
        progress_every=int(_pick_value(args.progress_every, cfg, "qinn_progress_every", 1)),
        seed=int(_pick_value(args.seed, cfg, "qinn_seed", 11)),
        split_energy_position=args.split_energy_position,
        max_events=int(args.max_events),
    )

    _, history = train_qinn(train_cfg, module_kwargs)
    print("[INFO] qINN train done", "epochs", len(history["epoch"]), "output", Path(args.output_path))


if __name__ == "__main__":
    parser = ArgumentParser(description="Standalone qINN trainer (separate from GAN).")
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--recon_weight", type=float, default=None)
    parser.add_argument("--prior_weight", type=float, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split_energy_position", type=str, default="", choices=["", "le12", "ge12", "ge12le18", "ge18"])
    parser.add_argument("--max_events", type=int, default=0)

    main(parser.parse_args())
