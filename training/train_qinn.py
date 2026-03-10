from argparse import ArgumentParser
import json
from pathlib import Path

from qinn_pipeline import QINNTrainConfig, train_qinn


def main(args):
    cfg = json.load(open(args.config, "r")) if args.config else {}
    module_kwargs = cfg.get("qinn_module_kwargs", {})

    train_cfg = QINNTrainConfig(
        input_file=args.input_file,
        output_dir=args.output_path,
        batch_size=int(cfg.get("qinn_batch_size", args.batch_size)),
        epochs=int(cfg.get("qinn_epochs", args.epochs)),
        lr=float(cfg.get("qinn_lr", args.lr)),
        weight_decay=float(cfg.get("qinn_weight_decay", args.weight_decay)),
        recon_weight=float(cfg.get("qinn_recon_weight", args.recon_weight)),
        prior_weight=float(cfg.get("qinn_prior_weight", args.prior_weight)),
        checkpoint_every=int(cfg.get("qinn_checkpoint_every", args.checkpoint_every)),
        seed=int(cfg.get("qinn_seed", args.seed)),
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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--prior_weight", type=float, default=0.1)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--split_energy_position", type=str, default="", choices=["", "le12", "ge12", "ge12le18", "ge18"])
    parser.add_argument("--max_events", type=int, default=0)

    main(parser.parse_args())
