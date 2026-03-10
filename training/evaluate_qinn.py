from argparse import ArgumentParser
import json
from pathlib import Path

from qinn_pipeline import QINNEvalConfig, evaluate_qinn


def main(args):
    cfg = json.load(open(args.config, "r")) if args.config else {}
    module_kwargs = cfg.get("qinn_module_kwargs", {})

    eval_cfg = QINNEvalConfig(
        input_file=args.input_file,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_path,
        batch_size=int(cfg.get("qinn_eval_batch_size", args.batch_size)),
        split_energy_position=args.split_energy_position,
        max_events=int(args.max_events),
    )

    results = evaluate_qinn(eval_cfg, module_kwargs)
    print("[INFO] qINN eval", results)
    print("[INFO] saved", Path(args.output_path) / "qinn_eval.json")


if __name__ == "__main__":
    parser = ArgumentParser(description="Standalone qINN evaluation (separate from GAN).")
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--split_energy_position", type=str, default="", choices=["", "le12", "ge12", "ge12le18", "ge18"])
    parser.add_argument("--max_events", type=int, default=0)

    main(parser.parse_args())
