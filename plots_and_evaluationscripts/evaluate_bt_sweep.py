import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from scSplit.plots_and_evaluationscripts.evaluate_bt import get_channel_stats, infer_frame, load_model, load_xt_normalizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-actin", required=True)
    parser.add_argument("--target-mito", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bt-channel", choices=("actin", "mito"), required=True)
    parser.add_argument("--t-values", nargs="+", type=float, required=True)
    parser.add_argument("--max-frames", type=int, default=15)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if any(not 0.0 < value < 1.0 for value in args.t_values):
        raise ValueError("All --t-values must be strictly between 0 and 1")

    device = torch.device(f"cuda:{args.gpu}")
    config = json.loads(Path(args.config).read_text())
    model, _ = load_model(args.config, args.checkpoint, args.gpu)
    checkpoint_state = torch.load(args.checkpoint, map_location="cpu")
    normalizers = [
        load_xt_normalizer(checkpoint_state, "indi1._xt_normalizer", device),
        load_xt_normalizer(checkpoint_state, "indi2._xt_normalizer", device),
    ]
    mean_channel, std_channel = get_channel_stats(config)
    input_stack = tifffile.imread(args.input)[:args.max_frames]
    target_actin = tifffile.imread(args.target_actin)[:args.max_frames]
    target_mito = tifffile.imread(args.target_mito)[:args.max_frames]
    if input_stack.shape != target_actin.shape or input_stack.shape != target_mito.shape:
        raise ValueError("Input and target stacks must have matching shapes")

    targets = np.stack([target_actin, target_mito], axis=1).astype(np.uint16)
    with torch.no_grad():
        for t_value in args.t_values:
            predictions = np.stack([
                infer_frame(model, normalizers, frame, args.bt_channel, t_value,
                            config["datasets"]["patch_size"], device)
                for frame in input_stack
            ])
            predictions = predictions * std_channel.reshape(1, 2, 1, 1) + mean_channel.reshape(1, 2, 1, 1)
            predictions = np.clip(np.rint(predictions), 0, 1993).astype(np.uint16)

            output_dir = Path(args.output_root) / f"t_{t_value:.2f}"
            output_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(output_dir / "predicted_actin.tif", predictions[:, 0])
            tifffile.imwrite(output_dir / "predicted_mito.tif", predictions[:, 1])
            tifffile.imwrite(output_dir / "target_actin.tif", targets[:, 0])
            tifffile.imwrite(output_dir / "target_mito.tif", targets[:, 1])

            metrics = {}
            for index, name in enumerate(("actin", "mito")):
                psnr = peak_signal_noise_ratio(targets[:, index], predictions[:, index], data_range=1993)
                ssim = np.mean([
                    structural_similarity(target, prediction, data_range=1993)
                    for target, prediction in zip(targets[:, index], predictions[:, index])
                ])
                metrics[name] = {"psnr": float(psnr), "ssim": float(ssim)}
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(f"t={t_value:.2f} {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
