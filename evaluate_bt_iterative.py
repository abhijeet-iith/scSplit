import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tifffile
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from core.logger import load_json
from model import create_model
from model.normalizer import NormalizerXT


def load_model(config_path, checkpoint, device):
    opt = load_json(str(config_path))
    opt["phase"] = "val"
    opt["gpu_ids"] = [0]
    opt["distributed"] = False
    opt["device"] = "cuda"
    opt["path"]["resume_state"] = None
    state_dict = torch.load(checkpoint, map_location="cpu")
    opt["model"]["xt_normalizer_1"] = load_xt_normalizer(
        state_dict,
        "indi1._xt_normalizer",
        device,
    )
    opt["model"]["xt_normalizer_2"] = load_xt_normalizer(
        state_dict,
        "indi2._xt_normalizer",
        device,
    )
    logging.basicConfig(level=logging.INFO)
    model = create_model(opt)
    model.netG.load_state_dict(state_dict, strict=True)
    model.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], "val")
    model.netG.eval()
    model.netG.to(device)
    return model, opt


def load_xt_normalizer(checkpoint_state, prefix, device):
    mean = checkpoint_state[f"{prefix}.data_mean"].cpu().numpy()
    std = checkpoint_state[f"{prefix}.data_std"].cpu().numpy()
    return NormalizerXT(data_mean=mean, data_std=std, data_count=1).to(device)


def extract_patches(stack, patch_size):
    frames, height, width = stack.shape
    patches = []
    locations = []
    for frame_idx in range(frames):
        for top in range(0, height - patch_size + 1, patch_size):
            for left in range(0, width - patch_size + 1, patch_size):
                patches.append(stack[frame_idx, top:top + patch_size, left:left + patch_size])
                locations.append((frame_idx, top, left))
    if not patches:
        raise ValueError("Input contains no complete patches")
    return (
        torch.from_numpy(np.asarray(patches, dtype=np.float32)).unsqueeze(1),
        locations,
    )


def normalize_at_time(values, normalizer, time_value):
    times = torch.full(
        (values.shape[0],),
        time_value,
        dtype=values.dtype,
        device=values.device,
    )
    return normalizer.normalize(values, times, update=False)


def denormalize_at_time(values, normalizer, time_value):
    time_bin = min(int(time_value * normalizer.num_bins), normalizer.num_bins - 1)
    mean = normalizer.data_mean[time_bin].to(values.device, values.dtype)
    std = normalizer.data_std[time_bin].to(values.device, values.dtype)
    return values * std + mean


def run_branch(
    raw_patches,
    indi,
    normalizer,
    initial_time,
    num_steps,
    batch_size,
    device,
    seed,
):
    current = normalize_at_time(raw_patches.to(device), normalizer, initial_time)
    step_size = initial_time / num_steps
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for step_index in range(num_steps):
        current_time = max(initial_time - step_index * step_size, step_size)
        next_time = max(current_time - step_size, 0.0)
        next_values = []
        for start in range(0, len(current), batch_size):
            batch = current[start:start + batch_size]
            noise = torch.randn(
                batch.shape,
                generator=generator,
                device=device,
                dtype=batch.dtype,
            )
            noisy = batch + noise * (0.01 * current_time)
            current_tensor = torch.full(
                (batch.shape[0],),
                current_time,
                dtype=batch.dtype,
                device=device,
            )
            step_tensor = torch.full_like(current_tensor, step_size)
            if getattr(indi, "partial_step_training", False):
                if getattr(indi, "partial_step_conditioning", "b") == "tb":
                    condition = torch.stack([current_tensor, step_tensor], dim=1)
                else:
                    condition = step_tensor
            else:
                condition = current_tensor
            predicted = indi.denoise_fn(noisy, condition).detach()
            # The prediction represents the raw state at next_time in the
            # time-dependent normalized representation used for training.
            # Explicitly convert through raw values and normalize again so
            # every network input follows the requested evaluation protocol.
            predicted_raw = denormalize_at_time(
                predicted,
                normalizer,
                next_time,
            )
            next_values.append(
                normalize_at_time(predicted_raw, normalizer, next_time)
            )

        current = torch.cat(next_values, dim=0)

    return denormalize_at_time(current, normalizer, 0.0).cpu()


def place_patches(patches, locations, shape, patch_size):
    output = np.zeros(shape, dtype=np.float32)
    for patch, (frame_idx, top, left) in zip(patches[:, 0], locations):
        output[frame_idx, top:top + patch_size, left:left + patch_size] = patch
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-actin", required=True)
    parser.add_argument("--target-mito", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bt-channel", choices=("actin", "mito"), required=True)
    parser.add_argument("--t", type=float, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=15)
    args = parser.parse_args()

    if not 0.0 < args.t < 1.0:
        raise ValueError("--t must be strictly between 0 and 1")
    if args.num_steps < 1 or args.batch_size < 1:
        raise ValueError("--num-steps and --batch-size must be positive")

    device = torch.device(f"cuda:{args.gpu}")
    model, _ = load_model(args.config, args.checkpoint, device)
    normalizers = [
        model.netG.indi1._xt_normalizer,
        model.netG.indi2._xt_normalizer,
    ]
    config = load_json(args.config)

    input_stack = tifffile.imread(args.input)[:args.max_frames]
    target_actin = tifffile.imread(args.target_actin)[:args.max_frames]
    target_mito = tifffile.imread(args.target_mito)[:args.max_frames]
    if input_stack.shape != target_actin.shape or input_stack.shape != target_mito.shape:
        raise ValueError("Input and target stacks must have matching shapes")

    patch_size = config["datasets"]["patch_size"]
    raw_patches, locations = extract_patches(input_stack, patch_size)
    if args.bt_channel == "actin":
        initial_times = (args.t, 1.0 - args.t)
    else:
        initial_times = (1.0 - args.t, args.t)

    with torch.no_grad():
        predicted_actin = run_branch(
            raw_patches,
            model.netG.indi1,
            normalizers[0],
            initial_times[0],
            args.num_steps,
            args.batch_size,
            device,
            args.seed,
        )
        predicted_mito = run_branch(
            raw_patches,
            model.netG.indi2,
            normalizers[1],
            initial_times[1],
            args.num_steps,
            args.batch_size,
            device,
            args.seed + 1,
        )

    prediction_shape = input_stack.shape
    prediction = np.stack([
        place_patches(predicted_actin.numpy(), locations, prediction_shape, patch_size),
        place_patches(predicted_mito.numpy(), locations, prediction_shape, patch_size),
    ], axis=1)
    prediction = np.clip(np.rint(prediction), 0, 1993).astype(np.uint16)
    targets = np.stack([target_actin, target_mito], axis=1).astype(np.uint16)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_dir / "predicted_actin.tif", prediction[:, 0])
    tifffile.imwrite(output_dir / "predicted_mito.tif", prediction[:, 1])
    tifffile.imwrite(output_dir / "target_actin.tif", targets[:, 0])
    tifffile.imwrite(output_dir / "target_mito.tif", targets[:, 1])

    metrics = {}
    for index, name in enumerate(("actin", "mito")):
        metrics[name] = {
            "psnr": float(
                peak_signal_noise_ratio(
                    targets[:, index], prediction[:, index], data_range=1993
                )
            ),
            "ssim": float(np.mean([
                structural_similarity(target, pred, data_range=1993)
                for target, pred in zip(targets[:, index], prediction[:, index])
            ])),
        }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
