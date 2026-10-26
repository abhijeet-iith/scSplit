import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tifffile
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from core.logger import load_json
from data.data_location import DataLocation
from data.split_dataset import SplitDataset
from model import create_model
from model.normalizer import NormalizerXT


def load_model(config_path, checkpoint, gpu):
    opt = load_json(str(config_path))
    opt["phase"] = "val"
    opt["gpu_ids"] = [gpu]
    opt["distributed"] = False
    opt["device"] = "cuda"
    opt["path"]["resume_state"] = None
    opt["model"]["beta_schedule"]["val"]["n_timestep"] = 1
    logging.basicConfig(level=logging.INFO)
    model = create_model(opt)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.netG.load_state_dict(state_dict, strict=False)
    model.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], "val")
    model.netG.eval()
    return model, opt


def load_xt_normalizer(checkpoint_state, prefix, device):
    mean = checkpoint_state[f"{prefix}.data_mean"].cpu().numpy()
    std = checkpoint_state[f"{prefix}.data_std"].cpu().numpy()
    return NormalizerXT(data_mean=mean, data_std=std, data_count=1).to(device)


def get_channel_stats(config):
    train_paths = config["datasets"]["train"]["datapath"]
    train_location = DataLocation(channelwise_fpath=(train_paths["ch0"], train_paths["ch1"]))
    train_set = SplitDataset(
        "Hagen",
        train_location,
        config["datasets"]["patch_size"],
        enable_transforms=False,
        random_patching=True,
        normalize_channels=False,
        input_from_normalized_target=True,
    )
    stats = train_set.get_input_target_normalization_dict()
    return stats["mean_channel"].reshape(2), stats["std_channel"].reshape(2)


def infer_frame(model, normalizer, image, bt_channel, t_value, patch_size, device):
    height, width = image.shape
    prediction = np.zeros((2, height, width), dtype=np.float32)
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            patch = image[top:top + patch_size, left:left + patch_size]
            if patch.shape != (patch_size, patch_size):
                continue
            input_tensor = torch.from_numpy(patch.astype(np.float32))[None, None].to(device)
            if bt_channel == "mito":
                normalized = normalizer[1].normalize(input_tensor, torch.tensor([t_value], device=device))
                pred_mito = model.netG.indi2.inference(normalized, num_timesteps=1, t_float_start=t_value)
                pred_actin = model.netG.indi1.inference(normalized, num_timesteps=1, t_float_start=1.0 - t_value)
            else:
                normalized = normalizer[0].normalize(input_tensor, torch.tensor([t_value], device=device))
                pred_actin = model.netG.indi1.inference(normalized, num_timesteps=1, t_float_start=t_value)
                pred_mito = model.netG.indi2.inference(normalized, num_timesteps=1, t_float_start=1.0 - t_value)
            prediction[0, top:top + patch_size, left:left + patch_size] = pred_actin[0, 0].detach().cpu().numpy()
            prediction[1, top:top + patch_size, left:left + patch_size] = pred_mito[0, 0].detach().cpu().numpy()
    return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-actin", required=True)
    parser.add_argument("--target-mito", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bt-channel", choices=("actin", "mito"), required=True)
    parser.add_argument("--t", type=float, default=0.1)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if not 0.0 < args.t < 1.0:
        raise ValueError("--t must be strictly between 0 and 1")

    device = torch.device(f"cuda:{args.gpu}")
    config = load_json(args.config)
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

    predictions = []
    with torch.no_grad():
        for frame in input_stack:
            predictions.append(infer_frame(model, normalizers, frame, args.bt_channel, args.t, config["datasets"]["patch_size"], device))
    predictions = np.stack(predictions)
    predictions = predictions * std_channel.reshape(1, 2, 1, 1) + mean_channel.reshape(1, 2, 1, 1)
    predictions = np.clip(np.rint(predictions), 0, 1993).astype(np.uint16)
    targets = np.stack([target_actin, target_mito], axis=1).astype(np.uint16)

    output_dir = Path(args.output_dir)
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
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
