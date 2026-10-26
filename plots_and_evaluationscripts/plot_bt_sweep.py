import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = {
    "mito_bt_010": "mito",
    "mito_bt_020": "mito",
    "mito_bt_040": "mito",
    "actin_bt_010": "actin",
    "actin_bt_020": "actin",
    "actin_bt_040": "actin",
}
METRICS = ("psnr", "ssim")
CHANNELS = ("actin", "mito")


def load_sweep(sweep_root):
    results = {}
    for model_name in MODELS:
        model_dir = sweep_root / model_name
        points = []
        for metrics_path in model_dir.glob("t_*/metrics.json"):
            t_value = float(metrics_path.parent.name.removeprefix("t_"))
            metrics = json.loads(metrics_path.read_text())
            points.append((t_value, metrics))
        points.sort(key=lambda item: item[0])
        if not points:
            raise FileNotFoundError(f"No metrics found under {model_dir}")
        results[model_name] = points
    return results


def plot_channel_metric(results, channel, metric, output_path):
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    model_names = [name for name, primary in MODELS.items() if primary == channel]
    for model_name in model_names:
        points = results[model_name]
        t_values = [t for t, _ in points]
        values = [metrics[channel][metric] for _, metrics in points]
        axis.plot(t_values, values, marker="o", linewidth=2, label=model_name)
    axis.set_title(f"{channel.capitalize()} {metric.upper()} vs inference t")
    axis.set_xlabel("Inference t")
    axis.set_ylabel(metric.upper() + (" (dB)" if metric == "psnr" else ""))
    axis.grid(True, alpha=0.3)
    axis.legend(title="BT model")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_all_models_metric(results, metric, output_path):
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for axis, (model_name, points) in zip(axes.flat, results.items()):
        t_values = [t for t, _ in points]
        for channel in CHANNELS:
            values = [metrics[channel][metric] for _, metrics in points]
            axis.plot(t_values, values, marker="o", label=channel)
        axis.set_title(model_name)
        axis.set_xlabel("Inference t")
        axis.set_ylabel(metric.upper() + (" (dB)" if metric == "psnr" else ""))
        axis.grid(True, alpha=0.3)
        axis.legend(title="Predicted channel")
    figure.suptitle(f"{metric.upper()} vs inference t - all models")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary(results, output_path):
    fields = [
        "model", "bt_channel", "t", "actin_psnr", "actin_ssim",
        "mito_psnr", "mito_ssim", "primary_channel",
        "primary_psnr", "primary_ssim",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for model_name, points in results.items():
            primary_channel = MODELS[model_name]
            for t_value, metrics in points:
                primary = metrics[primary_channel]
                writer.writerow({
                    "model": model_name,
                    "bt_channel": primary_channel,
                    "t": f"{t_value:.2f}",
                    "actin_psnr": metrics["actin"]["psnr"],
                    "actin_ssim": metrics["actin"]["ssim"],
                    "mito_psnr": metrics["mito"]["psnr"],
                    "mito_ssim": metrics["mito"]["ssim"],
                    "primary_channel": primary_channel,
                    "primary_psnr": primary["psnr"],
                    "primary_ssim": primary["ssim"],
                })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("evaluation_bt_results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_bt_results/plots"),
    )
    args = parser.parse_args()

    results = load_sweep(args.sweep_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for channel in CHANNELS:
        for metric in METRICS:
            plot_channel_metric(
                results,
                channel,
                metric,
                args.output_dir / f"{channel}_{metric}_vs_t.png",
            )
    for metric in METRICS:
        plot_all_models_metric(results, metric, args.output_dir / f"{metric}_vs_t.png")
    write_summary(results, args.output_dir / "sweep_metrics.csv")

    for model_name, points in results.items():
        target_channel = MODELS[model_name]
        best_psnr = max(points, key=lambda item: item[1][target_channel]["psnr"])
        best_ssim = max(points, key=lambda item: item[1][target_channel]["ssim"])
        print(
            f"{model_name} ({target_channel}): "
            f"best PSNR t={best_psnr[0]:.2f} "
            f"({best_psnr[1][target_channel]['psnr']:.4f}), "
            f"best SSIM t={best_ssim[0]:.2f} "
            f"({best_ssim[1][target_channel]['ssim']:.4f})"
        )
    print(f"Saved plots and CSV under {args.output_dir}")


if __name__ == "__main__":
    main()
