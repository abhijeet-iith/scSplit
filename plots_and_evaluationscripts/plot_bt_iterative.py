import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = {
    "actin_bt_010": "actin",
    "actin_bt_020": "actin",
    "actin_bt_040": "actin",
    "mito_bt_010": "mito",
    "mito_bt_020": "mito",
    "mito_bt_040": "mito",
}
STEPS = (1, 5, 10, 20)
METRICS = ("psnr", "ssim")


def load_results(results_root):
    results = {}
    for model_name, primary_channel in MODELS.items():
        metrics_by_steps = {}
        model_root = results_root / model_name
        for step_count in STEPS:
            metric_paths = list(model_root.glob(
                f"t_*/t_*_iterative_{step_count}step"
                if step_count == 1
                else f"t_*/t_*_iterative_{step_count}steps"
            ))
            if len(metric_paths) != 1:
                raise FileNotFoundError(
                    f"Expected one result directory for {model_name}, "
                    f"{step_count} steps under {model_root}; found {metric_paths}"
                )
            metrics_path = metric_paths[0] / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
            metrics_by_steps[step_count] = json.loads(metrics_path.read_text())
        results[model_name] = {
            "primary_channel": primary_channel,
            "metrics": metrics_by_steps,
        }
    return results


def plot_channel_metric(results, channel, metric, output_path):
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for model_name, result in results.items():
        if result["primary_channel"] != channel:
            continue
        values = [
            result["metrics"][step][channel][metric]
            for step in STEPS
        ]
        axis.plot(STEPS, values, marker="o", linewidth=2, label=model_name)
    axis.set_title(f"{channel.capitalize()} {metric.upper()} vs number of steps")
    axis.set_xlabel("Number of iterative steps")
    axis.set_ylabel(metric.upper() + (" (dB)" if metric == "psnr" else ""))
    axis.set_xticks(STEPS)
    axis.grid(True, alpha=0.3)
    axis.legend(title="BT model")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_all_models_metric(results, metric, output_path):
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for axis, (model_name, result) in zip(axes.flat, results.items()):
        for channel in ("actin", "mito"):
            values = [
                result["metrics"][step][channel][metric]
                for step in STEPS
            ]
            axis.plot(STEPS, values, marker="o", linewidth=2, label=channel)
        axis.set_title(model_name)
        axis.set_xlabel("Number of iterative steps")
        axis.set_ylabel(metric.upper() + (" (dB)" if metric == "psnr" else ""))
        axis.set_xticks(STEPS)
        axis.grid(True, alpha=0.3)
        axis.legend(title="Target channel")
    figure.suptitle(f"{metric.upper()} vs number of steps - all six models")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_model_metric(model_name, result, metric, output_path):
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for channel in ("actin", "mito"):
        values = [
            result["metrics"][step][channel][metric]
            for step in STEPS
        ]
        axis.plot(STEPS, values, marker="o", linewidth=2, label=channel)
    axis.set_title(f"{model_name}: {metric.upper()} vs number of steps")
    axis.set_xlabel("Number of iterative steps")
    axis.set_ylabel(metric.upper() + (" (dB)" if metric == "psnr" else ""))
    axis.set_xticks(STEPS)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Target channel")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary(results, output_path):
    fields = [
        "model", "primary_channel", "num_steps",
        "actin_psnr", "actin_ssim", "mito_psnr", "mito_ssim",
        "primary_psnr", "primary_ssim",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for model_name, result in results.items():
            primary_channel = result["primary_channel"]
            for step in STEPS:
                metrics = result["metrics"][step]
                primary = metrics[primary_channel]
                writer.writerow({
                    "model": model_name,
                    "primary_channel": primary_channel,
                    "num_steps": step,
                    "actin_psnr": metrics["actin"]["psnr"],
                    "actin_ssim": metrics["actin"]["ssim"],
                    "mito_psnr": metrics["mito"]["psnr"],
                    "mito_ssim": metrics["mito"]["ssim"],
                    "primary_psnr": primary["psnr"],
                    "primary_ssim": primary["ssim"],
                })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("evaluation_bt_results_xt_b"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_bt_results_xt_b/plots"),
    )
    args = parser.parse_args()

    results = load_results(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for channel in ("actin", "mito"):
        for metric in METRICS:
            plot_channel_metric(
                results,
                channel,
                metric,
                args.output_dir / f"{channel}_{metric}_vs_steps.png",
            )
    for metric in METRICS:
        plot_all_models_metric(
            results,
            metric,
            args.output_dir / f"{metric}_vs_steps_all_models.png",
        )
    per_model_dir = args.output_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    for model_name, result in results.items():
        for metric in METRICS:
            plot_model_metric(
                model_name,
                result,
                metric,
                per_model_dir / f"{model_name}_{metric}_vs_steps.png",
            )
    write_summary(results, args.output_dir / "iterative_metrics.csv")
    print(f"Saved six aggregate plots, twelve per-model plots, and CSV under {args.output_dir}")


if __name__ == "__main__":
    main()
