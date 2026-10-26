import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile



def display_range(*images):
    values = np.concatenate([image.astype(np.float32).ravel() for image in images])
    low, high = np.percentile(values, [1, 99])
    return float(low), float(high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index to render; omit to render every frame.",
    )
    args = parser.parse_args()

    input_stack = tifffile.imread(args.input)
    results_dir = Path(args.results)
    predicted_actin = tifffile.imread(results_dir / "predicted_actin.tif")
    predicted_mito = tifffile.imread(results_dir / "predicted_mito.tif")
    target_actin = tifffile.imread(results_dir / "target_actin.tif")
    target_mito = tifffile.imread(results_dir / "target_mito.tif")

    frame_indices = range(len(input_stack)) if args.frame is None else [args.frame]
    if args.frame is not None and not 0 <= args.frame < len(input_stack):
        raise ValueError(f"frame must be between 0 and {len(input_stack) - 1}")

    output_path = Path(args.output)
    if args.frame is None:
        output_path.mkdir(parents=True, exist_ok=True)

    for frame in frame_indices:
        input_image = input_stack[frame]
        images = {
            "Input BT": input_image,
            "Target actin": target_actin[frame],
            "Predicted actin": predicted_actin[frame],
            "Target mito": target_mito[frame],
            "Predicted mito": predicted_mito[frame],
        }
        # Use the same reference range for every panel. Predictions are excluded
        # because iterative artifacts can otherwise darken all clean images.
        display_low, display_high = display_range(
            images["Input BT"],
            images["Target actin"],
            images["Target mito"],
        )

        figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        panels = [
            (axes[0, 0], "Input BT", images["Input BT"]),
            (axes[0, 1], "Clean actin target", images["Target actin"]),
            (axes[0, 2], "Predicted actin", images["Predicted actin"]),
            (axes[1, 0], "Input BT", images["Input BT"]),
            (axes[1, 1], "Clean mito target", images["Target mito"]),
            (axes[1, 2], "Predicted mito", images["Predicted mito"]),
        ]
        for axis, title, image in panels:
            axis.imshow(image, cmap="gray", vmin=display_low, vmax=display_high)
            axis.set_title(title)
            axis.axis("off")

        figure.suptitle(f"Frame {frame}")
        frame_output = (
            output_path / f"frame_{frame:03d}.png"
            if args.frame is None
            else output_path
        )
        figure.savefig(frame_output, dpi=150)
        plt.close(figure)
        print(f"Saved {frame_output}")


if __name__ == "__main__":
    main()
