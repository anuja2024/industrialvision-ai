from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from app.data.dataset import VisADataset


CATEGORIES = ["pcb1", "pcb2", "pcb3", "pcb4"]

SPLIT_FILE = Path("data/raw/split_csv/1cls.csv")
DATASET_ROOT = Path("data/raw")
INFERENCE_DIR = Path("data/inference")
OUTPUT_DIR = Path("data/visualizations")


def overlay_heatmap(image, patch_scores, alpha=0.45):
    image_np = np.asarray(image.convert("RGB"))
    h, w = image_np.shape[:2]

    heatmap = np.asarray(patch_scores, dtype=np.float32).reshape(16, 16)
    heatmap = cv2.resize(
        heatmap,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap = heatmap - heatmap.min()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    colored = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB,
    )

    overlay = (
        image_np.astype(np.float32) * (1 - alpha)
        + colored.astype(np.float32) * alpha
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)


def draw_box(image, region, linewidth=3):
    output = image.copy()

    x1 = int(region["x1"])
    y1 = int(region["y1"])
    x2 = int(region["x2"])
    y2 = int(region["y2"])

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        linewidth,
    )

    return output


def visualize_sample(
    image,
    ground_truth_mask,
    patch_scores,
    predicted_region,
    title,
    output_path,
):
    image_np = np.asarray(image.convert("RGB"))
    gt = np.asarray(ground_truth_mask)

    heatmap_overlay = overlay_heatmap(
        image,
        patch_scores,
    )

    prediction = draw_box(
        image_np,
        predicted_region,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5),
    )

    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(
        gt,
        cmap="gray",
    )
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(heatmap_overlay)
    axes[2].set_title("PatchCore Heatmap")
    axes[2].axis("off")

    axes[3].imshow(prediction)
    axes[3].set_title("Predicted Region")
    axes[3].axis("off")

    fig.suptitle(title)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    dataset = VisADataset(
        root=DATASET_ROOT,
        split_file=SPLIT_FILE,
        categories=CATEGORIES,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for category in CATEGORIES:
        print()
        print("=" * 60)
        print(f"Visualization: {category}")
        print("=" * 60)

        inference_path = (
            INFERENCE_DIR
            / f"{category}_patchcore.pt"
        )

        results = torch.load(
            inference_path,
            map_location="cpu",
            weights_only=False,
        )

        samples = [
            sample
            for sample in dataset.samples("test")
            if sample.object_name == category
        ]

        patch_scores = results["patch_scores"]
        regions = results["anomaly_regions"]

        if len(samples) != len(patch_scores):
            raise ValueError(
                f"{category}: "
                f"{len(samples)} test samples but "
                f"{len(patch_scores)} patch-score rows."
            )

        anomaly_indices = [
            index
            for index, sample in enumerate(samples)
            if sample.label == "anomaly"
        ]

        if len(anomaly_indices) < 2:
            raise ValueError(
                f"{category}: fewer than 2 anomaly samples found."
            )

        for selected, index in enumerate(anomaly_indices[:2]):
            sample = samples[index]

            image = dataset.load_image(sample)
            mask = dataset.load_mask(sample)

            output_path = (
                OUTPUT_DIR
                / category
                / f"anomaly_{selected + 1}.png"
            )

            visualize_sample(
                image=image,
                ground_truth_mask=mask,
                patch_scores=patch_scores[index].numpy(),
                predicted_region=regions[index],
                title=(
                    f"{category.upper()} — "
                    f"Anomaly {selected + 1}"
                ),
                output_path=output_path,
            )

            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()