from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from app.data.dataset import VisADataset


CATEGORIES = ["pcb1", "pcb2", "pcb3", "pcb4"]


def normalize_map(scores: torch.Tensor) -> np.ndarray:
    score_map = scores.reshape(16, 16)

    score_map = torch.nn.functional.interpolate(
        score_map.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    score_map = score_map.numpy()

    minimum = score_map.min()
    maximum = score_map.max()

    if maximum > minimum:
        score_map = (score_map - minimum) / (maximum - minimum)

    return score_map


def plot_sample(
    ax,
    image_path: Path,
    patch_scores: torch.Tensor,
    mask,
    title: str,
    show_heatmap: bool = False,
    show_mask: bool = False,
) -> None:
    image = Image.open(image_path).convert("RGB")

    ax.imshow(image)

    if show_heatmap:
        heatmap = normalize_map(patch_scores)

        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            extent=(
                0,
                image.width,
                image.height,
                0,
            ),
        )

    if show_mask and mask is not None:
        mask = np.asarray(mask)

        ax.imshow(
            mask,
            cmap="gray",
            alpha=0.45,
            extent=(
                0,
                image.width,
                image.height,
                0,
            ),
        )

    ax.set_title(title)
    ax.axis("off")


def visualize_category(
    dataset: VisADataset,
    category: str,
    inference_dir: Path,
    output_dir: Path,
) -> None:
    samples = [
        sample
        for sample in dataset.samples("test")
        if sample.object_name == category
    ]

    results = torch.load(
        inference_dir / f"{category}_patchcore.pt",
        weights_only=True,
    )

    image_scores = results["image_scores"]
    patch_scores = results["patch_scores"]

    anomaly_indices = [
        i
        for i, sample in enumerate(samples)
        if sample.label == "anomaly"
    ]

    normal_indices = [
        i
        for i, sample in enumerate(samples)
        if sample.label == "normal"
    ]

    anomaly_index = max(
        anomaly_indices,
        key=lambda i: image_scores[i].item(),
    )

    normal_index = max(
        normal_indices,
        key=lambda i: image_scores[i].item(),
    )

    anomaly_sample = samples[anomaly_index]
    normal_sample = samples[normal_index]

    anomaly_mask = dataset.load_mask(anomaly_sample)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )

    plot_sample(
        axes[0, 0],
        anomaly_sample.image_path,
        patch_scores[anomaly_index],
        anomaly_mask,
        f"Anomaly | score={image_scores[anomaly_index]:.3f}",
    )

    plot_sample(
        axes[0, 1],
        anomaly_sample.image_path,
        patch_scores[anomaly_index],
        anomaly_mask,
        "PatchCore localization",
        show_heatmap=True,
    )

    plot_sample(
        axes[0, 2],
        anomaly_sample.image_path,
        patch_scores[anomaly_index],
        anomaly_mask,
        "Ground-truth mask",
        show_mask=True,
    )

    plot_sample(
        axes[1, 0],
        normal_sample.image_path,
        patch_scores[normal_index],
        None,
        f"Normal | score={image_scores[normal_index]:.3f}",
    )

    plot_sample(
        axes[1, 1],
        normal_sample.image_path,
        patch_scores[normal_index],
        None,
        "PatchCore localization",
        show_heatmap=True,
    )

    axes[1, 2].axis("off")
    axes[1, 2].set_title("No ground-truth defect")

    figure.suptitle(
        f"IndustrialVision AI — {category.upper()} PatchCore",
        fontsize=16,
    )

    figure.tight_layout()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"{category}_localization.png"

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")
    print(
        f"  anomaly: {anomaly_sample.image_path.name}"
    )
    print(
        f"  normal:  {normal_sample.image_path.name}"
    )


if __name__ == "__main__":
    dataset = VisADataset(
        root="data/raw",
        split_file="data/raw/split_csv/1cls.csv",
        categories=CATEGORIES,
    )

    inference_dir = Path("data/inference")
    output_dir = Path("data/inference/visualizations")

    for category in CATEGORIES:
        visualize_category(
            dataset,
            category,
            inference_dir,
            output_dir,
        )