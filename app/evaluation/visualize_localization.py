from pathlib import Path

import numpy as np
import torch

from app.data.dataset import VisADataset
from app.data.visualization import visualize_sample


ROOT = Path("data/raw")
INFERENCE_ROOT = Path("data/inference")
OUTPUT_ROOT = Path("data/inference/localization_review")

CATEGORIES = ["pcb1", "pcb2", "pcb3", "pcb4"]


def main():
    dataset = VisADataset(
        root=ROOT,
        split_file=ROOT / "split_csv" / "1cls.csv",
        categories=CATEGORIES,
    )

    test_samples = dataset.samples("test")

    for category in CATEGORIES:
        print()
        print("=" * 60)
        print(f"Visualizing {category.upper()}")
        print("=" * 60)

        category_samples = [
            sample
            for sample in test_samples
            if sample.object_name == category
        ]

        anomaly_samples = [
            sample
            for sample in category_samples
            if sample.label == "anomaly"
        ][:2]

        if len(anomaly_samples) < 2:
            print(
                f"Warning: only {len(anomaly_samples)} "
                f"anomaly samples found for {category}."
            )

        result_path = INFERENCE_ROOT / f"{category}_patchcore.pt"

        if not result_path.exists():
            raise FileNotFoundError(
                f"PatchCore result not found:\n{result_path}"
            )

        results = torch.load(
            result_path,
            weights_only=True,
        )

        if "patch_scores" not in results:
            raise KeyError(
                f"'patch_scores' not found in {result_path}"
            )

        if "anomaly_regions" not in results:
            raise KeyError(
                f"'anomaly_regions' not found in {result_path}"
            )

        patch_scores = results["patch_scores"]
        regions = results["anomaly_regions"]

        if isinstance(patch_scores, torch.Tensor):
            patch_scores = patch_scores.cpu().numpy()

        if isinstance(regions, torch.Tensor):
            regions = regions.cpu().numpy()

        if len(category_samples) != len(patch_scores):
            raise ValueError(
                f"{category}: sample count mismatch. "
                f"Dataset={len(category_samples)}, "
                f"PatchCore={len(patch_scores)}"
            )

        if len(category_samples) != len(regions):
            raise ValueError(
                f"{category}: region count mismatch. "
                f"Dataset={len(category_samples)}, "
                f"Regions={len(regions)}"
            )

        category_indices = {
            sample.image_path: index
            for index, sample in enumerate(category_samples)
        }

        for sample_number, sample in enumerate(
            anomaly_samples,
            start=1,
        ):
            if sample.image_path not in category_indices:
                raise KeyError(
                    f"Could not find sample in PatchCore results:\n"
                    f"{sample.image_path}"
                )

            index = category_indices[sample.image_path]

            image = dataset.load_image(sample)
            mask = dataset.load_mask(sample)

            if mask is None:
                raise ValueError(
                    f"No ground-truth mask found for:\n"
                    f"{sample.image_path}"
                )

            output_path = (
                OUTPUT_ROOT
                / category
                / f"{sample_number}_{Path(sample.image_path).stem}.png"
            )

            visualize_sample(
                image=image,
                ground_truth_mask=mask,
                patch_scores=patch_scores[index],
                predicted_region=regions[index],
                title=(
                    f"{category.upper()} — "
                    f"Anomaly {sample_number} — "
                    f"{Path(sample.image_path).name}"
                ),
                output_path=output_path,
            )

            print(f"Saved: {output_path}")

    print()
    print("=" * 60)
    print("Localization visualization complete.")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()