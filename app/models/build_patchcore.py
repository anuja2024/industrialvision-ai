from pathlib import Path

import torch

from app.models.patchcore import PatchCore


CATEGORIES = ["pcb1", "pcb2", "pcb3", "pcb4"]

FEATURE_DIR = Path("data/features")

SAMPLING_RATIO = 0.01
GRID_SIZE = 32


def build_memory_bank(category):
    feature_path = (
        FEATURE_DIR
        / f"{category}_train_dinov2_448.pt"
    )

    roi_path = (
        FEATURE_DIR
        / f"{category}_train_roi_32.pt"
    )

    output_path = (
        FEATURE_DIR
        / f"{category}_memory_bank_32.pt"
    )

    features = torch.load(
        feature_path,
        weights_only=True,
    )

    roi_masks = torch.load(
        roi_path,
        weights_only=True,
    )

    if features.ndim != 3:
        raise ValueError(
            f"{category}: expected features "
            f"[N, 1024, 384], got {features.shape}"
        )

    if features.shape[1] != GRID_SIZE * GRID_SIZE:
        raise ValueError(
            f"{category}: expected "
            f"{GRID_SIZE * GRID_SIZE} patches, "
            f"got {features.shape[1]}"
        )

    if roi_masks.shape != (
        len(features),
        GRID_SIZE,
        GRID_SIZE,
    ):
        raise ValueError(
            f"{category}: expected ROI shape "
            f"({len(features)}, {GRID_SIZE}, "
            f"{GRID_SIZE}), "
            f"got {tuple(roi_masks.shape)}"
        )

    model = PatchCore(
        sampling_ratio=SAMPLING_RATIO,
        grid_size=GRID_SIZE,
    )

    model.fit(
        features,
        roi_masks,
    )

    model.save(output_path)

    print(
        f"{category}: "
        f"features={tuple(features.shape)}, "
        f"memory_bank={tuple(model.memory_bank.shape)}, "
        f"saved={output_path}"
    )


def main():
    print()
    print("=" * 60)
    print("PatchCore 32x32 memory-bank construction")
    print("=" * 60)

    for category in CATEGORIES:
        build_memory_bank(category)


if __name__ == "__main__":
    main()