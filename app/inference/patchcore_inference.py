from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from app.data.dataset import VisADataset
from app.models.patchcore import PatchCore


CATEGORIES = [
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
]

FEATURE_DIR = Path("data/features")
INFERENCE_DIR = Path("data/inference")

PATCH_GRID_SIZE = 32

TOP_PATCH_PERCENTILE = 95.0
MIN_COMPONENT_AREA = 2


def load_samples(
    dataset: VisADataset,
    category: str,
    split: str,
):
    return [
        sample
        for sample in dataset.samples(split)
        if sample.object_name == category
    ]


def load_memory_bank(
    category: str,
) -> PatchCore:

    model = PatchCore(
        grid_size=PATCH_GRID_SIZE,
    )

    model.load(
        FEATURE_DIR
        / f"{category}_memory_bank_32.pt"
    )

    if model.grid_size != PATCH_GRID_SIZE:
        raise ValueError(
            f"{category}: expected PatchCore "
            f"grid size {PATCH_GRID_SIZE}, "
            f"got {model.grid_size}."
        )

    return model


def get_anomaly_regions(
    patch_scores: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[dict[str, float | int]]:

    scores = np.asarray(
        patch_scores,
        dtype=np.float32,
    ).reshape(-1)

    expected_patches = (
        PATCH_GRID_SIZE
        * PATCH_GRID_SIZE
    )

    if scores.size != expected_patches:
        raise ValueError(
            f"Expected {expected_patches} "
            f"patch scores, got "
            f"{scores.size}."
        )

    score_grid = scores.reshape(
        PATCH_GRID_SIZE,
        PATCH_GRID_SIZE,
    )

    minimum = float(
        score_grid.min()
    )

    maximum = float(
        score_grid.max()
    )

    if maximum > minimum:

        normalized = (
            score_grid - minimum
        ) / (
            maximum - minimum
        )

    else:

        normalized = np.zeros_like(
            score_grid,
            dtype=np.float32,
        )

    threshold = np.percentile(
        normalized,
        TOP_PATCH_PERCENTILE,
    )

    binary = (
        normalized >= threshold
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    components = []

    for component_id in range(
        1,
        count,
    ):

        x = int(
            stats[
                component_id,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                component_id,
                cv2.CC_STAT_TOP,
            ]
        )

        width = int(
            stats[
                component_id,
                cv2.CC_STAT_WIDTH,
            ]
        )

        height = int(
            stats[
                component_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < MIN_COMPONENT_AREA:
            continue

        component_mask = (
            labels == component_id
        )

        mean_score = float(
            normalized[
                component_mask
            ].mean()
        )

        max_score = float(
            normalized[
                component_mask
            ].max()
        )

        component_score = (
            mean_score
            * np.sqrt(
                max(area, 1)
            )
        )

        components.append(
            {
                "grid_x": x,
                "grid_y": y,
                "grid_width": width,
                "grid_height": height,
                "active_patches": area,
                "mean_score": mean_score,
                "max_score": max_score,
                "component_score": component_score,
            }
        )

    components.sort(
        key=lambda item: item[
            "component_score"
        ],
        reverse=True,
    )

    regions = []

    scale_x = (
        image_width
        / PATCH_GRID_SIZE
    )

    scale_y = (
        image_height
        / PATCH_GRID_SIZE
    )

    for component in components:

        x1 = component[
            "grid_x"
        ]

        y1 = component[
            "grid_y"
        ]

        x2 = (
            x1
            + component[
                "grid_width"
            ]
        )

        y2 = (
            y1
            + component[
                "grid_height"
            ]
        )

        pixel_x1 = int(
            round(
                x1 * scale_x
            )
        )

        pixel_y1 = int(
            round(
                y1 * scale_y
            )
        )

        pixel_x2 = int(
            round(
                x2 * scale_x
            )
        )

        pixel_y2 = int(
            round(
                y2 * scale_y
            )
        )

        pixel_x1 = max(
            0,
            min(
                pixel_x1,
                image_width - 1,
            ),
        )

        pixel_y1 = max(
            0,
            min(
                pixel_y1,
                image_height - 1,
            ),
        )

        pixel_x2 = max(
            pixel_x1 + 1,
            min(
                pixel_x2,
                image_width,
            ),
        )

        pixel_y2 = max(
            pixel_y1 + 1,
            min(
                pixel_y2,
                image_height,
            ),
        )

        regions.append(
            {
                "x1": pixel_x1,
                "y1": pixel_y1,
                "x2": pixel_x2,
                "y2": pixel_y2,
                "width": (
                    pixel_x2
                    - pixel_x1
                ),
                "height": (
                    pixel_y2
                    - pixel_y1
                ),
                "score": float(
                    component[
                        "mean_score"
                    ]
                ),
                "max_score": float(
                    component[
                        "max_score"
                    ]
                ),
                "active_patches": int(
                    component[
                        "active_patches"
                    ]
                ),
            }
        )

    if not regions:

        flat_index = int(
            np.argmax(score_grid)
        )

        grid_y, grid_x = (
            np.unravel_index(
                flat_index,
                score_grid.shape,
            )
        )

        pixel_x1 = int(
            round(
                grid_x
                * scale_x
            )
        )

        pixel_y1 = int(
            round(
                grid_y
                * scale_y
            )
        )

        pixel_x2 = int(
            round(
                (grid_x + 1)
                * scale_x
            )
        )

        pixel_y2 = int(
            round(
                (grid_y + 1)
                * scale_y
            )
        )

        regions.append(
            {
                "x1": pixel_x1,
                "y1": pixel_y1,
                "x2": pixel_x2,
                "y2": pixel_y2,
                "width": (
                    pixel_x2
                    - pixel_x1
                ),
                "height": (
                    pixel_y2
                    - pixel_y1
                ),
                "score": float(
                    score_grid[
                        grid_y,
                        grid_x,
                    ]
                ),
                "max_score": float(
                    score_grid[
                        grid_y,
                        grid_x,
                    ]
                ),
                "active_patches": 1,
            }
        )

    return regions


def run_category(
    dataset: VisADataset,
    category: str,
) -> None:

    samples = load_samples(
        dataset,
        category,
        "test",
    )

    if not samples:
        raise RuntimeError(
            f"{category}: no test samples found."
        )

    feature_path = (
        FEATURE_DIR
        / f"{category}_test_dinov2_448.pt"
    )

    roi_path = (
        FEATURE_DIR
        / f"{category}_test_roi_32.pt"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing features: "
            f"{feature_path}"
        )

    if not roi_path.exists():
        raise FileNotFoundError(
            f"Missing ROI masks: "
            f"{roi_path}"
        )

    features = torch.load(
        feature_path,
        weights_only=True,
    )

    roi_masks = torch.load(
        roi_path,
        weights_only=True,
    )

    expected_patches = (
        PATCH_GRID_SIZE
        * PATCH_GRID_SIZE
    )

    if features.ndim != 3:
        raise ValueError(
            f"{category}: expected feature "
            f"tensor [N, {expected_patches}, D], "
            f"got {tuple(features.shape)}."
        )

    if features.shape[1] != (
        expected_patches
    ):
        raise ValueError(
            f"{category}: expected "
            f"{expected_patches} patches, "
            f"got {features.shape[1]}."
        )

    if roi_masks.shape != (
        len(features),
        PATCH_GRID_SIZE,
        PATCH_GRID_SIZE,
    ):
        raise ValueError(
            f"{category}: expected ROI "
            f"shape "
            f"[N, {PATCH_GRID_SIZE}, "
            f"{PATCH_GRID_SIZE}], "
            f"got {tuple(roi_masks.shape)}."
        )

    if len(samples) != len(features):
        raise ValueError(
            f"{category}: sample count "
            f"{len(samples)} does not match "
            f"feature count {len(features)}."
        )

    model = load_memory_bank(
        category
    )

    image_scores = []
    patch_scores_all = []
    anomaly_regions = []

    print()
    print("=" * 60)
    print(
        f"PatchCore 32x32 ROI inference: "
        f"{category}"
    )
    print("=" * 60)

    for index in tqdm(
        range(len(samples)),
        desc=f"PatchCore {category}",
    ):

        feature = features[
            index
        ]

        roi_mask = roi_masks[
            index
        ]

        patch_scores, image_score = (
            model.predict(
                feature,
                roi_mask,
            )
        )

        patch_scores = np.asarray(
            patch_scores,
            dtype=np.float32,
        ).reshape(-1)

        if patch_scores.size != (
            expected_patches
        ):
            raise ValueError(
                f"{category} sample "
                f"{index}: expected "
                f"{expected_patches} "
                f"patch scores, got "
                f"{patch_scores.size}."
            )

        image_score = float(
            image_score
        )

        sample = samples[
            index
        ]

        with Image.open(
            sample.image_path
        ) as image:

            image_width, image_height = (
                image.size
            )

        regions = get_anomaly_regions(
            patch_scores,
            image_width,
            image_height,
        )

        image_scores.append(
            image_score
        )

        patch_scores_all.append(
            patch_scores
        )

        anomaly_regions.append(
            regions
        )

    image_scores_tensor = torch.tensor(
        image_scores,
        dtype=torch.float32,
    )

    patch_scores_tensor = torch.tensor(
        np.stack(
            patch_scores_all
        ),
        dtype=torch.float32,
    )

    output = {
        "image_scores": image_scores_tensor,
        "patch_scores": patch_scores_tensor,
        "anomaly_regions": anomaly_regions,
    }

    INFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        INFERENCE_DIR
        / f"{category}_patchcore_32.pt"
    )

    torch.save(
        output,
        output_path,
    )

    region_counts = [
        len(regions)
        for regions in anomaly_regions
    ]

    print()
    print(
        f"Saved: {output_path}"
    )

    print(
        f"Image scores: "
        f"{image_scores_tensor.shape}"
    )

    print(
        f"Patch scores: "
        f"{patch_scores_tensor.shape}"
    )

    print(
        f"Regions: "
        f"{len(anomaly_regions)}"
    )

    print(
        f"Average regions/image: "
        f"{np.mean(region_counts):.2f}"
    )

    print(
        f"Maximum regions/image: "
        f"{max(region_counts)}"
    )


def main() -> None:

    dataset = VisADataset(
        root="data/raw",
        split_file=(
            "data/raw/split_csv/1cls.csv"
        ),
        categories=CATEGORIES,
    )

    for category in CATEGORIES:

        run_category(
            dataset,
            category,
        )


if __name__ == "__main__":
    main()