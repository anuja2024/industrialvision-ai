from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from app.data.dataset import VisADataset


CATEGORIES = [
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
]

FEATURE_DIR = Path("data/features")
INFERENCE_DIR = Path("data/inference")

PATCH_GRID_SIZE = 32

P95 = 95.0
P99 = 99.0


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


def load_inference(
    category: str,
):
    path = (
        INFERENCE_DIR
        / f"{category}_patchcore_32.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing inference file: {path}"
        )

    return torch.load(
        path,
        weights_only=True,
    )


def validate_inference(
    category: str,
    inference: dict,
    sample_count: int,
):
    image_scores = inference[
        "image_scores"
    ]

    patch_scores = inference[
        "patch_scores"
    ]

    regions = inference[
        "anomaly_regions"
    ]

    expected_patches = (
        PATCH_GRID_SIZE
        * PATCH_GRID_SIZE
    )

    if image_scores.shape != (
        sample_count,
    ):
        raise ValueError(
            f"{category}: expected image "
            f"scores shape "
            f"({sample_count},), got "
            f"{tuple(image_scores.shape)}."
        )

    if patch_scores.shape != (
        sample_count,
        expected_patches,
    ):
        raise ValueError(
            f"{category}: expected patch "
            f"scores shape "
            f"({sample_count}, "
            f"{expected_patches}), got "
            f"{tuple(patch_scores.shape)}."
        )

    if len(regions) != sample_count:
        raise ValueError(
            f"{category}: expected "
            f"{sample_count} region lists, "
            f"got {len(regions)}."
        )


def get_labels(samples):
    return np.array(
        [
            1 if sample.label == "anomaly"
            else 0
            for sample in samples
        ],
        dtype=np.int64,
    )


def load_sample_mask(
    sample,
):
    if sample.label != "anomaly":

        with Image.open(
            sample.image_path
        ) as image:

            width, height = image.size

        return np.zeros(
            (height, width),
            dtype=bool,
        )

    if sample.mask_path is None:
        raise ValueError(
            "Anomaly sample has no mask path: "
            f"{sample.image_path}"
        )

    with Image.open(
        sample.mask_path
    ) as mask_image:

        mask = np.asarray(
            mask_image.convert("L"),
            dtype=np.uint8,
        )

    return mask > 0


def get_training_normal_scores(
    category: str,
    dataset: VisADataset,
):
    samples = load_samples(
        dataset,
        category,
        "train",
    )

    samples = [
        sample
        for sample in samples
        if sample.label == "normal"
    ]

    feature_path = (
        FEATURE_DIR
        / f"{category}_train_dinov2_448.pt"
    )

    roi_path = (
        FEATURE_DIR
        / f"{category}_train_roi_32.pt"
    )

    memory_bank_path = (
        FEATURE_DIR
        / f"{category}_memory_bank_32.pt"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing training features: "
            f"{feature_path}"
        )

    if not roi_path.exists():
        raise FileNotFoundError(
            f"Missing training ROI masks: "
            f"{roi_path}"
        )

    if not memory_bank_path.exists():
        raise FileNotFoundError(
            f"Missing memory bank: "
            f"{memory_bank_path}"
        )

    features = torch.load(
        feature_path,
        weights_only=True,
    )

    roi_masks = torch.load(
        roi_path,
        weights_only=True,
    )

    checkpoint = torch.load(
        memory_bank_path,
        weights_only=True,
    )

    if "memory_bank" not in checkpoint:
        raise ValueError(
            f"{category}: memory bank "
            f"checkpoint does not contain "
            f"'memory_bank'."
        )

    memory_bank = checkpoint[
        "memory_bank"
    ].float()

    expected_patches = (
        PATCH_GRID_SIZE
        * PATCH_GRID_SIZE
    )

    if features.ndim != 3:
        raise ValueError(
            f"{category}: expected training "
            f"features [N, {expected_patches}, D], "
            f"got {tuple(features.shape)}."
        )

    if features.shape[1] != (
        expected_patches
    ):
        raise ValueError(
            f"{category}: expected "
            f"{expected_patches} training "
            f"patches, got "
            f"{features.shape[1]}."
        )

    if roi_masks.shape != (
        len(features),
        PATCH_GRID_SIZE,
        PATCH_GRID_SIZE,
    ):
        raise ValueError(
            f"{category}: ROI shape mismatch. "
            f"Got {tuple(roi_masks.shape)}."
        )

    scores = []

    for index in range(
        len(samples)
    ):

        feature = features[
            index
        ].float()

        roi_mask = (
            roi_masks[
                index
            ]
            .bool()
            .reshape(-1)
        )

        if not roi_mask.any():
            roi_mask = torch.ones_like(
                roi_mask,
                dtype=torch.bool,
            )

        selected_features = (
            feature[roi_mask]
        )

        distances = torch.cdist(
            selected_features,
            memory_bank,
        )

        nearest_distances = (
            distances.min(
                dim=1
            ).values
        )

        scores.append(
            float(
                nearest_distances.max()
            )
        )

    return np.asarray(
        scores,
        dtype=np.float32,
    )


def get_thresholds(
    training_scores: np.ndarray,
):
    if len(training_scores) == 0:
        raise ValueError(
            "Cannot calculate thresholds "
            "from empty training scores."
        )

    return {
        "p95": float(
            np.percentile(
                training_scores,
                P95,
            )
        ),
        "p99": float(
            np.percentile(
                training_scores,
                P99,
            )
        ),
    }


def classify_decision(
    score: float,
    thresholds: dict[str, float],
) -> str:

    if score < thresholds["p95"]:
        return "PASS"

    if score < thresholds["p99"]:
        return "REVIEW"

    return "HOLD"


def summarize_decisions(
    decisions,
):
    return {
        "PASS": decisions.count(
            "PASS"
        ),
        "REVIEW": decisions.count(
            "REVIEW"
        ),
        "HOLD": decisions.count(
            "HOLD"
        ),
        "total": len(decisions),
    }


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
):
    predictions = (
        scores >= threshold
    ).astype(np.int64)

    tp = int(
        np.sum(
            (predictions == 1)
            & (labels == 1)
        )
    )

    tn = int(
        np.sum(
            (predictions == 0)
            & (labels == 0)
        )
    )

    fp = int(
        np.sum(
            (predictions == 1)
            & (labels == 0)
        )
    )

    fn = int(
        np.sum(
            (predictions == 0)
            & (labels == 1)
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(labels)
        if len(labels) > 0
        else 0.0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def mask_to_bbox(
    mask: np.ndarray,
):
    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def bbox_iou(
    predicted,
    ground_truth,
):
    if (
        predicted is None
        or ground_truth is None
    ):
        return 0.0

    px1, py1, px2, py2 = predicted
    gx1, gy1, gx2, gy2 = ground_truth

    ix1 = max(
        px1,
        gx1,
    )

    iy1 = max(
        py1,
        gy1,
    )

    ix2 = min(
        px2,
        gx2,
    )

    iy2 = min(
        py2,
        gy2,
    )

    intersection = (
        max(0, ix2 - ix1)
        * max(0, iy2 - iy1)
    )

    predicted_area = (
        max(0, px2 - px1)
        * max(0, py2 - py1)
    )

    ground_truth_area = (
        max(0, gx2 - gx1)
        * max(0, gy2 - gy1)
    )

    union = (
        predicted_area
        + ground_truth_area
        - intersection
    )

    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


def best_region_iou(
    regions,
    ground_truth,
):
    if (
        not regions
        or ground_truth is None
    ):
        return 0.0

    best_iou = 0.0

    for region in regions:

        predicted = (
            int(region["x1"]),
            int(region["y1"]),
            int(region["x2"]),
            int(region["y2"]),
        )

        iou = bbox_iou(
            predicted,
            ground_truth,
        )

        best_iou = max(
            best_iou,
            iou,
        )

    return best_iou


def localization_iou(
    samples,
    inference,
):
    values = []

    regions_all = inference[
        "anomaly_regions"
    ]

    for index, sample in enumerate(
        samples
    ):

        if sample.label != "anomaly":
            continue

        mask = load_sample_mask(
            sample
        )

        ground_truth = mask_to_bbox(
            mask
        )

        if ground_truth is None:
            continue

        regions = regions_all[
            index
        ]

        if isinstance(
            regions,
            dict,
        ):
            regions = [
                regions
            ]

        iou = best_region_iou(
            regions,
            ground_truth,
        )

        values.append(
            iou
        )

    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "iou50": 0.0,
        }

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    return {
        "mean": float(
            values.mean()
        ),
        "median": float(
            np.median(values)
        ),
        "iou50": float(
            np.mean(
                values >= 0.50
            )
        ),
    }


def pixel_auroc(
    samples,
    patch_scores,
):
    all_predictions = []
    all_targets = []

    expected_patches = (
        PATCH_GRID_SIZE
        * PATCH_GRID_SIZE
    )

    if patch_scores.shape[1] != (
        expected_patches
    ):
        raise ValueError(
            f"Expected "
            f"{expected_patches} patch "
            f"scores, got "
            f"{patch_scores.shape[1]}."
        )

    for index, sample in enumerate(
        samples
    ):

        mask = load_sample_mask(
            sample
        )

        height, width = (
            mask.shape
        )

        score_grid = torch.tensor(
            patch_scores[index],
            dtype=torch.float32,
        ).reshape(
            1,
            1,
            PATCH_GRID_SIZE,
            PATCH_GRID_SIZE,
        )

        score_map = F.interpolate(
            score_grid,
            size=(
                height,
                width,
            ),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        all_predictions.append(
            score_map.reshape(-1)
        )

        all_targets.append(
            mask.astype(
                np.uint8
            ).reshape(-1)
        )

    predictions = np.concatenate(
        all_predictions
    )

    targets = np.concatenate(
        all_targets
    )

    if np.unique(
        targets
    ).size < 2:
        raise ValueError(
            "Pixel AUROC requires both "
            "normal and anomalous pixels."
        )

    return float(
        roc_auc_score(
            targets,
            predictions,
        )
    )


def evaluate_category(
    dataset: VisADataset,
    category: str,
):

    samples = load_samples(
        dataset,
        category,
        "test",
    )

    if not samples:
        raise RuntimeError(
            f"{category}: no test samples."
        )

    inference = load_inference(
        category
    )

    validate_inference(
        category,
        inference,
        len(samples),
    )

    labels = get_labels(
        samples
    )

    image_scores = (
        inference[
            "image_scores"
        ]
        .cpu()
        .numpy()
    )

    patch_scores = (
        inference[
            "patch_scores"
        ]
        .cpu()
        .numpy()
    )

    training_scores = (
        get_training_normal_scores(
            category,
            dataset,
        )
    )

    thresholds = get_thresholds(
        training_scores
    )

    decisions = [
        classify_decision(
            float(score),
            thresholds,
        )
        for score in image_scores
    ]

    decision_summary = (
        summarize_decisions(
            decisions
        )
    )

    image_auc = roc_auc_score(
        labels,
        image_scores,
    )

    pixel_auc = pixel_auroc(
        samples,
        patch_scores,
    )

    binary = binary_metrics(
        labels,
        image_scores,
        thresholds["p99"],
    )

    localization = (
        localization_iou(
            samples,
            inference,
        )
    )

    return {
        "thresholds": thresholds,
        "decisions": decision_summary,
        "image_auc": image_auc,
        "pixel_auc": pixel_auc,
        "binary": binary,
        "localization": localization,
    }


def print_results(
    category: str,
    results,
):

    print()
    print("=" * 70)

    print(
        f"{category.upper()} — "
        f"PatchCore 32×32 Evaluation"
    )

    print("=" * 70)

    thresholds = results[
        "thresholds"
    ]

    print(
        f"P95 threshold: "
        f"{thresholds['p95']:.6f}"
    )

    print(
        f"P99 threshold: "
        f"{thresholds['p99']:.6f}"
    )

    decisions = results[
        "decisions"
    ]

    print()
    print(
        "Decision distribution:"
    )

    print(
        f"PASS:   "
        f"{decisions['PASS']}"
    )

    print(
        f"REVIEW: "
        f"{decisions['REVIEW']}"
    )

    print(
        f"HOLD:   "
        f"{decisions['HOLD']}"
    )

    print()
    print(
        f"Image AUROC: "
        f"{results['image_auc']:.4f}"
    )

    print(
        f"Pixel AUROC: "
        f"{results['pixel_auc']:.4f}"
    )

    binary = results[
        "binary"
    ]

    print()
    print(
        "P99 binary classification:"
    )

    print(
        f"Precision: "
        f"{binary['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{binary['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{binary['f1']:.4f}"
    )

    print(
        f"Accuracy:  "
        f"{binary['accuracy']:.4f}"
    )

    print(
        f"TP={binary['TP']} "
        f"TN={binary['TN']} "
        f"FP={binary['FP']} "
        f"FN={binary['FN']}"
    )

    localization = results[
        "localization"
    ]

    print()
    print(
        "Multi-region localization:"
    )

    print(
        f"Mean best-region IoU:   "
        f"{localization['mean']:.4f}"
    )

    print(
        f"Median best-region IoU: "
        f"{localization['median']:.4f}"
    )

    print(
        f"IoU@0.50:              "
        f"{localization['iou50']:.4f}"
    )


def main():

    dataset = VisADataset(
        root="data/raw",
        split_file=(
            "data/raw/split_csv/1cls.csv"
        ),
        categories=CATEGORIES,
    )

    results_all = {}

    for category in CATEGORIES:

        results = evaluate_category(
            dataset,
            category,
        )

        results_all[
            category
        ] = results

        print_results(
            category,
            results,
        )

    image_aucs = [
        result["image_auc"]
        for result in (
            results_all.values()
        )
    ]

    pixel_aucs = [
        result["pixel_auc"]
        for result in (
            results_all.values()
        )
    ]

    localization_means = [
        result[
            "localization"
        ]["mean"]
        for result in (
            results_all.values()
        )
    ]

    print()
    print("=" * 70)
    print(
        "OVERALL — PatchCore 32×32"
    )
    print("=" * 70)

    print(
        f"Mean Image AUROC: "
        f"{np.mean(image_aucs):.4f}"
    )

    print(
        f"Mean Pixel AUROC: "
        f"{np.mean(pixel_aucs):.4f}"
    )

    print(
        f"Mean Best-Region IoU: "
        f"{np.mean(localization_means):.4f}"
    )


if __name__ == "__main__":
    main()