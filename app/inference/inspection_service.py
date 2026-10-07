from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

from app.models.patchcore import PatchCore


MODEL_NAME = "facebook/dinov2-small"

FEATURE_DIR = Path("data/features")

PATCH_GRID_SIZE = 32
IMAGE_SIZE = 448

CATEGORIES = ["pcb1", "pcb2", "pcb3", "pcb4"]

THRESHOLDS = {
    "pcb1": {
        "review": 29.493319,
        "hold": 33.092995,
    },
    "pcb2": {
        "review": 26.761963,
        "hold": 29.274633,
    },
    "pcb3": {
        "review": 27.339272,
        "hold": 31.370409,
    },
    "pcb4": {
        "review": 28.061361,
        "hold": 30.672636,
    },
}


class InspectionService:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )

        self.dinov2 = AutoModel.from_pretrained(
            MODEL_NAME
        ).to(self.device)

        self.dinov2.eval()

        self.patchcore_models: dict[str, PatchCore] = {}

        self._load_patchcore_models()

    def _load_patchcore_models(self) -> None:
        for category in CATEGORIES:
            model_path = (
                FEATURE_DIR
                / f"{category}_memory_bank_32.pt"
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing PatchCore memory bank: {model_path}"
                )

            model = PatchCore(
                sampling_ratio=0.01,
                grid_size=PATCH_GRID_SIZE,
            )

            model.load(model_path)

            if model.grid_size != PATCH_GRID_SIZE:
                raise ValueError(
                    f"{category}: expected grid size "
                    f"{PATCH_GRID_SIZE}, got {model.grid_size}"
                )

            self.patchcore_models[category] = model

    @staticmethod
    def _prepare_image(
        image: Image.Image,
    ) -> Image.Image:
        image = image.convert("RGB")

        width, height = image.size

        scale = min(
            IMAGE_SIZE / width,
            IMAGE_SIZE / height,
        )

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BICUBIC,
        )

        image = ImageOps.pad(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            method=Image.Resampling.BICUBIC,
            color=(0, 0, 0),
            centering=(0.5, 0.5),
        )

        return image

    @staticmethod
    def _build_roi_mask(
        image: Image.Image,
    ) -> np.ndarray:
        original = np.asarray(
            image.convert("RGB")
        )

        height, width = original.shape[:2]

        scale = min(
            320 / width,
            320 / height,
            1.0,
        )

        small_width = max(1, round(width * scale))
        small_height = max(1, round(height * scale))

        small = cv2.resize(
            original,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_RGB2GRAY,
        )

        foreground = cv2.GC_PR_FGD
        probable_foreground = cv2.GC_PR_FGD
        probable_background = cv2.GC_PR_BGD

        mask = np.full(
            gray.shape,
            probable_foreground,
            dtype=np.uint8,
        )

        border = max(
            2,
            int(min(small_width, small_height) * 0.04),
        )

        mask[:border, :] = probable_background
        mask[-border:, :] = probable_background
        mask[:, :border] = probable_background
        mask[:, -border:] = probable_background

        rect = (
            border,
            border,
            max(1, small_width - 2 * border),
            max(1, small_height - 2 * border),
        )

        try:
            background_model = np.zeros(
                (1, 65),
                dtype=np.float64,
            )

            foreground_model = np.zeros(
                (1, 65),
                dtype=np.float64,
            )

            cv2.grabCut(
                small,
                mask,
                rect,
                background_model,
                foreground_model,
                3,
                cv2.GC_INIT_WITH_RECT,
            )

            foreground_mask = np.where(
                (mask == cv2.GC_FGD)
                | (mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)

            kernel = np.ones(
                (5, 5),
                dtype=np.uint8,
            )

            foreground_mask = cv2.morphologyEx(
                foreground_mask,
                cv2.MORPH_CLOSE,
                kernel,
            )

            foreground_mask = cv2.morphologyEx(
                foreground_mask,
                cv2.MORPH_OPEN,
                kernel,
            )

            contours, _ = cv2.findContours(
                foreground_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:
                raise RuntimeError(
                    "No foreground contour detected."
                )

            largest = max(
                contours,
                key=cv2.contourArea,
            )

            x, y, w, h = cv2.boundingRect(
                largest
            )

            if w <= 0 or h <= 0:
                raise RuntimeError(
                    "Invalid ROI bounding box."
                )

            expansion = 0.08

            x1 = max(
                0,
                int(x - w * expansion),
            )

            y1 = max(
                0,
                int(y - h * expansion),
            )

            x2 = min(
                small_width,
                int(x + w * (1 + expansion)),
            )

            y2 = min(
                small_height,
                int(y + h * (1 + expansion)),
            )

            roi_small = np.zeros(
                (small_height, small_width),
                dtype=np.uint8,
            )

            roi_small[
                y1:y2,
                x1:x2,
            ] = 1

            roi_full = cv2.resize(
                roi_small,
                (PATCH_GRID_SIZE, PATCH_GRID_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )

            return roi_full.astype(bool)

        except Exception:
            return np.ones(
                (
                    PATCH_GRID_SIZE,
                    PATCH_GRID_SIZE,
                ),
                dtype=bool,
            )

    def _extract_features(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, np.ndarray]:
        processed_image = self._prepare_image(
            image
        )

        roi_mask = self._build_roi_mask(
            image
        )

        inputs = self.processor(
            images=[processed_image],
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.dinov2(
                **inputs
            )

        patch_tokens = outputs.last_hidden_state[
            :,
            1:,
            :,
        ]

        expected_patches = (
            PATCH_GRID_SIZE
            * PATCH_GRID_SIZE
        )

        if patch_tokens.shape[1] != expected_patches:
            raise RuntimeError(
                "Unexpected DINOv2 patch count: "
                f"{patch_tokens.shape[1]}. "
                f"Expected {expected_patches}."
            )

        return (
            patch_tokens[0].float(),
            roi_mask,
        )

    @staticmethod
    def _normalize_heatmap(
        patch_scores: np.ndarray,
    ) -> np.ndarray:
        scores = np.asarray(
            patch_scores,
            dtype=np.float32,
        ).reshape(
            PATCH_GRID_SIZE,
            PATCH_GRID_SIZE,
        )

        minimum = float(scores.min())
        maximum = float(scores.max())

        if maximum > minimum:
            normalized = (
                (scores - minimum)
                / (maximum - minimum)
            )
        else:
            normalized = np.zeros_like(
                scores,
                dtype=np.float32,
            )

        return normalized

    @staticmethod
    def _extract_regions(
        patch_scores: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> list[dict]:
        score_grid = InspectionService._normalize_heatmap(
            patch_scores
        )

        threshold = np.percentile(
            score_grid,
            95.0,
        )

        binary = (
            score_grid >= threshold
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

            if area < 2:
                continue

            component_mask = (
                labels == component_id
            )

            mean_score = float(
                score_grid[
                    component_mask
                ].mean()
            )

            max_score = float(
                score_grid[
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
            x1 = component["grid_x"]
            y1 = component["grid_y"]

            x2 = (
                x1
                + component["grid_width"]
            )

            y2 = (
                y1
                + component["grid_height"]
            )

            pixel_x1 = int(
                round(x1 * scale_x)
            )

            pixel_y1 = int(
                round(y1 * scale_y)
            )

            pixel_x2 = int(
                round(x2 * scale_x)
            )

            pixel_y2 = int(
                round(y2 * scale_y)
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
                    "width": pixel_x2 - pixel_x1,
                    "height": pixel_y2 - pixel_y1,
                    "score": component[
                        "mean_score"
                    ],
                    "max_score": component[
                        "max_score"
                    ],
                    "active_patches": component[
                        "active_patches"
                    ],
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
                    "width": max(
                        1,
                        pixel_x2 - pixel_x1,
                    ),
                    "height": max(
                        1,
                        pixel_y2 - pixel_y1,
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

    @staticmethod
    def _decision(
        category: str,
        anomaly_score: float,
    ) -> str:
        thresholds = THRESHOLDS[
            category
        ]

        if anomaly_score >= thresholds["hold"]:
            return "HOLD"

        if anomaly_score >= thresholds["review"]:
            return "REVIEW"

        return "PASS"

    @staticmethod
    def _confidence(
        category: str,
        anomaly_score: float,
    ) -> float:
        thresholds = THRESHOLDS[
            category
        ]

        review = thresholds["review"]
        hold = thresholds["hold"]

        if anomaly_score >= hold:
            distance = (
                anomaly_score - hold
            )

            confidence = (
                0.80
                + min(
                    distance / max(
                        hold,
                        1e-6,
                    ),
                    1.0,
                )
                * 0.19
            )

        elif anomaly_score >= review:
            confidence = 0.60

        else:
            distance = (
                review - anomaly_score
            )

            confidence = (
                0.50
                + min(
                    distance / max(
                        review,
                        1e-6,
                    ),
                    1.0,
                )
                * 0.40
            )

        return round(
            float(
                min(
                    max(
                        confidence,
                        0.0,
                    ),
                    0.99,
                )
            ),
            3,
        )

    @staticmethod
    def _heatmap_image(
        patch_scores: np.ndarray,
    ) -> str:
        import base64

        normalized = (
            InspectionService._normalize_heatmap(
                patch_scores
            )
        )

        heatmap = (
            normalized * 255
        ).astype(np.uint8)

        heatmap = cv2.resize(
            heatmap,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET,
        )

        success, encoded = cv2.imencode(
            ".png",
            heatmap,
        )

        if not success:
            raise RuntimeError(
                "Failed to encode heatmap."
            )

        return base64.b64encode(
            encoded.tobytes()
        ).decode("utf-8")

    def inspect(
        self,
        image_bytes: bytes,
        category: str,
    ) -> dict:
        category = category.lower().strip()

        if category not in CATEGORIES:
            raise ValueError(
                f"Unsupported category: {category}"
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image_width, image_height = (
            image.size
        )

        features, roi_mask = (
            self._extract_features(
                image
            )
        )

        patchcore = (
            self.patchcore_models[
                category
            ]
        )

        patch_scores, image_score = (
            patchcore.predict(
                features,
                torch.from_numpy(
                    roi_mask
                ),
            )
        )

        patch_scores = np.asarray(
            patch_scores,
            dtype=np.float32,
        ).reshape(-1)

        anomaly_score = float(
            image_score
        )

        regions = self._extract_regions(
            patch_scores,
            image_width,
            image_height,
        )

        decision = self._decision(
            category,
            anomaly_score,
        )

        confidence = self._confidence(
            category,
            anomaly_score,
        )

        return {
            "category": category,
            "image": {
                "width": image_width,
                "height": image_height,
            },
            "anomaly_score": round(
                anomaly_score,
                4,
            ),
            "decision": decision,
            "confidence": confidence,
            "regions": regions,
            "heatmap": (
                self._heatmap_image(
                    patch_scores
                )
            ),
            "model": {
                "feature_extractor": MODEL_NAME,
                "image_size": IMAGE_SIZE,
                "patch_grid": (
                    f"{PATCH_GRID_SIZE}x"
                    f"{PATCH_GRID_SIZE}"
                ),
                "memory_bank_size": int(
                    patchcore.memory_bank.shape[0]
                ),
            },
        }


_service: InspectionService | None = None


def get_inspection_service() -> InspectionService:
    global _service

    if _service is None:
        _service = InspectionService()

    return _service