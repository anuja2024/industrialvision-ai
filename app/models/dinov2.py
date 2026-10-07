from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from app.data.dataset import VisADataset


MODEL_NAME = "facebook/dinov2-small"

IMAGE_SIZE = 448
PATCH_SIZE = 14
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FEATURE_DIR = Path("data/features")
CATEGORIES = ["pcb1", "pcb2", "pcb3", "pcb4"]


class DINOv2FeatureExtractor:

    def __init__(
        self,
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
    ):
        self.image_size = image_size
        self.patch_size = PATCH_SIZE
        self.grid_size = image_size // PATCH_SIZE

        self.processor = AutoImageProcessor.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        ).to(DEVICE)

        self.model.eval()

    def preprocess_image(self, image):
        image = image.convert("RGB")

        image.thumbnail(
            (self.image_size, self.image_size),
            Image.Resampling.BICUBIC,
        )

        image = ImageOps.pad(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.BICUBIC,
            color=(0, 0, 0),
            centering=(0.5, 0.5),
        )

        return image

    def create_roi_mask(
        self,
        image,
        grid_size=None,
    ):
        if grid_size is None:
            grid_size = self.grid_size

        image_np = np.asarray(
            image.convert("RGB")
        )

        height, width = image_np.shape[:2]

        scale = min(
            320 / max(height, width),
            1.0,
        )

        small_width = max(
            1,
            int(width * scale),
        )

        small_height = max(
            1,
            int(height * scale),
        )

        small = cv2.resize(
            image_np,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_RGB2GRAY,
        )

        mask = np.zeros(
            gray.shape,
            dtype=np.uint8,
        )

        border = max(
            1,
            int(min(gray.shape) * 0.02),
        )

        rect = (
            border,
            border,
            max(1, gray.shape[1] - 2 * border),
            max(1, gray.shape[0] - 2 * border),
        )

        try:
            cv2.grabCut(
                small,
                mask,
                rect,
                np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64),
                3,
                cv2.GC_INIT_WITH_RECT,
            )

            roi = np.where(
                (mask == cv2.GC_FGD)
                | (mask == cv2.GC_PR_FGD),
                1,
                0,
            ).astype(np.uint8)

            kernel = np.ones(
                (5, 5),
                np.uint8,
            )

            roi = cv2.morphologyEx(
                roi,
                cv2.MORPH_CLOSE,
                kernel,
            )

            roi = cv2.morphologyEx(
                roi,
                cv2.MORPH_OPEN,
                kernel,
            )

            contours, _ = cv2.findContours(
                roi,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if contours:
                largest = max(
                    contours,
                    key=cv2.contourArea,
                )

                x, y, w, h = cv2.boundingRect(
                    largest
                )

                expand_x = int(w * 0.08)
                expand_y = int(h * 0.08)

                x1 = max(0, x - expand_x)
                y1 = max(0, y - expand_y)

                x2 = min(
                    roi.shape[1],
                    x + w + expand_x,
                )

                y2 = min(
                    roi.shape[0],
                    y + h + expand_y,
                )

                roi[:] = 0
                roi[
                    y1:y2,
                    x1:x2,
                ] = 1

            else:
                roi[:] = 1

        except cv2.error:
            roi = np.ones(
                gray.shape,
                dtype=np.uint8,
            )

        roi = cv2.resize(
            roi,
            (grid_size, grid_size),
            interpolation=cv2.INTER_NEAREST,
        )

        return roi.astype(bool)

    @torch.no_grad()
    def extract_batch(
        self,
        images,
    ):
        processed_images = [
            self.preprocess_image(image)
            for image in images
        ]

        inputs = self.processor(
            images=processed_images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )

        pixel_values = inputs["pixel_values"].to(
            DEVICE
        )

        outputs = self.model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
        )

        patch_tokens = outputs.last_hidden_state[
            :,
            1:,
            :,
        ]

        expected_patches = (
            self.grid_size * self.grid_size
        )

        if patch_tokens.shape[1] != expected_patches:
            raise ValueError(
                f"Expected {expected_patches} patch tokens "
                f"for {self.image_size}x{self.image_size}, "
                f"got {patch_tokens.shape[1]}."
            )

        roi_masks = np.stack(
            [
                self.create_roi_mask(
                    image,
                    self.grid_size,
                )
                for image in images
            ]
        )

        return (
            patch_tokens.cpu(),
            torch.from_numpy(roi_masks),
        )

    def extract_dataset(
        self,
        dataset,
        split,
        category,
        batch_size=8,
    ):
        samples = [
            sample
            for sample in dataset.samples(split)
            if sample.object_name == category
        ]

        all_features = []
        all_roi_masks = []

        for start in tqdm(
            range(
                0,
                len(samples),
                batch_size,
            ),
            desc=f"DINOv2 {category} {split}",
        ):
            batch_samples = samples[
                start:start + batch_size
            ]

            images = [
                dataset.load_image(sample)
                for sample in batch_samples
            ]

            features, roi_masks = self.extract_batch(
                images
            )

            all_features.append(features)
            all_roi_masks.append(roi_masks)

        features = torch.cat(
            all_features,
            dim=0,
        )

        roi_masks = torch.cat(
            all_roi_masks,
            dim=0,
        )

        FEATURE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        feature_path = (
            FEATURE_DIR
            / f"{category}_{split}_dinov2_448.pt"
        )

        roi_path = (
            FEATURE_DIR
            / f"{category}_{split}_roi_32.pt"
        )

        torch.save(
            features,
            feature_path,
        )

        torch.save(
            roi_masks,
            roi_path,
        )

        print(
            f"{category} {split}: "
            f"features={features.shape}, "
            f"roi={roi_masks.shape}"
        )

        return features, roi_masks


def main():
    dataset = VisADataset(
        root="data/raw",
        split_file="data/raw/split_csv/1cls.csv",
        categories=CATEGORIES,
    )

    extractor = DINOv2FeatureExtractor()

    print()
    print("=" * 60)
    print("DINOv2 448x448 / 32x32 feature extraction")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Patch tokens: {GRID_SIZE * GRID_SIZE}")

    for category in CATEGORIES:
        extractor.extract_dataset(
            dataset=dataset,
            split="train",
            category=category,
        )

        extractor.extract_dataset(
            dataset=dataset,
            split="test",
            category=category,
        )


if __name__ == "__main__":
    main()