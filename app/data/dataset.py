from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class VisASample:
    object_name: str
    split: str
    label: str
    image_path: Path
    mask_path: Path | None


class VisADataset:
    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        categories: list[str],
    ) -> None:
        self.root = Path(root)
        self.split_file = Path(split_file)
        self.categories = categories

        if not self.split_file.exists():
            raise FileNotFoundError(
                f"VisA split file not found: {self.split_file}"
            )

        self.data = self._load_split()

    def _load_split(self) -> pd.DataFrame:
        df = pd.read_csv(self.split_file)

        required_columns = {"object", "split", "label", "image", "mask"}
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns in split file: {sorted(missing)}"
            )

        df = df[df["object"].isin(self.categories)].copy()

        if df.empty:
            raise ValueError(
                f"No samples found for categories: {self.categories}"
            )

        df["image_path"] = df["image"].apply(
            lambda x: self.root / str(x)
        )

        df["mask_path"] = df["mask"].apply(
            lambda x: None
            if pd.isna(x)
            else self.root / str(x)
        )

        return df.reset_index(drop=True)

    def samples(self, split: str | None = None) -> list[VisASample]:
        df = self.data

        if split is not None:
            if split not in {"train", "test"}:
                raise ValueError(
                    "split must be either 'train' or 'test'"
                )

            df = df[df["split"] == split]

        return [
            VisASample(
                object_name=row["object"],
                split=row["split"],
                label=row["label"],
                image_path=Path(row["image_path"]),
                mask_path=(
                    Path(row["mask_path"])
                    if row["mask_path"] is not None
                    else None
                ),
            )
            for _, row in df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.data)

    def summary(self) -> pd.DataFrame:
        return (
            self.data
            .groupby(["object", "split", "label"])
            .size()
            .reset_index(name="count")
        )

    def validate_paths(self) -> dict[str, int]:
        image_missing = 0
        mask_missing = 0

        for sample in self.samples():
            if not sample.image_path.exists():
                image_missing += 1

            if sample.mask_path is not None and not sample.mask_path.exists():
                mask_missing += 1

        return {
            "images_missing": image_missing,
            "masks_missing": mask_missing,
        }

    def load_image(self, sample: VisASample) -> Image.Image:
        if not sample.image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {sample.image_path}"
            )

        return Image.open(sample.image_path).convert("RGB")

    def load_mask(self, sample: VisASample) -> Image.Image | None:
        if sample.mask_path is None:
            return None

        if not sample.mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found: {sample.mask_path}"
            )

        return Image.open(sample.mask_path).convert("L")