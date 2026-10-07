from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("data/raw")
SPLIT_FILE = ROOT / "split_csv" / "1cls.csv"
CATEGORIES = ["pcb1", "pcb2", "pcb3", "pcb4"]


def main() -> None:
    df = pd.read_csv(SPLIT_FILE)

    df = df[df["object"].isin(CATEGORIES)].copy()

    print("\n=== DATASET AUDIT ===\n")

    print(f"Total samples: {len(df)}")

    print("\nSamples by PCB:")
    print(
        df.groupby("object")
        .size()
        .to_string()
    )

    print("\nSamples by split and label:")
    print(
        df.groupby(["object", "split", "label"])
        .size()
        .to_string()
    )

    print("\nSplit totals:")
    print(
        df.groupby(["split", "label"])
        .size()
        .to_string()
    )

    print("\nImage dimensions:")

    dimensions = []

    for _, row in df.iterrows():
        image_path = ROOT / row["image"]

        from PIL import Image

        with Image.open(image_path) as image:
            dimensions.append(
                {
                    "object": row["object"],
                    "split": row["split"],
                    "label": row["label"],
                    "width": image.width,
                    "height": image.height,
                }
            )

    dimension_df = pd.DataFrame(dimensions)

    print(
        dimension_df
        .groupby(["object", "width", "height"])
        .size()
        .reset_index(name="count")
        .to_string(index=False)
    )

    print("\nAnomaly mask coverage:")

    anomaly_df = df[df["label"] == "anomaly"]

    mask_exists = []

    for _, row in anomaly_df.iterrows():
        mask_path = ROOT / row["mask"]
        mask_exists.append(mask_path.exists())

    print(f"Anomaly samples: {len(anomaly_df)}")
    print(f"Valid masks: {sum(mask_exists)}")
    print(f"Missing masks: {len(mask_exists) - sum(mask_exists)}")


if __name__ == "__main__":
    main()