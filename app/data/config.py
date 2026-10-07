from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Project root not found.")


PROJECT_ROOT = _find_project_root()


class DatasetConfig(BaseModel):
    name: str = "visa"
    version: str = "20220922"
    source_url: str = (
        "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/"
        "VisA_20220922.tar"
    )
    s3_bucket: str = "amazon-visual-anomaly"
    s3_key: str = "VisA_20220922.tar"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    tar_filename: str = "VisA_20220922.tar"
    split_file: str = "split_csv/1cls.csv"

    @property
    def raw_path(self) -> Path:
        return PROJECT_ROOT / self.raw_dir

    @property
    def processed_path(self) -> Path:
        return PROJECT_ROOT / self.processed_dir

    @property
    def tar_path(self) -> Path:
        return self.raw_path / self.tar_filename

    @property
    def split_path(self) -> Path:
        return self.raw_path / self.split_file


class ProjectConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    categories: list[str] = Field(
        default_factory=lambda: [
            "candle",
            "capsules",
            "cashew",
            "chewinggum",
            "fryum",
            "macaroni1",
            "macaroni2",
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4",
            "pipe_fryum",
        ]
    )


def load_config(path: Path | None = None) -> ProjectConfig:
    config_path = path or PROJECT_ROOT / "configs" / "dataset.yaml"

    if not config_path.exists():
        return ProjectConfig()

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    return ProjectConfig.model_validate(data)