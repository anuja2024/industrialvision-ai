from __future__ import annotations

import argparse
import logging
import tarfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from tqdm import tqdm

from app.data.config import ProjectConfig, load_config

logger = logging.getLogger(__name__)


class VisADownloader:
    def __init__(
        self,
        config: ProjectConfig,
    ) -> None:
        self.config = config
        self.dataset = config.dataset
        self.target_dir = self.dataset.raw_path
        self.categories = config.categories

        self.s3 = boto3.client(
            "s3",
            region_name="us-west-2",
            config=Config(signature_version=UNSIGNED),
        )

    def download(self) -> Path:
        self.target_dir.mkdir(parents=True, exist_ok=True)

        if self.dataset.tar_path.exists():
            logger.info("Dataset archive already exists: %s", self.dataset.tar_path)
            return self.dataset.tar_path

        logger.info(
            "Downloading s3://%s/%s",
            self.dataset.s3_bucket,
            self.dataset.s3_key,
        )

        try:
            metadata = self.s3.head_object(
                Bucket=self.dataset.s3_bucket,
                Key=self.dataset.s3_key,
            )
            total_bytes = metadata["ContentLength"]

            progress = tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc="Downloading VisA",
            )

            transferred = 0

            def callback(bytes_transferred: int) -> None:
                nonlocal transferred
                progress.update(bytes_transferred - transferred)
                transferred = bytes_transferred

            self.s3.download_file(
                self.dataset.s3_bucket,
                self.dataset.s3_key,
                str(self.dataset.tar_path),
                Callback=callback,
            )

            progress.close()

        except (BotoCoreError, ClientError) as exc:
            if self.dataset.tar_path.exists():
                self.dataset.tar_path.unlink()
            raise RuntimeError(f"VisA download failed: {exc}") from exc

        logger.info("Download complete: %s", self.dataset.tar_path)
        return self.dataset.tar_path

    def extract(self) -> Path:
        tar_path = self.dataset.tar_path

        if not tar_path.exists():
            raise FileNotFoundError(
                f"Dataset archive not found: {tar_path}"
            )

        logger.info("Extracting VisA to %s", self.target_dir)

        with tarfile.open(tar_path, "r:*") as archive:
            root = self.target_dir.resolve()

            for member in tqdm(
                archive.getmembers(),
                desc="Extracting VisA",
            ):
                destination = (self.target_dir / member.name).resolve()

                if root not in destination.parents and destination != root:
                    raise RuntimeError(
                        f"Unsafe archive member: {member.name}"
                    )

                archive.extract(member, self.target_dir)

        logger.info("Extraction complete.")
        return self.target_dir

    def download_and_extract(self) -> Path:
        self.download()
        return self.extract()

    def verify_extraction(self) -> bool:
        required = (
            Path("Data/Images/Normal"),
            Path("Data/Images/Anomaly"),
            Path("Data/Masks/Anomaly"),
        )

        failed = []

        for category in self.categories:
            category_dir = self.target_dir / category

            for relative_path in required:
                path = category_dir / relative_path
                if not path.is_dir():
                    failed.append(path)

        if failed:
            for path in failed:
                logger.error("Missing directory: %s", path)
            return False

        logger.info(
            "Dataset structure verified for %d categories.",
            len(self.categories),
        )
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VisA dataset management CLI."
    )

    parser.add_argument(
        "command",
        choices=("download", "extract", "prepare", "verify"),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = load_config(args.config)
    downloader = VisADownloader(config)

    if args.command == "download":
        downloader.download()
    elif args.command == "extract":
        downloader.extract()
    elif args.command == "prepare":
        downloader.download_and_extract()
    elif args.command == "verify":
        if not downloader.verify_extraction():
            raise SystemExit(1)


if __name__ == "__main__":
    main()