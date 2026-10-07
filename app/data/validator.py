from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    message: str


class CategoryValidation(BaseModel):
    category: str
    checks: list[ValidationCheck]


class ValidationReport(BaseModel):
    timestamp: str
    dataset_path: str
    categories: list[CategoryValidation]
    summary: dict[str, int | str]


class VisAValidator:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        dataset_root: Path,
        categories: list[str],
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.categories = categories

    def validate(self) -> ValidationReport:
        results = [
            self._validate_category(category)
            for category in self.categories
        ]

        total = sum(len(result.checks) for result in results)
        passed = sum(
            check.passed
            for result in results
            for check in result.checks
        )

        return ValidationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_path=str(self.dataset_root),
            categories=results,
            summary={
                "total_checks": total,
                "passed": passed,
                "failed": total - passed,
            },
        )

    def _validate_category(
        self,
        category: str,
    ) -> CategoryValidation:
        category_dir = self.dataset_root / category

        checks = [
            self._check_directory_structure(category_dir),
        ]

        if category_dir.is_dir():
            checks.extend(
                [
                    self._check_images(
                        category_dir / "Data/Images/Normal",
                        "normal_images",
                    ),
                    self._check_images(
                        category_dir / "Data/Images/Anomaly",
                        "anomaly_images",
                    ),
                    self._check_images(
                        category_dir / "Data/Masks/Anomaly",
                        "anomaly_masks",
                    ),
                    self._check_alignment(category_dir),
                ]
            )

        return CategoryValidation(
            category=category,
            checks=checks,
        )

    def _check_directory_structure(
        self,
        category_dir: Path,
    ) -> ValidationCheck:
        required = (
            "Data/Images/Normal",
            "Data/Images/Anomaly",
            "Data/Masks/Anomaly",
        )

        missing = [
            relative
            for relative in required
            if not (category_dir / relative).is_dir()
        ]

        return ValidationCheck(
            name="directory_structure",
            passed=not missing,
            message=(
                "All required directories exist."
                if not missing
                else f"Missing: {missing}"
            ),
        )

    def _check_images(
        self,
        directory: Path,
        name: str,
    ) -> ValidationCheck:
        if not directory.is_dir():
            return ValidationCheck(
                name=name,
                passed=False,
                message=f"Directory not found: {directory}",
            )

        files = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.image_extensions
        ]

        if not files:
            return ValidationCheck(
                name=name,
                passed=False,
                message=f"No image files found: {directory}",
            )

        invalid = []

        for path in files:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                invalid.append(f"{path.name}: {exc}")

        return ValidationCheck(
            name=name,
            passed=not invalid,
            message=(
                f"{len(files)} files verified."
                if not invalid
                else f"{len(invalid)} invalid files."
            ),
        )

    def _check_alignment(
        self,
        category_dir: Path,
    ) -> ValidationCheck:
        image_dir = category_dir / "Data/Images/Anomaly"
        mask_dir = category_dir / "Data/Masks/Anomaly"

        images = {
            path.stem
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.image_extensions
        }

        masks = {
            path.stem
            for path in mask_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.image_extensions
        }

        missing = images - masks

        return ValidationCheck(
            name="image_mask_alignment",
            passed=not missing,
            message=(
                f"{len(images)} anomaly images have matching masks."
                if not missing
                else f"{len(missing)} anomaly images are missing masks."
            ),
        )

    def print_report(
        self,
        report: ValidationReport,
    ) -> None:
        console = Console()

        table = Table(title="VisA Dataset Validation")

        table.add_column("Category")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Message")

        for category in report.categories:
            for check in category.checks:
                table.add_row(
                    category.category,
                    check.name,
                    "PASS" if check.passed else "FAIL",
                    check.message,
                )

        console.print(table)

        summary = report.summary
        console.print(
            f"\n{summary['passed']}/{summary['total_checks']} checks passed."
        )

    def save_report(
        self,
        report: ValidationReport,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Validation report saved to %s", output_path)