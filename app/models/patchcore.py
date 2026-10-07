from __future__ import annotations

from pathlib import Path

import torch


class PatchCore:
    def __init__(
        self,
        sampling_ratio: float = 0.01,
        grid_size: int = 16,
        max_candidates: int = 50_000,
    ) -> None:

        if not 0 < sampling_ratio <= 1:
            raise ValueError(
                "sampling_ratio must be in (0, 1]."
            )

        if grid_size <= 0:
            raise ValueError(
                "grid_size must be positive."
            )

        if max_candidates <= 0:
            raise ValueError(
                "max_candidates must be positive."
            )

        self.sampling_ratio = float(
            sampling_ratio
        )

        self.grid_size = int(
            grid_size
        )

        self.max_candidates = int(
            max_candidates
        )

        self.memory_bank: torch.Tensor | None = None

    def _apply_roi(
        self,
        features: torch.Tensor,
        roi_masks: torch.Tensor,
    ) -> torch.Tensor:

        if features.ndim != 3:
            raise ValueError(
                "Expected features with shape "
                "[N, P, D], got "
                f"{tuple(features.shape)}."
            )

        if roi_masks.ndim != 3:
            raise ValueError(
                "Expected ROI masks with shape "
                "[N, H, W], got "
                f"{tuple(roi_masks.shape)}."
            )

        n_samples, n_patches, feature_dim = (
            features.shape
        )

        expected_patches = (
            self.grid_size
            * self.grid_size
        )

        if n_patches != expected_patches:
            raise ValueError(
                f"Expected {expected_patches} "
                f"patches for a "
                f"{self.grid_size}x"
                f"{self.grid_size} grid, "
                f"got {n_patches}."
            )

        if roi_masks.shape != (
            n_samples,
            self.grid_size,
            self.grid_size,
        ):
            raise ValueError(
                "ROI mask shape mismatch. "
                f"Expected "
                f"[{n_samples}, "
                f"{self.grid_size}, "
                f"{self.grid_size}], "
                f"got "
                f"{tuple(roi_masks.shape)}."
            )

        flat_masks = roi_masks.reshape(
            n_samples,
            n_patches,
        ).bool()

        selected_features = []

        for index in range(n_samples):

            mask = flat_masks[index]

            if not mask.any():
                mask = torch.ones_like(
                    mask,
                    dtype=torch.bool,
                )

            selected_features.append(
                features[index][mask]
            )

        if not selected_features:
            raise RuntimeError(
                "No ROI features were selected."
            )

        return torch.cat(
            selected_features,
            dim=0,
        ).contiguous()

    def _reduce_candidates(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        if features.shape[0] <= (
            self.max_candidates
        ):
            return features

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(42)

        indices = torch.randperm(
            features.shape[0],
            generator=generator,
        )[
            : self.max_candidates
        ]

        indices = indices.to(
            features.device
        )

        return features[indices]

    def _select_coreset(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        if features.ndim != 2:
            raise ValueError(
                "Expected features with shape "
                "[N, D], got "
                f"{tuple(features.shape)}."
            )

        n_features = features.shape[0]

        target_size = max(
            1,
            int(
                n_features
                * self.sampling_ratio
            ),
        )

        if target_size >= n_features:
            return features

        candidates = (
            self._reduce_candidates(
                features
            )
        )

        if target_size >= candidates.shape[0]:
            return candidates

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(42)

        initial_index = int(
            torch.randint(
                candidates.shape[0],
                (1,),
                generator=generator,
            ).item()
        )

        selected_indices = [
            initial_index
        ]

        selected = candidates[
            initial_index
        ].unsqueeze(0)

        min_distances = torch.cdist(
            candidates,
            selected,
        ).squeeze(1)

        for _ in range(
            1,
            target_size,
        ):

            next_index = int(
                torch.argmax(
                    min_distances
                ).item()
            )

            selected_indices.append(
                next_index
            )

            next_feature = candidates[
                next_index
            ].unsqueeze(0)

            distances = torch.cdist(
                candidates,
                next_feature,
            ).squeeze(1)

            min_distances = torch.minimum(
                min_distances,
                distances,
            )

        return candidates[
            torch.tensor(
                selected_indices,
                dtype=torch.long,
                device=candidates.device,
            )
        ]

    def fit(
        self,
        features: torch.Tensor,
        roi_masks: torch.Tensor | None = None,
    ) -> "PatchCore":

        features = features.float()

        if features.ndim == 2:

            memory_features = features

        elif features.ndim == 3:

            if roi_masks is None:
                raise ValueError(
                    "roi_masks are required "
                    "for [N, P, D] features."
                )

            memory_features = (
                self._apply_roi(
                    features,
                    roi_masks,
                )
            )

        else:
            raise ValueError(
                "Expected features with shape "
                "[N, D] or [N, P, D], got "
                f"{tuple(features.shape)}."
            )

        if memory_features.shape[0] == 0:
            raise RuntimeError(
                "No features available for "
                "PatchCore fitting."
            )

        memory_features = (
            memory_features.contiguous()
        )

        memory_bank = (
            self._select_coreset(
                memory_features
            )
        )

        self.memory_bank = (
            memory_bank.detach()
            .float()
            .cpu()
            .contiguous()
        )

        return self

    def predict(
        self,
        features: torch.Tensor,
        roi_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:

        if self.memory_bank is None:
            raise RuntimeError(
                "PatchCore must be fitted "
                "before prediction."
            )

        features = features.float()

        if features.ndim == 1:

            features = features.unsqueeze(0)

            single_vector = True

        elif features.ndim == 2:

            single_vector = False

        else:
            raise ValueError(
                "Expected features with shape "
                "[P, D] or [D], got "
                f"{tuple(features.shape)}."
            )

        expected_patches = (
            self.grid_size
            * self.grid_size
        )

        if roi_mask is not None:

            if single_vector:
                raise ValueError(
                    "roi_mask cannot be used "
                    "with a single feature vector."
                )

            if features.shape[0] != (
                expected_patches
            ):
                raise ValueError(
                    f"Expected "
                    f"{expected_patches} "
                    f"patch features, got "
                    f"{features.shape[0]}."
                )

            roi_mask = roi_mask.bool()

            if roi_mask.shape != (
                self.grid_size,
                self.grid_size,
            ):
                raise ValueError(
                    "Expected ROI mask shape "
                    f"({self.grid_size}, "
                    f"{self.grid_size}), "
                    f"got "
                    f"{tuple(roi_mask.shape)}."
                )

            selected_indices = (
                torch.where(
                    roi_mask.reshape(-1)
                )[0]
            )

            if selected_indices.numel() == 0:
                selected_indices = torch.arange(
                    expected_patches,
                    device=features.device,
                )

            selected_features = (
                features[
                    selected_indices
                ]
            )

        else:

            selected_indices = None
            selected_features = features

        memory_bank = (
            self.memory_bank.to(
                device=features.device,
                dtype=features.dtype,
            )
        )

        distances = torch.cdist(
            selected_features,
            memory_bank,
        )

        nearest_distances = (
            distances.min(dim=1).values
        )

        if single_vector:
            image_score = float(
                nearest_distances.max().item()
            )

            return (
                nearest_distances,
                image_score,
            )

        patch_scores = torch.zeros(
            expected_patches,
            dtype=nearest_distances.dtype,
            device=nearest_distances.device,
        )

        patch_scores[
            selected_indices
        ] = nearest_distances

        image_score = float(
            nearest_distances.max().item()
        )

        return (
            patch_scores.detach().cpu(),
            image_score,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:

        if self.memory_bank is None:
            raise RuntimeError(
                "Cannot save an unfitted "
                "PatchCore model."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "sampling_ratio": (
                self.sampling_ratio
            ),
            "grid_size": (
                self.grid_size
            ),
            "max_candidates": (
                self.max_candidates
            ),
            "memory_bank": (
                self.memory_bank
            ),
        }

        torch.save(
            checkpoint,
            path,
        )

    def load(
        self,
        path: str | Path,
    ) -> "PatchCore":

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"PatchCore checkpoint not found: "
                f"{path}"
            )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise ValueError(
                "Invalid PatchCore checkpoint. "
                "Expected a dictionary."
            )

        if "memory_bank" not in checkpoint:
            raise ValueError(
                "Invalid PatchCore checkpoint: "
                "'memory_bank' is missing."
            )

        memory_bank = checkpoint[
            "memory_bank"
        ]

        if not isinstance(
            memory_bank,
            torch.Tensor,
        ):
            raise ValueError(
                "Invalid memory bank. "
                "Expected a torch.Tensor."
            )

        if memory_bank.ndim != 2:
            raise ValueError(
                "Invalid memory bank shape: "
                f"{tuple(memory_bank.shape)}. "
                "Expected [N, D]."
            )

        self.sampling_ratio = float(
            checkpoint.get(
                "sampling_ratio",
                0.01,
            )
        )

        self.grid_size = int(
            checkpoint.get(
                "grid_size",
                16,
            )
        )

        self.max_candidates = int(
            checkpoint.get(
                "max_candidates",
                50_000,
            )
        )

        self.memory_bank = (
            memory_bank
            .float()
            .cpu()
            .contiguous()
        )

        expected_patches = (
            self.grid_size
            * self.grid_size
        )

        if self.grid_size <= 0:
            raise ValueError(
                f"Invalid grid size: "
                f"{self.grid_size}."
            )

        if self.memory_bank.shape[1] <= 0:
            raise ValueError(
                "Memory bank has zero "
                "feature dimensions."
            )

        return self