"""Concept Vector Data Structures and Serialization.

This module defines the core `ConceptVector` data structure used across VLM-Steer.
In Representation Engineering (RepE) and activation addition frameworks, a concept direction
is represented as a 1D tensor in the latent activation space of a transformer layer (residual stream).

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

import torch


@dataclass
class ConceptVector:
    """Represents an isolated semantic direction in a model's latent representation space.

    Attributes:
        name: Unique identifier for the concept (e.g., 'demographic_neutrality', 'safety_refusal').
        layer_idx: Zero-indexed integer indicating the target residual transformer layer.
        vector: 1D torch.Tensor holding the directional weights in activation space.
        description: Human-readable explanation of what this concept represents and mitigates.
        metadata: Arbitrary calibration metrics (separation margin, raw norm, method used, timestamps).
    """

    name: str
    layer_idx: int
    vector: torch.Tensor
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and sanitize vector types and dimensions upon instantiation."""
        # Convert raw lists or numpy arrays to float32 PyTorch tensors if necessary
        if not isinstance(self.vector, torch.Tensor):
            self.vector = torch.tensor(self.vector, dtype=torch.float32)

        # Flatten multi-dimensional tensors into a strict 1D direction vector
        if self.vector.ndim > 1:
            self.vector = self.vector.squeeze()

    @property
    def dimension(self) -> int:
        """Return the hidden dimension size (e.g., 768, 1024, 2048) of the vector."""
        return self.vector.shape[-1]

    def normalize(self) -> ConceptVector:
        """Return a unit-normalized (L2 norm = 1.0) copy of this concept vector.

        Unit normalization ensures that steering intensity is controlled strictly
        by the steering multiplier (alpha) rather than the arbitrary magnitude
        of raw residual stream activations.

        Returns:
            A new ConceptVector instance with L2 norm equal to 1.0.
        """
        norm = torch.norm(self.vector, p=2)
        if norm > 0:
            normed_vec = self.vector / norm
        else:
            normed_vec = self.vector.clone()

        return ConceptVector(
            name=self.name,
            layer_idx=self.layer_idx,
            vector=normed_vec,
            description=self.description,
            metadata={**self.metadata, "normalized": True, "original_norm": float(norm)},
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize vector weights and metadata to disk via PyTorch file format (.pt).

        Args:
            filepath: Target file destination path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "name": self.name,
            "layer_idx": self.layer_idx,
            "description": self.description,
            "metadata": self.metadata,
            # Detach and move to CPU to guarantee portable cross-device loading
            "vector": self.vector.detach().cpu(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, filepath: Union[str, Path], map_location: str = "cpu") -> ConceptVector:
        """Deserialize a saved ConceptVector from disk.

        Args:
            filepath: Path to the .pt file.
            map_location: Target device for loading weights ('cpu', 'cuda', etc.).

        Returns:
            Hydrated ConceptVector instance.

        Raises:
            FileNotFoundError: If the specified vector file does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Concept vector file not found at: {path}")

        payload = torch.load(path, map_location=map_location, weights_only=False)
        return cls(
            name=payload["name"],
            layer_idx=payload["layer_idx"],
            vector=payload["vector"],
            description=payload.get("description", ""),
            metadata=payload.get("metadata", {}),
        )
