"""Concept vector data structures and serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


@dataclass
class ConceptVector:
    """Represents an extracted concept direction in representation space."""

    name: str
    layer_idx: int
    vector: torch.Tensor
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.vector, torch.Tensor):
            self.vector = torch.tensor(self.vector, dtype=torch.float32)
        # Ensure 1D vector
        if self.vector.ndim > 1:
            self.vector = self.vector.squeeze()

    @property
    def dimension(self) -> int:
        return self.vector.shape[-1]

    def normalize(self) -> ConceptVector:
        """Return a unit-normalized copy of this concept vector."""
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
        """Save vector weights and metadata to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "name": self.name,
            "layer_idx": self.layer_idx,
            "description": self.description,
            "metadata": self.metadata,
            "vector": self.vector.detach().cpu(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, filepath: Union[str, Path], map_location: str = "cpu") -> ConceptVector:
        """Load a saved concept vector from disk."""
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
