"""Concept Vector Extractor using Representation Engineering (RepE) techniques."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.steering.vectors import ConceptVector


class ConceptVectorExtractor:
    """Extracts concept directions from contrastive positive and negative activations."""

    @staticmethod
    def diff_of_means(
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor,
        name: str,
        layer_idx: int,
        normalize: bool = True,
        description: str = "",
    ) -> ConceptVector:
        """Compute the Difference-of-Means concept vector: v = mean(pos) - mean(neg).

        Args:
            pos_acts: Positive activations [N_pos, hidden_dim]
            neg_acts: Negative activations [N_neg, hidden_dim]
            name: Concept name (e.g. "demographic_neutrality", "safety_refusal")
            layer_idx: Target layer index
            normalize: Whether to unit-normalize the resulting vector
            description: Description of the concept

        Returns:
            ConceptVector
        """
        assert pos_acts.ndim == 2 and neg_acts.ndim == 2, "Activations must be 2D [N, hidden_dim]"
        assert pos_acts.shape[-1] == neg_acts.shape[-1], "Hidden dimensions must match"

        mean_pos = pos_acts.float().mean(dim=0)
        mean_neg = neg_acts.float().mean(dim=0)

        raw_diff = mean_pos - mean_neg
        raw_norm = float(torch.norm(raw_diff, p=2))

        # Separation metric: Cosine distance between cluster centers
        cos_sim = float(F.cosine_similarity(mean_pos.unsqueeze(0), mean_neg.unsqueeze(0)))

        # Project positive and negative samples onto the vector
        unit_vec = raw_diff / (raw_norm + 1e-8)
        pos_proj = (pos_acts.float() @ unit_vec).mean().item()
        neg_proj = (neg_acts.float() @ unit_vec).mean().item()
        separation_margin = pos_proj - neg_proj

        metadata = {
            "method": "diff_of_means",
            "num_pos": len(pos_acts),
            "num_neg": len(neg_acts),
            "raw_norm": raw_norm,
            "center_cosine_similarity": cos_sim,
            "mean_pos_projection": pos_proj,
            "mean_neg_projection": neg_proj,
            "separation_margin": separation_margin,
        }

        final_vec = unit_vec if normalize else raw_diff
        return ConceptVector(
            name=name,
            layer_idx=layer_idx,
            vector=final_vec,
            description=description,
            metadata=metadata,
        )

    @staticmethod
    def contrastive_pca(
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor,
        name: str,
        layer_idx: int,
        description: str = "",
    ) -> ConceptVector:
        """Compute concept vector using Principal Component Analysis on paired differences.

        Useful when activations are paired: (pos_i, neg_i).
        If lengths differ, takes the minimum shared sample count.
        """
        n_pairs = min(len(pos_acts), len(neg_acts))
        diffs = (pos_acts[:n_pairs] - neg_acts[:n_pairs]).float()

        # Center differences
        centered = diffs - diffs.mean(dim=0, keepdim=True)

        # SVD on centered differences
        # centered = U @ diag(S) @ V^T -> top principal direction is V[:, 0]
        _, s, v_t = torch.linalg.svd(centered, full_matrices=False)
        top_direction = v_t[0]

        # Ensure the sign points in the positive direction
        mean_pos_proj = (pos_acts.float() @ top_direction).mean()
        mean_neg_proj = (neg_acts.float() @ top_direction).mean()
        if mean_pos_proj < mean_neg_proj:
            top_direction = -top_direction

        variance_explained = float((s[0] ** 2) / (s**2).sum())

        metadata = {
            "method": "contrastive_pca",
            "num_pairs": n_pairs,
            "top_singular_value": float(s[0]),
            "variance_explained_ratio": variance_explained,
        }

        return ConceptVector(
            name=name,
            layer_idx=layer_idx,
            vector=top_direction,
            description=description,
            metadata=metadata,
        )
