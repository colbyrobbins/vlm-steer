"""Concept Vector Extractor using Representation Engineering (RepE) Techniques.

This module provides mathematical algorithms for extracting semantic concept directions
from contrastive pairs of positive and negative model activations.

Mathematical Background:
1. Difference-of-Means (Diff-of-Means):
   Given N_pos positive activations {h_pos} and N_neg negative activations {h_neg}:
   mu_pos = (1 / N_pos) * sum(h_pos)
   mu_neg = (1 / N_neg) * sum(h_neg)
   v_raw = mu_pos - mu_neg
   v_concept = v_raw / ||v_raw||_2

2. Contrastive Principal Component Analysis (cPCA):
   Given paired contrastive activations (h_pos_i, h_neg_i):
   Delta_i = h_pos_i - h_neg_i
   Centered_Delta = Delta - mean(Delta)
   Singular Value Decomposition: Centered_Delta = U * S * V^T
   The primary concept direction corresponds to the first right-singular vector V[:, 0],
   which captures the axis of maximal contrastive variance.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

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
        """Extract a concept vector using the Difference-of-Means method.

        Computes the directional vector separating positive concept examples (e.g. unbiased,
        safe, or factually grounded) from negative examples (e.g. biased, toxic, or hallucinated).

        Args:
            pos_acts: Positive activations tensor of shape [N_pos, hidden_dim].
            neg_acts: Negative activations tensor of shape [N_neg, hidden_dim].
            name: String name identifier for the concept.
            layer_idx: Target residual transformer layer index.
            normalize: If True, scales the returned vector to unit L2 norm (1.0).
            description: Human-readable description of concept purpose.

        Returns:
            ConceptVector with computed direction weights and statistical separation metadata.
        """
        assert pos_acts.ndim == 2 and neg_acts.ndim == 2, (
            f"Activations must be 2D matrices [N, hidden_dim], got {pos_acts.shape} and {neg_acts.shape}"
        )
        assert pos_acts.shape[-1] == neg_acts.shape[-1], (
            f"Hidden dimensions must match: {pos_acts.shape[-1]} != {neg_acts.shape[-1]}"
        )

        # 1. Compute empirical centroids of positive and negative activation clusters
        mean_pos = pos_acts.float().mean(dim=0)
        mean_neg = neg_acts.float().mean(dim=0)

        # 2. Compute difference vector pointing from negative centroid to positive centroid
        raw_diff = mean_pos - mean_neg
        raw_norm = float(torch.norm(raw_diff, p=2))

        # 3. Calculate baseline separation diagnostics
        # Cosine similarity between cluster centers (values near -1 indicate strong separation)
        cos_sim = float(
            F.cosine_similarity(mean_pos.unsqueeze(0), mean_neg.unsqueeze(0)).item()
        )

        # 4. Project individual samples onto the unit direction vector
        unit_vec = raw_diff / (raw_norm + 1e-8)
        pos_proj = (pos_acts.float() @ unit_vec).mean().item()
        neg_proj = (neg_acts.float() @ unit_vec).mean().item()
        # Separation margin: how far apart the two classes are along this concept axis
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
        """Extract a concept vector using Principal Component Analysis on paired differences.

        Computes the primary axis of variance among contrastive pairs (pos_i - neg_i).
        Particularly effective when samples are strictly paired (e.g. same prompt with
        subtle demographic token changes or adversarial patch perturbations).

        Args:
            pos_acts: Positive activations tensor of shape [N_pairs, hidden_dim].
            neg_acts: Negative activations tensor of shape [N_pairs, hidden_dim].
            name: Concept name identifier.
            layer_idx: Target residual layer index.
            description: Description of concept purpose.

        Returns:
            ConceptVector aligned along the top principal contrastive direction.
        """
        # Ensure pairwise alignment
        n_pairs = min(len(pos_acts), len(neg_acts))
        # Element-wise difference for each contrastive pair
        diffs = (pos_acts[:n_pairs] - neg_acts[:n_pairs]).float()

        # Center differences to remove global offset
        centered = diffs - diffs.mean(dim=0, keepdim=True)

        # Compute Singular Value Decomposition: centered = U * diag(S) * V^T
        # v_t[0] corresponds to the first right-singular vector (principal axis)
        _, s, v_t = torch.linalg.svd(centered, full_matrices=False)
        top_direction = v_t[0]

        # Verify polarity: ensure vector points in direction of positive examples
        mean_pos_proj = (pos_acts.float() @ top_direction).mean()
        mean_neg_proj = (neg_acts.float() @ top_direction).mean()
        if mean_pos_proj < mean_neg_proj:
            # Flip vector sign if SVD inverted the orientation
            top_direction = -top_direction

        # Ratio of total variance explained by the top principal component
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
