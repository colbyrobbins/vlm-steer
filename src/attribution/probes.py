"""In-flight Latent Concept Probes for real-time model observability."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.harness.telemetry import ProbeResult
from src.steering.vectors import ConceptVector


class LatentConceptProbe:
    """Evaluates whether intermediate residual states align with a risk or safety concept direction."""

    def __init__(
        self,
        concept: ConceptVector,
        threshold: float = 0.5,
        temperature: float = 5.0,
    ):
        self.concept = concept
        self.threshold = threshold
        self.temperature = temperature
        # Ensure concept vector is unit normalized
        self.unit_vec = concept.vector / (torch.norm(concept.vector, p=2) + 1e-8)

    def evaluate(
        self,
        hidden_states: torch.Tensor,
        token_reduction: str = "last",
    ) -> ProbeResult:
        """Evaluate hidden states against the concept direction.

        Args:
            hidden_states: Hidden state tensor from hooked layer [batch, seq_len, hidden_dim]
                or [batch, hidden_dim]
            token_reduction: "last" (evaluates last token) or "mean" (evaluates mean across sequence)

        Returns:
            ProbeResult
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        vec = self.unit_vec.to(device=device, dtype=dtype)

        with torch.no_grad():
            if hidden_states.ndim == 3:
                if token_reduction == "last":
                    h = hidden_states[:, -1, :]  # [batch, hidden_dim]
                else:
                    h = hidden_states.mean(dim=1)  # [batch, hidden_dim]
            else:
                h = hidden_states

            # Cosine similarity against the unit concept vector: range [-1, 1]
            cos_sim = F.cosine_similarity(h, vec.unsqueeze(0), dim=-1).mean().item()

            # Map cosine similarity [-1, 1] to normalized risk [0, 1] via scaled sigmoid
            # Centered at 0: when cos_sim = 0, risk = 0.5
            normalized_risk = float(torch.sigmoid(torch.tensor(cos_sim * self.temperature)).item())

            triggered = normalized_risk >= self.threshold

        return ProbeResult(
            concept_name=self.concept.name,
            layer_idx=self.concept.layer_idx,
            raw_score=float(cos_sim),
            normalized_risk=normalized_risk,
            triggered=triggered,
        )
