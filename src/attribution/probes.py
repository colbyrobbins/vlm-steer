"""In-Flight Latent Concept Probes for Real-Time Model Observability.

This module implements `LatentConceptProbe`, which evaluates model hidden states in real time
during the forward pass to measure semantic alignment with known risk or safety concepts.

Mathematical Formulation:
1. Cosine Alignment:
   Given residual hidden state h in R^d and unit concept vector v in R^d:
   cos_sim = (h . v) / (||h||_2 * ||v||_2)  in [-1.0, 1.0]

2. Calibrated Normalized Risk Score:
   To translate raw cosine similarity into a probability-like risk score in [0.0, 1.0]:
   risk = sigma(temperature * cos_sim)
   - When cos_sim = 0 (orthogonal, neutral): risk = 0.5
   - When cos_sim > 0 (aligning with risk direction): risk -> 1.0
   - When cos_sim < 0 (opposing risk direction): risk -> 0.0
   - 'temperature' parameter tunes the steepness of the decision boundary.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.harness.telemetry import ProbeResult

from src.steering.vectors import ConceptVector


class LatentConceptProbe:
    """Evaluates whether intermediate residual states align with a risk or safety concept direction.

    Designed for lightweight, zero-latency execution inside model forward hooks.
    """

    def __init__(
        self,
        concept: ConceptVector,
        threshold: float = 0.5,
        temperature: float = 5.0,
    ) -> None:
        """Initialize the latent probe.

        Args:
            concept: ConceptVector instance representing the target direction.
            threshold: Normalized risk cutoff in [0, 1] above which probe triggers an alert.
            temperature: Scaling factor controlling the sensitivity/sharpness of the sigmoid mapping.
        """
        self.concept = concept
        self.threshold = threshold
        self.temperature = temperature

        # Pre-normalize concept vector to guarantee unit length for fast cosine dot products
        raw_norm = torch.norm(concept.vector, p=2)
        self.unit_vec = concept.vector / (raw_norm + 1e-8)

    def evaluate(
        self,
        hidden_states: torch.Tensor,
        token_reduction: str = "last",
    ) -> ProbeResult:
        """Evaluate hidden states against the concept direction.

        Args:
            hidden_states: Tensor from hooked layer [batch_size, seq_len, hidden_dim]
                           or [batch_size, hidden_dim].
            token_reduction: Method for reducing sequence tokens:
                - "last": Evaluates only the final sequence token (typical for autoregressive next-token).
                - "mean": Averages across all sequence tokens (typical for visual patch or prompt pooling).

        Returns:
            ProbeResult containing raw cosine similarity, normalized risk, and trigger flag.
        """
        # Synchronize device and precision with incoming hidden states
        device = hidden_states.device
        dtype = hidden_states.dtype
        vec = self.unit_vec.to(device=device, dtype=dtype)

        with torch.no_grad():
            # Reduce 3D sequence tensor to 2D [batch_size, hidden_dim]
            if hidden_states.ndim == 3:
                if token_reduction == "last":
                    h = hidden_states[:, -1, :]
                else:
                    h = hidden_states.mean(dim=1)
            else:
                h = hidden_states

            # Compute cosine similarity along hidden_dim dimension: range [-1.0, 1.0]
            cos_sim = F.cosine_similarity(h, vec.unsqueeze(0), dim=-1).mean().item()

            # Determine polarity: does vector encode risk directly or a target remediation direction?
            is_remediation_target = any(
                token in self.concept.name.lower()
                for token in ["neutrality", "refusal", "grounding", "safety", "remediation", "aligned"]
            )
            effective_score = -cos_sim if is_remediation_target else cos_sim

            # Map alignment/risk to normalized risk in [0, 1] via scaled sigmoid
            # sigma(temperature * effective_score)
            scaled_logit = torch.tensor(effective_score * self.temperature)
            normalized_risk = float(torch.sigmoid(scaled_logit).item())

            # Evaluate decision rule: does risk exceed the configured policy threshold?
            triggered = normalized_risk >= self.threshold

        from src.harness.telemetry import ProbeResult

        return ProbeResult(
            concept_name=self.concept.name,
            layer_idx=self.concept.layer_idx,
            raw_score=float(cos_sim),
            normalized_risk=normalized_risk,
            triggered=triggered,
        )
