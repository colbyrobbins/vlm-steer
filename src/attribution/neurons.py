"""Neuron-Level Attribution and Activation Profiling.

This module implements `NeuronAttributor`, which profiles individual neuron activations
and decomposes concept vectors down to individual feature dimensions.

Theoretical Purpose:
Mechanistic interpretability seeks to understand neural networks at the circuit and neuron level.
By isolating the highest-magnitude dimensions of a concept vector:
1. We can verify whether specific polysemantic or monosemantic neurons encode the concept.
2. We can monitor whether individual "safety neurons" or "bias neurons" are firing during inference.
3. We provide granular XAI diagnostic transparency to audit logs and UI dashboards.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from src.harness.telemetry import NeuronAttribution

from src.steering.vectors import ConceptVector


class NeuronAttributor:
    """Profiles individual neurons and feature channels driving model representations."""

    @staticmethod
    def _pool_last_token(hidden_states: torch.Tensor) -> torch.Tensor:
        """Reduce 3D/2D hidden states to a 1D vector [hidden_dim] over the final token."""
        if hidden_states.ndim == 3:
            return hidden_states[:, -1, :].mean(dim=0)
        if hidden_states.ndim == 2:
            return hidden_states.mean(dim=0)
        return hidden_states

    @staticmethod
    def causal_attribution_vector(
        concept: ConceptVector,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute signed per-neuron causal contributions to the concept projection.

        When hidden_states is provided, computes activation * unit_concept_vector (h_i * v_i),
        measuring each neuron's exact linear contribution to concept alignment (or ablation delta).
        Falls back to static concept vector weights v_i when hidden_states is omitted.
        """
        with torch.no_grad():
            vec = concept.vector.detach()
            unit_vec = vec / (torch.norm(vec, p=2) + 1e-8)
            if hidden_states is None:
                return vec
            h_pooled = NeuronAttributor._pool_last_token(
                hidden_states.detach().to(device=unit_vec.device, dtype=unit_vec.dtype)
            )
            return h_pooled * unit_vec

    @staticmethod
    def top_concept_neurons(
        concept: ConceptVector,
        top_k: int = 5,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> NeuronAttribution:
        """Identify top-k neuron channels driving the concept direction.

        When hidden_states is supplied, ranks neurons by empirical causal impact |h_i * v_i|.
        Otherwise ranks by static direction weight magnitude |v_i|.

        Args:
            concept: ConceptVector instance.
            top_k: Number of highest-magnitude neuron indices to retrieve.
            hidden_states: Optional residual activations [batch, seq_len, dim] or [batch, dim].

        Returns:
            NeuronAttribution containing top indices and their attribution magnitudes.
        """
        from src.harness.telemetry import NeuronAttribution

        causal_scores = NeuronAttributor.causal_attribution_vector(
            concept, hidden_states=hidden_states
        )
        abs_scores = causal_scores.abs()
        k = min(top_k, len(abs_scores))

        top_scores, top_indices = torch.topk(abs_scores, k=k)

        return NeuronAttribution(
            layer_idx=concept.layer_idx,
            top_neuron_indices=[int(idx) for idx in top_indices],
            top_neuron_scores=[float(score) for score in top_scores],
        )

    @staticmethod
    def top_active_neurons(
        hidden_states: torch.Tensor,
        layer_idx: int,
        top_k: int = 5,
    ) -> NeuronAttribution:
        """Identify which neuron dimensions have the highest activation magnitude in an inference state.

        Args:
            hidden_states: Layer activation tensor of shape [batch, seq_len, hidden_dim]
                           or [batch, hidden_dim].
            layer_idx: Residual layer index from which hidden states were extracted.
            top_k: Number of highest-activation neuron indices to retrieve.

        Returns:
            NeuronAttribution containing top active neuron indices and activation magnitudes.
        """
        from src.harness.telemetry import NeuronAttribution

        with torch.no_grad():
            # Pool across batch and sequence dimensions to obtain a 1D activation profile
            if hidden_states.ndim == 3:
                # Shape: [batch, seq_len, hidden_dim] -> average over batch and sequence
                profile = hidden_states.abs().mean(dim=[0, 1])
            elif hidden_states.ndim == 2:
                # Shape: [batch, hidden_dim] -> average over batch
                profile = hidden_states.abs().mean(dim=0)
            else:
                profile = hidden_states.abs()

            k = min(top_k, len(profile))
            top_scores, top_indices = torch.topk(profile, k=k)

            return NeuronAttribution(
                layer_idx=layer_idx,
                top_neuron_indices=[int(idx) for idx in top_indices],
                top_neuron_scores=[float(score) for score in top_scores],
            )
