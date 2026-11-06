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

from typing import List, Tuple

import torch

from src.harness.telemetry import NeuronAttribution
from src.steering.vectors import ConceptVector


class NeuronAttributor:
    """Profiles individual neurons and feature channels driving model representations."""

    @staticmethod
    def top_concept_neurons(
        concept: ConceptVector,
        top_k: int = 5,
    ) -> NeuronAttribution:
        """Identify which neuron indices contribute most strongly to the concept direction vector.

        Deconstructs the concept vector v by absolute weight magnitude: |v_i|.
        High-magnitude indices represent the primary neuron channels forming the concept circuit.

        Args:
            concept: ConceptVector instance.
            top_k: Number of highest-magnitude neuron indices to retrieve.

        Returns:
            NeuronAttribution containing top indices and their corresponding weights.
        """
        # Absolute magnitude reveals directional impact regardless of positive/negative orientation
        abs_weights = concept.vector.abs()
        k = min(top_k, len(abs_weights))

        # Retrieve top-k values and their positional column indices
        top_scores, top_indices = torch.topk(abs_weights, k=k)

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
