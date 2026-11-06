"""Neuron-level attribution and activation profiling."""

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
        """Identify which neuron indices contribute most strongly to the concept direction vector."""
        vec = concept.vector.abs()
        k = min(top_k, len(vec))
        top_scores, top_indices = torch.topk(vec, k=k)

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
        """Identify which neuron indices have the highest activation magnitude in a hidden state."""
        with torch.no_grad():
            if hidden_states.ndim == 3:
                # Average across sequence length
                h = hidden_states.abs().mean(dim=[0, 1])
            elif hidden_states.ndim == 2:
                h = hidden_states.abs().mean(dim=0)
            else:
                h = hidden_states.abs()

            k = min(top_k, len(h))
            top_scores, top_indices = torch.topk(h, k=k)

            return NeuronAttribution(
                layer_idx=layer_idx,
                top_neuron_indices=[int(idx) for idx in top_indices],
                top_neuron_scores=[float(score) for score in top_scores],
            )
