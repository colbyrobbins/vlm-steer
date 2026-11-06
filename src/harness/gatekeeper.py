"""Autonomous Decision Gatekeeper: Evaluates Latent Probes and Plans Interventions.

This module implements `DecisionGatekeeper`, the policy routing engine of `VLM-SteerHarness`.
During inference, it evaluates intercepted hidden states against configured risk thresholds
and dynamically plans steering interventions to remediate detected failure modes.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.attribution.probes import LatentConceptProbe
from src.harness.config import ProbeConfig, SteeringPolicy
from src.harness.telemetry import ProbeResult, SteeringRecord
from src.steering.vectors import ConceptVector


class DecisionGatekeeper:
    """Evaluates in-flight probes and dynamically triggers active steering interventions."""

    def __init__(self) -> None:
        """Initialize empty registries for probes, policies, and concept vectors."""
        self._probes: Dict[str, LatentConceptProbe] = {}
        self._policies: Dict[str, SteeringPolicy] = {}
        self._concept_vectors: Dict[str, ConceptVector] = {}

    def register_concept(
        self,
        concept: ConceptVector,
        probe_threshold: float = 0.5,
        default_alpha: float = 1.0,
        token_scope: str = "all",
    ) -> None:
        """Register a concept with an associated in-flight probe and default steering policy.

        Args:
            concept: ConceptVector instance holding directional weights and target layer.
            probe_threshold: Risk score cutoff in [0, 1] above which steering is triggered.
            default_alpha: Steering intensity multiplier applied when triggered.
            token_scope: "all" (applies to all sequence tokens) or "last" (final token).
        """
        # Store concept vector for retrieval during hook injection
        self._concept_vectors[concept.name] = concept

        # Instantiate latent probe for in-flight monitoring
        self._probes[concept.name] = LatentConceptProbe(
            concept=concept,
            threshold=probe_threshold,
        )

        # Configure default intervention policy
        self._policies[concept.name] = SteeringPolicy(
            concept_name=concept.name,
            target_layer_idx=concept.layer_idx,
            alpha=default_alpha,
            token_scope=token_scope,
        )

    def evaluate_and_plan(
        self,
        layer_activations: Dict[int, torch.Tensor],
    ) -> Tuple[List[ProbeResult], List[SteeringRecord]]:
        """Evaluate all registered probes on captured layer hidden states and plan interventions.

        Args:
            layer_activations: Dictionary mapping integer layer indices to intercepted hidden state tensors.

        Returns:
            Tuple of:
                - List[ProbeResult]: Diagnostic evaluations from all active probes.
                - List[SteeringRecord]: Planned steering actions for probes that breached threshold.
        """
        probe_results: List[ProbeResult] = []
        planned_steers: List[SteeringRecord] = []

        for name, probe in self._probes.items():
            layer_idx = probe.concept.layer_idx
            # Skip evaluation if the monitored layer was not captured in this forward pass
            if layer_idx not in layer_activations:
                continue

            # Evaluate probe on layer hidden states (using last token reduction by default)
            h = layer_activations[layer_idx]
            result = probe.evaluate(h, token_reduction="last")
            probe_results.append(result)

            # If the probe detects a risk breach, queue an active steering intervention
            if result.triggered:
                policy = self._policies.get(name)
                if policy:
                    planned_steers.append(
                        SteeringRecord(
                            concept_name=name,
                            layer_idx=policy.target_layer_idx,
                            alpha=policy.alpha,
                            token_scope=policy.token_scope,
                        )
                    )

        return probe_results, planned_steers

    def get_concept_vector(self, name: str) -> Optional[ConceptVector]:
        """Retrieve a registered ConceptVector by name.

        Args:
            name: Concept name identifier.

        Returns:
            ConceptVector if found, otherwise None.
        """
        return self._concept_vectors.get(name)
