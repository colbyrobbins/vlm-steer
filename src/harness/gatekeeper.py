"""Autonomous Decision Gatekeeper: Evaluates latent probes and decides interventions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.attribution.probes import LatentConceptProbe
from src.harness.config import ProbeConfig, SteeringPolicy
from src.harness.telemetry import ProbeResult, SteeringRecord
from src.steering.vectors import ConceptVector


class DecisionGatekeeper:
    """Evaluates in-flight probes and triggers active steering interventions."""

    def __init__(self):
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
        """Register a concept with an associated in-flight probe and default steering policy."""
        self._concept_vectors[concept.name] = concept
        self._probes[concept.name] = LatentConceptProbe(
            concept=concept,
            threshold=probe_threshold,
        )
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
        """Evaluate all registered probes on the captured layer hidden states.

        Returns:
            Tuple of:
                - List[ProbeResult]: Diagnostic probe evaluations
                - List[SteeringRecord]: Active steering interventions to execute
        """
        probe_results: List[ProbeResult] = []
        planned_steers: List[SteeringRecord] = []

        for name, probe in self._probes.items():
            layer_idx = probe.concept.layer_idx
            if layer_idx not in layer_activations:
                continue

            h = layer_activations[layer_idx]
            result = probe.evaluate(h, token_reduction="last")
            probe_results.append(result)

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
        return self._concept_vectors.get(name)
