"""Core SteerHarness runtime: Orchestrates ingestion, probing, steering, and telemetry."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

import torch
import torch.nn as nn

from src.attribution.neurons import NeuronAttributor
from src.harness.config import HarnessConfig, SteeringPolicy
from src.harness.gatekeeper import DecisionGatekeeper
from src.harness.telemetry import AuditTrace, NeuronAttribution, SteeringRecord
from src.models.wrapper import BaseVLM, MockVLM
from src.steering.hook import ActivationHookManager
from src.steering.vectors import ConceptVector


class SteerHarness:
    """Production runtime harness integrating in-flight XAI observability and active steering."""

    def __init__(
        self,
        model: Optional[BaseVLM] = None,
        config: Optional[HarnessConfig] = None,
    ):
        self.config = config or HarnessConfig()
        self.model = model or MockVLM()
        self.hook_manager = ActivationHookManager()
        self.gatekeeper = DecisionGatekeeper()

    def register_concept(
        self,
        concept: ConceptVector,
        probe_threshold: float = 0.5,
        default_alpha: float = 1.0,
        token_scope: str = "all",
    ) -> None:
        """Register a concept direction with its threshold and steering policy."""
        self.gatekeeper.register_concept(
            concept=concept,
            probe_threshold=probe_threshold,
            default_alpha=default_alpha,
            token_scope=token_scope,
        )

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        manual_policies: Optional[List[SteeringPolicy]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, AuditTrace]:
        """Execute the full inference lifecycle with in-flight XAI observability and active steering.

        Args:
            prompt: Text prompt input.
            image: Optional PIL Image input.
            manual_policies: Optional override steering policies for manual experimentation.
            max_new_tokens: Max new tokens to generate.

        Returns:
            Tuple of:
                - str: Generated output text (remediated if steering was applied)
                - AuditTrace: Comprehensive XAI governance audit record
        """
        start_time = time.perf_counter()
        tokens_to_gen = max_new_tokens or self.config.max_new_tokens

        # Step 1: Pre-flight Diagnostic Forward Pass
        # Attach capture hooks to all layers monitored by probes
        captured_layers: Dict[int, torch.Tensor] = {}

        # Run forward pass to extract hidden states across monitored layers
        with torch.no_grad():
            _, layer_states = self.model.forward_hidden_states(image=image, text=prompt)
            for idx, h in enumerate(layer_states):
                captured_layers[idx] = h

        # Step 2: In-Flight XAI Observability Probing
        probe_results, planned_steers = self.gatekeeper.evaluate_and_plan(captured_layers)

        # Merge with any manual policies passed by developer
        if manual_policies:
            for mp in manual_policies:
                planned_steers.append(
                    SteeringRecord(
                        concept_name=mp.concept_name,
                        layer_idx=mp.target_layer_idx,
                        alpha=mp.alpha,
                        token_scope=mp.token_scope,
                    )
                )

        # Step 3: Neuron Attribution Profiling
        neuron_attributions: List[NeuronAttribution] = []
        for pr in probe_results:
            concept = self.gatekeeper.get_concept_vector(pr.concept_name)
            if concept:
                neuron_attr = NeuronAttributor.top_concept_neurons(concept, top_k=5)
                neuron_attributions.append(neuron_attr)

        # Step 4: Decision Gate & Active Steering Generation
        applied_steers: List[SteeringRecord] = []
        steered = len(planned_steers) > 0

        if steered:
            # Register active steering hooks for each triggered policy
            handles = []
            try:
                for plan in planned_steers:
                    concept = self.gatekeeper.get_concept_vector(plan.concept_name)
                    if not concept:
                        continue
                    layer_mod = self.model.get_residual_layer(plan.layer_idx)
                    handle = self.hook_manager.register_steering_hook(
                        module=layer_mod,
                        steering_vector=concept,
                        alpha=plan.alpha,
                        token_scope=plan.token_scope,
                    )
                    handles.append(handle)
                    applied_steers.append(plan)

                # Generate output with active steering in place
                output_text = self.model.generate(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=tokens_to_gen,
                )
            finally:
                # Always safely remove steering hooks
                for h in handles:
                    h.remove()
                self.hook_manager.clear()
        else:
            # Baseline execution without intervention
            output_text = self.model.generate(
                image=image,
                prompt=prompt,
                max_new_tokens=tokens_to_gen,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        trace = AuditTrace(
            prompt=prompt,
            has_image=image is not None,
            probes=probe_results,
            steered=steered,
            steering_applied=applied_steers,
            neuron_attributions=neuron_attributions,
            output_text=output_text,
            latency_ms=latency_ms,
        )

        return output_text, trace
