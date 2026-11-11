"""Core SteerHarness Runtime: Orchestrates Ingestion, Probing, Steering, and Telemetry.

This module implements `SteerHarness`, the top-level SDK interface for `VLM-Steer`.
It wraps any supported vision-language model backbone to provide an end-to-end,
production-grade inference lifecycle with integrated mechanistic XAI observability,
active latent steering, and governance telemetry.

Lifecycle Workflow:
1. Ingestion & Pre-Flight Diagnostic Pass:
   Forward pass intercepts intermediate residual stream hidden states across monitored layers.
2. In-Flight XAI Observability:
   Latent probes compute cosine alignment and normalized risk scores against registered concepts.
3. Neuron Attribution Profiling:
   Identifies the top-k neurons/channels contributing to active concept circuits.
4. Autonomous Decision Gate & Active Steering:
   If a risk threshold is breached, forward hooks dynamically inject calibrated steering vectors
   into the residual stream during generation, remediating the output in real time.
5. Telemetry & Governance Logging:
   Emits structured `AuditTrace` objects documenting the entire diagnostic and intervention trace.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

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
    ) -> None:
        """Initialize the SteerHarness runtime.

        Args:
            model: BaseVLM instance (defaults to MockVLM for instant local testing).
            config: HarnessConfig specifying default policies and telemetry levels.
        """
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
        """Register a concept direction with its threshold and default steering policy.

        Args:
            concept: ConceptVector instance containing directional weights and target layer.
            probe_threshold: Risk score cutoff in [0, 1] that triggers steering.
            default_alpha: Steering intensity multiplier applied when triggered.
            token_scope: "all" (applies to all sequence tokens) or "last" (final token).
        """
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
            prompt: Text prompt input string.
            image: Optional PIL Image input providing visual context.
            manual_policies: Optional override steering policies for manual experimentation.
            max_new_tokens: Maximum token budget for generation.

        Returns:
            Tuple of:
                - str: Generated output text (remediated if steering was applied).
                - AuditTrace: Comprehensive XAI governance audit record.
        """
        # Start high-precision timer to measure end-to-end latency overhead
        start_time = time.perf_counter()
        tokens_to_gen = max_new_tokens or self.config.max_new_tokens

        # -------------------------------------------------------------------------
        # Step 1: Pre-Flight Diagnostic Forward Pass
        # -------------------------------------------------------------------------
        # Intercept intermediate hidden states across all monitored residual layers
        captured_layers: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            _, layer_states = self.model.forward_hidden_states(image=image, text=prompt)
            for idx, h in enumerate(layer_states):
                captured_layers[idx] = h

        # -------------------------------------------------------------------------
        # Step 2: In-Flight XAI Observability Probing
        # -------------------------------------------------------------------------
        # Evaluate latent probes against captured hidden states to detect risk breaches
        probe_results, planned_steers = self.gatekeeper.evaluate_and_plan(captured_layers)

        # Merge with any manual override policies provided for experimentation
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

        # -------------------------------------------------------------------------
        # Step 3: Neuron Attribution Profiling
        # -------------------------------------------------------------------------
        # Extract the highest-magnitude neuron dimensions forming the active concept circuits
        neuron_attributions: List[NeuronAttribution] = []
        for pr in probe_results:
            concept = self.gatekeeper.get_concept_vector(pr.concept_name)
            if concept:
                neuron_attr = NeuronAttributor.top_concept_neurons(
                    concept,
                    top_k=5,
                    hidden_states=captured_layers.get(pr.layer_idx),
                )
                neuron_attributions.append(neuron_attr)

        # -------------------------------------------------------------------------
        # Step 4: Decision Gate & Active Steering Generation
        # -------------------------------------------------------------------------
        applied_steers: List[SteeringRecord] = []
        steered = len(planned_steers) > 0

        if steered:
            # Attach active PyTorch forward hooks for each triggered steering policy
            handles = []
            try:
                for plan in planned_steers:
                    concept = self.gatekeeper.get_concept_vector(plan.concept_name)
                    if not concept:
                        continue
                    # Retrieve the specific transformer block module to hook
                    layer_mod = self.model.get_residual_layer(plan.layer_idx)
                    handle = self.hook_manager.register_steering_hook(
                        module=layer_mod,
                        steering_vector=concept,
                        alpha=plan.alpha,
                        token_scope=plan.token_scope,
                    )
                    handles.append(handle)
                    applied_steers.append(plan)

                # Generate output with active steering vector injection in place
                output_text = self.model.generate(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=tokens_to_gen,
                )
            finally:
                # Guarantees complete cleanup of all forward hooks even on failure
                for h in handles:
                    h.remove()
                self.hook_manager.clear()
        else:
            # Baseline execution: generate without residual stream intervention
            output_text = self.model.generate(
                image=image,
                prompt=prompt,
                max_new_tokens=tokens_to_gen,
            )

        # Calculate latency overhead in milliseconds
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Construct structured audit record
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
