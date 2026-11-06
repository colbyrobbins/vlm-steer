"""Structured Telemetry and XAI Governance Audit Records.

This module defines standardized audit records emitted by `VLM-SteerHarness`.
In enterprise and safety-critical deployments (e.g. NIST AI RMF, EU AI Act compliance),
explainability cannot be an ephemeral print statement—it must produce structured,
reproducible audit telemetry that tracks:
1. Input prompts and modal context (image presence).
2. Diagnostic risk probe scores prior to intervention.
3. Causal mechanistic interventions applied (steering vectors, layer indices, alpha values).
4. Granular neuron attributions identifying the active concept circuits.
5. Latency metrics quantifying runtime overhead.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProbeResult(BaseModel):
    """Result of an in-flight latent probe evaluation.

    Attributes:
        concept_name: Target concept evaluated.
        layer_idx: Layer index where probe inspected hidden states.
        raw_score: Raw cosine similarity between hidden state and concept vector [-1.0, 1.0].
        normalized_risk: Scaled sigmoid risk probability [0.0, 1.0].
        triggered: Boolean flag indicating whether normalized_risk >= threshold.
    """

    concept_name: str
    layer_idx: int
    raw_score: float
    normalized_risk: float
    triggered: bool


class SteeringRecord(BaseModel):
    """Record of an active steering intervention executed during generation.

    Attributes:
        concept_name: Concept vector applied.
        layer_idx: Layer index where vector was injected via forward hook.
        alpha: Steering multiplier used during inference.
        token_scope: Token scope ('all' or 'last').
    """

    concept_name: str
    layer_idx: int
    alpha: float
    token_scope: str


class NeuronAttribution(BaseModel):
    """Top-k neurons correlated with active concept directions.

    Attributes:
        layer_idx: Residual layer index.
        top_neuron_indices: List of integer column indices corresponding to top neurons.
        top_neuron_scores: Absolute weight or activation magnitude for each top neuron.
    """

    layer_idx: int
    top_neuron_indices: List[int]
    top_neuron_scores: List[float]


class AuditTrace(BaseModel):
    """Comprehensive governance and XAI audit trace for an inference lifecycle.

    Attributes:
        trace_id: Unique 8-character UUID identifying the inference request.
        timestamp: Unix epoch timestamp of request execution.
        prompt: User input prompt string.
        has_image: True if an image was provided as multimodal context.
        probes: List of diagnostic ProbeResult evaluations.
        steered: True if one or more active steering interventions were executed.
        steering_applied: Detailed records of steering vectors and alpha values injected.
        neuron_attributions: Top active neurons and concept circuit profiles.
        output_text: Final generated text output.
        latency_ms: Total end-to-end inference latency in milliseconds.
    """

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    prompt: str
    has_image: bool
    probes: List[ProbeResult] = Field(default_factory=list)
    steered: bool = False
    steering_applied: List[SteeringRecord] = Field(default_factory=list)
    neuron_attributions: List[NeuronAttribution] = Field(default_factory=list)
    output_text: str = ""
    latency_ms: float = 0.0

    def to_json(self, indent: int = 2) -> str:
        """Serialize audit trace to standard formatted JSON string.

        Args:
            indent: JSON indentation spacing (default: 2).

        Returns:
            JSON string representation of the audit trace.
        """
        return self.model_dump_json(indent=indent)

    def summary(self) -> str:
        """Human-readable one-line summary for developer logs and telemetry pipelines."""
        status = "STEERED" if self.steered else "BASELINE"
        num_triggered = sum(p.triggered for p in self.probes)
        total_probes = len(self.probes)
        return (
            f"[Trace {self.trace_id}] {status} | Latency: {self.latency_ms:.1f}ms | "
            f"Probes Triggered: {num_triggered}/{total_probes}"
        )
