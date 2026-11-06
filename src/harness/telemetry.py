"""Structured Telemetry and XAI Governance Audit Records."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProbeResult(BaseModel):
    """Result of an in-flight latent probe evaluation."""

    concept_name: str
    layer_idx: int
    raw_score: float
    normalized_risk: float
    triggered: bool


class SteeringRecord(BaseModel):
    """Record of an active steering intervention applied during generation."""

    concept_name: str
    layer_idx: int
    alpha: float
    token_scope: str


class NeuronAttribution(BaseModel):
    """Top-k neurons correlated with active concept directions."""

    layer_idx: int
    top_neuron_indices: List[int]
    top_neuron_scores: List[float]


class AuditTrace(BaseModel):
    """Comprehensive governance and XAI audit trace for an inference lifecycle."""

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
        """Serialize audit trace to standard JSON format."""
        return self.model_dump_json(indent=indent)

    def summary(self) -> str:
        """Human-readable one-line summary for developer logs."""
        status = "STEERED" if self.steered else "BASELINE"
        return f"[Trace {self.trace_id}] {status} | Latency: {self.latency_ms:.1f}ms | Probes Triggered: {sum(p.triggered for p in self.probes)}/{len(self.probes)}"
