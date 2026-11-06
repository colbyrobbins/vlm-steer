"""Configuration Schemas and Policy Definitions for VLM-SteerHarness.

This module defines Pydantic configuration models that specify the behavioral parameters
of the runtime harness, including latent probe thresholds, steering intensities (alpha),
and telemetry granularity.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SteeringPolicy(BaseModel):
    """Configuration for an active steering intervention on a specific concept.

    Attributes:
        concept_name: Identifier of the target concept vector to apply.
        target_layer_idx: Layer index where forward hooks will inject the vector.
        alpha: Steering multiplier. Positive values reinforce alignment; negative values red-team.
        token_scope: "all" (applies to all sequence tokens) or "last" (applies strictly to final token).
    """

    concept_name: str = Field(..., description="Unique concept vector identifier.")
    target_layer_idx: int = Field(..., ge=0, description="Zero-indexed target residual layer.")
    alpha: float = Field(
        default=1.0,
        description="Steering intensity. Positive remediates; negative red-teams.",
    )
    token_scope: Literal["all", "last"] = Field(
        default="all",
        description="Token scope for residual addition ('all' or 'last').",
    )


class ProbeConfig(BaseModel):
    """Configuration for an in-flight latent concept probe.

    Attributes:
        concept_name: Identifier of the concept vector monitored by this probe.
        target_layer_idx: Layer index where hidden states are captured and evaluated.
        threshold: Normalized risk score cutoff in [0, 1] that triggers an alert or intervention.
        auto_steer: If True, automatically activates the associated SteeringPolicy upon trigger.
    """

    concept_name: str = Field(..., description="Monitored concept identifier.")
    target_layer_idx: int = Field(..., ge=0, description="Layer index monitored by probe.")
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Trigger threshold in [0.0, 1.0].",
    )
    auto_steer: bool = Field(
        default=True,
        description="Whether to automatically trigger steering when threshold is exceeded.",
    )


class HarnessConfig(BaseModel):
    """Global configuration for the VLM-SteerHarness runtime.

    Attributes:
        model_id: Identifier of model backbone ('mock-vlm-v1', 'paligemma-3b-pt-224', etc.).
        device: Compute device ('cpu', 'cuda', or None for auto-detection).
        enabled_probes: List of ProbeConfig instances active during inference.
        default_policies: Dictionary mapping concept names to default SteeringPolicy rules.
        telemetry_level: Logging detail level ('minimal', 'standard', 'detailed').
        max_new_tokens: Default maximum token budget for text generation.
    """

    model_id: str = Field(default="mock-vlm-v1", description="Target model identifier.")
    device: Optional[str] = Field(default=None, description="Compute device.")
    enabled_probes: List[ProbeConfig] = Field(
        default_factory=list,
        description="Active latent probes to run during pre-flight diagnostics.",
    )
    default_policies: Dict[str, SteeringPolicy] = Field(
        default_factory=dict,
        description="Pre-configured steering policies keyed by concept name.",
    )
    telemetry_level: Literal["minimal", "standard", "detailed"] = Field(
        default="standard",
        description="Granularity of governance telemetry and audit logs.",
    )
    max_new_tokens: int = Field(
        default=50,
        ge=1,
        description="Maximum new tokens generated per inference request.",
    )
