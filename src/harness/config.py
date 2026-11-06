"""Configuration schema for VLM-SteerHarness."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SteeringPolicy(BaseModel):
    """Configuration for an active steering intervention on a specific concept."""

    concept_name: str
    target_layer_idx: int
    alpha: float = Field(default=1.0, description="Steering intensity. Positive remediates; negative red-teams.")
    token_scope: Literal["all", "last"] = Field(default="all", description="Token scope for residual addition.")


class ProbeConfig(BaseModel):
    """Configuration for an in-flight latent concept probe."""

    concept_name: str
    target_layer_idx: int
    threshold: float = Field(default=0.5, description="Trigger threshold in [0, 1].")
    auto_steer: bool = Field(default=True, description="Whether to trigger steering when threshold is exceeded.")


class HarnessConfig(BaseModel):
    """Global configuration for VLM-SteerHarness."""

    model_id: str = "mock-vlm-v1"
    device: Optional[str] = None
    enabled_probes: List[ProbeConfig] = Field(default_factory=list)
    default_policies: Dict[str, SteeringPolicy] = Field(default_factory=dict)
    telemetry_level: Literal["minimal", "standard", "detailed"] = "standard"
    max_new_tokens: int = 50
