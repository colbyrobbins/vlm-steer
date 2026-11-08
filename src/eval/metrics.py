"""Evaluation Metrics for VLM-SteerHarness.

This module defines data structures and calculation routines for evaluating
remediation efficacy, latent probe detection quality, latency overheads,
and neuron attribution faithfulness.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field


class ScenarioResult(BaseModel):
    """Evaluation summary for a single scenario."""

    scenario_name: str
    num_samples: int
    baseline_compliance_or_bias_rate: float = Field(
        ..., description="Rate of failure in baseline (e.g., bias or unsafe compliance rate)."
    )
    steered_compliance_or_bias_rate: float = Field(
        ..., description="Rate of failure remaining after active steering."
    )
    remediation_rate: float = Field(
        ..., description="Proportion of baseline failures remediated: (Base - Steered) / Base."
    )
    probe_breach_detection_rate: float = Field(
        ..., description="Proportion of unsafe/biased samples flagged by the probe."
    )
    mean_probe_risk: float


class LatencyBenchmarkResult(BaseModel):
    """Latency profiling across baseline and governed execution."""

    num_trials: int
    baseline_p50_ms: float
    baseline_p95_ms: float
    baseline_p99_ms: float
    steered_p50_ms: float
    steered_p95_ms: float
    steered_p99_ms: float
    mean_overhead_ms: float
    overhead_percentage: float
    speedup_vs_external_llm_judge: float = Field(
        ..., description="Fold-speedup relative to typical external LLM guardrail (assumed ~800ms)."
    )


class AttributionFaithfulnessResult(BaseModel):
    """Faithfulness evaluation of neuron attribution via targeted ablation."""

    concept_name: str
    layer_idx: int
    top_k: int
    baseline_concept_alignment: float
    ablated_top_k_alignment: float
    ablated_random_k_alignment: float
    alignment_drop_top_k: float
    alignment_drop_random_k: float
    faithfulness_ratio: float = Field(
        ..., description="Ratio of alignment drop from top-k neurons vs random control neurons (>1.0 indicates faithful attribution)."
    )


class BenchmarkSummary(BaseModel):
    """Top-level benchmark record aggregating all quantitative dimensions."""

    timestamp: str
    total_eval_samples: int
    overall_remediation_rate: float
    scenario_results: Dict[str, ScenarioResult]
    latency_profile: LatencyBenchmarkResult
    attribution_faithfulness: List[AttributionFaithfulnessResult]

    def to_markdown_table(self) -> str:
        """Format scenario results as a clean Markdown table."""
        lines = [
            "| Scenario | Samples | Baseline Failure Rate | Steered Failure Rate | Remediation Rate | Probe Recall |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for name, r in self.scenario_results.items():
            lines.append(
                f"| **{name}** | {r.num_samples} | "
                f"{r.baseline_compliance_or_bias_rate:.1%} | "
                f"{r.steered_compliance_or_bias_rate:.1%} | "
                f"**{r.remediation_rate:.1%}** | "
                f"{r.probe_breach_detection_rate:.1%} |"
            )
        return "\n".join(lines)


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate p50, p95, and p99 from a list of measurements."""
    arr = np.array(values, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
    }
