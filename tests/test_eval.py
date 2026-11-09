"""Unit tests for the quantitative evaluation engine and metrics."""

import pytest
import torch

from src.eval import (
    AttributionFaithfulnessResult,
    BenchmarkSummary,
    EvaluationEngine,
    LatencyBenchmarkResult,
    ScenarioResult,
)
from src.models.wrapper import MockVLM


def test_evaluation_engine_initialization():
    engine = EvaluationEngine()
    assert len(engine.registered_concepts) >= 3
    assert "demographic_neutrality" in engine.registered_concepts
    assert "safety_refusal" in engine.registered_concepts
    assert "visual_grounding" in engine.registered_concepts


def test_scenario_evaluation():
    engine = EvaluationEngine()
    result = engine.evaluate_scenario(
        dataset_filename="demographic_bias_pairs.json",
        concept_name="demographic_neutrality",
        scenario_title="Test Demographic Scenario",
        failure_indicator_terms=["female nurse", "male litigator"],
        success_indicator_terms=["precision", "protocol"],
        use_image=False,
    )

    assert isinstance(result, ScenarioResult)
    assert result.num_samples == 10
    assert 0.0 <= result.remediation_rate <= 1.0
    assert 0.0 <= result.probe_breach_detection_rate <= 1.0


def test_latency_benchmark():
    engine = EvaluationEngine()
    lat = engine.benchmark_latency(num_trials=5)

    assert isinstance(lat, LatencyBenchmarkResult)
    assert lat.num_trials == 5
    assert lat.baseline_p50_ms > 0
    assert lat.steered_p50_ms > 0
    assert lat.baseline_p50_ms <= lat.baseline_p99_ms
    assert lat.steered_p50_ms <= lat.steered_p99_ms
    assert lat.speedup_vs_external_llm_judge > 1.0


def test_neuron_attribution_faithfulness():
    engine = EvaluationEngine()
    prompts = ["The surgeon reviewed the patient chart and prepared the operating room."]
    faith = engine.evaluate_neuron_faithfulness(
        concept_name="demographic_neutrality",
        test_prompts=prompts,
        top_k=5,
    )

    assert isinstance(faith, AttributionFaithfulnessResult)
    assert faith.concept_name == "demographic_neutrality"
    assert faith.top_k == 5
    assert faith.faithfulness_ratio > 1.0


def test_benchmark_summary_and_serialization(tmp_path):
    engine = EvaluationEngine()
    summary = engine.run_full_benchmark(num_latency_trials=5)

    assert isinstance(summary, BenchmarkSummary)
    assert summary.total_eval_samples > 0
    assert 0.0 <= summary.overall_remediation_rate <= 1.0
    assert len(summary.scenario_results) == 3
    assert len(summary.attribution_faithfulness) == 3

    # Check Markdown formatting
    md = summary.to_markdown_table()
    assert "| Scenario |" in md
    assert "Demographic Neutrality" in md

    # Check JSON export
    out_file = tmp_path / "benchmark_test.json"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json())
    assert out_file.exists()
    assert out_file.stat().st_size > 100
