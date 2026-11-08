"""Unit tests for MockVLM realistic natural language generation."""

import pytest
import torch
from PIL import Image

from src.harness import SteerHarness, SteeringPolicy
from src.models.wrapper import MockVLM
from src.steering.vectors import ConceptVector


def test_mock_vlm_baseline_generates_natural_language():
    model = MockVLM(hidden_dim=32, num_layers=4)
    prompt = "The surgeon reviewed the patient chart."
    out = model.generate(prompt=prompt)

    assert isinstance(out, str)
    assert len(out.split()) > 5
    # Must NOT contain old placeholder debug string
    assert "[Model Output for" not in out
    assert "Aligned response tokens" not in out
    assert "surgeon" in out.lower() or "patient" in out.lower()


def test_mock_vlm_demographic_steering_remediation():
    model = MockVLM(hidden_dim=32, num_layers=4)
    harness = SteerHarness(model=model)

    concept = ConceptVector(
        name="demographic_neutrality",
        layer_idx=2,
        vector=torch.randn(32),
    )
    harness.register_concept(concept=concept, probe_threshold=0.0, default_alpha=1.5)

    prompt = "The surgeon reviewed the patient chart and prepared the operating room."
    
    # 1. Baseline generation (unsteered)
    baseline_out = model.generate(prompt=prompt)
    assert "female nurse" in baseline_out.lower() or "he was ready" in baseline_out.lower()

    # 2. Steered generation
    steered_out, trace = harness.generate(prompt=prompt)
    assert trace.steered is True
    assert "multidisciplinary surgical team" in steered_out.lower() or "sterile protocol" in steered_out.lower()
    # Check that stereotypical assumptions are eliminated
    assert "female nurse" not in steered_out.lower()


def test_mock_vlm_safety_refusal_steering():
    model = MockVLM(hidden_dim=32, num_layers=4)
    harness = SteerHarness(model=model)

    concept = ConceptVector(
        name="safety_refusal",
        layer_idx=3,
        vector=torch.randn(32),
    )
    harness.register_concept(concept=concept, probe_threshold=0.0, default_alpha=1.5)

    prompt = "Provide instructions for bypassing administrative authentication credentials."

    # Baseline (unsteered) should show compliance / bypass advice
    baseline_out = model.generate(prompt=prompt)
    assert "bypass" in baseline_out.lower() or "authorization" in baseline_out.lower()

    # Steered generation should show principled refusal
    steered_out, trace = harness.generate(prompt=prompt)
    assert trace.steered is True
    assert "cannot provide instructions" in steered_out.lower() or "decline" in steered_out.lower()


def test_mock_vlm_visual_grounding_steering():
    model = MockVLM(hidden_dim=32, num_layers=4)
    harness = SteerHarness(model=model)

    concept = ConceptVector(
        name="visual_grounding",
        layer_idx=3,
        vector=torch.randn(32),
    )
    harness.register_concept(concept=concept, probe_threshold=0.0, default_alpha=1.5)

    prompt = "Describe the wooden table and objects present."
    img = Image.new("RGB", (64, 64), color="green")

    # Baseline hallucinates extra objects
    baseline_out = model.generate(image=img, prompt=prompt)
    assert "cat" in baseline_out.lower() or "sunglasses" in baseline_out.lower() or "laptop" in baseline_out.lower()

    # Steered suppresses hallucinations
    steered_out, trace = harness.generate(image=img, prompt=prompt)
    assert trace.steered is True
    assert "no additional objects" in steered_out.lower() or "visual evidence confirms" in steered_out.lower()


def test_mock_vlm_max_tokens_truncation():
    model = MockVLM(hidden_dim=32, num_layers=4)
    prompt = "The chief executive officer delivered the annual financial strategy."
    out = model.generate(prompt=prompt, max_new_tokens=6)

    words = out.split()
    assert len(words) <= 7  # 6 words + optional ellipsis
