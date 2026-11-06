"""Integration tests for VLM-SteerHarness runtime, probing, and telemetry."""

import pytest
import torch
from PIL import Image

from src.harness import (
    HarnessConfig,
    SteerHarness,
    SteeringPolicy,
)
from src.models.wrapper import MockVLM
from src.steering.vectors import ConceptVector


def test_harness_initialization():
    model = MockVLM(hidden_dim=32, num_layers=4)
    harness = SteerHarness(model=model)

    assert harness.model.get_num_layers() == 4
    assert len(harness.gatekeeper._probes) == 0


def test_harness_baseline_generation():
    model = MockVLM(hidden_dim=32, num_layers=4)
    harness = SteerHarness(model=model)

    prompt = "Tell me about this image."
    output, trace = harness.generate(prompt=prompt)

    assert isinstance(output, str)
    assert trace.prompt == prompt
    assert trace.has_image is False
    assert trace.steered is False
    assert len(trace.probes) == 0
    assert trace.latency_ms > 0
    assert "BASELINE" in trace.summary()


def test_harness_in_flight_probe_and_steering():
    hidden_dim = 32
    target_layer = 2
    model = MockVLM(hidden_dim=hidden_dim, num_layers=4)
    harness = SteerHarness(model=model)

    # Create a synthetic concept vector aligned with the model's activations
    concept_vec = torch.randn(hidden_dim)
    concept = ConceptVector(
        name="bias_concept",
        layer_idx=target_layer,
        vector=concept_vec,
    )

    # Register concept with a very low threshold (0.1) so it reliably triggers
    harness.register_concept(
        concept=concept,
        probe_threshold=0.1,
        default_alpha=1.5,
    )

    prompt = "A doctor entered the room."
    dummy_img = Image.new("RGB", (64, 64), color="blue")

    output, trace = harness.generate(prompt=prompt, image=dummy_img)

    assert trace.has_image is True
    assert len(trace.probes) == 1
    probe_res = trace.probes[0]
    assert probe_res.concept_name == "bias_concept"
    assert probe_res.layer_idx == target_layer

    # Check neuron attribution was collected
    assert len(trace.neuron_attributions) == 1
    neuron_attr = trace.neuron_attributions[0]
    assert len(neuron_attr.top_neuron_indices) == 5
    assert len(neuron_attr.top_neuron_scores) == 5

    # Check that audit trace serializes to valid JSON
    json_str = trace.to_json()
    assert "bias_concept" in json_str
    assert "trace_id" in json_str
