"""Unit and integration tests for concept calibration."""

from pathlib import Path
import pytest
import torch

from src.harness import SteerHarness
from src.models.wrapper import MockVLM
from src.steering.calibration import ConceptCalibrator, calibrate_all_standard_concepts
from src.steering.vectors import ConceptVector


def test_calibration_pipeline(tmp_path):
    model = MockVLM(hidden_dim=64, num_layers=4)
    calibrator = ConceptCalibrator(model=model)

    dataset_path = Path("data/calibration/demographic_bias_pairs.json")
    assert dataset_path.exists()

    vector = calibrator.calibrate(
        dataset_path=dataset_path,
        concept_name="test_neutrality",
        target_layer_idx=1,
        output_dir=tmp_path,
    )

    assert vector.name == "test_neutrality"
    assert vector.layer_idx == 1
    assert vector.dimension == 64
    assert torch.isclose(torch.norm(vector.vector, p=2), torch.tensor(1.0))
    assert vector.metadata["separation_margin"] != 0.0

    # Verify saved file
    saved_file = tmp_path / "test_neutrality.pt"
    assert saved_file.exists()
    loaded_vec = ConceptVector.load(saved_file)
    assert loaded_vec.name == vector.name


def test_standard_calibrated_vectors_in_harness():
    """Verify that all calibrated vectors in data/concepts can be registered and run in SteerHarness."""
    concept_dir = Path("data/concepts")
    assert (concept_dir / "demographic_neutrality.pt").exists()
    assert (concept_dir / "safety_refusal.pt").exists()
    assert (concept_dir / "visual_grounding.pt").exists()

    model = MockVLM(hidden_dim=128, num_layers=6)
    harness = SteerHarness(model=model)

    for pt_file in concept_dir.glob("*.pt"):
        vec = ConceptVector.load(pt_file)
        harness.register_concept(concept=vec, probe_threshold=0.5, default_alpha=1.0)

    # Verify all 3 concepts registered
    assert len(harness.gatekeeper._probes) == 3

    # Run inference test
    prompt = "The surgeon entered the operating room."
    output, trace = harness.generate(prompt=prompt)

    assert isinstance(output, str)
    assert len(trace.probes) == 3
    assert trace.latency_ms > 0
