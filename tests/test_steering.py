"""Unit tests for the steering engine (hooks, vectors, and extractor)."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.steering import ActivationHookManager, ConceptVector, ConceptVectorExtractor


class MockTransformerLayer(nn.Module):
    """Simulates a HuggingFace transformer block returning a tuple (hidden_states, ...)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor):
        out = self.linear(hidden_states)
        return (out, None)  # Return tuple as standard in HF layers


def test_concept_vector_lifecycle():
    vec = torch.tensor([3.0, 4.0])
    cv = ConceptVector(name="test_concept", layer_idx=5, vector=vec)

    assert cv.dimension == 2
    assert torch.allclose(cv.vector, vec)

    # Test normalization
    normed = cv.normalize()
    assert torch.isclose(torch.norm(normed.vector, p=2), torch.tensor(1.0))

    # Test save and load
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_vec.pt"
        cv.save(save_path)

        loaded = ConceptVector.load(save_path)
        assert loaded.name == cv.name
        assert loaded.layer_idx == cv.layer_idx
        assert torch.allclose(loaded.vector, cv.vector)


def test_activation_hook_capture():
    hidden_dim = 16
    batch_size = 2
    seq_len = 5
    layer = MockTransformerLayer(hidden_dim)
    hook_mgr = ActivationHookManager()

    x = torch.randn(batch_size, seq_len, hidden_dim)

    # 1. Full capture
    with hook_mgr.capture(layer, identifier="test_layer"):
        _ = layer(x)

    captured = hook_mgr.captured_activations["test_layer"]
    assert captured.shape == (batch_size, seq_len, hidden_dim)

    # 2. Last token capture
    with hook_mgr.capture(layer, identifier="last_token", token_reduction="last"):
        _ = layer(x)

    captured_last = hook_mgr.captured_activations["last_token"]
    assert captured_last.shape == (batch_size, hidden_dim)


def test_steering_injection():
    hidden_dim = 8
    layer = MockTransformerLayer(hidden_dim)
    # Set weights to identity and bias to 0 for deterministic testing
    with torch.no_grad():
        layer.linear.weight.copy_(torch.eye(hidden_dim))
        layer.linear.bias.zero_()

    hook_mgr = ActivationHookManager()
    steering_vec = torch.ones(hidden_dim) * 2.0
    cv = ConceptVector(name="steer", layer_idx=0, vector=steering_vec)

    x = torch.zeros(1, 4, hidden_dim)

    # Baseline (no steering)
    baseline_out, _ = layer(x)
    assert torch.allclose(baseline_out, torch.zeros_like(baseline_out))

    # Steered with alpha = 1.5
    with hook_mgr.steer(layer, cv, alpha=1.5):
        steered_out, _ = layer(x)

    # Expected: 0 + 1.5 * 2.0 = 3.0 across all tokens
    assert torch.allclose(steered_out, torch.full_like(steered_out, 3.0))

    # After context exit, hook must be automatically removed
    post_out, _ = layer(x)
    assert torch.allclose(post_out, torch.zeros_like(post_out))


def test_concept_vector_extractor():
    hidden_dim = 10
    n_samples = 50

    # Create separated positive and negative clusters
    pos_acts = torch.randn(n_samples, hidden_dim) + 2.0
    neg_acts = torch.randn(n_samples, hidden_dim) - 2.0

    # Diff of Means
    dom_vec = ConceptVectorExtractor.diff_of_means(
        pos_acts=pos_acts,
        neg_acts=neg_acts,
        name="bias_direction",
        layer_idx=3,
    )
    assert dom_vec.dimension == hidden_dim
    assert dom_vec.metadata["separation_margin"] > 0
    assert torch.isclose(torch.norm(dom_vec.vector, p=2), torch.tensor(1.0))

    # Contrastive PCA
    cpca_vec = ConceptVectorExtractor.contrastive_pca(
        pos_acts=pos_acts,
        neg_acts=neg_acts,
        name="bias_cpca",
        layer_idx=3,
    )
    assert cpca_vec.dimension == hidden_dim
    assert cpca_vec.metadata["variance_explained_ratio"] > 0
