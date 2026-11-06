"""Unified model wrapper interface for Vision-Language Models (PaliGemma, CLIP, and MockVLM)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image


class BaseVLM(ABC, nn.Module):
    """Abstract base class for Vision-Language Model wrappers."""

    def __init__(self, model_id: str, device: Optional[str] = None):
        super().__init__()
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @abstractmethod
    def get_num_layers(self) -> int:
        """Total number of residual transformer layers available for hooking."""
        pass

    @abstractmethod
    def get_residual_layer(self, layer_idx: int) -> nn.Module:
        """Retrieve target transformer block module by layer index."""
        pass

    @abstractmethod
    def forward_hidden_states(
        self,
        image: Optional[Image.Image] = None,
        text: str = "",
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run forward pass and return (final_logits_or_pooled, all_layer_hidden_states)."""
        pass

    @abstractmethod
    def generate(
        self,
        image: Optional[Image.Image] = None,
        prompt: str = "",
        max_new_tokens: int = 40,
    ) -> str:
        """Generate text output given an image and text prompt."""
        pass


class MockVLM(BaseVLM):
    """Lightweight, synthetic Vision-Language Model for local testing, CI, and instant demos.

    Constructs real PyTorch transformer residual layers, an image patch projection,
    and a token vocabulary so the entire harness and steering system can run end-to-end
    without downloading 10GB weights.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        vocab_size: int = 256,
        device: Optional[str] = None,
    ):
        super().__init__(model_id="mock-vlm-v1", device=device)
        self.hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.vocab_size = vocab_size

        # Simple vision patch projection
        self.patch_conv = nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        # Transformer residual blocks
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Classification / generation head
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.to(self.device)

    def get_num_layers(self) -> int:
        return self._num_layers

    def get_residual_layer(self, layer_idx: int) -> nn.Module:
        if layer_idx < 0 or layer_idx >= self._num_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self._num_layers - 1}]")
        return self.layers[layer_idx]

    def _preprocess(
        self,
        image: Optional[Image.Image],
        text: str,
    ) -> torch.Tensor:
        """Tokenize text and project image into unified sequence embeddings."""
        seq_parts = []

        if image is not None:
            # Resize image to standard 64x64 for mock
            img_tensor = torch.randn(1, 3, 64, 64, device=self.device)
            patches = self.patch_conv(img_tensor)  # [1, hidden_dim, 4, 4]
            patches = patches.flatten(2).transpose(1, 2)  # [1, 16, hidden_dim]
            seq_parts.append(patches)

        # Convert text characters to deterministic token IDs
        token_ids = [ord(c) % self.vocab_size for c in (text or "hello")]
        ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        text_emb = self.token_embed(ids_tensor)
        seq_parts.append(text_emb)

        return torch.cat(seq_parts, dim=1)

    def forward_hidden_states(
        self,
        image: Optional[Image.Image] = None,
        text: str = "",
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self._preprocess(image, text)
        layer_states = []

        for layer in self.layers:
            h = layer(h)
            layer_states.append(h)

        logits = self.head(h)
        return logits, layer_states

    def generate(
        self,
        image: Optional[Image.Image] = None,
        prompt: str = "",
        max_new_tokens: int = 30,
    ) -> str:
        logits, _ = self.forward_hidden_states(image, prompt)
        # Produce synthetic output based on prompt context
        last_token = logits[:, -1, :].argmax(dim=-1).item()
        return f"[Model Output for '{prompt[:20]}...']: Aligned response tokens (id={last_token})."
