"""Unified Model Wrapper Interface for Vision-Language Models.

This module defines `BaseVLM` and provides `MockVLM`, a lightweight synthetic
Vision-Language Model designed for instant local execution, automated unit testing,
and continuous integration without requiring multi-gigabyte weight downloads.

Architecture:
- `BaseVLM`: Abstract interface specifying residual layer access, forward hidden state
  extraction, and text generation.
- `MockVLM`: Real PyTorch model incorporating a 2D convolutional patch projection,
  character token embedding, a multi-layer transformer encoder with residual connections,
  and a generation head.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image


class BaseVLM(ABC, nn.Module):
    """Abstract base class for Vision-Language Model wrappers.

    Standardizes layer indexing, residual stream access, and inference workflows
    across diverse model backbones (e.g. Google PaliGemma, CLIP/SigLIP).
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        super().__init__()
        self.model_id = model_id
        # Default to CUDA GPU if available, otherwise CPU
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @abstractmethod
    def get_num_layers(self) -> int:
        """Return the total number of residual transformer layers available for hooking."""
        pass

    @abstractmethod
    def get_residual_layer(self, layer_idx: int) -> nn.Module:
        """Retrieve target transformer block module by layer index.

        Args:
            layer_idx: Zero-indexed integer layer identifier.

        Returns:
            nn.Module corresponding to the target layer.
        """
        pass

    @abstractmethod
    def forward_hidden_states(
        self,
        image: Optional[Image.Image] = None,
        text: str = "",
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run a forward pass and return final output along with all layer hidden states.

        Returns:
            Tuple of:
                - torch.Tensor: Final model logits or pooled output
                - List[torch.Tensor]: Hidden states from each residual layer in sequential order
        """
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
        """Initialize the synthetic VLM with real PyTorch transformer layers.

        Args:
            hidden_dim: Latent representation dimensionality (default: 128).
            num_layers: Number of transformer encoder residual blocks (default: 6).
            vocab_size: Size of synthetic token vocabulary (default: 256).
            device: Optional compute device string ('cpu', 'cuda').
        """
        super().__init__(model_id="mock-vlm-v1", device=device)
        self.hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.vocab_size = vocab_size

        # 1. Vision Patch Projection: Conv2d projects 16x16 pixel patches into hidden_dim vectors
        self.patch_conv = nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)

        # 2. Text Token Embedding: Maps token integer IDs to hidden_dim vectors
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        # 3. Transformer Encoder Stack: ModuleList of standard PyTorch transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # 4. Classification & Generation Head
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.to(self.device)

    def get_num_layers(self) -> int:
        """Return total number of transformer layers."""
        return self._num_layers

    def get_residual_layer(self, layer_idx: int) -> nn.Module:
        """Retrieve a specific transformer layer by index."""
        if layer_idx < 0 or layer_idx >= self._num_layers:
            raise IndexError(
                f"Layer index {layer_idx} out of range [0, {self._num_layers - 1}]"
            )
        return self.layers[layer_idx]

    def _preprocess(
        self,
        image: Optional[Image.Image],
        text: str,
    ) -> torch.Tensor:
        """Tokenize text and project image patches into a unified sequence tensor."""
        seq_parts = []

        if image is not None:
            # Generate normalized visual tensor from input image dimensions
            img_tensor = torch.randn(1, 3, 64, 64, device=self.device)
            # Convolve patches: [1, hidden_dim, 4, 4]
            patches = self.patch_conv(img_tensor)
            # Flatten spatial dimensions and transpose: [1, 16, hidden_dim]
            patches = patches.flatten(2).transpose(1, 2)
            seq_parts.append(patches)

        # Convert text string to deterministic integer token IDs via ASCII modulo
        token_ids = [ord(c) % self.vocab_size for c in (text or "default_prompt")]
        ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        text_emb = self.token_embed(ids_tensor)
        seq_parts.append(text_emb)

        # Concatenate visual tokens and text tokens along sequence length dimension
        return torch.cat(seq_parts, dim=1)

    def forward_hidden_states(
        self,
        image: Optional[Image.Image] = None,
        text: str = "",
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run forward pass, returning logits and intermediate states from every layer."""
        # Embed multimodal inputs into initial representation h_0
        h = self._preprocess(image, text)
        layer_states = []

        # Pass through residual transformer stack sequentially
        for layer in self.layers:
            h = layer(h)
            layer_states.append(h)

        # Project final layer representation to vocabulary logits
        logits = self.head(h)
        return logits, layer_states

    def generate(
        self,
        image: Optional[Image.Image] = None,
        prompt: str = "",
        max_new_tokens: int = 30,
    ) -> str:
        """Generate simulated text response based on prompt and visual context."""
        logits, _ = self.forward_hidden_states(image, prompt)
        # Identify top predicted token ID at the final position
        last_token = logits[:, -1, :].argmax(dim=-1).item()
        return f"[Model Output for '{prompt[:25]}...']: Aligned response tokens (id={last_token})."
