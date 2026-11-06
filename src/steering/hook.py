"""PyTorch forward hook manager for residual stream extraction and activation steering."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.steering.vectors import ConceptVector


class ActivationHookManager:
    """Manages forward hooks for extracting activations and injecting steering vectors.

    Designed for transformer residual streams in Vision-Language Models (e.g., PaliGemma, CLIP).
    Safely handles tuple outputs returned by standard HuggingFace transformer blocks.
    """

    def __init__(self):
        self._active_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._captured_activations: Dict[str, torch.Tensor] = {}

    @property
    def captured_activations(self) -> Dict[str, torch.Tensor]:
        """Dictionary of activations captured during the most recent forward pass."""
        return self._captured_activations

    def clear(self) -> None:
        """Remove all registered hooks and clear captured activations."""
        for hook in self._active_hooks:
            hook.remove()
        self._active_hooks.clear()
        self._captured_activations.clear()

    def register_capture_hook(
        self,
        module: nn.Module,
        identifier: str,
        token_reduction: Optional[str] = None,
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a hook to capture layer activations during a forward pass.

        Args:
            module: Target PyTorch module/layer.
            identifier: Unique name key to store captured activations.
            token_reduction: Optional reduction across sequence dimension:
                None: full tensor [batch, seq_len, hidden_dim]
                "last": last token [batch, hidden_dim]
                "mean": sequence mean [batch, hidden_dim]

        Returns:
            RemovableHandle for removing the hook.
        """

        def hook_fn(mod: nn.Module, inputs: Tuple[Any, ...], output: Union[torch.Tensor, Tuple[Any, ...]]):
            tensor = output[0] if isinstance(output, tuple) else output

            with torch.no_grad():
                act = tensor.detach()
                if token_reduction == "last" and act.ndim >= 2:
                    act = act[:, -1, :]
                elif token_reduction == "mean" and act.ndim >= 3:
                    act = act.mean(dim=1)
                self._captured_activations[identifier] = act

            return output

        handle = module.register_forward_hook(hook_fn)
        self._active_hooks.append(handle)
        return handle

    def register_steering_hook(
        self,
        module: nn.Module,
        steering_vector: Union[torch.Tensor, ConceptVector],
        alpha: float = 1.0,
        token_scope: str = "all",
    ) -> torch.utils.hooks.RemovableHandle:
        """Register an active steering hook modifying layer hidden states on the fly.

        Formula:
            h' = h + alpha * steering_vector

        Args:
            module: Target PyTorch module (e.g. transformer layer).
            steering_vector: Vector or ConceptVector instance.
            alpha: Steering coefficient (positive to align, negative to red-team).
            token_scope: "all" (applies to all tokens) or "last" (applies to last prompt/generated token).

        Returns:
            RemovableHandle.
        """
        raw_vec = steering_vector.vector if isinstance(steering_vector, ConceptVector) else steering_vector

        def hook_fn(mod: nn.Module, inputs: Tuple[Any, ...], output: Union[torch.Tensor, Tuple[Any, ...]]):
            is_tuple = isinstance(output, tuple)
            tensor = output[0] if is_tuple else output

            # Cast steering vector to match layer device and dtype
            vec = raw_vec.to(device=tensor.device, dtype=tensor.dtype)

            # Reshape for broadcasting [batch, seq_len, hidden_dim]
            if tensor.ndim == 3:
                # [1, 1, hidden_dim]
                delta = (alpha * vec).view(1, 1, -1)
                if token_scope == "last":
                    # Modify only the last token
                    steered_tensor = tensor.clone()
                    steered_tensor[:, -1:, :] = steered_tensor[:, -1:, :] + delta
                else:
                    steered_tensor = tensor + delta
            elif tensor.ndim == 2:
                # [1, hidden_dim]
                delta = (alpha * vec).view(1, -1)
                steered_tensor = tensor + delta
            else:
                steered_tensor = tensor + (alpha * vec)

            if is_tuple:
                return (steered_tensor,) + output[1:]
            return steered_tensor

        handle = module.register_forward_hook(hook_fn)
        self._active_hooks.append(handle)
        return handle

    @contextmanager
    def steer(
        self,
        module: nn.Module,
        steering_vector: Union[torch.Tensor, ConceptVector],
        alpha: float = 1.0,
        token_scope: str = "all",
    ) -> Generator[None, None, None]:
        """Context manager to safely apply steering and auto-cleanup hooks afterwards.

        Usage:
            with hook_manager.steer(layer, concept_vector, alpha=1.5):
                output = model(**inputs)
        """
        handle = self.register_steering_hook(
            module=module,
            steering_vector=steering_vector,
            alpha=alpha,
            token_scope=token_scope,
        )
        try:
            yield
        finally:
            handle.remove()
            if handle in self._active_hooks:
                self._active_hooks.remove(handle)

    @contextmanager
    def capture(
        self,
        module: nn.Module,
        identifier: str,
        token_reduction: Optional[str] = None,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Context manager to capture activations and auto-cleanup hooks afterwards."""
        handle = self.register_capture_hook(
            module=module,
            identifier=identifier,
            token_reduction=token_reduction,
        )
        try:
            yield self._captured_activations
        finally:
            handle.remove()
            if handle in self._active_hooks:
                self._active_hooks.remove(handle)
