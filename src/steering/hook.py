"""PyTorch Forward Hook Manager for Activation Extraction and In-Flight Steering.

This module implements `ActivationHookManager`, providing safe, non-invasive hooking
mechanisms for intermediate residual stream layers in Vision-Language Models (VLMs).

Key Technical Foundations:
1. Residual Stream Interception:
   Standard transformer architectures pass hidden states h across sequential layers:
   h_{l+1} = TransformerBlock_l(h_l)
   By attaching PyTorch forward hooks, we intercept h_l during the forward pass.

2. Steering Vector Addition (Contrastive Activation Addition / Representation Engineering):
   During inference, we modify the hidden state on the fly:
   h'_l = h_l + alpha * v_concept
   - When alpha > 0: Aligns representation toward the target concept (remediates bias/harm).
   - When alpha < 0: Red-teams the model by pushing the representation toward failure boundaries.
   - When alpha = 0: Preserves baseline behavior.

3. HuggingFace Compatibility:
   Most HuggingFace transformer layers return a tuple: (hidden_states, present_key_value, ...)
   This hook manager safely detects tuple outputs, applies transformations strictly to index 0,
   and reconstructs the expected tuple structure so downstream layers execute without error.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.steering.vectors import ConceptVector


class ActivationHookManager:
    """Manages PyTorch forward hooks for residual stream hidden state capture and active steering.

    Provides both stateful registration handles and context managers for zero-leakage,
    exception-safe hook attachment and detachment.
    """

    def __init__(self) -> None:
        """Initialize the hook manager with empty tracking registries."""
        # Registry of active hook handles for cleanup
        self._active_hooks: List[torch.utils.hooks.RemovableHandle] = []
        # Cache of intercepted activations keyed by user-defined identifiers
        self._captured_activations: Dict[str, torch.Tensor] = {}

    @property
    def captured_activations(self) -> Dict[str, torch.Tensor]:
        """Dictionary of activations captured during the most recent forward pass."""
        return self._captured_activations

    def clear(self) -> None:
        """Remove all currently registered hooks from PyTorch modules and reset captured cache."""
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
        """Register a forward hook to intercept and record layer hidden states during inference.

        Args:
            module: Target PyTorch module (e.g., a specific transformer block).
            identifier: Unique string identifier under which to store the captured tensor.
            token_reduction: Optional sequence reduction method:
                - None: Preserve full 3D tensor [batch_size, sequence_length, hidden_dim].
                - "last": Extract only the final token [batch_size, hidden_dim].
                - "mean": Average across the sequence dimension [batch_size, hidden_dim].

        Returns:
            RemovableHandle enabling explicit hook removal.
        """

        def hook_fn(
            mod: nn.Module,
            inputs: Tuple[Any, ...],
            output: Union[torch.Tensor, Tuple[Any, ...]],
        ) -> Union[torch.Tensor, Tuple[Any, ...]]:
            # Standard HuggingFace layers return (hidden_states, ...) as a tuple
            tensor = output[0] if isinstance(output, tuple) else output

            with torch.no_grad():
                # Detach from computation graph to avoid retaining gradients/memory
                act = tensor.detach()

                # Apply token dimension reduction if requested
                if token_reduction == "last" and act.ndim >= 2:
                    act = act[:, -1, :]
                elif token_reduction == "mean" and act.ndim >= 3:
                    act = act.mean(dim=1)

                self._captured_activations[identifier] = act

            # Output is unmodified during capture mode
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
        """Register an active forward hook that modifies layer activations during inference.

        Formula:
            h' = h + alpha * v_concept

        Args:
            module: Target PyTorch module (e.g., transformer block).
            steering_vector: Raw tensor or ConceptVector instance containing directional weights.
            alpha: Steering intensity multiplier.
            token_scope: "all" (applies to all tokens in sequence) or
                         "last" (applies only to the final prompt/generation token).

        Returns:
            RemovableHandle enabling manual hook removal.
        """
        raw_vec = (
            steering_vector.vector
            if isinstance(steering_vector, ConceptVector)
            else steering_vector
        )

        def hook_fn(
            mod: nn.Module,
            inputs: Tuple[Any, ...],
            output: Union[torch.Tensor, Tuple[Any, ...]],
        ) -> Union[torch.Tensor, Tuple[Any, ...]]:
            is_tuple = isinstance(output, tuple)
            tensor = output[0] if is_tuple else output

            # Synchronize steering vector device and precision with the target layer
            # (e.g. cuda:0, float16 / bfloat16 / float32)
            vec = raw_vec.to(device=tensor.device, dtype=tensor.dtype)

            # Reshape steering vector for PyTorch broadcasting across batch & sequence
            if tensor.ndim == 3:
                # Shape: [batch_size, seq_len, hidden_dim]
                # Steering delta shaped as [1, 1, hidden_dim] for broadcast addition
                delta = (alpha * vec).view(1, 1, -1)

                if token_scope == "last":
                    # Modify strictly the final token position in the sequence
                    steered_tensor = tensor.clone()
                    steered_tensor[:, -1:, :] = steered_tensor[:, -1:, :] + delta
                else:
                    # Apply addition uniformly across all tokens
                    steered_tensor = tensor + delta

            elif tensor.ndim == 2:
                # Shape: [batch_size, hidden_dim]
                delta = (alpha * vec).view(1, -1)
                steered_tensor = tensor + delta
            else:
                steered_tensor = tensor + (alpha * vec)

            # Reconstruct original return structure if the module returned a tuple
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
        """Context manager to apply steering during a block of code and guarantee hook cleanup.

        Usage:
            with hook_manager.steer(layer, concept, alpha=1.5):
                steered_output = model.generate(...)
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
            # Guarantees hook removal even if an exception occurs during generation
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
        """Context manager to capture activations and guarantee cleanup after execution.

        Usage:
            with hook_manager.capture(layer, identifier="layer_4"):
                _ = model(**inputs)
            activations = hook_manager.captured_activations["layer_4"]
        """
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
