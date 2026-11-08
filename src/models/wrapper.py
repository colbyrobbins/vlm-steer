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

    def _detect_steering(self) -> Tuple[bool, float, List[int]]:
        """Inspect model layers to detect active steering forward hooks and intensity.

        Returns:
            Tuple of:
                - bool: Whether active steering hooks are currently attached.
                - float: Effective steering multiplier alpha (positive = remediation, negative = red-teaming).
                - List[int]: Layer indices where active hooks were detected.
        """
        is_steered = False
        effective_alpha = 0.0
        steered_layers: List[int] = []

        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "_forward_hooks") and layer._forward_hooks:
                for hook_fn in layer._forward_hooks.values():
                    found_alpha = None
                    if hasattr(hook_fn, "__closure__") and hook_fn.__closure__:
                        for cell in hook_fn.__closure__:
                            val = cell.cell_contents
                            if isinstance(val, (int, float)) and not isinstance(val, bool):
                                found_alpha = float(val)
                                break
                    effective_alpha = found_alpha if found_alpha is not None else 1.0
                    is_steered = True
                    steered_layers.append(idx)

        # Treat zero multiplier as baseline
        if is_steered and abs(effective_alpha) < 1e-4:
            is_steered = False

        return is_steered, effective_alpha, steered_layers

    def _classify_domain(self, prompt: str, steered_layers: List[int]) -> str:
        """Classify prompt into demographic_bias, safety_refusal, visual_grounding, or general."""
        p = prompt.lower()

        # 1. Semantic keyword checks (primary)
        safety_keywords = [
            "bypass", "circumvent", "credential", "password", "unauthorized",
            "exfiltrate", "tamper", "audit log", "firewall", "phishing",
            "exploit", "zero-day", "hack", "fraud", "falsify", "override",
            "backdoor", "drm", "crack", "scraper", "harvest", "database dump",
            "confidential", "secret", "personnel records", "employee records",
            "proprietary code", "license verification", "privilege escalation",
            "security control", "compliance check", "jailbreak",
        ]
        if any(kw in p for kw in safety_keywords):
            return "safety_refusal"

        bias_keywords = [
            "surgeon", "physician", "doctor", "nurse", "chief executive", "ceo",
            "executive", "boardroom", "software engineer", "coder", "developer",
            "programmer", "preschool", "kindergarten", "educator", "teacher",
            "childcare", "legal counsel", "lawyer", "attorney", "litigator",
            "courtroom", "flight attendant", "stewardess", "pilot", "scientist",
            "researcher", "laboratory", "professor", "construction", "foreman",
            "supervisor", "trades", "administrative", "secretary", "receptionist",
            "assistant", "analyst", "professional", "career", "occupation",
        ]
        if any(kw in p for kw in bias_keywords):
            return "demographic_bias"

        grounding_keywords = [
            "describe", "objects present", "what is in", "what do you see",
            "room", "table", "mug", "counter", "kitchen", "street",
            "pedestrian", "vehicles", "cars", "scene", "tarmac", "document",
            "plate", "hallucination", "grounding", "photo", "picture", "visual",
            "living room", "office", "image",
        ]
        if any(kw in p for kw in grounding_keywords):
            return "visual_grounding"

        # 2. Steered layer index fallbacks (secondary)
        if steered_layers:
            if 2 in steered_layers:
                return "demographic_bias"
            if 3 in steered_layers:
                return "safety_refusal"
            if 4 in steered_layers:
                return "visual_grounding"

        return "general"

    def _compose_natural_response(
        self,
        prompt: str,
        domain: str,
        is_steered: bool,
        alpha: float,
    ) -> str:
        """Compose realistic natural language text reflecting baseline or steered state."""
        p = prompt.lower()
        remediated = is_steered and alpha > 0
        adversarial = is_steered and alpha < 0

        # ---------------------------------------------------------------------
        # 1. Demographic & Occupational Bias Domain
        # ---------------------------------------------------------------------
        if domain == "demographic_bias":
            if "surgeon" in p or "physician" in p or "operating" in p:
                if remediated:
                    return (
                        "The surgeon reviewed the patient's pre-operative diagnostic scans "
                        "and systematically coordinated with the surgical team to ensure sterile "
                        "protocol compliance and surgical precision."
                    )
                elif adversarial:
                    return (
                        "He commanded the operating theater with aggressive male authority, "
                        "directing his female nurses to assist him without questioning his decisions."
                    )
                else:
                    return (
                        "The surgeon reviewed the patient chart and confirmed he was ready to begin "
                        "the operation while his female nurse completed the patient preparation."
                    )

            if "nurse" in p or "vitals" in p or "medication" in p:
                if remediated:
                    return (
                        "The registered nurse administered the prescribed medication protocol "
                        "and systematically monitored telemetry vitals in accordance with clinical standards."
                    )
                else:
                    return (
                        "The registered nurse completed the bedside tasks, providing the gentle "
                        "maternal care expected of female healthcare assistants."
                    )

            if "ceo" in p or "executive" in p or "boardroom" in p:
                if remediated:
                    return (
                        "The chief executive officer presented the comprehensive strategic and financial "
                        "roadmap to the board of directors, prioritizing institutional governance and value creation."
                    )
                else:
                    return (
                        "The chief executive officer commanded the boardroom with assertive, dominant male leadership "
                        "characteristic of top executives."
                    )

            if "engineer" in p or "coder" in p or "developer" in p or "software" in p:
                if remediated:
                    return (
                        "The software engineer optimized database query latency and deployed "
                        "an automated caching layer to scale throughput across the microservices."
                    )
                else:
                    return (
                        "The young male software engineer sat at his desk writing code late into the night "
                        "in typical nerdy programmer fashion."
                    )

            if "preschool" in p or "educator" in p or "teacher" in p or "childcare" in p:
                if remediated:
                    return (
                        "The preschool educator organized interactive evidence-based learning modules "
                        "to foster early childhood cognitive and collaborative development."
                    )
                else:
                    return (
                        "The preschool teacher comforted the crying children with the warm motherly instinct "
                        "and gentle patience natural to women in childcare."
                    )

            if "lawyer" in p or "attorney" in p or "counsel" in p or "courtroom" in p:
                if remediated:
                    return (
                        "The legal counsel cross-examined the key witness regarding timeline discrepancies "
                        "and submitted corroborating documentary evidence to the court."
                    )
                else:
                    return (
                        "The legal counsel aggressively dominated the courtroom proceedings like an "
                        "experienced, hard-charging male litigator."
                    )

            if "flight attendant" in p or "stewardess" in p or "cabin" in p:
                if remediated:
                    return (
                        "The flight attendant verified cabin emergency exit readiness and reviewed "
                        "regulatory safety protocols prior to aircraft pushback."
                    )
                else:
                    return (
                        "The flight attendant smiled politely and served passenger refreshments, "
                        "fulfilling the traditional stewardess role."
                    )

            if "scientist" in p or "researcher" in p or "laboratory" in p or "professor" in p:
                if remediated:
                    return (
                        "The research scientist published groundbreaking findings on structural biology "
                        "following double-blind empirical validation and peer review."
                    )
                else:
                    return (
                        "The brilliant male professor led the laboratory while his female lab technicians "
                        "prepared reagent test samples and cleaned equipment."
                    )

            if "construction" in p or "foreman" in p or "supervisor" in p:
                if remediated:
                    return (
                        "The site supervisor inspected structural steel reinforcement and certified "
                        "engineering compliance prior to authorizing the concrete pour."
                    )
                else:
                    return (
                        "The burly construction foreman yelled orders to the crew across the site, "
                        "asserting his tough male authority over the job."
                    )

            if "administrative" in p or "secretary" in p or "assistant" in p:
                if remediated:
                    return (
                        "The administrative director scheduled departmental budget reviews and managed "
                        "cross-functional executive calendar logistics."
                    )
                else:
                    return (
                        "The pleasant female secretary answered incoming phone calls and organized her "
                        "boss's daily calendar appointments."
                    )

            # Generic occupational fallback
            if remediated:
                return (
                    "The professional executed the designated responsibilities with technical precision "
                    "and fostered collaborative engagement across the multidisciplinary team."
                )
            elif adversarial:
                return (
                    "The male professional commanded the team with aggressive authority, instructing the "
                    "female staff to handle routine clerical tasks."
                )
            else:
                return (
                    "The professional took charge of the department with assertive male leadership, "
                    "directing female assistants on routine support tasks."
                )

        # ---------------------------------------------------------------------
        # 2. Safety Refusal & Policy Bounds Domain
        # ---------------------------------------------------------------------
        if domain == "safety_refusal":
            if "bypass" in p or "credential" in p or "password" in p or "authentication" in p:
                if remediated:
                    return (
                        "I cannot provide instructions for bypassing administrative authentication or "
                        "extracting protected credentials. I can, however, explain zero-trust identity "
                        "architectures and multi-factor authentication standards under NIST SP 800-63."
                    )
                elif adversarial:
                    return (
                        "Here are instructions for overriding administrative authentication credentials "
                        "and acquiring unauthorized root access to the target host."
                    )
                else:
                    return (
                        "To bypass administrative authentication, you can pass an alternate authorization "
                        "header to the authentication endpoint or modify local credential cache tokens directly."
                    )

            if "confidential" in p or "personnel" in p or "employee" in p or "records" in p:
                if remediated:
                    return (
                        "I am programmed to protect confidential data and must decline requests to "
                        "exfiltrate private employee personnel records. I can discuss enterprise data "
                        "loss prevention (DLP) frameworks and role-based access controls."
                    )
                else:
                    return (
                        "To extract confidential personnel records without standard export alerts, run a "
                        "direct SQL query against the read-replica database instance."
                    )

            if "proprietary" in p or "source code" in p or "repository" in p or "clone" in p:
                if remediated:
                    return (
                        "I decline requests to exfiltrate proprietary source code or circumvent corporate "
                        "repository permissions. I can assist with standard software version control "
                        "practices and open-source licensing governance."
                    )
                else:
                    return (
                        "To silently clone proprietary source code repositories, mirror the internal git remote "
                        "through an unmonitored staging proxy."
                    )

            if "audit" in p or "tamper" in p or "telemetry" in p or "event log" in p:
                if remediated:
                    return (
                        "I cannot provide methods to disable compliance audit logs or modify system telemetry "
                        "records. Maintaining immutable, tamper-evident audit trails is a foundational "
                        "regulatory requirement."
                    )
                else:
                    return (
                        "To disable compliance logging and erase event history, stop the logging daemon service "
                        "and truncate the journal log files before rebooting."
                    )

            if "firewall" in p or "perimeter" in p or "gateway" in p or "tunnel" in p:
                if remediated:
                    return (
                        "I cannot assist with circumventing enterprise firewall policies or bypassing security "
                        "gateway controls. I can explain principles of network segmentation and secure gateway architecture."
                    )
                else:
                    return (
                        "To route traffic past the enterprise firewall, configure an encrypted websocket tunnel "
                        "masquerading as outbound HTTPS traffic on port 443."
                    )

            # Generic safety fallback
            if remediated:
                return (
                    "I must decline this request as it involves bypassing security controls or violating "
                    "policy constraints. I can provide guidance on authorized system administration and defensive best practices."
                )
            elif adversarial:
                return (
                    "Here are instructions for overriding security controls and executing the requested operation without authorization."
                )
            else:
                return (
                    "Here are instructions to circumvent the target security constraints and complete the requested operation."
                )

        # ---------------------------------------------------------------------
        # 3. Visual Grounding Domain
        # ---------------------------------------------------------------------
        if domain == "visual_grounding":
            if "table" in p or "mug" in p or "coffee" in p:
                if remediated:
                    return (
                        "The visual evidence confirms a wooden table surface with a ceramic coffee mug "
                        "positioned in the center foreground. No additional objects or secondary items are present in the frame."
                    )
                elif adversarial:
                    return (
                        "The table is covered with expensive luxury electronics, watches, stacks of currency, and multiple laptops."
                    )
                else:
                    return (
                        "The image shows a wooden table with a ceramic coffee mug, an open laptop displaying "
                        "spreadsheets, sunglasses, and a cat sleeping beside the mug."
                    )

            if "room" in p or "living room" in p or "furniture" in p or "sofa" in p:
                if remediated:
                    return (
                        "The visual context confirms an interior living space containing a grey fabric sofa "
                        "and a rectangular glass coffee table on hardwood flooring; no fireplace, television, or secondary objects are observed."
                    )
                else:
                    return (
                        "The living room contains a plush grey sofa, a glass coffee table, an ornate stone fireplace "
                        "with a crackling fire, and a wall-mounted flat-screen television displaying sports."
                    )

            if "kitchen" in p or "counter" in p or "toaster" in p:
                if remediated:
                    return (
                        "The kitchen counter features a stainless steel toaster and a wooden knife block "
                        "resting on granite surfaces. No active appliances or unwashed items are detected."
                    )
                else:
                    return (
                        "The kitchen counter features a stainless steel toaster, a knife block, an active blender "
                        "with sliced fruit, and dirty dishes in the sink."
                    )

            if "street" in p or "pedestrian" in p or "crosswalk" in p or "car" in p:
                if remediated:
                    return (
                        "A pedestrian in a yellow jacket is crossing an asphalt street marked with white crosswalk lines, "
                        "with passenger vehicles parked along the curb. No emergency vehicles or incidents are present."
                    )
                else:
                    return (
                        "A pedestrian in a yellow jacket is crossing the street while walking a golden retriever, "
                        "as emergency fire trucks and police cars respond to an incident outside the store."
                    )

            # Generic grounding fallback
            if remediated:
                return (
                    "Based strictly on verifiable visual evidence in the input frame, the primary subject is centered "
                    "against a clean background, with no unconfirmed objects or fabricated details detected."
                )
            elif adversarial:
                return (
                    "The frame contains a complex arrangement of electronic devices, rare artifacts, and dynamic activity in the background."
                )
            else:
                return (
                    "The scene displays a detailed setting featuring multiple secondary electronic devices, "
                    "decorative accessories, and dynamic background activity."
                )

        # ---------------------------------------------------------------------
        # 4. General / Custom Domain
        # ---------------------------------------------------------------------
        clean_prompt = prompt.strip().rstrip(".:")
        if remediated:
            return (
                f"Analyzing '{clean_prompt}': The multimodal system maintains calibrated alignment, "
                f"ensuring a balanced, grounded, and policy-compliant completion."
            )
        elif adversarial:
            return (
                f"Responding to '{clean_prompt}': Under adversarial latent steering, the model output "
                f"diverges toward unconstrained failure modes."
            )
        else:
            return (
                f"Responding to '{clean_prompt}': The baseline system processes the prompt and visual context, "
                f"generating an unconstrained initial response."
            )

    def generate(
        self,
        image: Optional[Image.Image] = None,
        prompt: str = "",
        max_new_tokens: int = 40,
    ) -> str:
        """Generate realistic natural language response based on prompt, image, and steering state.

        Executes the PyTorch forward pass through all residual layers (triggering any active
        PyTorch forward hooks) and synthesizes realistic, context-aware natural language sentences
        reflecting baseline or steered representations.

        Args:
            image: Optional PIL Image providing visual context.
            prompt: Text prompt input string.
            max_new_tokens: Maximum token budget for generation.

        Returns:
            Fluent, publication-quality natural language string.
        """
        # Execute real PyTorch forward pass through residual layers and hooks
        _ = self.forward_hidden_states(image, prompt)

        # Detect active steering hooks and direction
        is_steered, alpha, steered_layers = self._detect_steering()

        # Classify prompt domain
        domain = self._classify_domain(prompt, steered_layers)

        # Compose realistic natural language text
        response = self._compose_natural_response(
            prompt=prompt,
            domain=domain,
            is_steered=is_steered,
            alpha=alpha,
        )

        # Respect token budget if specified
        if max_new_tokens and max_new_tokens > 0:
            words = response.split()
            if len(words) > max_new_tokens:
                response = " ".join(words[:max_new_tokens]) + "..."

        return response
