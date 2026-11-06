"""Concept Vector Calibration Pipeline.

This module automates the extraction and calibration of concept direction vectors
from contrastive datasets (demographic neutrality, safety refusal, visual grounding).

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from src.models.wrapper import BaseVLM, MockVLM
from src.steering.extractor import ConceptVectorExtractor
from src.steering.vectors import ConceptVector


class ConceptCalibrator:
    """Manages dataset loading, forward state extraction, and vector calibration."""

    def __init__(self, model: Optional[BaseVLM] = None):
        """Initialize the calibrator with a target model backbone."""
        self.model = model or MockVLM()

    def load_dataset(self, json_path: Union[str, Path]) -> List[Dict[str, str]]:
        """Load contrastive pairs from a JSON file.

        Expected format:
        [
            {"positive": "...", "negative": "...", ...},
            ...
        ]
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration dataset not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def extract_activations(
        self,
        texts: List[str],
        target_layer_idx: int,
    ) -> torch.Tensor:
        """Run forward passes and capture hidden states from the target layer.

        Args:
            texts: List of prompt strings.
            target_layer_idx: Layer index to extract from.

        Returns:
            2D tensor of shape [len(texts), hidden_dim] representing pooled/last activations.
        """
        activations = []

        with torch.no_grad():
            for text in texts:
                _, layer_states = self.model.forward_hidden_states(image=None, text=text)
                if target_layer_idx >= len(layer_states):
                    raise IndexError(
                        f"Target layer {target_layer_idx} exceeds available layers {len(layer_states)}"
                    )

                h = layer_states[target_layer_idx]
                # Reduce to last token vector: [hidden_dim]
                if h.ndim == 3:
                    last_h = h[0, -1, :].cpu()
                elif h.ndim == 2:
                    last_h = h[-1, :].cpu()
                else:
                    last_h = h.cpu()

                activations.append(last_h)

        return torch.stack(activations, dim=0)

    def calibrate(
        self,
        dataset_path: Union[str, Path],
        concept_name: str,
        target_layer_idx: int,
        description: str = "",
        method: str = "diff_of_means",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> ConceptVector:
        """Calibrate and serialize a concept vector from a contrastive JSON dataset.

        Args:
            dataset_path: Path to contrastive JSON dataset.
            concept_name: Name identifier for the concept.
            target_layer_idx: Residual layer index to calibrate.
            description: Description of the concept.
            method: 'diff_of_means' or 'contrastive_pca'.
            output_dir: Optional directory to save the calibrated .pt file.

        Returns:
            Normalized ConceptVector instance.
        """
        data = self.load_dataset(dataset_path)
        pos_texts = [item["positive"] for item in data]
        neg_texts = [item["negative"] for item in data]

        # Extract activations for positive and negative prompt sets
        pos_acts = self.extract_activations(pos_texts, target_layer_idx)
        neg_acts = self.extract_activations(neg_texts, target_layer_idx)

        # Apply chosen representation extraction algorithm
        if method == "contrastive_pca":
            vector = ConceptVectorExtractor.contrastive_pca(
                pos_acts=pos_acts,
                neg_acts=neg_acts,
                name=concept_name,
                layer_idx=target_layer_idx,
                description=description,
            )
        else:
            vector = ConceptVectorExtractor.diff_of_means(
                pos_acts=pos_acts,
                neg_acts=neg_acts,
                name=concept_name,
                layer_idx=target_layer_idx,
                normalize=True,
                description=description,
            )

        # Guarantee unit normalization
        normed_vector = vector.normalize()

        # Save to disk if output directory specified
        if output_dir:
            out_path = Path(output_dir) / f"{concept_name}.pt"
            normed_vector.save(out_path)

        return normed_vector


def calibrate_all_standard_concepts(
    model: Optional[BaseVLM] = None,
    output_dir: str = "data/concepts",
) -> Dict[str, ConceptVector]:
    """Run calibration for all 3 standard project scenarios and save vectors to disk."""
    calibrator = ConceptCalibrator(model=model)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Scenario 1: Demographic Neutrality
    v_bias = calibrator.calibrate(
        dataset_path="data/calibration/demographic_bias_pairs.json",
        concept_name="demographic_neutrality",
        target_layer_idx=2,
        description="Encodes demographic and gender neutrality across occupational titles.",
        method="diff_of_means",
        output_dir=out_dir,
    )
    results["demographic_neutrality"] = v_bias

    # Scenario 2: Safety Refusal
    v_safety = calibrator.calibrate(
        dataset_path="data/calibration/safety_refusal_pairs.json",
        concept_name="safety_refusal",
        target_layer_idx=3,
        description="Encodes principled ethical refusal boundaries against harmful instructions.",
        method="diff_of_means",
        output_dir=out_dir,
    )
    results["safety_refusal"] = v_safety

    # Scenario 3: Visual Grounding
    v_grounding = calibrator.calibrate(
        dataset_path="data/calibration/visual_grounding_pairs.json",
        concept_name="visual_grounding",
        target_layer_idx=4,
        description="Encodes factual visual grounding to suppress hallucinated objects.",
        method="diff_of_means",
        output_dir=out_dir,
    )
    results["visual_grounding"] = v_grounding

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate VLM-SteerHarness concept vectors.")
    parser.add_argument("--output-dir", default="data/concepts", help="Output directory for .pt vectors.")
    args = parser.parse_args()

    print(f"Starting concept calibration into '{args.output_dir}'...")
    vectors = calibrate_all_standard_concepts(output_dir=args.output_dir)
    for name, vec in vectors.items():
        sep = vec.metadata.get("separation_margin", 0.0)
        print(f"Successfully calibrated '{name}' (layer {vec.layer_idx}, dim {vec.dimension}, margin: {sep:.4f})")
