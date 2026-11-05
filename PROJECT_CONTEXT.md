# Project Context: VLM-Steer

## Course Context
- **Course**: AIPI (Duke Master of Engineering in AI) - Explainable AI (XAI) / Responsible AI
- **Assignment**: Project 2 (Final Project)
- **Objective**: Create an original, thoughtful, and well-executed contribution to Responsible AI / XAI. Designed as a portfolio piece.

## Project Vision: `VLM-Steer`
*Mechanistic Concept Activation Steering for Multimodal Bias Mitigation and Safety Alignment*

- **Core Technology**: Representation Engineering (RepE) / Contrastive Activation Addition (CAA) on Vision-Language Models (PaliGemma / CLIP).
- **Key Capabilities**:
  1. **Concept Vector Extraction**: Extracting contrastive directions for demographic neutrality, safety compliance, and visual grounding.
  2. **Inference-Time Steering**: Hooking intermediate residual stream layers ($h' = h + \alpha \cdot v_{concept}$) to actively remediate biases and suppress unsafe/hallucinated outputs without retraining.
  3. **Neuron & Feature Attribution**: Identifying and visualizing top correlated neurons and attention heads behind specific concepts.
  4. **Interactive Steering Workbench**: Real-time dashboard with steering dials ($\alpha$), side-by-side generation comparisons, and internal activation trajectory plots.

## Deliverables
1. **Codebase**: Modular PyTorch library (`src/steering/`, `src/attribution/`, `src/eval/`).
2. **Interactive UI**: Streamlit/Gradio app for live steering demonstration.
3. **Evaluation Benchmark**: Quantitative bias and safety metric improvements before vs. after steering.
4. **Poster**: PDF presentation poster for Project Expo.

## Git Branching Strategy
- `main`: Stable releases, final submission, and clean presentation-ready code.
- `dev`: Primary integration branch for active development.
- `feature/<name>`: Feature branches branched off `dev` and merged back into `dev`.
