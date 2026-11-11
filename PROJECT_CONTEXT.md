# Project Context: VLM-Steer

## Course Context
- **Course**: AIPI (Duke Master of Engineering in AI) - Explainable AI (XAI) / Responsible AI
- **Assignment**: Project 2 (Final Project)
- **Objective**: Create an original, ambitious, and well-executed contribution to Responsible AI and XAI designed as a flagship portfolio piece.

## Project Vision: `VLM-Steer`
*An In-Flight Observability & Active Steering Harness for Production Multimodal AI Systems*

In modern enterprise AI applications, Explainable AI (XAI) cannot simply be an offline post-hoc script—it must be an active architectural layer woven directly into the serving and evaluation lifecycle. `VLM-Steer` is a production-style runtime middleware that wraps open Vision-Language Models (e.g., PaliGemma, CLIP/SigLIP) to provide:

1. **In-Flight XAI Observability**: During the forward pass, real-time latent concept probes evaluate the model's intermediate representations for risk (demographic stereotyping, visual toxicity/jailbreaks, hallucination drift).
2. **Autonomous Mechanistic Steering**: If latent risk exceeds a user-configured policy threshold, the harness dynamically injects calibrated concept vectors ($h' = h + \alpha \cdot v_{concept}$) into the residual stream, remediating the output in real time without retraining or calling slow external LLM guardrails.
3. **Neuron & Patch Attribution**: Pinpoints the exact vision encoder patches and language backbone neurons driving the classification/generation.
4. **Structured Governance Telemetry**: Emits standard JSON audit traces documenting probe scores, neuron activations, patch attributions, and applied steering interventions.

---

## Architectural Lifecycle

```text
User Request [Image + Prompt]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│                       VLM-Steer                        │
│                                                        │
│  1. Ingestion & Pre-Flight Validation                  │
│                                                        │
│  2. Hooked Model Execution (PaliGemma / CLIP)          │
│     - Residual stream hidden state interception        │
│                                                        │
│  3. In-Flight XAI Observability Engine                 │
│     - Latent Concept Probes (Bias, Safety, Grounding)  │
│     - Visual Patch Saliency / Cross-Attention Map      │
│     - Neuron Activation Profiler                       │
│                                                        │
│  4. Autonomous Decision Gate & Active Steering         │
│     ├─ IF probe_score > threshold:                     │
│     │   ► Dynamically inject steering vector (α > 0)   │
│     │   ► Log neuron attribution & remediation delta   │
│     └─ ELSE: Proceed with standard baseline            │
│                                                        │
│  5. Telemetry & Governance Logger                      │
│     - Emits structured JSON audit trace                │
└────────────────────────────────────────────────────────┘
           │
           ▼
Remediated Safe Output + XAI Audit Record
```

---

## Target Use Cases & Scenarios
1. **Multimodal Demographic & Occupational Bias**:
   - Ambiguous human images prompted with occupational queries.
   - Harness probes demographic bias direction and steers along a neutrality vector to ensure fair descriptions.
2. **Visual Red-Teaming & Safety Jailbreaks**:
   - Adversarial visual patches or typographic triggers designed to bypass text guardrails.
   - Harness detects unsafe latent activation and dynamically steers into principled refusal or risk analysis.
3. **Multimodal Hallucination Suppression**:
   - Suggestive prompts on complex visual scenes (POPE benchmark style).
   - Harness probes visual grounding vector and suppresses ungrounded hallucinated tokens.

---

## Deliverables
1. **Core Python SDK (`src/`)**: Clean, modular, well-tested Python library.
2. **Interactive Workbench (`web/` & `app/`)**: Next.js Control Center and Streamlit fallback comparing Unprotected Baseline vs. `VLM-Steer` in real time.
3. **Quantitative Evaluation (`eval/`)**: Benchmarks measuring remediation rate, latency overhead, and attribution faithfulness.
4. **Project Poster (`poster/`)**: PDF poster for Project Expo presentation.

---

## Git Workflow
- `main`: Production-ready releases and final submission.
- `dev`: Primary integration branch.
- `feature/<name>`: Feature branches branched off `dev` and merged back into `dev`.
