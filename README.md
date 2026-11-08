# VLM-SteerHarness 🛡️

**An In-Flight Observability & Active Steering Harness for Production Multimodal AI Systems**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Developed with Antigravity](https://img.shields.io/badge/Developed%20with-Google%20Antigravity-4285F4.svg?logo=google&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Executive Summary

In enterprise and high-stakes AI applications (healthcare, legal, content moderation, autonomous decision systems), Explainable AI (XAI) cannot simply be an offline post-hoc script or a slow "LLM-as-a-judge" wrapper. 

**`VLM-SteerHarness`** is a production-style runtime middleware that wraps open Vision-Language Models (e.g. Google PaliGemma, CLIP/SigLIP) to provide:
1. **In-Flight XAI Observability**: During the forward pass, real-time latent concept probes evaluate intermediate hidden states for risk (demographic stereotyping, visual toxicity/jailbreaks, hallucination drift).
2. **Autonomous Mechanistic Steering**: If latent risk breaches a policy threshold, the harness dynamically injects calibrated concept vectors ($\mathbf{h}' = \mathbf{h} + \alpha \vec{v}_{\text{concept}}$) into the residual stream via PyTorch forward hooks—remediating the output in milliseconds without retraining or weight modifications.
3. **Neuron-Level Circuit Profiling**: Isolates the top-$k$ feature dimensions driving active concept directions.
4. **Structured Governance Telemetry**: Emits NIST AI RMF / EU AI Act compliant JSON audit traces documenting probe scores, neuron attributions, and applied interventions.

---

## Architectural Lifecycle

```text
                                User Request [Image + Text Prompt]
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             VLM-SteerHarness                                     │
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │ 1. Ingestion & Pre-Flight Validation                                     │   │
│   │    - Image preprocessing, tokenization, config initialization            │   │
│   └────────────────────────────────────┬─────────────────────────────────────┘   │
│                                        ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │ 2. Hooked Model Execution (PaliGemma / CLIP / MockVLM)                   │   │
│   │    - Intercepts hidden states across intermediate residual layers        │   │
│   └────────────────────────────────────┬─────────────────────────────────────┘   │
│                                        ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │ 3. In-Flight XAI Observability Engine                                    │   │
│   │    - Latent Concept Probes (Computes real-time Bias / Safety / POPE risk)│   │
│   │    - Neuron Activation Profiler (identifies active concept circuits)     │   │
│   └────────────────────────────────────┬─────────────────────────────────────┘   │
│                                        ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │ 4. Autonomous Decision Gate & Active Steering                            │   │
│   │    - IF Risk Score > Configured Policy Threshold:                        │   │
│   │        ► Dynamically inject steering vector:  h' = h + (α * v_concept)   │   │
│   │        ► Suppress biased / toxic / hallucinated tokens in real time      │   │
│   │    - ELSE: Proceed with standard baseline execution                      │   │
│   └────────────────────────────────────┬─────────────────────────────────────┘   │
│                                        ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │ 5. Telemetry & Governance Logger                                         │   │
│   │    - Emits structured JSON audit trace (NIST AI RMF / EU AI Act ready)   │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
                 Remediated Safe Output + XAI Audit Record
```

---

## Target Evaluation Scenarios

| Scenario | Risk Failure Mode | Steering Concept Vector | Layer |
| :--- | :--- | :--- | :--- |
| **1. Occupational Neutrality** | Defaulting to gender/demographic stereotypes in professional roles | `demographic_neutrality.pt` | Layer 2 |
| **2. Visual Safety & Red-Teaming** | Visual triggers or typographic attacks bypassing guardrails | `safety_refusal.pt` | Layer 3 |
| **3. Visual Grounding** | Fabricating objects not present in the visual frame (POPE style) | `visual_grounding.pt` | Layer 4 |

---

## Project Structure

```text
xai_project_2/
├── app/
│   └── app.py                  # Interactive Streamlit Control Center
├── data/
│   ├── calibration/            # Curated contrastive JSON datasets
│   └── concepts/               # Calibrated unit concept vectors (.pt)
├── src/
│   ├── harness/
│   │   ├── core.py             # SteerHarness main runtime SDK
│   │   ├── config.py           # Pydantic schemas (HarnessConfig, SteeringPolicy)
│   │   ├── gatekeeper.py       # Autonomous decision gatekeeper
│   │   └── telemetry.py        # Structured AuditTrace logger
│   ├── steering/
│   │   ├── hook.py             # PyTorch residual stream forward hook manager
│   │   ├── extractor.py        # Difference-of-Means & Contrastive PCA algorithms
│   │   ├── vectors.py          # ConceptVector data structures & normalization
│   │   └── calibration.py      # Automated vector calibration pipeline
│   ├── attribution/
│   │   ├── probes.py           # In-flight latent concept probes (cosine & risk mapping)
│   │   └── neurons.py          # Top-k neuron circuit profiler
│   └── models/
│       └── wrapper.py          # Unified model wrapper & zero-download MockVLM
├── tests/                      # Full integration & unit test suite (pytest)
├── requirements.txt            # Pinned dependencies
└── PROJECT_CONTEXT.md          # Project design specifications & workflow
```

---

## Quickstart & Installation

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/colbyrobbins/xai-project-2.git
cd xai-project-2

# Initialize virtual environment with Python 3.10
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
python -m pytest -v tests\
```

### 3. Launch the Interactive Control Center
```powershell
streamlit run app\app.py
```

### 4. Run Quantitative Evaluation Benchmarks
```powershell
python eval\run_benchmarks.py
```

---

## Python SDK Example

```python
from PIL import Image
from src.harness import SteerHarness, HarnessConfig
from src.steering.vectors import ConceptVector

# 1. Initialize Harness
harness = SteerHarness(config=HarnessConfig(telemetry_level="detailed"))

# 2. Register Calibrated Concept Vector
concept = ConceptVector.load("data/concepts/demographic_neutrality.pt")
harness.register_concept(
    concept=concept,
    probe_threshold=0.5,
    default_alpha=1.5,
)

# 3. Execute Ingest & Forward Pass
image = Image.open("sample.jpg")
prompt = "The surgeon reviewed the patient chart."

output, audit_trace = harness.generate(prompt=prompt, image=image)

print("Generated Output:", output)
print("Audit Trace Summary:", audit_trace.summary())
print("JSON Telemetry:\n", audit_trace.to_json(indent=2))
```

---

## Authors & Acknowledgments

* **Developer**: Colby Robbins
* **Program**: AIPI (Master of Engineering in AI), Duke University
* **Course**: Explainable AI (XAI) / Responsible AI — Final Project 2
* **Theoretical Foundations**: Representation Engineering (RepE), Contrastive Activation Addition (CAA), Google TCAV, and Anthropic Mechanistic Interpretability.
* **Engineering Acceleration**: Pair-programmed and accelerated with **Google Antigravity** (Google DeepMind Advanced Agentic Coding).

---

<p align="center">
  <sub>Developed with <b>Google Antigravity</b></sub>
</p>
