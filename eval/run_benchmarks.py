"""VLM-SteerHarness Quantitative Benchmark Execution Script.

Runs the complete quantitative evaluation across:
1. Scenario Remediation Rates (Demographic Bias, Safety Refusal, Visual Grounding).
2. Latency & Computational Overhead (p50, p95, p99 vs external LLM guardrails).
3. Neuron Attribution Faithfulness (Mechanistic Ablation of Top-K vs Random-K Channels).

Outputs formatted Markdown tables and saves structured JSON results.

Authors/Maintainers: Colby Robbins
Course: AIPI (Duke Master of Engineering in AI) - Responsible AI / XAI Project 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.benchmark import EvaluationEngine


def main():
    parser = argparse.ArgumentParser(description="Run VLM-SteerHarness Quantitative Benchmarks.")
    parser.add_argument(
        "--output-json",
        default="eval/benchmark_results.json",
        help="Path to save output JSON benchmark results.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of latency benchmark trials.",
    )
    args = parser.parse_args()

    print("================================================================================")
    print(" [VLM-SteerHarness] Quantitative Evaluation Benchmark Suite")
    print("================================================================================")
    print("Initializing Evaluation Engine and loading pre-calibrated concept vectors...")

    engine = EvaluationEngine()
    print(f"Loaded {len(engine.registered_concepts)} concept vectors: {list(engine.registered_concepts.keys())}")
    print("\nRunning benchmarks across scenarios, latency profiling, and neuron faithfulness...")

    summary = engine.run_full_benchmark(num_latency_trials=args.trials)

    # 1. Print Scenario Remediation Results
    print("\n--------------------------------------------------------------------------------")
    print(" 1. SCENARIO REMEDIATION & DETECTION BENCHMARK")
    print("--------------------------------------------------------------------------------")
    print(summary.to_markdown_table())
    print(f"\nOverall Mean Remediation Rate: {summary.overall_remediation_rate:.1%}")

    # 2. Print Latency Profiling Results
    lat = summary.latency_profile
    print("\n--------------------------------------------------------------------------------")
    print(" 2. IN-FLIGHT LATENCY & COMPUTATIONAL OVERHEAD PROFILING")
    print("--------------------------------------------------------------------------------")
    print(f"Baseline Latency (Unhooked):")
    print(f"  - p50: {lat.baseline_p50_ms:.2f} ms | p95: {lat.baseline_p95_ms:.2f} ms | p99: {lat.baseline_p99_ms:.2f} ms")
    print(f"SteerHarness Latency (Probing + Dynamic Steering):")
    print(f"  - p50: {lat.steered_p50_ms:.2f} ms | p95: {lat.steered_p95_ms:.2f} ms | p99: {lat.steered_p99_ms:.2f} ms")
    print(f"Overhead Delta:")
    print(f"  - In-Flight Middleware Overhead: +{lat.mean_overhead_ms:.2f} ms ({lat.overhead_percentage:.1f}%)")
    print(f"  - Speedup vs. External LLM Guardrail (~800ms API roundtrip): {lat.speedup_vs_external_llm_judge:.1f}x faster")

    # 3. Print Neuron Attribution Faithfulness Results
    print("\n--------------------------------------------------------------------------------")
    print(" 3. MECHANISTIC NEURON ATTRIBUTION FAITHFULNESS (ABLATION PROFILING)")
    print("--------------------------------------------------------------------------------")
    print("| Concept | Layer | Top-K | Baseline Alignment | Top-K Ablated | Random-K Ablated | Faithfulness Ratio |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for f in summary.attribution_faithfulness:
        print(
            f"| **{f.concept_name}** | {f.layer_idx} | {f.top_k} | "
            f"{f.baseline_concept_alignment:.3f} | {f.ablated_top_k_alignment:.3f} | "
            f"{f.ablated_random_k_alignment:.3f} | **{f.faithfulness_ratio:.2f}x** |"
        )
    print("\n* Faithfulness Ratio > 1.0 proves top-attributed neurons causally drive the concept circuit.")

    # 4. Save JSON results
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))

    print(f"\n[OK] Structured benchmark results saved to: {out_path.resolve()}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
