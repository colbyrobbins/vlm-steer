"use client";

import { useEffect, useState } from "react";
import {
  BarChart3,
  Clock,
  Cpu,
  Loader2,
  RefreshCw,
  ShieldCheck,
  TrendingDown,
  Zap,
} from "lucide-react";
import { fetchBenchmarks, runLiveBenchmarks } from "@/lib/api";
import type { BenchmarkSummary } from "@/types/harness";

export default function BenchmarksPage() {
  const [data, setData] = useState<BenchmarkSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    fetchBenchmarks()
      .then((res) => {
        if (mounted) {
          setData(res);
          setError(null);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(
            err instanceof Error ? err.message : "Unable to load benchmark telemetry."
          );
        }
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handleRerun = async () => {
    setRunning(true);
    setError(null);
    try {
      const updated = await runLiveBenchmarks(20);
      setData(updated);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Benchmark execution failed."
      );
    } finally {
      setRunning(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-zinc-400 gap-2 text-sm">
        <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
        Loading quantitative evaluation telemetry...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-xl border border-rose-500/35 bg-rose-950/20 p-6 text-center space-y-3">
        <p className="text-sm text-rose-200">
          {error || "No benchmark results available."}
        </p>
        <button
          type="button"
          onClick={handleRerun}
          disabled={running}
          className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-xs font-medium text-white cursor-pointer"
        >
          {running ? "Running Benchmarks..." : "Run Evaluation Suite Now"}
        </button>
      </div>
    );
  }

  const scenarioEntries = Object.entries(data.scenario_results);
  const meanFaithfulness =
    data.attribution_faithfulness.reduce(
      (acc, f) => acc + f.faithfulness_ratio,
      0
    ) / Math.max(data.attribution_faithfulness.length, 1);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-lg sm:text-xl font-semibold tracking-tight text-zinc-100">
            Quantitative Benchmarks &amp; Causal Efficacy
          </h1>
          <p className="text-xs sm:text-sm text-zinc-400 mt-0.5">
            Empirical validation across remediation rates, computational latency overhead, and neuron ablation faithfulness.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-[11px] font-mono text-zinc-500 hidden md:inline">
            Last Run: {new Date(data.timestamp).toLocaleString()}
          </span>
          <button
            type="button"
            onClick={handleRerun}
            disabled={running}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-700/80 text-xs font-medium text-zinc-100 transition-colors disabled:opacity-50 cursor-pointer"
          >
            {running ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-400" />
                Evaluating (20 trials)...
              </>
            ) : (
              <>
                <RefreshCw className="h-3.5 w-3.5 text-indigo-400" />
                Re-Run Benchmarks
              </>
            )}
          </button>
        </div>
      </div>

      {/* Top KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-xl border border-emerald-500/25 bg-[#111116] p-4">
          <div className="flex items-center justify-between text-xs text-zinc-400 mb-2">
            <span>Mean Remediation Rate</span>
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-emerald-300">
            {(data.overall_remediation_rate * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-zinc-500 mt-1">
            Across {data.total_eval_samples} contrastive boundary samples
          </p>
        </div>

        <div className="rounded-xl border border-indigo-500/25 bg-[#111116] p-4">
          <div className="flex items-center justify-between text-xs text-zinc-400 mb-2">
            <span>Guardrail Speedup</span>
            <Zap className="h-4 w-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-indigo-300">
            {data.latency_profile.speedup_vs_external_llm_judge.toFixed(1)}x
          </div>
          <p className="text-[11px] text-zinc-500 mt-1">
            Faster than ~800ms external LLM API judge
          </p>
        </div>

        <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-4">
          <div className="flex items-center justify-between text-xs text-zinc-400 mb-2">
            <span>Steered Latency (p50)</span>
            <Clock className="h-4 w-4 text-amber-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-zinc-100">
            {data.latency_profile.steered_p50_ms.toFixed(1)} ms
          </div>
          <p className="text-[11px] text-zinc-500 mt-1">
            Middleware delta: +{data.latency_profile.mean_overhead_ms.toFixed(1)} ms
          </p>
        </div>

        <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-4">
          <div className="flex items-center justify-between text-xs text-zinc-400 mb-2">
            <span>Mean Neuron Faithfulness</span>
            <Cpu className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-cyan-300">
            {meanFaithfulness.toFixed(2)}x
          </div>
          <p className="text-[11px] text-zinc-500 mt-1">
            Top-K causal impact vs. 10x random control
          </p>
        </div>
      </div>

      {/* Scenario Remediation + Latency Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Scenario Remediation Rates */}
        <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-5 space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingDown className="h-4 w-4 text-emerald-400" />
              <h2 className="text-sm font-semibold text-zinc-100">
                Failure Rate Reduction by Domain
              </h2>
            </div>
            <span className="text-[11px] text-zinc-500">Lower failure rate is better</span>
          </div>

          <div className="space-y-5">
            {scenarioEntries.map(([key, metric]) => {
              const basePct = metric.baseline_compliance_or_bias_rate * 100;
              const steerPct = metric.steered_compliance_or_bias_rate * 100;
              const remPct = metric.remediation_rate * 100;

              return (
                <div key={key} className="space-y-2 pb-4 border-b border-zinc-800/60 last:border-b-0 last:pb-0">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-zinc-200">{key}</span>
                    <span className="px-2 py-0.5 rounded-full font-mono text-[11px] bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                      {remPct.toFixed(0)}% Remediated
                    </span>
                  </div>

                  {/* Baseline Bar */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px] text-zinc-400">
                      <span>Unprotected Baseline Failure</span>
                      <span className="font-mono text-rose-300">{basePct.toFixed(0)}%</span>
                    </div>
                    <div className="h-2 w-full bg-zinc-900 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-rose-500/80 rounded-full"
                        style={{ width: `${basePct}%` }}
                      />
                    </div>
                  </div>

                  {/* Steered Bar */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px] text-zinc-400">
                      <span>Steered Harness Residual Failure</span>
                      <span className="font-mono text-emerald-300">{steerPct.toFixed(0)}%</span>
                    </div>
                    <div className="h-2 w-full bg-zinc-900 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 rounded-full"
                        style={{ width: `${Math.max(steerPct, 2)}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Latency Distribution Profile */}
        <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-5 space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-indigo-400" />
              <h2 className="text-sm font-semibold text-zinc-100">
                Inference Latency Profile ({data.latency_profile.num_trials} Trials)
              </h2>
            </div>
            <span className="text-[11px] font-mono text-zinc-500">ms (log comparison)</span>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "p50 Median", base: data.latency_profile.baseline_p50_ms, steer: data.latency_profile.steered_p50_ms },
              { label: "p95 Tail", base: data.latency_profile.baseline_p95_ms, steer: data.latency_profile.steered_p95_ms },
              { label: "p99 Tail", base: data.latency_profile.baseline_p99_ms, steer: data.latency_profile.steered_p99_ms },
            ].map((item) => (
              <div
                key={item.label}
                className="rounded-lg bg-[#09090b] border border-zinc-800/90 p-3 space-y-2"
              >
                <span className="text-[11px] text-zinc-500 block">{item.label}</span>
                <div>
                  <div className="text-xs text-zinc-400">Baseline</div>
                  <div className="text-sm font-mono font-semibold text-zinc-200">
                    {item.base.toFixed(1)} ms
                  </div>
                </div>
                <div className="pt-1 border-t border-zinc-800/80">
                  <div className="text-xs text-indigo-300">Steered</div>
                  <div className="text-sm font-mono font-semibold text-indigo-200">
                    {item.steer.toFixed(1)} ms
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Architectural Comparison vs External Guardrail */}
          <div className="space-y-3 pt-2 border-t border-zinc-800/80">
            <span className="text-xs font-medium text-zinc-300 block">
              End-to-End Safety Overhead vs. External Guardrail LLM
            </span>

            <div className="space-y-2">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-emerald-300">VLM-SteerHarness (In-Flight Hooks)</span>
                  <span className="font-mono text-emerald-300">
                    {data.latency_profile.steered_p50_ms.toFixed(1)} ms
                  </span>
                </div>
                <div className="h-3 bg-zinc-900 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full"
                    style={{
                      width: `${Math.min(
                        (data.latency_profile.steered_p50_ms / 800) * 100,
                        100
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-zinc-400">External Post-Hoc LLM Judge API</span>
                  <span className="font-mono text-zinc-400">~800.0 ms</span>
                </div>
                <div className="h-3 bg-zinc-900 rounded-full overflow-hidden">
                  <div className="h-full bg-zinc-600 rounded-full w-full" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mechanistic Attribution Faithfulness Table */}
      <div className="rounded-xl border border-zinc-800/80 bg-[#111116] overflow-hidden">
        <div className="px-5 py-4 border-b border-zinc-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold text-zinc-100">
              Mechanistic Neuron Attribution Faithfulness (Ablation Analysis)
            </h2>
            <p className="text-xs text-zinc-400 mt-0.5">
              Zeroing the top-K attributed neurons triggers markedly sharper concept alignment drops than random-K Monte Carlo controls (ratio &gt; 1.0x proves causal fidelity).
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-zinc-900/70 text-zinc-400 uppercase font-mono text-[11px] border-b border-zinc-800">
              <tr>
                <th className="px-5 py-3">Concept Vector</th>
                <th className="px-4 py-3">Layer</th>
                <th className="px-4 py-3">Top-K</th>
                <th className="px-4 py-3">Baseline Proj</th>
                <th className="px-4 py-3">Top-K Drop</th>
                <th className="px-4 py-3">Random-K Drop</th>
                <th className="px-5 py-3 text-right">Faithfulness Ratio</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/70">
              {data.attribution_faithfulness.map((row) => (
                <tr key={row.concept_name} className="hover:bg-zinc-900/40">
                  <td className="px-5 py-3.5 font-mono font-medium text-zinc-100">
                    {row.concept_name}
                  </td>
                  <td className="px-4 py-3.5 font-mono text-zinc-300">
                    Layer {row.layer_idx}
                  </td>
                  <td className="px-4 py-3.5 font-mono text-zinc-400">
                    k={row.top_k}
                  </td>
                  <td className="px-4 py-3.5 font-mono text-zinc-300">
                    {row.baseline_concept_alignment.toFixed(3)}
                  </td>
                  <td className="px-4 py-3.5 font-mono text-rose-300">
                    -{row.alignment_drop_top_k.toFixed(3)}
                  </td>
                  <td className="px-4 py-3.5 font-mono text-zinc-400">
                    -{row.alignment_drop_random_k.toFixed(3)}
                  </td>
                  <td className="px-5 py-3.5 text-right">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full font-mono font-semibold bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                      {row.faithfulness_ratio.toFixed(2)}x
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
