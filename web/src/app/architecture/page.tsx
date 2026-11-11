"use client";

import { useEffect, useState } from "react";
import {
  ArrowRight,
  Binary,
  BookOpen,
  Cpu,
  Database,
  Layers,
  Loader2,
  Shield,
} from "lucide-react";
import { fetchCalibration, fetchConcepts } from "@/lib/api";
import type { CalibrationPair, ConceptMetadata } from "@/types/harness";

const DATASET_LABELS: Record<string, string> = {
  demographic_bias_pairs: "Demographic & Occupational Bias",
  safety_refusal_pairs: "Enterprise Safety Refusal",
  visual_grounding_pairs: "Visual Grounding & Hallucination",
};

export default function ArchitecturePage() {
  const [concepts, setConcepts] = useState<ConceptMetadata[]>([]);
  const [datasets, setDatasets] = useState<Record<string, CalibrationPair[]>>({});
  const [activeDatasetKey, setActiveDatasetKey] = useState<string>(
    "demographic_bias_pairs"
  );
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    Promise.all([fetchConcepts(), fetchCalibration()])
      .then(([c, d]) => {
        if (!mounted) return;
        setConcepts(c);
        setDatasets(d);
        const keys = Object.keys(d);
        if (keys.length > 0 && !d[activeDatasetKey]) {
          setActiveDatasetKey(keys[0]);
        }
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [activeDatasetKey]);

  const layers = [0, 1, 2, 3, 4, 5];
  const activePairs = datasets[activeDatasetKey] || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-lg sm:text-xl font-semibold tracking-tight text-zinc-100">
          Mechanistic Architecture &amp; Contrastive Calibration
        </h1>
        <p className="text-xs sm:text-sm text-zinc-400 mt-0.5">
          Residual-stream hook topology, linear representation geometry, and contrastive dataset pairs driving active steering.
        </p>
      </div>

      {/* 6-Layer Residual Stream Pipeline */}
      <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-5 space-y-5">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-indigo-400" />
            <h2 className="text-sm font-semibold text-zinc-100">
              Transformer Residual Stream Hook Topology (6 Layers &times; 128D)
            </h2>
          </div>
          <span className="text-[11px] font-mono text-zinc-500">
            Conv2d Patch Encoder + Token Embed &rarr; Transformer Stack &rarr; LM Head
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
          {layers.map((layerIdx) => {
            const attached = concepts.find((c) => c.layer_idx === layerIdx);
            return (
              <div
                key={layerIdx}
                className={`rounded-xl border p-3.5 flex flex-col justify-between min-h-[140px] transition-all ${
                  attached
                    ? "bg-indigo-500/10 border-indigo-500/40 shadow-[0_0_24px_-12px_rgba(99,102,241,0.45)]"
                    : "bg-[#09090b] border-zinc-800/80"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-xs font-semibold text-zinc-200">
                      Layer {layerIdx}
                    </span>
                    {attached ? (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                        HOOKED
                      </span>
                    ) : (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-zinc-900 text-zinc-500 border border-zinc-800">
                        PASS
                      </span>
                    )}
                  </div>

                  {attached ? (
                    <div className="space-y-1.5">
                      <p className="text-xs font-medium text-indigo-200 break-words">
                        {attached.name}
                      </p>
                      <p className="text-[11px] text-zinc-400 leading-snug">
                        Margin:{" "}
                        <span className="font-mono text-emerald-300">
                          {attached.separation_margin.toFixed(3)}
                        </span>
                      </p>
                    </div>
                  ) : (
                    <p className="text-[11px] text-zinc-500 mt-1">
                      Unmodified residual pass-through.
                    </p>
                  )}
                </div>

                <div className="pt-2 mt-2 border-t border-zinc-800/60 flex items-center justify-between text-[10px] font-mono text-zinc-500">
                  <span>{attached ? "Probe + Steer" : "Standard Block"}</span>
                  <ArrowRight className="h-3 w-3 text-zinc-600" />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Theoretical Formulation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-4 space-y-2">
          <div className="flex items-center gap-2 text-indigo-300">
            <Binary className="h-4 w-4" />
            <h3 className="text-sm font-semibold text-zinc-100">
              1. Concept Calibration (CAA)
            </h3>
          </div>
          <div className="rounded-lg bg-[#09090b] border border-zinc-800 px-3 py-2 font-mono text-xs text-indigo-200">
            v_concept = normalize(mean(H+) - mean(H-))
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Difference-in-means isolates the linear concept direction separating aligned target completions (H+) from harmful or biased failures (H-).
          </p>
        </div>

        <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-4 space-y-2">
          <div className="flex items-center gap-2 text-amber-300">
            <Shield className="h-4 w-4" />
            <h3 className="text-sm font-semibold text-zinc-100">
              2. In-Flight Latent Probing
            </h3>
          </div>
          <div className="rounded-lg bg-[#09090b] border border-zinc-800 px-3 py-2 font-mono text-xs text-amber-200">
            risk = sigmoid(-tau * cos(h_l, v_concept))
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Because concept vectors point toward the safe target manifold, negative cosine alignment flags unsafe drift and triggers steering above threshold &theta;.
          </p>
        </div>

        <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-4 space-y-2">
          <div className="flex items-center gap-2 text-emerald-300">
            <Cpu className="h-4 w-4" />
            <h3 className="text-sm font-semibold text-zinc-100">
              3. Dynamic Residual Steering
            </h3>
          </div>
          <div className="rounded-lg bg-[#09090b] border border-zinc-800 px-3 py-2 font-mono text-xs text-emerald-200">
            h&apos;_l = h_l + alpha * v_concept
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Forward hooks modify intermediate residual activations on the fly during generation, steering semantics without updating weights.
          </p>
        </div>
      </div>

      {/* Loaded Concept Vector Registry */}
      {concepts.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {concepts.map((c) => (
            <div
              key={c.name}
              className="rounded-xl border border-zinc-800/80 bg-[#111116] p-4 space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-semibold text-indigo-300">
                  {c.name}.pt
                </span>
                <span className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 font-mono text-[10px] text-zinc-400">
                  L{c.layer_idx} &bull; {c.dimension}D
                </span>
              </div>
              <p className="text-xs text-zinc-400 leading-relaxed">
                {c.description || "Calibrated residual steering direction vector."}
              </p>
              <div className="pt-2 border-t border-zinc-800/60 flex items-center justify-between text-[11px] font-mono text-zinc-500">
                <span>Margin: {c.separation_margin.toFixed(3)}</span>
                <span>||H+||={c.positive_norm.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Contrastive Calibration Dataset Explorer */}
      <div className="rounded-xl border border-zinc-800/80 bg-[#111116] overflow-hidden">
        <div className="px-5 py-4 border-b border-zinc-800/80 flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-emerald-400" />
            <div>
              <h2 className="text-sm font-semibold text-zinc-100">
                Contrastive Calibration Pairs Explorer
              </h2>
              <p className="text-xs text-zinc-400">
                Sanitized representation engineering pairs from{" "}
                <code className="font-mono text-zinc-300">data/calibration/</code>
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-1 bg-[#09090b] p-1 rounded-lg border border-zinc-800">
            {Object.keys(datasets).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => setActiveDatasetKey(key)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
                  activeDatasetKey === key
                    ? "bg-zinc-800 text-zinc-100 border border-zinc-700"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                {DATASET_LABELS[key] || key}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="p-10 flex items-center justify-center text-sm text-zinc-400 gap-2">
            <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
            Loading contrastive datasets...
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-zinc-900/70 text-zinc-400 uppercase font-mono text-[11px] border-b border-zinc-800">
                <tr>
                  <th className="px-4 py-3 w-28">ID / Tag</th>
                  <th className="px-4 py-3">Positive Aligned Target (H+)</th>
                  <th className="px-4 py-3">Negative Contrastive Failure (H-)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/70">
                {activePairs.map((pair) => {
                  const tag = pair.domain || pair.category || pair.type || "pair";
                  return (
                    <tr key={pair.id} className="hover:bg-zinc-900/35 align-top">
                      <td className="px-4 py-3.5">
                        <div className="font-mono font-semibold text-zinc-200">
                          {pair.id}
                        </div>
                        <span className="inline-block mt-1 px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 font-mono text-[10px] text-zinc-400">
                          {tag}
                        </span>
                      </td>
                      <td className="px-4 py-3.5 text-emerald-200/90 leading-relaxed">
                        <div className="flex items-start gap-2">
                          <BookOpen className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                          <span>{pair.positive}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3.5 text-rose-200/85 leading-relaxed">
                        {pair.negative}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
