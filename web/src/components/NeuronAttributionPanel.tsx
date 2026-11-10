"use client";

import { useState } from "react";
import type { EnrichedNeuronAttribution } from "@/types/harness";
import { Cpu } from "lucide-react";

interface NeuronAttributionPanelProps {
  attributions: EnrichedNeuronAttribution[];
}

export function NeuronAttributionPanel({
  attributions,
}: NeuronAttributionPanelProps) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  if (!attributions || attributions.length === 0) {
    return (
      <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-6 text-center text-sm text-zinc-500">
        Execute a forward pass to inspect causal neuron circuits.
      </div>
    );
  }

  const active = attributions[Math.min(selectedIdx, attributions.length - 1)];
  const maxScore = Math.max(...active.top_neuron_scores, 1e-6);

  return (
    <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-zinc-100">
              Causal Neuron Attribution Profiler
            </h3>
          </div>
          <p className="text-xs text-zinc-400 mt-0.5">
            Top residual channels driving the active concept vector in Layer{" "}
            <span className="font-mono text-zinc-300">{active.layer_idx}</span>
          </p>
        </div>

        {/* Layer Switcher Tabs */}
        <div className="flex items-center gap-1 bg-zinc-900/90 p-1 rounded-lg border border-zinc-800">
          {attributions.map((attr, idx) => (
            <button
              key={attr.layer_idx}
              type="button"
              onClick={() => setSelectedIdx(idx)}
              className={`px-2.5 py-1 rounded-md text-xs font-mono transition-colors cursor-pointer ${
                idx === selectedIdx
                  ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/35"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              L{attr.layer_idx}
            </button>
          ))}
        </div>
      </div>

      {/* Horizontal Neuron Bars */}
      <div className="space-y-3">
        {active.top_neuron_indices.map((neuronId, i) => {
          const magnitude = active.top_neuron_scores[i];
          const signed = active.signed_neuron_scores?.[i] ?? magnitude;
          const isPositive = signed >= 0;
          const barWidth = Math.max((magnitude / maxScore) * 100, 6);

          return (
            <div key={neuronId} className="group">
              <div className="flex items-center justify-between text-xs mb-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono font-medium text-zinc-200 w-24">
                    Neuron #{neuronId}
                  </span>
                  <span
                    className={`px-1.5 py-0.2 rounded text-[10px] font-mono border ${
                      isPositive
                        ? "bg-indigo-500/10 text-indigo-300 border-indigo-500/25"
                        : "bg-amber-500/10 text-amber-300 border-amber-500/25"
                    }`}
                  >
                    {isPositive ? "+ALIGN" : "-INHIBIT"}
                  </span>
                </div>
                <span className="font-mono text-zinc-300">
                  {signed >= 0 ? "+" : ""}
                  {signed.toFixed(4)}
                </span>
              </div>

              <div className="h-2 w-full bg-zinc-900 rounded-full overflow-hidden border border-zinc-800/80">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isPositive
                      ? "bg-gradient-to-r from-indigo-600 to-cyan-400"
                      : "bg-gradient-to-r from-amber-600 to-orange-400"
                  }`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-3 border-t border-zinc-800/70 flex items-center justify-between text-[11px] text-zinc-500">
        <span>
          Concept Circuit:{" "}
          <code className="text-zinc-300 font-mono">{active.concept_name}</code>
        </span>
        <span>Coordinate basis: 128D residual stream</span>
      </div>
    </div>
  );
}
