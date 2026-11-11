"use client";

import type { ProbeResult } from "@/types/harness";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

interface RiskProbeCardProps {
  probe: ProbeResult;
  threshold: number;
}

const CONCEPT_LABELS: Record<string, string> = {
  demographic_neutrality: "Demographic Neutrality",
  safety_refusal: "Enterprise Safety Refusal",
  visual_grounding: "Visual Grounding",
};

export function RiskProbeCard({ probe, threshold }: RiskProbeCardProps) {
  const riskPct = Math.min(Math.max(probe.normalized_risk * 100, 0), 100);
  const thresholdPct = Math.min(Math.max(threshold * 100, 0), 100);
  const label = CONCEPT_LABELS[probe.concept_name] || probe.concept_name;

  return (
    <div
      className={`rounded-xl border p-4 transition-all ${
        probe.triggered
          ? "bg-rose-950/20 border-rose-500/35 shadow-[0_0_24px_-12px_rgba(244,63,94,0.45)]"
          : "bg-[#111116] border-zinc-800/80"
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-3">
        <div>
          <h4 className="text-sm font-medium text-zinc-100">{label}</h4>
          <span className="inline-block mt-0.5 text-[11px] font-mono text-zinc-500">
            Layer {probe.layer_idx} &bull; {probe.concept_name}
          </span>
        </div>

        {probe.triggered ? (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-rose-500/15 text-rose-300 border border-rose-500/30">
            <AlertTriangle className="h-3 w-3" />
            BREACH
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
            <CheckCircle2 className="h-3 w-3" />
            SAFE
          </span>
        )}
      </div>

      {/* Risk Progress Bar with Threshold Marker */}
      <div className="relative h-2.5 w-full bg-zinc-800/90 rounded-full overflow-visible my-3">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            probe.triggered
              ? "bg-gradient-to-r from-amber-500 to-rose-500"
              : "bg-gradient-to-r from-emerald-500 to-teal-400"
          }`}
          style={{ width: `${riskPct}%` }}
        />
        {/* Vertical Cutoff Marker */}
        <div
          title={`Policy Cutoff: ${thresholdPct.toFixed(0)}%`}
          className="absolute -top-1 -bottom-1 w-0.5 bg-zinc-100 shadow-[0_0_6px_rgba(255,255,255,0.8)] z-10 transition-all"
          style={{ left: `${thresholdPct}%` }}
        />
      </div>

      <div className="flex items-center justify-between text-xs mt-2 pt-2 border-t border-zinc-800/60">
        <div>
          <span className="text-zinc-500">Latent Risk: </span>
          <span
            className={`font-mono font-semibold ${
              probe.triggered ? "text-rose-300" : "text-emerald-300"
            }`}
          >
            {riskPct.toFixed(1)}%
          </span>
        </div>
        <div className="font-mono text-zinc-400">
          cos: {probe.raw_score >= 0 ? "+" : ""}
          {probe.raw_score.toFixed(3)}
        </div>
      </div>
    </div>
  );
}
