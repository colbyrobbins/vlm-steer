"use client";

import { useState } from "react";
import type { AuditTrace } from "@/types/harness";
import {
  Check,
  ChevronDown,
  ChevronUp,
  Copy,
  Download,
  FileJson,
} from "lucide-react";

interface AuditTraceDrawerProps {
  trace: AuditTrace | null;
}

export function AuditTraceDrawer({ trace }: AuditTraceDrawerProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!trace) return null;

  const jsonString = JSON.stringify(trace, null, 2);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownload = () => {
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `audit_trace_${trace.trace_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="rounded-xl border border-zinc-800/80 bg-[#111116] overflow-hidden">
      <div className="px-5 py-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-zinc-900 border border-zinc-800">
            <FileJson className="h-4 w-4 text-emerald-400" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-sm font-semibold text-zinc-100">
                Governance Audit Telemetry
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-zinc-800 text-zinc-300 border border-zinc-700">
                ID: {trace.trace_id}
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-indigo-500/10 text-indigo-300 border border-indigo-500/25">
                NIST AI RMF 1.0
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/25">
                EU AI Act Art. 14
              </span>
            </div>
            <p className="text-xs text-zinc-400 mt-0.5">
              Immutable JSON compliance record capturing probe evaluations, injected hooks, and circuit attributions.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-700/80 transition-colors cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-emerald-400" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" />
                Copy JSON
              </>
            )}
          </button>

          <button
            type="button"
            onClick={handleDownload}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/35 transition-colors cursor-pointer"
          >
            <Download className="h-3.5 w-3.5" />
            Download
          </button>

          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 transition-colors cursor-pointer"
          >
            {expanded ? (
              <>
                Hide <ChevronUp className="h-3.5 w-3.5" />
              </>
            ) : (
              <>
                Inspect <ChevronDown className="h-3.5 w-3.5" />
              </>
            )}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-zinc-800/80 bg-[#09090b] p-4 max-h-96 overflow-auto">
          <pre className="text-xs font-mono text-zinc-300 leading-relaxed whitespace-pre-wrap">
            {jsonString}
          </pre>
        </div>
      )}
    </div>
  );
}
