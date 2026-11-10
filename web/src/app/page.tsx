"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertOctagon,
  ArrowRight,
  ImagePlus,
  Loader2,
  Play,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  Terminal,
  Zap,
} from "lucide-react";
import {
  executeInference,
  fetchConcepts,
  fetchScenarios,
} from "@/lib/api";
import type {
  ConceptMetadata,
  InferenceResponse,
  ScenarioPreset,
} from "@/types/harness";
import { RiskProbeCard } from "@/components/RiskProbeCard";
import { NeuronAttributionPanel } from "@/components/NeuronAttributionPanel";
import { AuditTraceDrawer } from "@/components/AuditTraceDrawer";

export default function SteeringStudioPage() {
  const [scenarios, setScenarios] = useState<ScenarioPreset[]>([]);
  const [concepts, setConcepts] = useState<ConceptMetadata[]>([]);
  const [activeScenarioId, setActiveScenarioId] = useState<string>("demographic_bias");

  const [prompt, setPrompt] = useState<string>(
    "The surgeon reviewed the patient chart and prepared the operating room with precision."
  );
  const [customImageB64, setCustomImageB64] = useState<string | null>(null);
  const [useVisualContext, setUseVisualContext] = useState<boolean>(true);
  const [probeThreshold, setProbeThreshold] = useState<number>(0.5);
  const [steeringMode, setSteeringMode] = useState<"autonomous" | "manual">("autonomous");
  const [manualAlpha, setManualAlpha] = useState<number>(1.5);

  const [loading, setLoading] = useState<boolean>(false);
  const [bootError, setBootError] = useState<string | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const autoRanRef = useRef<boolean>(false);

  const activeScenario =
    scenarios.find((s) => s.id === activeScenarioId) || scenarios[0] || null;
  const previewImageSrc = customImageB64 || activeScenario?.default_image_b64 || null;

  const runPass = useCallback(
    async (opts?: {
      overridePrompt?: string;
      overrideImage?: string | null;
      overrideMode?: "autonomous" | "manual";
      overrideAlpha?: number;
      overrideThreshold?: number;
      overrideUseVisual?: boolean;
    }) => {
      setLoading(true);
      setInferenceError(null);
      try {
        const useVisual = opts?.overrideUseVisual ?? useVisualContext;
        const imgPayload = useVisual
          ? opts?.overrideImage !== undefined
            ? opts.overrideImage
            : customImageB64 || activeScenario?.default_image_b64 || null
          : null;

        const res = await executeInference({
          prompt: opts?.overridePrompt ?? prompt,
          image_base64: imgPayload,
          use_default_image: useVisual,
          probe_threshold: opts?.overrideThreshold ?? probeThreshold,
          steering_mode: opts?.overrideMode ?? steeringMode,
          manual_alpha: opts?.overrideAlpha ?? manualAlpha,
          max_new_tokens: 60,
        });
        setResult(res);
      } catch (err) {
        setInferenceError(
          err instanceof Error ? err.message : "Failed to execute forward pass."
        );
      } finally {
        setLoading(false);
      }
    },
    [
      activeScenario,
      customImageB64,
      manualAlpha,
      probeThreshold,
      prompt,
      steeringMode,
      useVisualContext,
    ]
  );

  useEffect(() => {
    let mounted = true;
    const init = async () => {
      try {
        const [loadedScenarios, loadedConcepts] = await Promise.all([
          fetchScenarios(),
          fetchConcepts(),
        ]);
        if (!mounted) return;
        setScenarios(loadedScenarios);
        setConcepts(loadedConcepts);
        setBootError(null);

        if (loadedScenarios.length > 0 && !autoRanRef.current) {
          autoRanRef.current = true;
          const first = loadedScenarios[0];
          setActiveScenarioId(first.id);
          setPrompt(first.default_prompt);
          // Execute initial telemetry pass automatically so workspace renders live results
          const initialRes = await executeInference({
            prompt: first.default_prompt,
            image_base64: first.default_image_b64,
            use_default_image: true,
            probe_threshold: 0.5,
            steering_mode: "autonomous",
            manual_alpha: 1.5,
            max_new_tokens: 60,
          });
          if (mounted) {
            setResult(initialRes);
          }
        }
      } catch (err) {
        if (!mounted) return;
        setBootError(
          err instanceof Error
            ? err.message
            : "Could not connect to VLM-SteerHarness Python API."
        );
      }
    };
    init();
    return () => {
      mounted = false;
    };
  }, []);

  const handleSelectScenario = (scenario: ScenarioPreset) => {
    setActiveScenarioId(scenario.id);
    setPrompt(scenario.default_prompt);
    setCustomImageB64(null);
    setUseVisualContext(true);
    runPass({
      overridePrompt: scenario.default_prompt,
      overrideImage: scenario.default_image_b64,
      overrideUseVisual: true,
    });
  };

  const handleFileUpload = (file: File | undefined) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setCustomImageB64(reader.result);
        setUseVisualContext(true);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runPass();
    }
  };

  const alphaZoneLabel =
    manualAlpha > 0.05
      ? { text: "SAFE REMEDIATION", cls: "text-emerald-400" }
      : manualAlpha < -0.05
      ? { text: "ADVERSARIAL RED-TEAM", cls: "text-rose-400" }
      : { text: "UNSTEERED NEUTRAL", cls: "text-zinc-400" };

  return (
    <div className="space-y-6">
      {/* Backend Offline Callout */}
      {bootError && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/25 p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <AlertOctagon className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-semibold text-rose-200">
                Python Inference Backend Unreachable
              </h3>
              <p className="text-xs text-rose-300/80 mt-0.5">
                {bootError}. Start the Starlette API server from your repository root:
              </p>
              <code className="inline-flex items-center gap-2 mt-2 px-3 py-1.5 rounded-md bg-zinc-950/90 border border-rose-500/30 font-mono text-xs text-zinc-200">
                <Terminal className="h-3.5 w-3.5 text-rose-400" />
                .\.venv\Scripts\uvicorn api.server:app --host 127.0.0.1 --port 8000
              </code>
            </div>
          </div>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 text-xs font-medium border border-rose-500/35 cursor-pointer shrink-0"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* Top Scenario Selection Strip */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h1 className="text-lg sm:text-xl font-semibold tracking-tight text-zinc-100">
              In-Flight Observability &amp; Active Steering Studio
            </h1>
            <p className="text-xs sm:text-sm text-zinc-400">
              Probe residual streams in real time and inject calibrated steering vectors (
              <span className="font-mono text-indigo-300">h&apos; = h + &alpha;v</span>) without retraining.
            </p>
          </div>
          {concepts.length > 0 && (
            <div className="hidden md:flex items-center gap-2 text-xs text-zinc-400">
              <span className="font-mono text-zinc-500">Registered Vectors:</span>
              {concepts.map((c) => (
                <span
                  key={c.name}
                  className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 font-mono text-[11px] text-zinc-300"
                >
                  L{c.layer_idx}:{c.name.split("_")[0]}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {scenarios.map((scenario) => {
            const isSelected = scenario.id === activeScenarioId;
            return (
              <button
                key={scenario.id}
                type="button"
                onClick={() => handleSelectScenario(scenario)}
                className={`text-left rounded-xl border p-3.5 transition-all cursor-pointer ${
                  isSelected
                    ? "bg-indigo-500/10 border-indigo-500/45 shadow-[0_0_24px_-12px_rgba(99,102,241,0.5)]"
                    : "bg-[#111116] border-zinc-800/80 hover:border-zinc-700 hover:bg-zinc-900/60"
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-1.5">
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium border ${
                      isSelected
                        ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/30"
                        : "bg-zinc-900 text-zinc-400 border-zinc-800"
                    }`}
                  >
                    {scenario.badge}
                  </span>
                  <span className="text-[11px] font-mono text-zinc-500">
                    Layer {scenario.target_layer}
                  </span>
                </div>
                <h3 className="text-sm font-medium text-zinc-100 line-clamp-1">
                  {scenario.title}
                </h3>
                <p className="text-xs text-zinc-400 mt-1 line-clamp-2 leading-relaxed">
                  {scenario.description}
                </p>
              </button>
            );
          })}
        </div>
      </section>

      {/* Main 2-Column Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT CONTROL PANEL */}
        <aside className="lg:col-span-4 space-y-4">
          {/* Prompt & Sample Variations */}
          <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-4 space-y-3">
            <div className="flex items-center justify-between">
              <label
                htmlFor="prompt-editor"
                className="text-xs font-semibold uppercase tracking-wider text-zinc-400"
              >
                Input Prompt
              </label>
              <span className="text-[11px] font-mono text-zinc-500">
                Ctrl+Enter to run
              </span>
            </div>

            <textarea
              id="prompt-editor"
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter an evaluation prompt or test an adversarial query..."
              className="w-full rounded-lg bg-[#09090b] border border-zinc-800 focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/40 px-3 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none resize-y transition-colors"
            />

            {activeScenario && activeScenario.sample_prompts.length > 1 && (
              <div className="space-y-1.5">
                <span className="text-[11px] text-zinc-500 block">
                  Quick test cases:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {activeScenario.sample_prompts.map((sample, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => {
                        setPrompt(sample);
                        runPass({ overridePrompt: sample });
                      }}
                      className={`px-2 py-1 rounded text-[11px] border transition-colors cursor-pointer truncate max-w-full ${
                        prompt === sample
                          ? "bg-indigo-500/15 border-indigo-500/40 text-indigo-300"
                          : "bg-zinc-900/90 border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:border-zinc-700"
                      }`}
                      title={sample}
                    >
                      Case #{idx + 1}: {sample.slice(0, 34)}...
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Visual Context Uploader */}
          <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                Visual Context
              </span>
              <label className="inline-flex items-center gap-2 text-xs text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useVisualContext}
                  onChange={(e) => setUseVisualContext(e.target.checked)}
                  className="accent-indigo-500 rounded cursor-pointer"
                />
                Attach Image
              </label>
            </div>

            {useVisualContext ? (
              <div className="flex items-center gap-3">
                <div className="h-20 w-20 rounded-lg border border-zinc-800 bg-[#09090b] overflow-hidden shrink-0 flex items-center justify-center">
                  {previewImageSrc ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={previewImageSrc}
                      alt="Multimodal Visual Context"
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <ImagePlus className="h-6 w-6 text-zinc-600" />
                  )}
                </div>

                <div className="flex-1 space-y-2">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={(e) => handleFileUpload(e.target.files?.[0])}
                    className="hidden"
                  />
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-700/80 text-xs text-zinc-200 transition-colors cursor-pointer"
                    >
                      <ImagePlus className="h-3.5 w-3.5 text-indigo-400" />
                      Upload Image
                    </button>
                    {customImageB64 && (
                      <button
                        type="button"
                        onClick={() => setCustomImageB64(null)}
                        className="inline-flex items-center gap-1 px-2 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-xs text-zinc-400 cursor-pointer"
                      >
                        <RotateCcw className="h-3 w-3" />
                        Reset
                      </button>
                    )}
                  </div>
                  <p className="text-[11px] text-zinc-500">
                    {customImageB64
                      ? "Custom image attached (Conv2d 16x16 patches)."
                      : "Using scenario synthetic visual patch context."}
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-xs text-zinc-500 italic py-2">
                Visual encoder bypassed — running text-only prompt evaluation.
              </p>
            )}
          </div>

          {/* Gatekeeper & Steering Policy Controls */}
          <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-4 space-y-4">
            <div className="flex items-center gap-2">
              <Sliders className="h-4 w-4 text-indigo-400" />
              <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                Gatekeeper &amp; Steering Policy
              </span>
            </div>

            {/* Mode Switcher */}
            <div className="grid grid-cols-2 gap-1 bg-[#09090b] p-1 rounded-lg border border-zinc-800">
              <button
                type="button"
                onClick={() => setSteeringMode("autonomous")}
                className={`py-1.5 px-2 rounded-md text-xs font-medium transition-all cursor-pointer ${
                  steeringMode === "autonomous"
                    ? "bg-indigo-600/25 text-indigo-200 border border-indigo-500/40"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                Autonomous Gate
              </button>
              <button
                type="button"
                onClick={() => setSteeringMode("manual")}
                className={`py-1.5 px-2 rounded-md text-xs font-medium transition-all cursor-pointer ${
                  steeringMode === "manual"
                    ? "bg-amber-500/20 text-amber-200 border border-amber-500/40"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                Manual &alpha; Override
              </button>
            </div>

            {/* Probe Threshold Slider */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-zinc-300">Probe Risk Threshold (&theta;)</span>
                <span className="font-mono font-semibold text-indigo-300">
                  {(probeThreshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={probeThreshold}
                onChange={(e) => setProbeThreshold(parseFloat(e.target.value))}
                className="steer-slider"
              />
              <p className="text-[11px] text-zinc-500">
                Autonomous steering engages if latent probe risk meets or exceeds &theta;.
              </p>
            </div>

            {/* Manual Alpha Slider */}
            {steeringMode === "manual" && (
              <div className="space-y-1.5 pt-2 border-t border-zinc-800/80">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-300">Steering Multiplier (&alpha;)</span>
                  <span className={`font-mono font-semibold ${alphaZoneLabel.cls}`}>
                    {manualAlpha >= 0 ? "+" : ""}
                    {manualAlpha.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={-3.0}
                  max={3.0}
                  step={0.25}
                  value={manualAlpha}
                  onChange={(e) => setManualAlpha(parseFloat(e.target.value))}
                  className="steer-slider"
                />
                <div className="flex items-center justify-between text-[10px] font-mono">
                  <span className="text-rose-400">-3.0 Red-Team</span>
                  <span className={alphaZoneLabel.cls}>{alphaZoneLabel.text}</span>
                  <span className="text-emerald-400">+3.0 Remediate</span>
                </div>
              </div>
            )}

            {/* Execute Button */}
            <button
              type="button"
              disabled={loading || !prompt.trim()}
              onClick={() => runPass()}
              className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium shadow-lg shadow-indigo-950/50 flex items-center justify-center gap-2 transition-all cursor-pointer"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Running Forward Pass...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-current" />
                  Execute Harness Forward Pass
                </>
              )}
            </button>

            {inferenceError && (
              <p className="text-xs text-rose-400 bg-rose-950/30 border border-rose-500/30 rounded-lg px-3 py-2">
                {inferenceError}
              </p>
            )}
          </div>
        </aside>

        {/* RIGHT TELEMETRY & OUTPUT STAGE */}
        <section className="lg:col-span-8 space-y-5">
          {/* KPI Metric Ribbon */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-3.5">
              <span className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">
                Governance State
              </span>
              <div className="mt-1.5 flex items-center gap-2">
                {result?.audit_trace.steered ? (
                  <>
                    <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />
                    <span className="text-sm font-semibold text-emerald-300">
                      STEERED
                    </span>
                  </>
                ) : (
                  <>
                    <Activity className="h-4 w-4 text-zinc-400 shrink-0" />
                    <span className="text-sm font-semibold text-zinc-200">
                      {result ? "BASELINE" : "READY"}
                    </span>
                  </>
                )}
              </div>
              <span className="text-[11px] text-zinc-500 mt-1 block">
                {result?.steering_mode === "manual"
                  ? "Manual override active"
                  : "Autonomous policy gate"}
              </span>
            </div>

            <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-3.5">
              <span className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">
                Harness Latency
              </span>
              <div className="mt-1.5 flex items-baseline gap-1.5">
                <span className="text-lg font-mono font-semibold text-zinc-100">
                  {result ? `${result.audit_trace.latency_ms.toFixed(1)} ms` : "--"}
                </span>
              </div>
              <span className="text-[11px] text-emerald-400/90 mt-0.5 flex items-center gap-1">
                <Zap className="h-3 w-3" />
                {result
                  ? `${result.speedup_vs_external_llm}x faster than LLM judge`
                  : "Sub-100ms in-flight"}
              </span>
            </div>

            <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-3.5">
              <span className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">
                Probes Breached
              </span>
              <div className="mt-1.5">
                <span
                  className={`text-lg font-mono font-semibold ${
                    result &&
                    result.audit_trace.probes.some((p) => p.triggered)
                      ? "text-rose-400"
                      : "text-emerald-300"
                  }`}
                >
                  {result
                    ? `${result.audit_trace.probes.filter((p) => p.triggered).length} / ${result.audit_trace.probes.length}`
                    : "--"}
                </span>
              </div>
              <span className="text-[11px] text-zinc-500 mt-0.5 block">
                Cutoff &theta; = {(probeThreshold * 100).toFixed(0)}%
              </span>
            </div>

            <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-3.5">
              <span className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">
                Active Hooks
              </span>
              <div className="mt-1.5">
                <span className="text-lg font-mono font-semibold text-indigo-300">
                  {result ? result.audit_trace.steering_applied.length : "--"}
                </span>
              </div>
              <span className="text-[11px] text-zinc-500 mt-0.5 block truncate font-mono">
                {result && result.audit_trace.steering_applied.length > 0
                  ? result.audit_trace.steering_applied
                      .map((s) => `L${s.layer_idx}(α=${s.alpha >= 0 ? "+" : ""}${s.alpha})`)
                      .join(", ")
                  : "No residual injection"}
              </span>
            </div>
          </div>

          {/* Side-by-Side Output Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Unprotected Baseline Card */}
            <div className="rounded-xl border border-rose-500/25 bg-[#111116] flex flex-col overflow-hidden">
              <div className="px-4 py-3 border-b border-zinc-800/80 bg-rose-950/15 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ShieldAlert className="h-4 w-4 text-rose-400" />
                  <h3 className="text-sm font-semibold text-rose-200">
                    Unprotected Baseline
                  </h3>
                </div>
                <span className="text-[11px] font-mono text-zinc-400">
                  {result ? `${result.baseline_latency_ms.toFixed(1)} ms` : "Unhooked"}
                </span>
              </div>
              <div className="p-4 flex-1 flex flex-col justify-between gap-3">
                <p className="text-sm text-zinc-200 leading-relaxed">
                  {result
                    ? result.baseline_output
                    : "Run a forward pass to inspect baseline VLM output without steering."}
                </p>
                <div className="text-[11px] text-zinc-500 pt-2 border-t border-zinc-800/60">
                  Standard unhooked transformer forward pass. Vulnerable to latent bias, jailbreaks, or fabricated scene objects.
                </div>
              </div>
            </div>

            {/* Steered Output Card */}
            <div
              className={`rounded-xl border flex flex-col overflow-hidden transition-colors ${
                result?.audit_trace.steered
                  ? "border-emerald-500/35 bg-[#111116] shadow-[0_0_30px_-14px_rgba(16,185,129,0.35)]"
                  : "border-zinc-800/80 bg-[#111116]"
              }`}
            >
              <div
                className={`px-4 py-3 border-b border-zinc-800/80 flex items-center justify-between ${
                  result?.audit_trace.steered ? "bg-emerald-950/20" : "bg-zinc-900/50"
                }`}
              >
                <div className="flex items-center gap-2">
                  <Sparkles
                    className={`h-4 w-4 ${
                      result?.audit_trace.steered
                        ? "text-emerald-400"
                        : "text-indigo-400"
                    }`}
                  />
                  <h3
                    className={`text-sm font-semibold ${
                      result?.audit_trace.steered
                        ? "text-emerald-200"
                        : "text-zinc-200"
                    }`}
                  >
                    VLM-SteerHarness Output
                  </h3>
                </div>
                {result?.audit_trace.steered ? (
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                    REMEDIATED
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-zinc-800 text-zinc-300 border border-zinc-700">
                    PASSED CLEAN
                  </span>
                )}
              </div>
              <div className="p-4 flex-1 flex flex-col justify-between gap-3">
                <p className="text-sm text-zinc-100 leading-relaxed">
                  {result
                    ? result.steered_output
                    : "Governed output will appear here alongside active hook interventions."}
                </p>
                <div className="text-[11px] text-zinc-400 pt-2 border-t border-zinc-800/60 flex items-center gap-1.5 flex-wrap">
                  {result?.audit_trace.steered ? (
                    <>
                      <ArrowRight className="h-3 w-3 text-emerald-400 shrink-0" />
                      <span>
                        Dynamic residual injection active across{" "}
                        {result.audit_trace.steering_applied.length} concept vector(s).
                      </span>
                    </>
                  ) : (
                    <span>
                      All latent risk probes remained within policy cutoff &theta;.
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* In-Flight Latent Concept Probes */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-zinc-200">
                In-Flight Latent Concept Probes
              </h2>
              <span className="text-xs text-zinc-500">
                White marker denotes active risk cutoff (&theta; = {(probeThreshold * 100).toFixed(0)}%)
              </span>
            </div>

            {result?.audit_trace.probes ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {result.audit_trace.probes.map((probe) => (
                  <RiskProbeCard
                    key={probe.concept_name}
                    probe={probe}
                    threshold={probeThreshold}
                  />
                ))}
              </div>
            ) : (
              <div className="rounded-xl border border-zinc-800/80 bg-[#111116] p-6 text-center text-sm text-zinc-500">
                Waiting for first forward pass...
              </div>
            )}
          </div>

          {/* Causal Neuron Attribution */}
          <NeuronAttributionPanel
            attributions={result?.enriched_attributions || []}
          />

          {/* Audit Trace Telemetry */}
          <AuditTraceDrawer trace={result?.audit_trace || null} />
        </section>
      </div>
    </div>
  );
}
