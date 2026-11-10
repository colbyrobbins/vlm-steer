export interface HealthStatus {
  status: string;
  model_id: string;
  hidden_dim: number;
  num_layers: number;
  device: string;
  concepts_loaded: number;
}

export interface ConceptMetadata {
  name: string;
  layer_idx: number;
  dimension: number;
  separation_margin: number;
  description: string;
  positive_norm: number;
  negative_norm: number;
}

export interface ScenarioPreset {
  id: string;
  title: string;
  concept_name: string;
  target_layer: number;
  badge: string;
  description: string;
  default_prompt: string;
  sample_prompts: string[];
  default_image_b64: string;
}

export interface ProbeResult {
  concept_name: string;
  layer_idx: number;
  raw_score: number;
  normalized_risk: number;
  triggered: boolean;
}

export interface SteeringRecord {
  concept_name: string;
  layer_idx: number;
  alpha: number;
  token_scope: string;
}

export interface EnrichedNeuronAttribution {
  layer_idx: number;
  concept_name: string;
  top_neuron_indices: number[];
  top_neuron_scores: number[];
  signed_neuron_scores: number[];
}

export interface AuditTrace {
  trace_id: string;
  timestamp: number;
  prompt: string;
  has_image: boolean;
  probes: ProbeResult[];
  steered: boolean;
  steering_applied: SteeringRecord[];
  neuron_attributions: {
    layer_idx: number;
    top_neuron_indices: number[];
    top_neuron_scores: number[];
  }[];
  output_text: string;
  latency_ms: number;
}

export interface InferenceResponse {
  baseline_output: string;
  baseline_latency_ms: number;
  steered_output: string;
  audit_trace: AuditTrace;
  enriched_attributions: EnrichedNeuronAttribution[];
  probe_threshold: number;
  steering_mode: "autonomous" | "manual";
  manual_alpha: number | null;
  speedup_vs_external_llm: number;
}

export interface InferencePayload {
  prompt: string;
  image_base64?: string | null;
  use_default_image: boolean;
  probe_threshold: number;
  steering_mode: "autonomous" | "manual";
  manual_alpha: number;
  max_new_tokens?: number;
}

export interface ScenarioBenchmarkMetric {
  scenario_name: string;
  num_samples: number;
  baseline_compliance_or_bias_rate: number;
  steered_compliance_or_bias_rate: number;
  remediation_rate: number;
  probe_breach_detection_rate: number;
  mean_probe_risk: number;
}

export interface LatencyProfileMetric {
  num_trials: number;
  baseline_p50_ms: number;
  baseline_p95_ms: number;
  baseline_p99_ms: number;
  steered_p50_ms: number;
  steered_p95_ms: number;
  steered_p99_ms: number;
  mean_overhead_ms: number;
  overhead_percentage: number;
  speedup_vs_external_llm_judge: number;
}

export interface AttributionFaithfulnessMetric {
  concept_name: string;
  layer_idx: number;
  top_k: number;
  baseline_concept_alignment: number;
  ablated_top_k_alignment: number;
  ablated_random_k_alignment: number;
  alignment_drop_top_k: number;
  alignment_drop_random_k: number;
  faithfulness_ratio: number;
}

export interface BenchmarkSummary {
  timestamp: string;
  total_eval_samples: number;
  overall_remediation_rate: number;
  scenario_results: Record<string, ScenarioBenchmarkMetric>;
  latency_profile: LatencyProfileMetric;
  attribution_faithfulness: AttributionFaithfulnessMetric[];
}

export interface CalibrationPair {
  id: string;
  positive: string;
  negative: string;
  domain?: string;
  category?: string;
  type?: string;
}
