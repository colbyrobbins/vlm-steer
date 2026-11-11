import type {
  BenchmarkSummary,
  CalibrationPair,
  ConceptMetadata,
  HealthStatus,
  InferencePayload,
  InferenceResponse,
  ScenarioPreset,
} from "@/types/harness";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const errBody = await res.json();
      if (errBody?.error) detail = errBody.error;
    } catch {
      // Ignore JSON parse errors on non-JSON failure responses
    }
    throw new Error(detail);
  }

  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthStatus> {
  return requestJson<HealthStatus>("/api/health");
}

export async function fetchConcepts(): Promise<ConceptMetadata[]> {
  const data = await requestJson<{ concepts: ConceptMetadata[] }>("/api/concepts");
  return data.concepts;
}

export async function fetchScenarios(): Promise<ScenarioPreset[]> {
  const data = await requestJson<{ scenarios: ScenarioPreset[] }>("/api/scenarios");
  return data.scenarios;
}

export async function fetchCalibration(): Promise<Record<string, CalibrationPair[]>> {
  const data = await requestJson<{ datasets: Record<string, CalibrationPair[]> }>(
    "/api/calibration"
  );
  return data.datasets;
}

export async function fetchBenchmarks(): Promise<BenchmarkSummary> {
  return requestJson<BenchmarkSummary>("/api/benchmarks");
}

export async function runLiveBenchmarks(trials = 20): Promise<BenchmarkSummary> {
  return requestJson<BenchmarkSummary>("/api/benchmarks/run", {
    method: "POST",
    body: JSON.stringify({ trials }),
  });
}

export async function executeInference(
  payload: InferencePayload
): Promise<InferenceResponse> {
  return requestJson<InferenceResponse>("/api/inference", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
