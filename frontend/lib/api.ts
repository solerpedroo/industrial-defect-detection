import type {
  DatasetCatalog,
  ExperimentsResponse,
  HealthResponse,
  ModelsResponse,
} from "@/types/api";

export type { DatasetCatalog, ExperimentProtocol, HealthResponse, ModelCard, ModelsResponse } from "@/types/api";

const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      return new ApiError(body.detail, response.status);
    }
  } catch {
    return new ApiError(`API request failed: ${response.status}`, response.status);
  }
  return new ApiError(`API request failed: ${response.status}`, response.status);
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, init);
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  return (await response.json()) as T;
}

export function getHealth() {
  return request<HealthResponse>("/api/health");
}

export function getDatasets() {
  return request<DatasetCatalog>("/api/datasets");
}

export function getModels() {
  return request<ModelsResponse>("/api/models");
}

export function getExperiments() {
  return request<ExperimentsResponse>("/api/experiments");
}

export function postInference(form: FormData) {
  return request<unknown>("/api/inference", { method: "POST", body: form });
}

export function postCompare(form: FormData) {
  return request<unknown>("/api/inference/compare", { method: "POST", body: form });
}
