export type CategorySummary = {
  id: string;
  name: string;
  kind: string;
  purpose: string;
  availability: "missing" | "present";
  train_normal: number;
  validation_normal: number;
  test_normal: number;
  test_anomalous: number;
  mask_count: number;
  defect_types: readonly { id: string; test_images: number }[];
  validation_split: "not_created" | "created" | "invalid";
};

export type DatasetCatalog = {
  id: string;
  name: string;
  license: string;
  license_url: string;
  commercial_use: boolean;
  storage_root: string;
  categories: readonly CategorySummary[];
};

export type ModelCard = {
  id: string;
  name: string;
  family: string;
  status: "planned" | "ready";
  backbone: string | null;
  description: string;
  ready_categories: readonly string[];
  synthetic_categories: readonly string[];
};

export type ModelsResponse = {
  models: readonly ModelCard[];
};

export type ExperimentProtocol = {
  id: string;
  title: string;
  status: "planned" | "complete";
  question: string;
  categories: readonly string[];
  models: readonly string[];
  metrics: readonly string[];
};

export type ExperimentsResponse = {
  experiments: readonly ExperimentProtocol[];
};

export type HealthResponse = {
  status: "ok";
  api_version: string;
};

export type InferenceResponse = {
  model_id: string;
  category_id: string;
  score: number;
  label: "normal" | "anomalous";
  threshold: number;
  pixel_threshold: number;
  input_png: string;
  anomaly_map_png: string;
  overlay_png: string;
  mask_png: string;
  reconstruction_png: string | null;
  display_max: number;
  map_width: number;
  map_height: number;
  data_kind: "synthetic" | "mvtec_ad";
  artifact_sha256: string;
  calibration_sha256: string;
  config_hash: string;
  inference_ms: number;
  request_ms: number;
};

export type CompareResponse = {
  category_id: string;
  results: readonly InferenceResponse[];
  request_ms: number;
};
