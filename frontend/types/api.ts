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
