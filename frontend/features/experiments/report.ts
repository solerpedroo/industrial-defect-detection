/** Read-only projection of Wave 5 reports. No metric or threshold computation. */
export const REPORT_LIMIT = 16 * 1024 * 1024;

export const FIELDS = [
  ["image_auroc", "Image AUROC", "ratio", "metrics"],
  ["image_aupr", "Image average precision", "ratio", "metrics"],
  ["precision", "Image precision", "ratio", "metrics"],
  ["recall", "Image recall", "ratio", "metrics"],
  ["f1", "Image F1", "ratio", "metrics"],
  ["false_positive_rate", "Image false-positive rate", "ratio", "metrics"],
  ["false_negative_rate", "Image false-negative rate", "ratio", "metrics"],
  ["tp", "True positives", "count", "metrics"],
  ["fp", "False positives", "count", "metrics"],
  ["tn", "True negatives", "count", "metrics"],
  ["fn", "False negatives", "count", "metrics"],
  ["pixel_auroc", "Pixel AUROC", "ratio", "metrics"],
  ["pixel_aupr", "Pixel average precision", "ratio", "metrics"],
  ["aupro", "Pixel AUPRO", "ratio", "metrics"],
  ["iou", "Pixel IoU", "ratio", "metrics"],
  ["dice", "Pixel Dice", "ratio", "metrics"],
  ["inference_latency_ms", "Mean warmed inference latency", "ms", "compute"],
  ["latency_median_ms", "Median warmed inference latency", "ms", "compute"],
  ["latency_p95_ms", "P95 warmed inference latency", "ms", "compute"],
  ["throughput_images_per_s", "Throughput", "images/s", "compute"],
  ["prepare_time_s", "Preparation time", "s", "compute"],
  ["artifact_size_bytes", "Artifact size", "bytes", "compute"],
  ["peak_memory_bytes", "Approximate peak memory", "bytes", "compute"],
] as const;

type Cell = { value: number | null; reason: string };
export type ReportRow = {
  category: string; model: string; modelHash: string; artifactHash: string;
  imageThreshold: number; pixelThreshold: number; testCount: number;
  memorySource: string; prepareSource: string; cells: Record<string, Cell>;
};
export type SavedReport = {
  dataKind: "synthetic" | "mvtec_ad"; scope: "partial" | "complete";
  protocol: string; seed: number; imageSize: number; warmup: number; auproLimit: number;
  configHash: string; calibrationHash: string; categories: string[]; models: string[];
  device: string; deviceName: string; rows: ReportRow[];
};
type ObjectValue = Record<string, unknown>;
function requireValue(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Invalid report: ${message}`);
}
function object(value: unknown): ObjectValue {
  requireValue(value !== null && typeof value === "object" && !Array.isArray(value), "expected an object.");
  return value as ObjectValue;
}
function string(value: unknown): string {
  requireValue(typeof value === "string" && value.length > 0 && value.length <= 2048, "missing or oversized text.");
  return value;
}
function hash(value: unknown): string {
  requireValue(typeof value === "string" && /^[a-f0-9]{64}$/.test(value), "missing SHA-256 provenance.");
  return value;
}
function number(value: unknown, integer = false): number {
  requireValue(typeof value === "number" && Number.isFinite(value) && value >= 0 && (!integer || Number.isSafeInteger(value)), "invalid numeric value.");
  return value;
}
function strings(value: unknown): string[] {
  requireValue(Array.isArray(value) && value.length > 0, "empty membership.");
  const result = value.map(string);
  requireValue(new Set(result).size === result.length, "duplicate membership.");
  return result;
}
function same(a: unknown, b: unknown) { return JSON.stringify(a) === JSON.stringify(b); }
function threshold(value: unknown, target: unknown): number {
  const item = object(value);
  const targetFpr = number(item.target_fpr);
  requireValue(item.source === "normal_validation" && item.method === "higher" && item.comparator === ">", "threshold must be frozen on normal validation.");
  requireValue(targetFpr > 0 && targetFpr < 1 && targetFpr === target &&
    item.quantile === 1 - targetFpr && number(item.sample_count, true) > 0 && number(item.observed_fpr) <= targetFpr,
  "inconsistent calibration convention.");
  return number(item.threshold);
}

export function parseReport(text: string, categoryIds: readonly string[], modelIds: readonly string[]): SavedReport {
  requireValue(new TextEncoder().encode(text).byteLength <= REPORT_LIMIT, "file exceeds 16 MiB.");
  let parsed: unknown;
  try { parsed = JSON.parse(text); } catch { throw new Error("Invalid report: expected JSON."); }
  const root = object(parsed), config = object(root.config);
  requireValue(root.schema_version === 1 && root.complete === true && root.phase === "official_test_report", "select a completed Wave 5 results.json, not calibration.json.");
  requireValue(root.data_kind === "synthetic" || root.data_kind === "mvtec_ad", "unknown data kind.");
  requireValue(root.protocol_scope === "complete" || root.protocol_scope === "partial", "unknown protocol coverage.");
  const categories = strings(config.categories), models = strings(config.models);
  requireValue(categories.every((id) => categoryIds.includes(id)) && models.length === 2 && models.every((id) => modelIds.includes(id)), "unknown catalog IDs or model coverage.");
  const complete = categoryIds.length === categories.length && categoryIds.every((id) => categories.includes(id));
  requireValue((root.protocol_scope === "complete") === complete, "incorrect category coverage label.");
  const configHash = hash(root.config_hash), calibrationHash = hash(root.calibration_sha256);
  const seed = number(config.seed, true), imageSize = number(config.image_size, true);
  requireValue(seed < 2 ** 32 && imageSize >= 32 && imageSize <= 4096 && imageSize % 32 === 0, "invalid seed or resolution.");
  const warmup = number(config.warmup_images, true), auproLimit = number(config.aupro_fpr_limit);
  requireValue(warmup > 0 && auproLimit > 0 && auproLimit <= 1, "missing warmup or AUPRO limit.");
  const tests = object(root.tests), calibration = object(root.calibration);
  requireValue(Array.isArray(root.results) && root.results.length === categories.length * models.length, "incomplete result rows.");
  const seen = new Set<string>(), labelsByCategory = new Map<string, unknown>();
  const rows = root.results.map((value): ReportRow => {
    const row = object(value), category = string(row.category), model = string(row.model_id);
    requireValue(categories.includes(category) && models.includes(model) && !seen.has(`${category}/${model}`), "duplicate or unexpected result row.");
    seen.add(`${category}/${model}`);
    requireValue(row.config_hash === configHash && row.seed === seed, "row config/seed mismatch.");
    const test = object(tests[category]), ids = strings(row.test_ids);
    requireValue(same(ids, strings(test.test_ids)) && ids.every((id) => id.startsWith(`${category}/test/`) && !id.split("/").includes("..")), "models must share ordered official test IDs.");
    const checksums = object(test.source_checksums);
    ids.forEach((id) => hash(checksums[id]));
    requireValue(Array.isArray(row.labels) && row.labels.length === ids.length && row.labels.every((label) => typeof label === "boolean"), "invalid test labels.");
    if (labelsByCategory.has(category)) requireValue(same(labelsByCategory.get(category), row.labels), "models must share test labels.");
    labelsByCategory.set(category, row.labels);
    requireValue(Array.isArray(row.scores) && row.scores.length === ids.length, "invalid score membership.");
    row.scores.forEach((score) => number(score));
    const frozen = object(object(calibration[category])[model]), thresholds = object(row.thresholds);
    const modelHash = hash(row.model_config_hash);
    requireValue(modelHash === hash(frozen.model_config_hash), "model config provenance mismatch.");
    for (const level of ["image", "pixel"] as const) {
      const stored = object(frozen[level]), reported = object(thresholds[level]);
      for (const key of ["threshold", "source", "method", "comparator", "target_fpr", "quantile", "sample_count", "observed_fpr"]) {
        requireValue(stored[key] === reported[key], "result threshold differs from frozen calibration.");
      }
    }
    const imageThreshold = threshold(thresholds.image, config.target_image_fpr);
    const pixelThreshold = threshold(thresholds.pixel, config.target_pixel_fpr);
    const metrics = object(row.metrics), compute = object(row.compute), undefinedReasons = object(metrics.undefined);
    requireValue(compute.timing_scope === "prediction_and_output_copy_no_file_decode_or_model_load", "unsupported latency scope.");
    const memorySource = string(compute.peak_memory_source), prepareSource = string(frozen.prepare_time_source);
    const cells: Record<string, Cell> = {};
    for (const [key, , unit, group] of FIELDS) {
      const entry = (group === "metrics" ? metrics : compute)[key];
      if (entry === null) {
        requireValue(unit !== "count" && (group === "metrics" || key === "prepare_time_s" || key === "peak_memory_bytes"), "missing required value.");
        cells[key] = { value: null, reason: group === "metrics" ? string(undefinedReasons[key]) : key === "prepare_time_s" ? prepareSource : memorySource };
      } else {
        const numeric = number(entry, unit === "count" || unit === "bytes");
        requireValue(unit !== "ratio" || numeric <= 1, "metric is outside [0,1].");
        cells[key] = { value: numeric, reason: "" };
      }
    }
    const positives = row.labels.filter(Boolean).length;
    const tp = cells.tp.value!, fp = cells.fp.value!, tn = cells.tn.value!, fn = cells.fn.value!;
    requireValue(tp + fn === positives && tn + fp === ids.length - positives,
      "image confusion counts do not match recorded test labels.");
    requireValue(compute.artifact_size_bytes === frozen.artifact_size_bytes && compute.prepare_time_s === frozen.prepare_time_s, "compute provenance mismatch.");
    return { category, model, modelHash, artifactHash: hash(frozen.artifact_sha256), imageThreshold, pixelThreshold,
      testCount: ids.length, memorySource, prepareSource, cells };
  });
  const environment = object(root.environment);
  return { dataKind: root.data_kind, scope: root.protocol_scope, protocol: string(config.protocol_id),
    seed, imageSize, warmup, auproLimit, configHash, calibrationHash, categories, models,
    device: string(environment.device), deviceName: string(environment.device_name), rows };
}

export function csvField(value: string | number | null): string {
  const text = value === null ? "" : String(value);
  // Quoting alone does not prevent spreadsheet formula evaluation.
  const safe = /^[\s\u0000-\u001f]*[=+@-]/.test(text) || /^[\t\r\n]/.test(text) ? `'${text}` : text;
  return `"${safe.replaceAll('"', '""')}"`;
}

export function reportCsv(report: SavedReport, category: string): string {
  const rows = report.models.map((model) => report.rows.find((row) => row.category === category && row.model === model)!);
  requireValue(rows.every(Boolean), "category is not in this report.");
  const headers = ["category", "split", "data_kind", "protocol_scope", "seed", "image_size", "config_sha256", "calibration_sha256", "device", "warmup_images", "aupro_fpr_limit", "metric", "unit",
    ...rows.flatMap((row) => [row.model, `${row.model}_undefined_reason`, `${row.model}_config_sha256`, `${row.model}_artifact_sha256`, `${row.model}_image_threshold`, `${row.model}_pixel_threshold`, `${row.model}_test_count`, `${row.model}_memory_source`, `${row.model}_prepare_source`])];
  const values = FIELDS.map(([key, , unit]) => [category, "official_test", report.dataKind, report.scope, report.seed, report.imageSize, report.configHash, report.calibrationHash, report.device, report.warmup, report.auproLimit, key, unit,
    ...rows.flatMap((row) => [row.cells[key].value, row.cells[key].reason, row.modelHash, row.artifactHash, row.imageThreshold, row.pixelThreshold, row.testCount, row.memorySource, row.prepareSource])]);
  return [headers, ...values].map((line) => line.map(csvField).join(",")).join("\r\n") + "\r\n";
}
