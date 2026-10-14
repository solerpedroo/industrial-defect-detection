import { describe, expect, it } from "vitest";
import { csvField, FIELDS, parseReport, REPORT_LIMIT, reportCsv } from "./report";

// Contract-only numbers, not benchmark evidence. Browser tests use the actual
// Python evaluator's generated report. No metric is calculated in this module.
function fixture() {
  const digest = "a".repeat(64);
  const categories = ["bottle", "leather"];
  const models = ["autoencoder", "patchcore"];
  const threshold = { threshold: 0.25, target_fpr: 0.5, quantile: 0.5, sample_count: 2,
    observed_fpr: 0, source: "normal_validation", method: "higher", comparator: ">" };
  const row = (category: string, model_id: string) => ({
    category, model_id, seed: 42, config_hash: digest, model_config_hash: digest,
    test_ids: [`${category}/test/good/1.png`, `${category}/test/broken/2.png`],
    labels: [false, true], scores: [0.1, 0.8],
    thresholds: { image: { ...threshold }, pixel: { ...threshold } },
    metrics: { ...Object.fromEntries(FIELDS.filter((field) => field[3] === "metrics").map(([key, , unit]) => [key, unit === "count" ? 1 : 0.5])), tp: 1, fp: 0, tn: 1, fn: 0, undefined: {} },
    compute: { ...Object.fromEntries(FIELDS.filter((field) => field[3] === "compute").map(([key]) => [key, 1])),
      timing_scope: "prediction_and_output_copy_no_file_decode_or_model_load", peak_memory_source: "process_lifetime_high_water_rss" },
  });
  return {
    schema_version: 1, complete: true, phase: "official_test_report", data_kind: "synthetic", protocol_scope: "partial",
    config: { protocol_id: "exp-a-autoencoder-vs-patchcore", categories, models, seed: 42, image_size: 32,
      warmup_images: 1, aupro_fpr_limit: 0.3, target_image_fpr: 0.5, target_pixel_fpr: 0.5 },
    config_hash: digest, calibration_sha256: digest,
    tests: Object.fromEntries(categories.map((category) => [category, { test_ids: row(category, models[0]).test_ids,
      source_checksums: Object.fromEntries(row(category, models[0]).test_ids.map((id) => [id, digest])) }])),
    calibration: Object.fromEntries(categories.map((category) => [category, Object.fromEntries(models.map((model) => [model, {
      image: { ...threshold }, pixel: { ...threshold }, artifact_sha256: digest, model_config_hash: digest,
      artifact_size_bytes: 1, prepare_time_s: 1, prepare_time_source: "checkpoint_metadata",
    }]))])),
    results: categories.flatMap((category) => models.map((model) => row(category, model))),
    environment: { device: "cpu", device_name: "CPU" },
  };
}
const parse = (value: unknown) => parseReport(JSON.stringify(value), ["bottle", "leather", "metal_nut"], ["autoencoder", "patchcore"]);

describe("saved report boundary", () => {
  it("projects both models/categories without changing values and exports only the selected category", () => {
    const report = parse(fixture());
    expect(report.rows).toHaveLength(4);
    const csv = reportCsv(report, "leather");
    expect(csv).toContain('"leather","official_test","synthetic","partial"');
    expect(csv).not.toContain('"bottle"');
    expect(csv).toContain('"image_auroc","ratio","0.5"');
    expect(csv).toContain(report.calibrationHash);
    expect(() => reportCsv(report, "metal_nut")).toThrow();
  });
  it("keeps undefined values blank in CSV and retains their reason", () => {
    const raw = fixture();
    Object.assign(raw.results[0].metrics, { image_auroc: null, undefined: { image_auroc: "Absent class." } });
    const report = parse(raw);
    expect(report.rows[0].cells.image_auroc).toEqual({ value: null, reason: "Absent class." });
    expect(reportCsv(report, "bottle")).toContain('"image_auroc","ratio","","Absent class."');
  });
  it.each([
    ["incomplete", (r: ReturnType<typeof fixture>) => { r.complete = false; }],
    ["schema", (r: ReturnType<typeof fixture>) => { r.schema_version = 2; }],
    ["calibration file", (r: ReturnType<typeof fixture>) => { r.phase = "thresholds_frozen_before_test"; }],
    ["unknown data", (r: ReturnType<typeof fixture>) => { r.data_kind = "real"; }],
    ["false complete label", (r: ReturnType<typeof fixture>) => { r.protocol_scope = "complete"; }],
    ["missing row", (r: ReturnType<typeof fixture>) => { r.results.pop(); }],
    ["duplicate row", (r: ReturnType<typeof fixture>) => { r.results[1] = r.results[0]; }],
    ["unknown category", (r: ReturnType<typeof fixture>) => { r.config.categories[0] = "unknown"; }],
    ["unknown model", (r: ReturnType<typeof fixture>) => { r.results[0].model_id = "unknown"; }],
    ["different IDs", (r: ReturnType<typeof fixture>) => { r.results[1].test_ids.reverse(); }],
    ["different labels", (r: ReturnType<typeof fixture>) => { r.results[1].labels.reverse(); }],
    ["duplicate IDs", (r: ReturnType<typeof fixture>) => { r.results[0].test_ids[1] = r.results[0].test_ids[0]; }],
    ["hash mismatch", (r: ReturnType<typeof fixture>) => { r.results[0].config_hash = "b".repeat(64); }],
    ["missing source hash", (r: ReturnType<typeof fixture>) => { r.tests.bottle.source_checksums = {}; }],
    ["negative metric", (r: ReturnType<typeof fixture>) => { Object.assign(r.results[0].metrics, { image_auroc: -1 }); }],
    ["metric above one", (r: ReturnType<typeof fixture>) => { Object.assign(r.results[0].metrics, { image_auroc: 2 }); }],
    ["undefined without reason", (r: ReturnType<typeof fixture>) => { Object.assign(r.results[0].metrics, { image_auroc: null }); }],
    ["unsafe count", (r: ReturnType<typeof fixture>) => { Object.assign(r.results[0].metrics, { tp: 2 ** 53 }); }],
    ["impossible confusion counts", (r: ReturnType<typeof fixture>) => { r.results[0].metrics.tp = 999; }],
    ["counts assigned to the wrong class", (r: ReturnType<typeof fixture>) => { r.results[0].metrics.tp = 2; r.results[0].metrics.tn = 0; }],
    ["threshold mismatch", (r: ReturnType<typeof fixture>) => { r.results[0].thresholds.image.threshold = 9; }],
    ["test calibration", (r: ReturnType<typeof fixture>) => { r.results[0].thresholds.image.source = "test"; r.calibration.bottle.autoencoder.image.source = "test"; }],
    ["timing scope", (r: ReturnType<typeof fixture>) => { r.results[0].compute.timing_scope = "unknown"; }],
  ])("rejects %s", (_, mutate) => {
    const raw = fixture(); mutate(raw); expect(() => parse(raw)).toThrow(/Invalid report/);
  });
  it("rejects malformed, oversized and non-finite JSON input", () => {
    for (const text of ["{", "null", " ".repeat(REPORT_LIMIT + 1), JSON.stringify(fixture()).replace('"image_auroc":0.5', '"image_auroc":1e999')]) {
      expect(() => parseReport(text, ["bottle", "leather", "metal_nut"], ["autoencoder", "patchcore"])).toThrow();
    }
  });
  it("escapes CSV quotes, delimiters, lines and spreadsheet formula prefixes", () => {
    expect(csvField('a,"b"\nc')).toBe('"a,""b""\nc"');
    for (const text of ["=1+1", " +cmd", "@SUM(1)", "-1+1", "\tformula", "\r=cmd"]) {
      expect(csvField(text).startsWith('"\'')).toBe(true);
    }
    expect(csvField(0.5)).toBe('"0.5"');
  });
});
