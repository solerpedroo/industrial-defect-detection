"use client";

import { useEffect, useRef, useState } from "react";
import { FormPanel } from "@/components/layout/form-panel";
import { Button } from "@/components/ui/button";
import type { CategorySummary, ModelCard } from "@/types/api";
import { FIELDS, parseReport, REPORT_LIMIT, reportCsv, type SavedReport } from "./report";

export function SavedComparison({ categories, models }: {
  categories: readonly CategorySummary[]; models: readonly ModelCard[];
}) {
  const [report, setReport] = useState<SavedReport | null>(null);
  const [filename, setFilename] = useState("");
  const [category, setCategory] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const generation = useRef(0);
  useEffect(() => () => { generation.current += 1; }, []);

  async function open(file?: File) {
    const version = ++generation.current;
    setReport(null); setMessage(null); setFilename(""); setCategory(""); setPending(false);
    if (!file) return;
    if (file.size > REPORT_LIMIT) { setMessage("Report exceeds the 16 MiB limit."); return; }
    setPending(true);
    try {
      const text = await file.text();
      if (version !== generation.current) return;
      const parsed = parseReport(text, categories.map((item) => item.id), models.map((item) => item.id));
      setReport(parsed); setFilename(file.name); setCategory(parsed.categories[0]);
    } catch (error) {
      if (version === generation.current) setMessage(error instanceof Error ? error.message : "Could not read report.");
    } finally {
      if (version === generation.current) setPending(false);
    }
  }

  function download() {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([reportCsv(report, category)], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url; link.download = `comparison-${category}.csv`; link.click();
    // Give the browser time to start consuming the download before releasing it.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const rows = report?.models.map((model) => report.rows.find((row) => row.category === category && row.model === model)!) ?? [];
  return <FormPanel title="Saved experiment metrics"
    subtitle="Open a completed Wave 5 results.json (up to 16 MiB). The file stays in this browser tab; nothing is uploaded or persisted.">
    <label className="block text-sm">Experiment report
      <input className="field mt-2" type="file" accept=".json,application/json" onChange={(event) => void open(event.target.files?.[0])} />
    </label>
    {pending ? <p role="status" className="mt-3 text-sm">Reading report…</p> : null}
    {message ? <p role="alert" className="mt-3 text-sm">{message}</p> : null}
    {!report && !pending && !message ? <p className="mt-4 text-sm text-muted-foreground">No saved report selected. Live upload results are independent of this table.</p> : null}
    {report ? <section aria-label="Saved metrics" className="mt-5 space-y-4">
      <p className="break-all text-sm">File: {filename} · {report.protocol} · {report.scope} category coverage</p>
      {report.dataKind === "synthetic" ? <p role="note" className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
        Synthetic experiment — engineering evidence only, not an MVTec benchmark.
      </p> : null}
      <p className="text-sm text-muted-foreground">Recorded official test split; thresholds frozen on normal validation.
        These saved metrics do not evaluate the current upload or guarantee the same installed checkpoints.
        Structural validation does not authenticate the file or recompute its metrics.</p>
      <div className="flex flex-wrap items-end gap-4">
        <label className="text-sm">Saved category
          <select className="field mt-2" value={category} onChange={(event) => setCategory(event.target.value)}>
            {report.categories.map((id) => <option key={id} value={id}>{categories.find((item) => item.id === id)?.name ?? id}</option>)}
          </select>
        </label>
        <Button type="button" onClick={download}>Export visible table CSV</Button>
      </div>
      <p className="text-sm">Seed {report.seed} · {report.imageSize} × {report.imageSize} model pixels · {report.deviceName} ({report.device}) · {report.warmup} warmup images · {rows[0]?.testCount} test images per model</p>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <caption className="pb-3 text-left font-semibold">Official test metrics — {categories.find((item) => item.id === category)?.name ?? category}</caption>
          <thead><tr className="border-b border-border"><th scope="col" className="p-3">Metric / unit</th>
            {rows.map((row) => <th key={row.model} scope="col" className="p-3">{models.find((item) => item.id === row.model)?.name ?? row.model}</th>)}
          </tr></thead>
          <tbody>{FIELDS.map(([key, label, unit]) => <tr key={key} className="border-b border-border/60">
            <th scope="row" className="p-3 font-normal">{label} <span className="text-muted-foreground">({unit})</span></th>
            {rows.map((row) => <td key={row.model} className="p-3 tabular-nums">
              {row.cells[key].value === null ? <span>N/A — {row.cells[key].reason}</span> : row.cells[key].value.toString()}
            </td>)}
          </tr>)}</tbody>
        </table>
      </div>
      <p className="text-xs text-muted-foreground">Average precision is non-interpolated AP, not trapezoidal PR area.
        AUPRO is normalized up to background FPR {report.auproLimit}. Pixel metrics use model resolution.
        Latency includes warmed prediction and output copy, excluding file decoding/model loading.
        CPU peak memory is a process lifetime high-water mark, not isolated per-model usage; CUDA uses allocator peak including the loaded model.</p>
      <details className="space-y-2 break-all text-xs text-muted-foreground"><summary>Saved provenance and operating points</summary>
        <p>Evaluation config SHA-256: {report.configHash}</p><p>Calibration SHA-256: {report.calibrationHash}</p>
        {rows.map((row) => <div key={row.model} className="mt-3 space-y-1">
          <p>{row.model}: image threshold {row.imageThreshold}; pixel threshold {row.pixelThreshold}; strict &gt;</p>
          <p>Model config SHA-256: {row.modelHash}</p><p>Artifact SHA-256: {row.artifactHash}</p>
          <p>Memory source: {row.memorySource}; preparation source: {row.prepareSource}</p>
        </div>)}
      </details>
    </section> : null}
  </FormPanel>;
}
