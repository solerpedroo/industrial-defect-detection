"use client";

import Image from "next/image";
import { useEffect, useRef, useState, type FormEvent } from "react";

import { AsyncState } from "@/components/async-state";
import { Button } from "@/components/ui/button";
import { FormPanel } from "@/components/layout/form-panel";
import { ApiError, postCompare } from "@/lib/api";
import { useApiGet } from "@/lib/use-api";
import type { CompareResponse, DatasetCatalog, ModelsResponse } from "@/types/api";
import { SavedComparison } from "@/features/experiments/saved-comparison";

export function ComparisonBoard() {
  const datasets = useApiGet<DatasetCatalog>("/api/datasets");
  const models = useApiGet<ModelsResponse>("/api/models");
  const [message, setMessage] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [categoryId, setCategoryId] = useState("");
  const [modelA, setModelA] = useState("");
  const [modelB, setModelB] = useState("");
  const [opacity, setOpacity] = useState(50);
  const generation = useRef(0);
  const active = useRef<AbortController | null>(null);
  const category = categoryId || datasets.data?.categories[0]?.id || "";
  const installed = models.data?.models.filter((model) => model.ready_categories.includes(category)) ?? [];
  const a = modelA || installed[0]?.id || "";
  const b = modelB || installed.find((model) => model.id !== a)?.id || "";
  const ready = a !== b && installed.some((model) => model.id === a) && installed.some((model) => model.id === b);

  useEffect(() => () => { generation.current += 1; active.current?.abort(); }, []);

  function invalidate() {
    generation.current += 1;
    active.current?.abort();
    setResult(null);
    setMessage(null);
    setPending(false);
    setOpacity(50);
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!ready) return;
    invalidate();
    const controller = new AbortController();
    active.current = controller;
    const version = generation.current;
    setPending(true);
    try {
      const response = await postCompare(new FormData(event.currentTarget), controller.signal);
      if (version === generation.current) setResult(response);
    } catch (error) {
      if (version === generation.current && !controller.signal.aborted) {
        setMessage(error instanceof ApiError ? error.message : "Comparison failed.");
      }
    } finally {
      if (version === generation.current) setPending(false);
    }
  }

  return (
    <div className="space-y-6">
      <FormPanel title="Same image, two models"
        subtitle="Live inspection with two distinct, calibrated models. PNG or JPEG, up to 10 MiB and 4096 × 4096.">
        <AsyncState loading={datasets.loading || models.loading} error={datasets.error ?? models.error}>
          <form className="grid gap-4 sm:grid-cols-2" onSubmit={onSubmit} onChange={invalidate}>
            <label className="block text-sm sm:col-span-2">
              <span className="label-caps">Image</span>
              <input className="field mt-2" name="image" type="file" accept="image/png,image/jpeg" required />
            </label>
            <label className="block text-sm sm:col-span-2">
              <span className="label-caps">Category</span>
              <select className="field mt-2" name="category_id" value={category} onChange={(event) => {
                setCategoryId(event.target.value); setModelA(""); setModelB("");
              }}>
                {datasets.data?.categories.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
            </label>
            {([ ["Model A", "model_a", a, setModelA], ["Model B", "model_b", b, setModelB] ] as const).map(([label, name, value, setter]) => (
              <label key={name} className="block text-sm">
                <span className="label-caps">{label}</span>
                <select className="field mt-2" name={name} value={value} onChange={(event) => setter(event.target.value)}>
                  <option value="" disabled>Select an installed model</option>
                  {models.data?.models.map((model) => <option key={model.id} value={model.id}
                    disabled={!model.ready_categories.includes(category)}>
                    {model.name}{model.ready_categories.includes(category) ? "" : " — not installed"}
                  </option>)}
                </select>
              </label>
            ))}
            <div className="sm:col-span-2">
              <Button type="submit" disabled={pending || !ready}>{pending ? "Comparing…" : "Compare models"}</Button>
              {!ready ? <p role="status" className="mt-3 text-sm text-muted-foreground">
                Select two distinct models installed for this category.
              </p> : null}
            </div>
          </form>
          {message ? <p role="alert" className="mt-4 text-sm">{message}</p> : null}
          {result ? <section aria-label="Live comparison" className="mt-6 space-y-4">
            <h2 className="text-xl font-semibold">Live comparison</h2>
            <p className="text-sm text-muted-foreground">Both models received the same decoded upload.
              Scores and heat colors use separate model scales, not severity or probability.
              Maps use each model’s resolution; differences are not a quality ranking.</p>
            <label className="block text-sm">Shared overlay opacity: {opacity}%
              <input aria-label="Shared overlay opacity" type="range" min="0" max="100" step="5" value={opacity}
                onChange={(event) => setOpacity(Number(event.target.value))} className="mt-2 w-full" />
            </label>
            <div className="grid gap-5 md:grid-cols-2">
              {result.results.map((item) => <article key={item.model_id} aria-label={`Result ${item.model_id}`}
                className="min-w-0 space-y-3 rounded-lg border border-border p-4">
                <h3 className="font-semibold">{models.data?.models.find((model) => model.id === item.model_id)?.name ?? item.model_id}</h3>
                {item.data_kind === "synthetic" ? <p role="note" className="text-sm text-amber-400">
                  Synthetic demonstration — not validated for industrial inspection.
                </p> : null}
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  <dt>Decision</dt><dd>{item.label}</dd>
                  <dt>Anomaly score</dt><dd>{item.score.toPrecision(5)}</dd>
                  <dt>Image threshold</dt><dd>{item.threshold.toPrecision(5)}</dd>
                  <dt>Pixel threshold</dt><dd>{item.pixel_threshold.toPrecision(5)}</dd>
                  <dt>Inference</dt><dd>{item.inference_ms.toFixed(1)} ms</dd>
                  <dt>Resolution</dt><dd>{item.map_width} × {item.map_height}</dd>
                </dl>
                <figure className="space-y-2">
                  <Image src={`data:image/png;base64,${item.anomaly_map_png}`} alt={`${item.model_id} heatmap`}
                    width={item.map_width} height={item.map_height} unoptimized className="aspect-square w-full rounded" />
                  <figcaption className="text-xs">Heatmap: blue (0) → red ({item.display_max ? item.display_max.toPrecision(5) : "nonzero"}). Fixed calibration scale.</figcaption>
                </figure>
                <figure className="space-y-2">
                  <div className="relative aspect-square">
                    <Image src={`data:image/png;base64,${item.input_png}`} alt={`${item.model_id} input`} width={item.map_width}
                      height={item.map_height} unoptimized className="absolute inset-0 h-full w-full rounded" />
                    <Image src={`data:image/png;base64,${item.anomaly_map_png}`} alt={`${item.model_id} overlay`} width={item.map_width}
                      height={item.map_height} unoptimized className="absolute inset-0 h-full w-full rounded" style={{ opacity: opacity / 100 }} />
                  </div>
                  <figcaption className="text-sm">Overlay at {opacity}%</figcaption>
                </figure>
                <a className="text-sm text-primary" href={`data:image/png;base64,${item.mask_png}`}
                  download={`${item.category_id}-${item.model_id}-mask.png`}>Download {item.model_id} mask PNG</a>
                <details className="break-all text-xs text-muted-foreground"><summary>Installed provenance</summary>
                  <p>Artifact SHA-256: {item.artifact_sha256}</p><p>Calibration SHA-256: {item.calibration_sha256}</p>
                  <p>Model config SHA-256: {item.config_hash}</p>
                </details>
              </article>)}
            </div>
            <p className="text-xs text-muted-foreground">Decisions use frozen normal-validation thresholds, strictly greater than.
              Inference time includes preprocessing, excludes decoding/PNG encoding and is not a benchmark average.
              API request time: {result.request_ms.toFixed(1)} ms (excludes network/JSON serialization).</p>
          </section> : null}
        </AsyncState>
      </FormPanel>
      <AsyncState loading={datasets.loading || models.loading} error={datasets.error ?? models.error}>
        {datasets.data && models.data ? <SavedComparison categories={datasets.data.categories} models={models.data.models} /> : null}
      </AsyncState>
    </div>
  );
}
