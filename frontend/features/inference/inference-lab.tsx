"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";

import { AsyncState } from "@/components/async-state";
import { Button } from "@/components/ui/button";
import { FormPanel } from "@/components/layout/form-panel";
import { ApiError, postInference } from "@/lib/api";
import { useApiGet } from "@/lib/use-api";
import type { DatasetCatalog, InferenceResponse, ModelsResponse } from "@/types/api";

import { InferenceResult } from "./result-view";

export function InferenceLab() {
  const datasets = useApiGet<DatasetCatalog>("/api/datasets");
  const models = useApiGet<ModelsResponse>("/api/models");
  const [message, setMessage] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [categoryId, setCategoryId] = useState("");
  const [modelId, setModelId] = useState("");
  const generation = useRef(0);
  const active = useRef<AbortController | null>(null);
  const category = categoryId || datasets.data?.categories[0]?.id || "";
  const selectedModel = modelId || models.data?.models.find((model) => model.ready_categories.includes(category))?.id
    || models.data?.models[0]?.id || "";
  const ready = models.data?.models.some((model) => model.id === selectedModel && model.ready_categories.includes(category));

  useEffect(() => () => { generation.current += 1; active.current?.abort(); }, []);

  function invalidate() {
    generation.current += 1;
    active.current?.abort();
    setResult(null);
    setMessage(null);
    setPending(false);
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    const version = ++generation.current;
    setPending(true);
    setMessage(null);
    setResult(null);
    const form = new FormData(event.currentTarget);
    try {
      const response = await postInference(form, controller.signal);
      if (version === generation.current) setResult(response);
    } catch (error) {
      if (version === generation.current && !controller.signal.aborted) {
        setMessage(error instanceof ApiError ? error.message : "Inference failed.");
      }
    } finally {
      if (version === generation.current) setPending(false);
    }
  }

  return (
    <FormPanel
      title="Inspect one image"
      subtitle="PNG or JPEG, up to 10 MiB and 4096 × 4096. Only installed, calibrated category/model pairs can score."
    >
      <AsyncState loading={datasets.loading || models.loading} error={datasets.error ?? models.error}>
        <form className="grid gap-4 sm:grid-cols-2" onSubmit={onSubmit} onChange={invalidate}>
          <label className="block text-sm sm:col-span-2">
            <span className="label-caps">Image</span>
            <input className="field mt-2" name="image" type="file" accept="image/png,image/jpeg" required />
          </label>
          <label className="block text-sm">
            <span className="label-caps">Category</span>
            <select className="field mt-2" name="category_id" value={category} onChange={(event) => {
              setCategoryId(event.target.value); setModelId("");
            }}>
              {datasets.data?.categories.map((category) => (
                <option key={category.id} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-sm">
            <span className="label-caps">Model</span>
            <select className="field mt-2" name="model_id" value={selectedModel} onChange={(event) => setModelId(event.target.value)}>
              {models.data?.models.map((model) => (
                <option key={model.id} value={model.id} disabled={!model.ready_categories.includes(category)}>
                  {model.name}{model.ready_categories.includes(category) ? "" : " — not installed"}
                </option>
              ))}
            </select>
          </label>
          <div className="sm:col-span-2">
            <Button type="submit" disabled={pending || !ready}>
              {pending ? "Inspecting…" : "Run inference"}
            </Button>
            {!ready ? <p role="status" className="mt-3 text-sm text-muted-foreground">
              No calibrated artifact is installed for this category/model. Configure the API serving manifest first.
            </p> : null}
          </div>
        </form>
        {message ? (
          <p role="alert" className="mt-4 text-sm text-muted-foreground">
            {message}
          </p>
        ) : null}
        {result ? <InferenceResult key={`${result.model_id}-${result.artifact_sha256}`} result={result} /> : null}
      </AsyncState>
    </FormPanel>
  );
}
