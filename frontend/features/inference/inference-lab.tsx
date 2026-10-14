"use client";

import { useState, type FormEvent } from "react";

import { AsyncState } from "@/components/async-state";
import { Button } from "@/components/ui/button";
import { FormPanel } from "@/components/layout/form-panel";
import { ApiError, postInference } from "@/lib/api";
import { useApiGet } from "@/lib/use-api";
import type { DatasetCatalog, ModelsResponse } from "@/types/api";

export function InferenceLab() {
  const datasets = useApiGet<DatasetCatalog>("/api/datasets");
  const models = useApiGet<ModelsResponse>("/api/models");
  const [message, setMessage] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPending(true);
    setMessage(null);
    const form = new FormData(event.currentTarget);
    try {
      await postInference(form);
      setMessage("Inference returned a payload.");
    } catch (error) {
      setMessage(error instanceof ApiError ? error.message : "Inference failed.");
    } finally {
      setPending(false);
    }
  }

  return (
    <FormPanel
      title="Inspect one image"
      subtitle="PNG or JPEG, up to 10 MiB and 4096 × 4096. Scoring starts when a model artifact is ready."
    >
      <AsyncState loading={datasets.loading || models.loading} error={datasets.error ?? models.error}>
        <form className="grid gap-4 sm:grid-cols-2" onSubmit={onSubmit}>
          <label className="block text-sm sm:col-span-2">
            <span className="label-caps">Image</span>
            <input className="field mt-2" name="image" type="file" accept="image/png,image/jpeg" required />
          </label>
          <label className="block text-sm">
            <span className="label-caps">Category</span>
            <select className="field mt-2" name="category_id" defaultValue="bottle">
              {datasets.data?.categories.map((category) => (
                <option key={category.id} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-sm">
            <span className="label-caps">Model</span>
            <select className="field mt-2" name="model_id" defaultValue="autoencoder">
              {models.data?.models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>
          <div className="sm:col-span-2">
            <Button type="submit" disabled={pending}>
              {pending ? "Checking…" : "Run inference"}
            </Button>
          </div>
        </form>
        {message ? (
          <p role="status" className="mt-4 text-sm text-muted-foreground">
            {message}
          </p>
        ) : null}
      </AsyncState>
    </FormPanel>
  );
}
