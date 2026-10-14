"use client";

import { useState, type FormEvent } from "react";

import { AsyncState } from "@/components/async-state";
import { Button } from "@/components/ui/button";
import { FormPanel } from "@/components/layout/form-panel";
import { ApiError, postCompare } from "@/lib/api";
import { useApiGet } from "@/lib/use-api";
import type { DatasetCatalog, ModelsResponse } from "@/types/api";

export function ComparisonBoard() {
  const datasets = useApiGet<DatasetCatalog>("/api/datasets");
  const models = useApiGet<ModelsResponse>("/api/models");
  const [message, setMessage] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPending(true);
    setMessage(null);
    try {
      await postCompare(new FormData(event.currentTarget));
      setMessage("Comparison returned a payload.");
    } catch (error) {
      setMessage(error instanceof ApiError ? error.message : "Comparison failed.");
    } finally {
      setPending(false);
    }
  }

  return (
    <FormPanel
      title="Same image, two models"
      subtitle="Heatmaps and metric tables arrive with saved experiment artifacts. This form already checks the request."
    >
      <AsyncState loading={datasets.loading || models.loading} error={datasets.error ?? models.error}>
        <form className="grid gap-4 sm:grid-cols-2" onSubmit={onSubmit}>
          <label className="block text-sm sm:col-span-2">
            <span className="label-caps">Image</span>
            <input className="field mt-2" name="image" type="file" accept="image/png,image/jpeg" required />
          </label>
          <label className="block text-sm sm:col-span-2">
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
            <span className="label-caps">Model A</span>
            <select className="field mt-2" name="model_a" defaultValue="autoencoder">
              {models.data?.models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-sm">
            <span className="label-caps">Model B</span>
            <select className="field mt-2" name="model_b" defaultValue="patchcore">
              {models.data?.models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>
          <div className="sm:col-span-2">
            <Button type="submit" disabled={pending}>
              {pending ? "Checking…" : "Compare models"}
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
