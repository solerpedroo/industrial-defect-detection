"use client";

import { AsyncState } from "@/components/async-state";
import { StatGrid } from "@/components/stat-grid";
import { FormPanel } from "@/components/layout/form-panel";
import { useApiGet } from "@/lib/use-api";
import type { DatasetCatalog, ExperimentsResponse, ModelsResponse } from "@/types/api";

export function OverviewPanel() {
  const models = useApiGet<ModelsResponse>("/api/models");
  const experiments = useApiGet<ExperimentsResponse>("/api/experiments");
  const datasets = useApiGet<DatasetCatalog>("/api/datasets");

  return (
    <div className="space-y-6">
      <StatGrid
        items={[
          { label: "Categories", value: "3" },
          { label: "Models", value: "2" },
          { label: "Protocols", value: "3" },
          { label: "Training", value: "Normal only" },
        ]}
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <FormPanel title="Models" subtitle="Reconstruction and feature memory. Both stay planned until their waves land.">
          <AsyncState loading={models.loading} error={models.error}>
            <ul className="space-y-4">
              {models.data?.models.map((model) => (
                <li key={model.id}>
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-medium">{model.name}</p>
                    <span className="font-mono text-[11px] uppercase tracking-wide text-brand">{model.status}</span>
                  </div>
                  <p className="mt-1 text-sm leading-6 text-muted-foreground">{model.description}</p>
                </li>
              ))}
            </ul>
          </AsyncState>
        </FormPanel>

        <FormPanel title="Protocols" subtitle="Questions for the later benchmark. No measured scores yet.">
          <AsyncState loading={experiments.loading} error={experiments.error}>
            <ul className="space-y-4">
              {experiments.data?.experiments.map((item) => (
                <li key={item.id}>
                  <p className="text-sm font-medium">{item.title}</p>
                  <p className="mt-1 text-sm leading-6 text-muted-foreground">{item.question}</p>
                </li>
              ))}
            </ul>
          </AsyncState>
        </FormPanel>
      </div>

      <FormPanel title="Dataset" subtitle="Local MVTec AD copy. Images are not stored in Git.">
        <AsyncState loading={datasets.loading} error={datasets.error}>
          {datasets.data ? (
            <p className="text-sm leading-6 text-muted-foreground">
              {datasets.data.name} · {datasets.data.license}. Place files in{" "}
              <span className="font-mono text-foreground">{datasets.data.storage_root}</span>. Commercial use is
              disabled.
            </p>
          ) : null}
        </AsyncState>
      </FormPanel>
    </div>
  );
}
