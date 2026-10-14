"use client";

import { AsyncState } from "@/components/async-state";
import { FormPanel } from "@/components/layout/form-panel";
import { StatGrid } from "@/components/stat-grid";
import { useApiGet } from "@/lib/use-api";
import type { DatasetCatalog } from "@/types/api";

export function DatasetBrowser() {
  const datasets = useApiGet<DatasetCatalog>("/api/datasets");

  return (
    <AsyncState loading={datasets.loading} error={datasets.error}>
      {datasets.data ? (
        <div className="space-y-4">
          <FormPanel title={datasets.data.name} subtitle={datasets.data.license}>
            <p className="text-sm leading-6 text-muted-foreground">
              Non-commercial license. The repository documents the layout and does not redistribute images. Expected
              root: <span className="font-mono text-foreground">{datasets.data.storage_root}</span>.
            </p>
            <a
              className="mt-3 inline-flex text-sm text-brand underline-offset-4 hover:underline"
              href={datasets.data.license_url}
            >
              {datasets.data.license}
            </a>
          </FormPanel>

          <div className="grid gap-4 lg:grid-cols-3">
            {datasets.data.categories.map((category) => (
              <FormPanel key={category.id} title={category.name} subtitle={category.purpose}>
                <p className="mb-3 font-mono text-[11px] uppercase tracking-wide text-brand">
                  {category.kind} · {category.availability}
                </p>
                <StatGrid
                  items={[
                    { label: "Train normal", value: String(category.train_normal) },
                    { label: "Validation normal", value: String(category.validation_normal) },
                    { label: "Test normal", value: String(category.test_normal) },
                    { label: "Test anomalous", value: String(category.test_anomalous) },
                    { label: "Masks", value: String(category.mask_count) },
                  ]}
                />
                <p className="mt-3 text-xs text-muted-foreground">
                  Validation split: {category.validation_split.replaceAll("_", " ")}
                </p>
                {category.validation_split === "invalid" ? (
                  <p role="alert" className="mt-2 text-xs text-destructive">
                    The split manifest is invalid or training files have changed. Audit the dataset and recreate the split before training.
                  </p>
                ) : null}
              </FormPanel>
            ))}
          </div>
        </div>
      ) : null}
    </AsyncState>
  );
}
