"use client";

import { AsyncState } from "@/components/async-state";
import { FormPanel } from "@/components/layout/form-panel";
import { useApiGet } from "@/lib/use-api";
import type { ExperimentsResponse } from "@/types/api";

export function ExperimentList() {
  const experiments = useApiGet<ExperimentsResponse>("/api/experiments");

  return (
    <AsyncState loading={experiments.loading} error={experiments.error}>
      <div className="grid gap-4">
        {experiments.data?.experiments.map((item) => (
          <FormPanel key={item.id} title={item.title} subtitle={item.id}>
            <p className="text-sm leading-6 text-muted-foreground">{item.question}</p>
            <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-3">
              <div>
                <dt className="label-caps">Status</dt>
                <dd className="mt-1 font-mono text-brand">{item.status}</dd>
              </div>
              <div>
                <dt className="label-caps">Categories</dt>
                <dd className="mt-1">{item.categories.join(", ")}</dd>
              </div>
              <div>
                <dt className="label-caps">Models</dt>
                <dd className="mt-1">{item.models.join(", ")}</dd>
              </div>
            </dl>
            <p className="mt-4 text-xs leading-5 text-muted-foreground">Metrics: {item.metrics.join(", ")}</p>
          </FormPanel>
        ))}
      </div>
    </AsyncState>
  );
}
