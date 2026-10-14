import type { Metadata } from "next";

import { PageHeader } from "@/components/layout/page-header";
import { ExperimentList } from "@/features/experiments/experiment-list";

export const metadata: Metadata = {
  title: "Experiments",
};

export default function ExperimentsPage() {
  return (
    <>
      <PageHeader
        label="Protocols"
        title="Experiments"
        description="Three comparisons structure the later benchmark: model family, image degradation, and the amount of normal training data."
      />
      <ExperimentList />
    </>
  );
}
