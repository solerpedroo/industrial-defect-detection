import type { Metadata } from "next";

import { PageHeader } from "@/components/layout/page-header";
import { ComparisonBoard } from "@/features/comparison/comparison-board";

export const metadata: Metadata = {
  title: "Comparison",
};

export default function ComparisonPage() {
  return (
    <>
      <PageHeader
        label="Module 08"
        title="Model Comparison"
        description="Compare the autoencoder and PatchCore on one upload. Bounding boxes, when added later, come from the anomaly mask."
      />
      <ComparisonBoard />
    </>
  );
}
