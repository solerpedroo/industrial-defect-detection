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
        description="Inspect one image with two calibrated models, then explore saved test metrics by category and export the visible comparison."
      />
      <ComparisonBoard />
    </>
  );
}
