import type { Metadata } from "next";

import { PageHeader } from "@/components/layout/page-header";
import { InferenceLab } from "@/features/inference/inference-lab";

export const metadata: Metadata = {
  title: "Inference Lab",
};

export default function InferencePage() {
  return (
    <>
      <PageHeader
        label="Module 06"
        title="Inference Lab"
        description="Upload one image and choose a category plus a model. Heatmaps, overlays, and masks appear after a checkpoint exists."
      />
      <InferenceLab />
    </>
  );
}
