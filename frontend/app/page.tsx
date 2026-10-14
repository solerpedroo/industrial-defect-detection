import type { Metadata } from "next";

import { PageHeader } from "@/components/layout/page-header";
import { OverviewPanel } from "@/features/overview/overview-panel";

export const metadata: Metadata = {
  title: "Overview",
};

export default function HomePage() {
  return (
    <>
      <PageHeader
        label="Inspection system"
        title="Industrial Defect Detection"
        description="Identify whether an industrial image is anomalous, where the unusual region is, and how large that region is. Scores are algorithm outputs, not a physical measure of severity."
      />
      <OverviewPanel />
    </>
  );
}
