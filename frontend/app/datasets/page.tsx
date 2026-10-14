import type { Metadata } from "next";

import { PageHeader } from "@/components/layout/page-header";
import { DatasetBrowser } from "@/features/datasets/dataset-browser";

export const metadata: Metadata = {
  title: "Datasets",
};

export default function DatasetsPage() {
  return (
    <>
      <PageHeader
        label="Module 01"
        title="Datasets"
        description="Bottle, Leather, and Metal Nut from MVTec AD. Official test images stay sealed until evaluation. Validation splits are created later from normal training images."
      />
      <DatasetBrowser />
    </>
  );
}
