import Image from "next/image";
import { useState } from "react";

import type { InferenceResponse } from "@/types/api";

function pngData(encoded: string) {
  return `data:image/png;base64,${encoded}`;
}

export function InferenceResult({ result }: { result: InferenceResponse }) {
  const [opacity, setOpacity] = useState(50);
  const dimensions = `${result.map_width} × ${result.map_height}`;
  const filename = `${result.category_id}-${result.model_id}`;
  const images = [
    ["Model-resolution input", result.input_png],
    ["Anomaly heatmap", result.anomaly_map_png],
    ["Binary anomaly mask", result.mask_png],
    ...(result.reconstruction_png ? [["Reconstruction", result.reconstruction_png]] : []),
  ];

  return (
    <section className="mt-6 space-y-5" aria-label="Inference result">
      <h2 className="text-xl font-semibold">Inspection result</h2>
      {result.data_kind === "synthetic" ? (
        <p role="note" className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
          Synthetic demonstration — not validated for industrial inspection.
        </p>
      ) : null}
      <div className="grid gap-3 text-sm sm:grid-cols-3">
        <p>Decision: <strong>{result.label}</strong></p>
        <p>Anomaly score: <strong>{result.score.toPrecision(5)}</strong></p>
        <p>Image threshold: {result.threshold.toPrecision(5)}</p>
        <p>Pixel threshold: {result.pixel_threshold.toPrecision(5)}</p>
        <p>Inference: {result.inference_ms.toFixed(1)} ms</p>
        <p>Resolution: {dimensions}</p>
      </div>
      <p className="text-sm text-muted-foreground">
        Decisions use frozen normal-validation thresholds (strictly greater than).
        Scores are model-specific, not severity or probability. Maps use model resolution,
        not original-image coordinates. A normal image decision can still contain highlighted pixels.
      </p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {images.map(([title, encoded]) => (
          <figure key={title} className="min-w-0 space-y-2 rounded-lg border border-border p-3">
            <Image src={pngData(encoded)} alt={title} width={result.map_width} height={result.map_height}
              unoptimized className="aspect-square h-auto w-full rounded object-contain" />
            <figcaption className="text-sm">{title}</figcaption>
          </figure>
        ))}
        <figure className="space-y-3 rounded-lg border border-border p-3">
          <div className="relative aspect-square">
            <Image src={pngData(result.input_png)} alt="Overlay input" width={result.map_width}
              height={result.map_height} unoptimized className="absolute inset-0 h-full w-full rounded" />
            <Image src={pngData(result.anomaly_map_png)} alt="Adjustable anomaly overlay"
              width={result.map_width} height={result.map_height} unoptimized
              className="absolute inset-0 h-full w-full rounded" style={{ opacity: opacity / 100 }} />
          </div>
          <figcaption className="text-sm">Overlay</figcaption>
          <label className="block text-sm">
            Overlay opacity: {opacity}%
            <input aria-label="Overlay opacity" type="range" min="0" max="100" step="5"
              value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} className="mt-2 w-full" />
          </label>
        </figure>
      </div>
      <p className="text-xs text-muted-foreground">
        Heat colors: blue (0) → red ({result.display_max ? result.display_max.toPrecision(5) : "nonzero"}).
        Fixed calibration scale; saturation is display-only, not score normalization.
      </p>
      <div className="flex flex-wrap gap-4 text-sm text-primary">
        <a href={pngData(result.mask_png)} download={`${filename}-mask.png`}>Download mask PNG</a>
        <a href={pngData(result.anomaly_map_png)} download={`${filename}-heatmap.png`}>Download heatmap PNG</a>
        <a href={pngData(result.overlay_png)} download={`${filename}-overlay.png`}>Download 50% overlay PNG</a>
      </div>
    </section>
  );
}
