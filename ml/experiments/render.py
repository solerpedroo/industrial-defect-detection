"""Write a Markdown, CSV, JSON and optional PNG report package.

Figures are charts of recorded numbers. This module does not score images,
fit models or choose thresholds. Missing measurements stay blocked instead of
being filled with a placeholder metric.
"""

import csv
import json
import os
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ml.experiments import PROTOCOL_IDS
from ml.experiments.subsets import plan_normal_subsets

LIMITATION = (
    "Anomaly scores, heatmap intensity, and mask area are not physical severity "
    "or financial loss."
)
REPO_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_OUTPUT_ROOTS = ("sample_outputs", "runs", "artifacts", "checkpoints")


def _atomic_write(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _reject_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT) and not any(
        resolved.is_relative_to(REPO_ROOT / name) and resolved != REPO_ROOT / name
        for name in _ALLOWED_OUTPUT_ROOTS
    ):
        raise ValueError("Reports must be written under sample_outputs, runs, artifacts or checkpoints.")


def _protocol_id(document: dict, expected: str) -> str:
    if expected not in PROTOCOL_IDS:
        raise ValueError(f"Unknown protocol '{expected}'.")
    found = document.get("protocol_id") or document.get("config", {}).get("protocol_id")
    if found != expected:
        raise ValueError("Result protocol does not match the requested report.")
    return expected


def _metric_rows(document: dict) -> list[dict]:
    rows = []
    for item in document.get("results") or []:
        metrics = item.get("metrics") or {}
        value = metrics.get("image_auroc")
        label = " / ".join(
            str(part)
            for part in (
                item.get("category"),
                item.get("model_id"),
                item.get("condition"),
                item.get("intensity"),
            )
            if part not in (None, "")
        )
        rows.append(
            {
                "label": label or "result",
                "category": item.get("category"),
                "model_id": item.get("model_id"),
                "condition": item.get("condition"),
                "intensity": item.get("intensity"),
                "image_auroc": value,
                "pixel_auroc": metrics.get("pixel_auroc"),
                "threshold_source": item.get("threshold_source")
                or item.get("thresholds", {}).get("image", {}).get("source"),
            }
        )
    return rows


def _chart(rows: list[dict], path: Path) -> None:
    usable = [row for row in rows if isinstance(row["image_auroc"], (int, float)) and np.isfinite(row["image_auroc"])]
    if not usable:
        return
    width, bar_h, margin = 720, 28, 16
    height = margin * 2 + bar_h * len(usable)
    canvas = Image.new("RGB", (width, height), (22, 28, 40))
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(usable):
        top = margin + index * bar_h
        span = max(0.0, min(1.0, float(row["image_auroc"])))
        draw.rectangle((180, top + 6, 180 + int(500 * span), top + 22), fill=(92, 148, 230))
        draw.text((12, top + 6), row["label"][:32], fill=(230, 236, 245))
    canvas.save(path, format="PNG")


def _markdown(protocol_id: str, document: dict, rows: list[dict], figure: str | None) -> str:
    complete = document.get("complete") is True and bool(rows)
    lines = [
        f"# Report: {protocol_id}",
        "",
        "## Objective",
        "",
        "Compare detection behavior under the recorded protocol. The conclusion is not assumed in advance.",
        "",
        "## Limitations",
        "",
        f"- {LIMITATION}",
        "- Thresholds, when present, were frozen before the reported test read.",
        "- A synthetic or partial run is engineering evidence, not a MVTec benchmark claim.",
        "",
    ]
    if not complete:
        reason = document.get("blocked_reason") or "No completed measurement is available for this protocol."
        lines.extend(["## Results", "", f"Blocked: {reason}", ""])
    else:
        lines.extend(
            [
                "## Results",
                "",
                "Image AUROC distinguishes normal and anomalous images. It is not a severity grade.",
                "",
                "| Category | Model | Condition | Intensity | Image AUROC | Pixel AUROC | Threshold source |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| {category} | {model_id} | {condition} | {intensity} | {image_auroc} | "
                "{pixel_auroc} | {threshold_source} |".format(
                    category=row["category"] or "",
                    model_id=row["model_id"] or "",
                    condition=row["condition"] or "",
                    intensity="" if row["intensity"] is None else row["intensity"],
                    image_auroc=row["image_auroc"],
                    pixel_auroc=row["pixel_auroc"],
                    threshold_source=row["threshold_source"] or "",
                )
            )
        lines.append("")
    plan = document.get("subset_plan") or []
    if plan:
        lines.extend(
            [
                "## Planned normal-training subsets",
                "",
                "These ids are a draw plan, not measured metrics.",
                "",
            ]
        )
        for item in plan:
            lines.append(
                f"- fraction {item['fraction']}, repeat {item['repeat']}: {', '.join(item['train_ids'])}"
            )
        lines.append("")
    if figure:
        lines.extend(["## Figure", "", f"Recorded image AUROC chart: `{figure}`.", ""])
    return "\n".join(lines)


def blocked_document(protocol_id: str, reason: str, *, train_ids: list[str] | None = None) -> dict:
    """A report body with an explicit block and no invented metrics."""
    if protocol_id not in PROTOCOL_IDS:
        raise ValueError(f"Unknown protocol '{protocol_id}'.")
    document = {
        "protocol_id": protocol_id,
        "complete": False,
        "blocked_reason": reason,
        "results": [],
        "limitations": [LIMITATION, reason],
    }
    if protocol_id == "exp-c-dataset-size" and train_ids:
        document["subset_plan"] = [dict(item) for item in plan_normal_subsets(train_ids)]
    return document


def render_report(document: dict, output_dir: Path, *, protocol_id: str) -> Path:
    """Write report.md, results.json, results.csv and image_auroc.png when numbers exist."""
    if not isinstance(document, dict):
        raise TypeError("Report document must be a JSON object.")
    _protocol_id(document, protocol_id)
    output = Path(output_dir)
    _reject_output(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = _metric_rows(document) if document.get("complete") is True else []
    figure = None
    if any(isinstance(row["image_auroc"], (int, float)) for row in rows):
        figure = "image_auroc.png"
        _chart(rows, output / figure)
    payload = {
        "protocol_id": protocol_id,
        "complete": document.get("complete") is True and bool(rows),
        "blocked_reason": document.get("blocked_reason"),
        "data_kind": document.get("data_kind"),
        "limitations": [LIMITATION],
        "results": document.get("results") if rows else [],
        "subset_plan": document.get("subset_plan") or [],
    }
    _atomic_write(output / "results.json", json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    stream = StringIO(newline="")
    fieldnames = [
        "category",
        "model_id",
        "condition",
        "intensity",
        "image_auroc",
        "pixel_auroc",
        "threshold_source",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})
    _atomic_write(output / "results.csv", stream.getvalue())
    markdown = _markdown(protocol_id, document if rows else payload, rows, figure)
    _atomic_write(output / "report.md", markdown)
    if LIMITATION not in markdown:
        raise ValueError("Report omitted the severity limitation.")
    return output / "report.md"
