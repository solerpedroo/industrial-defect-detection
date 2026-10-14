"""Read a generated report package without training or scoring."""

from pathlib import Path

MAX_REPORT_BYTES = 1_000_000


class ReportNotFoundError(FileNotFoundError):
    """The protocol exists, but no report package has been generated."""


class ReportTooLargeError(ValueError):
    """The markdown artifact exceeds the response limit."""


def load_report(experiment_id: str, root: Path) -> dict:
    """Return markdown and PNG names from ``root/<experiment_id>/``.

    ``experiment_id`` must already be a catalog id. Symlinks and paths that
    leave ``root`` are rejected. Figure bytes stay on disk.
    """
    base = Path(root).resolve()
    if (
        not experiment_id
        or "/" in experiment_id
        or "\\" in experiment_id
        or experiment_id in {".", ".."}
    ):
        raise ReportNotFoundError(experiment_id)
    candidate = base / experiment_id
    if candidate.is_symlink() or not candidate.is_dir():
        raise ReportNotFoundError(experiment_id)
    directory = candidate.resolve()
    markdown_path = directory / "report.md"
    if directory.parent != base or markdown_path.is_symlink() or not markdown_path.is_file():
        raise ReportNotFoundError(experiment_id)
    payload = markdown_path.read_bytes()
    if len(payload) > MAX_REPORT_BYTES:
        raise ReportTooLargeError("Report markdown exceeds 1 MiB.")
    figures = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".png"
    )
    return {
        "experiment_id": experiment_id,
        "markdown": payload.decode("utf-8"),
        "figures": figures,
    }
