from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def resolve_run_dir(
    run_dir: Optional[Union[Path, str]],
    runs_root: Union[Path, str],
    run_name: Optional[str],
) -> Path:
    """Resolve the path to a run directory from CLI inputs."""

    if run_dir is not None:
        candidate = Path(run_dir)
    elif run_name is not None:
        candidate = Path(runs_root) / run_name
    else:
        raise ValueError("Provide either --run-dir or --run-name to locate a run.")

    if not candidate.exists():
        raise FileNotFoundError(f"Run directory not found: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {candidate}")

    return candidate
