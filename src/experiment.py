from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_projector_resolution(blender_dir: Path) -> Tuple[int, int]:
    """
    Infer projector resolution from an existing pattern image.

    Falls back to a reasonable default if the file is missing or unreadable.
    """
    pattern_path = blender_dir / "projection_pattern.png"
    if pattern_path.exists():
        img = cv2.imread(str(pattern_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            height, width = img.shape[:2]
            return width, height

    # Default to 1024x768 if nothing is available.
    return 1024, 768


def _render_with_blender(blender_dir: Path) -> None:
    script_path = blender_dir / "render_scene.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Blender render script not found at: {script_path}")

    subprocess.run(["bash", str(script_path)], check=True)


@dataclass
class ExperimentRun:
    """Core experiment runner shared by all control algorithms."""

    blender_dir: Path
    run_dir: Path
    algorithm: str
    projector_width: int
    projector_height: int
    start_time: str
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    measurement_index: int = 0

    def capture(
        self, pattern: np.ndarray
    ) -> Tuple[np.ndarray, str, str, int]:
        """
        Project a pattern, trigger Blender render, and archive images.

        Returns
        -------
        captured_img : np.ndarray
            Grayscale captured image array.
        rel_proj : str
            Relative path to stored projection pattern within the run_dir.
        rel_cap : str
            Relative path to stored captured image within the run_dir.
        index : int
            1-based measurement index.
        """
        # Save pattern where Blender expects it.
        pattern_path = self.blender_dir / "projection_pattern.png"
        cv2.imwrite(str(pattern_path), pattern)

        # Render via Blender.
        _render_with_blender(self.blender_dir)

        # Load captured image.
        captured_path = self.blender_dir / "captured_image.png"
        if not captured_path.exists():
            raise FileNotFoundError(f"Captured image not found at: {captured_path}")

        captured_img = cv2.imread(str(captured_path), cv2.IMREAD_GRAYSCALE)
        if captured_img is None:
            raise RuntimeError(f"Failed to read captured image at: {captured_path}")

        # Archive both pattern and captured image into run_dir.
        self.measurement_index += 1
        idx = self.measurement_index

        proj_dest = self.run_dir / f"projection_pattern_{idx:04d}.png"
        cap_dest = self.run_dir / f"captured_image_{idx:04d}.png"

        shutil.move(str(pattern_path), proj_dest)
        shutil.move(str(captured_path), cap_dest)

        rel_proj = proj_dest.relative_to(self.run_dir).as_posix()
        rel_cap = cap_dest.relative_to(self.run_dir).as_posix()

        return captured_img, rel_proj, rel_cap, idx

    def log_measurement(self, record: Dict[str, Any]) -> None:
        """Append a single measurement record to the run log."""
        self.measurements.append(record)

    def finalize(
        self,
        summary: Optional[Dict[str, Any]] = None,
        extra_outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist run metadata, measurement log, and any extra outputs.

        Parameters
        ----------
        summary:
            Additional high-level metadata (e.g., counters) to merge into
            metadata.json.
        extra_outputs:
            Mapping from name -> data. Each entry is written as a separate
            JSON file '<name>.json' under the run directory.
        """
        end_time = datetime.now().isoformat(timespec="seconds")

        metadata: Dict[str, Any] = {
            "run_name": self.run_dir.name,
            "algorithm": self.algorithm,
            "start_time": self.start_time,
            "end_time": end_time,
            "blender_dir": str(self.blender_dir.resolve()),
            "projector_width": self.projector_width,
            "projector_height": self.projector_height,
            "num_measurements": self.measurement_index,
        }
        metadata.update(self.extra_metadata)
        if summary:
            metadata.update(summary)

        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with (self.run_dir / "measurements.json").open("w", encoding="utf-8") as f:
            json.dump(self.measurements, f, indent=2)

        if extra_outputs:
            for name, data in extra_outputs.items():
                out_path = self.run_dir / f"{name}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)


def create_experiment_run(
    algorithm: str,
    blender_dir: Union[Path, str] = Path("blender-virtual-experiment"),
    runs_root: Union[Path, str] = Path("runs"),
    run_name: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> ExperimentRun:
    """
    Create a new experiment run directory and context.

    All algorithms should use this helper so that run structure and
    metadata are consistent.
    """
    blender_dir = Path(blender_dir)
    runs_root = Path(runs_root)

    _ensure_directory(runs_root)

    if run_name is None:
        run_name = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    projector_width, projector_height = _load_projector_resolution(blender_dir)

    return ExperimentRun(
        blender_dir=blender_dir,
        run_dir=run_dir,
        algorithm=algorithm,
        projector_width=projector_width,
        projector_height=projector_height,
        start_time=datetime.now().isoformat(timespec="seconds"),
        extra_metadata=extra_metadata or {},
    )

