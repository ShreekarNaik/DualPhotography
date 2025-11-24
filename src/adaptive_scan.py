from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from experiment import ExperimentRun, create_experiment_run
from structured_light_products import MappingProducts, save_mapping_products


@dataclass
class Region:
    """Axis-aligned rectangular region in projector coordinates."""

    level: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def subdivide(self) -> List["Region"]:
        """Subdivide region into up to four quadrants."""
        if self.width <= 1 and self.height <= 1:
            return []

        mid_x = self.x0 + max(1, self.width // 2)
        mid_y = self.y0 + max(1, self.height // 2)

        children = [
            Region(self.level + 1, self.x0, self.y0, mid_x, mid_y),
            Region(self.level + 1, mid_x, self.y0, self.x1, mid_y),
            Region(self.level + 1, self.x0, mid_y, mid_x, self.y1),
            Region(self.level + 1, mid_x, mid_y, self.x1, self.y1),
        ]

        return [r for r in children if r.area > 0]


def _generate_pattern_for_region(
    region: Region, width: int, height: int
) -> np.ndarray:
    """
    Generate a binary illumination pattern image for a given region.

    Pixels inside the region are set to 255 (ON) and others to 0 (OFF).
    """
    pattern = np.zeros((height, width), dtype=np.uint8)
    pattern[region.y0 : region.y1, region.x0 : region.x1] = 255
    return pattern


def _capture_reference_frame(
    experiment: ExperimentRun,
    value: int,
    label: str,
) -> np.ndarray:
    """
    Capture a reference frame (all-ON or all-OFF) and log it.

    This mirrors the basis-scan workflow so that adaptive runs
    produce the same albedo and signal-strength products needed
    for relighting and projector POV utilities.
    """
    pattern = np.full(
        (experiment.projector_height, experiment.projector_width), value, dtype=np.uint8
    )
    captured_img, rel_proj, rel_cap, measurement_index = experiment.capture(pattern)
    experiment.log_measurement(
        {
            "measurement_index": measurement_index,
            "type": "reference",
            "reference_label": label,
            "projection_pattern": rel_proj,
            "captured_image": rel_cap,
            "mean_intensity": float(captured_img.mean()),
            "energy_sum": float(captured_img.sum()),
        }
    )
    return captured_img


def _build_adaptive_mapping_products(
    run_dir: Path,
    reference_frames: Dict[str, np.ndarray],
    terminal_mappings: List[Dict[str, object]],
    noise_threshold: float,
    projector_width: int,
    projector_height: int,
    min_relative_intensity: float = 0.05,
) -> MappingProducts:
    """
    Construct MappingProducts from adaptive terminal measurements.

    For each confident camera pixel (mask=True), assign the projector
    coordinate corresponding to the strongest single-pixel response
    observed among all terminal regions. This approximates
    argmax_j T_ij and is compatible with relight/projector_pov.
    """
    if "white" not in reference_frames or "black" not in reference_frames:
        raise RuntimeError("Missing reference frames required for adaptive decoding.")

    white_img = reference_frames["white"]
    black_img = reference_frames["black"]
    if white_img.shape != black_img.shape:
        raise ValueError("Reference frames I_white and I_black must have same shape.")

    camera_height, camera_width = white_img.shape

    white_f = white_img.astype(np.float32)
    black_f = black_img.astype(np.float32)

    signal_strength = np.clip(white_f - black_f, a_min=0.0, a_max=None)
    mask = signal_strength > float(noise_threshold)

    map_x = np.full((camera_height, camera_width), -1, dtype=np.int32)
    map_y = np.full((camera_height, camera_width), -1, dtype=np.int32)
    best_response = np.zeros((camera_height, camera_width), dtype=np.float32)

    safe_signal = np.where(signal_strength > 1e-6, signal_strength, 1.0)

    for entry in terminal_mappings:
        x_proj = int(entry["x"])
        y_proj = int(entry["y"])
        rel_cap = Path(str(entry["captured_image"]))
        cap_path = run_dir / rel_cap

        capture = cv2.imread(str(cap_path), cv2.IMREAD_GRAYSCALE)
        if capture is None:
            raise FileNotFoundError(f"Failed to read terminal capture at {cap_path}")

        if capture.shape != (camera_height, camera_width):
            raise ValueError(
                "Terminal capture resolution mismatch: expected "
                f"{camera_width}x{camera_height}, got "
                f"{capture.shape[1]}x{capture.shape[0]}"
            )

        capture_f = capture.astype(np.float32)
        response = np.clip(capture_f - black_f, a_min=0.0, a_max=None)

        relative = response / safe_signal
        candidate = mask & (response > best_response) & (
            relative >= min_relative_intensity
        )

        map_x[candidate] = x_proj
        map_y[candidate] = y_proj
        best_response[candidate] = response[candidate]

    return MappingProducts(
        map_x=map_x,
        map_y=map_y,
        mask=mask,
        signal_strength=signal_strength,
        i_white=white_img.copy(),
        i_black=black_img.copy(),
        noise_threshold=float(noise_threshold),
        projector_size=(projector_width, projector_height),
    )


def run_adaptive_scan(
    noise_threshold: float = 1e5,
    max_depth: int = 10,
    max_measurements: Optional[int] = None,
    projector_width: int = 1024,
    projector_height: int = 768,
    blender_dir: Union[Path, str] = Path("blender-virtual-experiment"),
    runs_root: Union[Path, str] = Path("runs"),
    run_name: Optional[str] = None,
) -> Path:
    """
    Run Algorithm 3: Adaptive Hierarchical Feedback Scan.

    This function:
      - Generates illumination patterns for hierarchical regions.
      - Uses ExperimentRun.capture to communicate with Blender and archive images.
      - Computes measurement energy and adaptively subdivides regions.
      - Logs metadata and measurement details via the shared ExperimentRun.
    """
    experiment: ExperimentRun = create_experiment_run(
        algorithm="adaptive_hierarchical_scan",
        blender_dir=blender_dir,
        runs_root=runs_root,
        run_name=run_name,
        projector_width=projector_width,
        projector_height=projector_height,
        extra_metadata={
            "noise_threshold": noise_threshold,
            "max_depth": max_depth,
            "max_measurements": max_measurements,
        },
    )

    projector_width = experiment.projector_width
    projector_height = experiment.projector_height

    reference_frames: Dict[str, np.ndarray] = {}
    reference_frames["black"] = _capture_reference_frame(experiment, 0, "black")
    reference_frames["white"] = _capture_reference_frame(experiment, 255, "white")

    # Queue-based BFS over regions.
    initial_region = Region(0, 0, 0, projector_width, projector_height)
    queue: deque[Region] = deque([initial_region])

    terminal_mappings: List[dict] = []

    while queue:
        if max_measurements is not None and experiment.measurement_index >= max_measurements:
            break

        region = queue.popleft()

        # Construct control input u_k: pattern for this region.
        pattern = _generate_pattern_for_region(region, projector_width, projector_height)

        captured_img, rel_proj, rel_cap, measurement_index = experiment.capture(
            pattern
        )

        # Compute total energy E = sum_i y_k(i).
        energy = float(captured_img.sum())
        mean_intensity = float(captured_img.mean())

        if energy < noise_threshold:
            decision = "pruned"
        elif region.area <= 1 or region.level >= max_depth:
            decision = "terminal"
            terminal_mappings.append(
                {
                    "x": region.x0,
                    "y": region.y0,
                    "level": region.level,
                    "measurement_index": measurement_index,
                    "energy_sum": energy,
                    "captured_image": rel_cap,
                    "projection_pattern": rel_proj,
                }
            )
        else:
            decision = "subdivided"
            for child in region.subdivide():
                if child.area > 0:
                    queue.append(child)

        region_record = {
            "level": region.level,
            "x0": region.x0,
            "y0": region.y0,
            "x1": region.x1,
            "y1": region.y1,
            "width": region.width,
            "height": region.height,
            "area": region.area,
        }

        experiment.log_measurement(
            {
                "measurement_index": measurement_index,
                "region": region_record,
                "energy_sum": energy,
                "mean_intensity": mean_intensity,
                "decision": decision,
                "projection_pattern": rel_proj,
                "captured_image": rel_cap,
            }
        )

    mapping_products = _build_adaptive_mapping_products(
        run_dir=experiment.run_dir,
        reference_frames=reference_frames,
        terminal_mappings=terminal_mappings,
        noise_threshold=noise_threshold,
        projector_width=projector_width,
        projector_height=projector_height,
    )
    mapping_path = save_mapping_products(experiment.run_dir, mapping_products)

    experiment.finalize(
        summary={
            "num_terminal_mappings": len(terminal_mappings),
            "mapping_products_file": mapping_path.name,
        },
        extra_outputs={"terminal_mappings": terminal_mappings},
    )

    return experiment.run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive Hierarchical Feedback Scan (Algorithm 3) "
            "using Blender-based virtual experiment."
        )
    )
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=1e5,
        help="Energy threshold for pruning regions (default: 1e5).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum subdivision depth (root is level 0).",
    )
    parser.add_argument(
        "--max-measurements",
        type=int,
        default=None,
        help="Optional cap on total number of measurements.",
    )
    parser.add_argument(
        "--projector-width",
        type=int,
        default=1024,
        help="Projector resolution width in pixels (default: 1024).",
    )
    parser.add_argument(
        "--projector-height",
        type=int,
        default=768,
        help="Projector resolution height in pixels (default: 768).",
    )
    parser.add_argument(
        "--blender-dir",
        type=Path,
        default=Path("blender-virtual-experiment"),
        help="Directory containing render_scene.sh and Blender scene.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory where per-run folders are created.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit name for the run folder.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    run_dir = run_adaptive_scan(
        noise_threshold=args.noise_threshold,
        max_depth=args.max_depth,
        max_measurements=args.max_measurements,
        projector_width=args.projector_width,
        projector_height=args.projector_height,
        blender_dir=args.blender_dir,
        runs_root=args.runs_root,
        run_name=args.run_name,
    )

    print(f"Adaptive scan completed. Results stored in: {run_dir}")


if __name__ == "__main__":
    main()
