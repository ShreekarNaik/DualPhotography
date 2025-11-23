from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from experiment import ExperimentRun, create_experiment_run


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


def run_adaptive_scan(
    noise_threshold: float = 1e5,
    max_depth: int = 10,
    max_measurements: Optional[int] = None,
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
        extra_metadata={
            "noise_threshold": noise_threshold,
            "max_depth": max_depth,
            "max_measurements": max_measurements,
        },
    )

    projector_width = experiment.projector_width
    projector_height = experiment.projector_height

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

        experiment.log_measurement(
            {
                "measurement_index": measurement_index,
                "region": {
                    "level": region.level,
                    "x0": region.x0,
                    "y0": region.y0,
                    "x1": region.x1,
                    "y1": region.y1,
                    "width": region.width,
                    "height": region.height,
                    "area": region.area,
                },
                "energy_sum": energy,
                "mean_intensity": mean_intensity,
                "decision": decision,
                "projection_pattern": rel_proj,
                "captured_image": rel_cap,
            }
        )

    experiment.finalize(
        summary={
            "num_terminal_mappings": len(terminal_mappings),
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
        blender_dir=args.blender_dir,
        runs_root=args.runs_root,
        run_name=args.run_name,
    )

    print(f"Adaptive scan completed. Results stored in: {run_dir}")


if __name__ == "__main__":
    main()

