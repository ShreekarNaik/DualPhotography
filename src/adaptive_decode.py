from __future__ import annotations

"""
Decoders for adaptive hierarchical scans.

This module reconstructs a sparse MappingProducts object directly
from an adaptive_hierarchical_scan run directory, without modifying
the acquisition code. It uses the measurement tree to build a
piecewise-constant approximation of the projector-to-camera mapping.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from structured_light_products import MappingProducts


@dataclass
class _RegionMeasurement:
    """Single region measurement from an adaptive scan."""

    level: int
    x0: int
    y0: int
    x1: int
    y1: int
    captured_image: str  # relative to run_dir
    measurement_index: int
    energy_sum: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def contains(self, other: "._RegionMeasurement") -> bool:
        return (
            other.x0 >= self.x0
            and other.y0 >= self.y0
            and other.x1 <= self.x1
            and other.y1 <= self.y1
        )


def _load_metadata(run_dir: Path) -> Dict[str, object]:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in run directory: {run_dir}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_measurements(run_dir: Path) -> List[Dict[str, object]]:
    path = run_dir / "measurements.json"
    if not path.exists():
        raise FileNotFoundError(f"measurements.json not found in run directory: {run_dir}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_reference_frames(run_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load I_black and I_white from the stored reference measurements.
    """
    measurements = _load_measurements(run_dir)
    frames: Dict[str, np.ndarray] = {}

    for rec in measurements:
        if rec.get("type") != "reference":
            continue
        label = rec.get("reference_label")
        if not label:
            continue
        rel_cap = rec.get("captured_image")
        if not rel_cap:
            continue
        cap_path = run_dir / rel_cap
        img = cv2.imread(str(cap_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read reference frame at {cap_path}")
        frames[label] = img

    if "white" not in frames or "black" not in frames:
        raise RuntimeError(
            "Adaptive decoder requires both 'white' and 'black' reference frames."
        )

    white = frames["white"]
    black = frames["black"]
    if white.shape != black.shape:
        raise ValueError("Reference frames I_white and I_black must have same shape.")

    return frames


def _load_region_measurements(run_dir: Path) -> List[_RegionMeasurement]:
    """
    Parse all region measurements from an adaptive scan's measurement log.
    """
    records = _load_measurements(run_dir)
    regions: List[_RegionMeasurement] = []

    for rec in records:
        region = rec.get("region")
        if not region:
            continue
        regions.append(
            _RegionMeasurement(
                level=int(region["level"]),
                x0=int(region["x0"]),
                y0=int(region["y0"]),
                x1=int(region["x1"]),
                y1=int(region["y1"]),
                captured_image=str(rec["captured_image"]),
                measurement_index=int(rec["measurement_index"]),
                energy_sum=float(rec.get("energy_sum", 0.0)),
            )
        )

    return regions


def _find_frontier_regions(regions: List[_RegionMeasurement]) -> List[_RegionMeasurement]:
    """
    Determine the frontier (leaf) regions in the measured quadtree.

    A region is treated as a leaf if none of its measured children
    (level+1, contained within it) appear in the log. This naturally
    includes true terminal leaves and any regions where the scan
    stopped due to measurement budget limits.
    """
    if not regions:
        return []

    leaves: List[_RegionMeasurement] = []
    for i, parent in enumerate(regions):
        has_child = False
        for j, child in enumerate(regions):
            if i == j:
                continue
            if child.level == parent.level + 1 and parent.contains(child):
                has_child = True
                break
        if not has_child:
            leaves.append(parent)
    return leaves


def build_frontier_mapping_from_adaptive_run(
    run_dir: Union[Path, str],
    pixel_noise_threshold: Optional[float] = None,
    min_relative_intensity: float = 0.05,
) -> MappingProducts:
    """
    Reconstruct a sparse MappingProducts from an adaptive_hierarchical_scan run.

    This decoder:
      - loads the stored I_black and I_white,
      - rebuilds the measured region tree from measurements.json,
      - extracts the frontier (leaf) regions as a piecewise-constant basis,
      - for each camera pixel, assigns the projector coordinate corresponding
        to the frontier region that produces the strongest *relative* response.

    The result is compatible with relight.py and projector_pov.py:
    `map_x`, `map_y`, `mask`, `signal_strength`, `i_white`, `i_black`,
    and `projector_size` are all populated.
    """
    run_dir = Path(run_dir)
    metadata = _load_metadata(run_dir)
    algorithm = metadata.get("algorithm", "")
    if algorithm != "adaptive_hierarchical_scan":
        raise ValueError(
            "Frontier adaptive decoding is only defined for runs with "
            "algorithm == 'adaptive_hierarchical_scan'. "
            f"Found algorithm='{algorithm}'."
        )

    projector_width = int(metadata["projector_width"])
    projector_height = int(metadata["projector_height"])

    reference_frames = _load_reference_frames(run_dir)
    white_img = reference_frames["white"]
    black_img = reference_frames["black"]

    camera_height, camera_width = white_img.shape

    white_f = white_img.astype(np.float32)
    black_f = black_img.astype(np.float32)

    signal_strength = np.clip(white_f - black_f, a_min=0.0, a_max=None)

    # Choose a per-pixel threshold for the confidence mask. If the caller
    # does not supply one, derive it from the signal statistics so that
    # low-contrast scenes still yield a useful mask but extremely dark
    # pixels are suppressed.
    if pixel_noise_threshold is None:
        max_signal = float(signal_strength.max()) if signal_strength.size else 0.0
        pixel_noise_threshold = max(5.0, 0.02 * max_signal)

    pixel_noise_threshold = float(pixel_noise_threshold)
    mask = signal_strength > pixel_noise_threshold

    map_x = np.full((camera_height, camera_width), -1, dtype=np.int32)
    map_y = np.full((camera_height, camera_width), -1, dtype=np.int32)
    best_response = np.zeros((camera_height, camera_width), dtype=np.float32)

    safe_signal = np.where(signal_strength > 1e-6, signal_strength, 1.0)

    regions = _load_region_measurements(run_dir)
    if not regions:
        # No region measurements at all – return an empty mapping but keep
        # the stored reference frames for consistency.
        return MappingProducts(
            map_x=map_x,
            map_y=map_y,
            mask=mask,
            signal_strength=signal_strength,
            i_white=white_img.copy(),
            i_black=black_img.copy(),
            noise_threshold=pixel_noise_threshold,
            projector_size=(projector_width, projector_height),
        )

    frontier = _find_frontier_regions(regions)

    for region in frontier:
        # Represent each frontier region by its integer center coordinate.
        cx = (region.x0 + region.x1 - 1) // 2
        cy = (region.y0 + region.y1 - 1) // 2
        rel_cap = Path(region.captured_image)
        cap_path = run_dir / rel_cap

        capture = cv2.imread(str(cap_path), cv2.IMREAD_GRAYSCALE)
        if capture is None:
            raise FileNotFoundError(f"Failed to read region capture at {cap_path}")

        if capture.shape != (camera_height, camera_width):
            raise ValueError(
                "Region capture resolution mismatch: expected "
                f"{camera_width}x{camera_height}, got "
                f"{capture.shape[1]}x{capture.shape[0]}"
            )

        capture_f = capture.astype(np.float32)
        response = np.clip(capture_f - black_f, a_min=0.0, a_max=None)

        relative = response / safe_signal
        candidate = mask & (response > best_response) & (
            relative >= float(min_relative_intensity)
        )

        map_x[candidate] = cx
        map_y[candidate] = cy
        best_response[candidate] = response[candidate]

    return MappingProducts(
        map_x=map_x,
        map_y=map_y,
        mask=mask,
        signal_strength=signal_strength,
        i_white=white_img.copy(),
        i_black=black_img.copy(),
        noise_threshold=pixel_noise_threshold,
        projector_size=(projector_width, projector_height),
    )

