from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from experiment import ExperimentRun, create_experiment_run
from structured_light_products import build_gray_code_products, save_mapping_products


@dataclass
class GrayCodeMeasurement:
    """Container for a single Gray-code capture used during decoding."""

    bit_plane: int
    polarity: str
    image: np.ndarray
    measurement_index: int


def _capture_reference_frame(
    experiment: ExperimentRun,
    value: int,
    label: str,
) -> Tuple[np.ndarray, int]:
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
    return captured_img, measurement_index


def run_decomposition_basis_scan(
    projector_width: int = 1024,
    projector_height: int = 768,
    blender_dir: Union[Path, str] = Path("blender-virtual-experiment"),
    runs_root: Union[Path, str] = Path("runs"),
    run_name: Optional[str] = None,
    include_inverse: bool = True,
    max_planes: Optional[int] = None,
    noise_threshold: float = 5.0,
) -> Path:
    """
    Run Algorithm 2: Logarithmic Basis Scan using Gray Codes.

    This implementation focuses on structured acquisition:
      - For each Gray-code bit plane, generates a binary illumination pattern
        where pixels are ON if that bit is 1.
      - Optionally also projects the inverse pattern for robustness
        (include_inverse=True).
      - For every pattern, it:
          * uses ExperimentRun.capture to interface with Blender and
            archive images,
          * logs per-measurement metadata via ExperimentRun.
    """
    experiment: ExperimentRun = create_experiment_run(
        algorithm="decomposition_basis_scan",
        blender_dir=blender_dir,
        runs_root=runs_root,
        run_name=run_name,
        projector_width=projector_width,
        projector_height=projector_height,
        extra_metadata={
            "include_inverse": include_inverse,
            "max_planes": max_planes,
            "noise_threshold": noise_threshold,
        },
    )
    projector_width = experiment.projector_width
    projector_height = experiment.projector_height
    num_pixels = projector_width * projector_height

    # Number of Gray-code bit planes.
    num_planes = int(math.ceil(math.log2(max(1, num_pixels))))
    if max_planes is not None:
        num_planes = min(num_planes, max_planes)

    # Precompute Gray codes for all projector pixels (linear index).
    indices = np.arange(num_pixels, dtype=np.uint32)
    gray_codes = indices ^ (indices >> 1)

    reference_frames: Dict[str, np.ndarray] = {}
    reference_frames["black"], _ = _capture_reference_frame(experiment, 0, "black")
    reference_frames["white"], _ = _capture_reference_frame(experiment, 255, "white")

    graycode_measurements: List[GrayCodeMeasurement] = []

    for plane in range(num_planes):
        bit_mask = ((gray_codes >> plane) & 1).astype(np.uint8)

        # Pattern where Gray-code bit == 1.
        pattern_on = (bit_mask * 255).reshape((projector_height, projector_width))

        polarities = ["on"]
        if include_inverse:
            polarities.append("off")

        for polarity in polarities:
            if polarity == "on":
                pattern = pattern_on
            else:
                pattern = ((1 - bit_mask) * 255).reshape(
                    (projector_height, projector_width)
                )

            captured_img, rel_proj, rel_cap, measurement_index = experiment.capture(
                pattern
            )

            graycode_measurements.append(
                GrayCodeMeasurement(
                    bit_plane=plane,
                    polarity=polarity,
                    image=captured_img,
                    measurement_index=measurement_index,
                )
            )

            energy = float(captured_img.sum())
            mean_intensity = float(captured_img.mean())

            experiment.log_measurement(
                {
                    "measurement_index": measurement_index,
                    "type": "gray_code",
                    "bit_plane": plane,
                    "polarity": polarity,
                    "energy_sum": energy,
                    "mean_intensity": mean_intensity,
                    "projection_pattern": rel_proj,
                    "captured_image": rel_cap,
                }
            )

    on_plane_images: Dict[int, np.ndarray] = {}
    for measurement in graycode_measurements:
        if measurement.polarity == "on":
            on_plane_images[measurement.bit_plane] = measurement.image

    if "white" not in reference_frames or "black" not in reference_frames:
        raise RuntimeError("Missing reference frames required for Gray-code decoding.")

    mapping_products = build_gray_code_products(
        white_img=reference_frames["white"],
        black_img=reference_frames["black"],
        bit_planes=on_plane_images,
        projector_size=(projector_width, projector_height),
        noise_threshold=noise_threshold,
    )

    mapping_path = save_mapping_products(experiment.run_dir, mapping_products)

    experiment.finalize(
        summary={
            "num_pixels": int(num_pixels),
            "num_planes": int(num_planes),
            "mapping_products_file": mapping_path.name,
        }
    )

    return experiment.run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Logarithmic Gray-code basis scan (Algorithm 2) "
            "using Blender-based virtual experiment."
        )
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
        "--include-inverse",
        action="store_true",
        default=True,
        help="Project both bit-plane and its inverse (default: True).",
    )
    parser.add_argument(
        "--no-include-inverse",
        dest="include_inverse",
        action="store_false",
        help="Disable inverse pattern projections.",
    )
    parser.add_argument(
        "--max-planes",
        type=int,
        default=None,
        help="Optional cap on the number of Gray-code planes.",
    )
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=5.0,
        help=(
            "Signal strength threshold (I_white - I_black) used for the confidence "
            "mask in decoded mappings."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    run_dir = run_decomposition_basis_scan(
        projector_width=args.projector_width,
        projector_height=args.projector_height,
        blender_dir=args.blender_dir,
        runs_root=args.runs_root,
        run_name=args.run_name,
        include_inverse=args.include_inverse,
        max_planes=args.max_planes,
        noise_threshold=args.noise_threshold,
    )

    print(f"Basis scan completed. Results stored in: {run_dir}")


if __name__ == "__main__":
    main()
