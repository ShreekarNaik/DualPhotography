from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np

from experiment import ExperimentRun, create_experiment_run


def run_decomposition_basis_scan(
    blender_dir: Union[Path, str] = Path("blender-virtual-experiment"),
    runs_root: Union[Path, str] = Path("runs"),
    run_name: Optional[str] = None,
    include_inverse: bool = True,
    max_planes: Optional[int] = None,
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
        extra_metadata={
            "include_inverse": include_inverse,
            "max_planes": max_planes,
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

            energy = float(captured_img.sum())
            mean_intensity = float(captured_img.mean())

            experiment.log_measurement(
                {
                    "measurement_index": measurement_index,
                    "bit_plane": plane,
                    "polarity": polarity,
                    "energy_sum": energy,
                    "mean_intensity": mean_intensity,
                    "projection_pattern": rel_proj,
                    "captured_image": rel_cap,
                }
            )

    experiment.finalize(
        summary={
            "num_pixels": int(num_pixels),
            "num_planes": int(num_planes),
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
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    run_dir = run_decomposition_basis_scan(
        blender_dir=args.blender_dir,
        runs_root=args.runs_root,
        run_name=args.run_name,
        include_inverse=args.include_inverse,
        max_planes=args.max_planes,
    )

    print(f"Basis scan completed. Results stored in: {run_dir}")


if __name__ == "__main__":
    main()

