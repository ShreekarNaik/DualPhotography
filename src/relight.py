from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from run_helpers import resolve_run_dir
from structured_light_products import load_mapping_products


def run_relighting(
    run_dir: Union[Path, str],
    pattern_path: Union[Path, str],
    output_path: Optional[Union[Path, str]] = None,
    mapping_file: str = "mapping_products.npz",
) -> Path:
    run_dir = Path(run_dir)
    pattern_path = Path(pattern_path)
    mapping = load_mapping_products(run_dir, filename=mapping_file)

    pattern = cv2.imread(str(pattern_path), cv2.IMREAD_COLOR)
    if pattern is None:
        raise FileNotFoundError(f"Failed to read projector pattern: {pattern_path}")

    proj_height, proj_width, channels = pattern.shape
    expected_width, expected_height = mapping.projector_size
    if (proj_width, proj_height) != (expected_width, expected_height):
        raise ValueError(
            "Projector pattern resolution mismatch: expected "
            f"{expected_width}x{expected_height}, got {proj_width}x{proj_height}"
        )

    pattern = pattern.astype(np.float32)

    camera_height, camera_width = mapping.camera_shape
    output = np.zeros((camera_height, camera_width, channels), dtype=np.float32)

    valid = mapping.mask & (mapping.map_x >= 0) & (mapping.map_y >= 0)
    if np.count_nonzero(valid) == 0:
        raise RuntimeError("Decoded maps contain no valid pixels to relight.")

    projector_samples = pattern[mapping.map_y[valid], mapping.map_x[valid]]
    albedo = mapping.i_white.astype(np.float32) / 255.0
    output[valid] = projector_samples * albedo[valid, None]

    output = np.clip(output, 0.0, 255.0)
    result = output.astype(np.uint8)

    if channels == 1:
        result = result[:, :, 0]

    if output_path is None:
        stem = pattern_path.stem
        output_path = run_dir / f"relit_{stem}.png"
    else:
        output_path = Path(output_path)

    cv2.imwrite(str(output_path), result)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Relight the captured scene using stored Gray-code mappings and a new "
            "virtual projector pattern."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to an existing acquisition run directory.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory containing acquisition runs (used with --run-name).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the run folder under runs-root (alternative to --run-dir).",
    )
    parser.add_argument(
        "--pattern",
        type=Path,
        required=True,
        help="Path to the virtual projector image to apply.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the relit camera image.",
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        default="mapping_products.npz",
        help="Filename of the stored mapping data inside the run directory.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    run_dir = resolve_run_dir(args.run_dir, args.runs_root, args.run_name)
    output_path = run_relighting(
        run_dir=run_dir,
        pattern_path=args.pattern,
        output_path=args.output,
        mapping_file=args.mapping_file,
    )

    print(f"Relit image saved to {output_path}")


if __name__ == "__main__":
    main()
