from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from run_helpers import resolve_run_dir
from structured_light_products import load_mapping_products
from adaptive_decode import build_frontier_mapping_from_adaptive_run


def run_relighting(
    run_dir: Union[Path, str],
    pattern_path: Union[Path, str],
    output_path: Optional[Union[Path, str]] = None,
    mapping_file: str = "mapping_products.npz",
    decoder: str = "mapping",
) -> Path:
    run_dir = Path(run_dir)
    pattern_path = Path(pattern_path)

    if decoder == "mapping":
        mapping = load_mapping_products(run_dir, filename=mapping_file)
    elif decoder == "adaptive":
        mapping = build_frontier_mapping_from_adaptive_run(run_dir)
    else:
        raise ValueError(f"Unknown decoder mode: {decoder!r}")

    pattern = cv2.imread(str(pattern_path), cv2.IMREAD_COLOR)
    if pattern is None:
        raise FileNotFoundError(f"Failed to read projector pattern: {pattern_path}")

    proj_height, proj_width = pattern.shape[:2]
    expected_width, expected_height = mapping.projector_size
    if (proj_width, proj_height) != (expected_width, expected_height):
        # Automatically resize the input pattern to the projector's
        # native resolution so that we can still use the stored
        # projector-to-camera mappings even if the user supplies a
        # camera-view image or a mismatched resolution.
        if proj_width <= 0 or proj_height <= 0:
            raise ValueError(
                f"Invalid projector pattern resolution: {proj_width}x{proj_height}"
            )
        # Use area interpolation when downsampling, linear when upsampling.
        if proj_width > expected_width or proj_height > expected_height:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        pattern = cv2.resize(
            pattern, (expected_width, expected_height), interpolation=interpolation
        )
        proj_height, proj_width = pattern.shape[:2]

    channels = pattern.shape[2] if pattern.ndim == 3 else 1

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
    parser.add_argument(
        "--decoder",
        type=str,
        default="mapping",
        choices=["mapping", "adaptive"],
        help=(
            "Decoding strategy: 'mapping' uses stored mapping_products.npz "
            "(Gray-code / dense), 'adaptive' reconstructs a sparse mapping "
            "from an adaptive hierarchical scan."
        ),
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
        decoder=args.decoder,
    )

    print(f"Relit image saved to {output_path}")


if __name__ == "__main__":
    main()
