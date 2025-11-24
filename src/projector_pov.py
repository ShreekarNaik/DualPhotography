from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from run_helpers import resolve_run_dir
from structured_light_products import MappingProducts, load_mapping_products
from adaptive_decode import build_frontier_mapping_from_adaptive_run


def run_projector_pov(
    run_dir: Union[Path, str],
    scene_image: Optional[Union[Path, str]] = None,
    output_path: Optional[Union[Path, str]] = None,
    counter_output: Optional[Union[Path, str]] = None,
    mapping_file: str = "mapping_products.npz",
    decoder: str = "mapping",
) -> Tuple[Path, Path]:
    run_dir = Path(run_dir)
    if decoder == "mapping":
        mapping = load_mapping_products(run_dir, filename=mapping_file)
    elif decoder == "adaptive":
        mapping = build_frontier_mapping_from_adaptive_run(run_dir)
    else:
        raise ValueError(f"Unknown decoder mode: {decoder!r}")

    camera_img = _load_camera_image(scene_image, mapping)
    camera_height, camera_width = mapping.camera_shape
    if camera_img.shape[:2] != (camera_height, camera_width):
        raise ValueError(
            "Camera image resolution mismatch: expected "
            f"{camera_width}x{camera_height}, got {camera_img.shape[1]}x{camera_img.shape[0]}"
        )

    proj_width, proj_height = mapping.projector_size
    proj_shape = (proj_height, proj_width)

    channels = camera_img.shape[2]
    accum = np.zeros((proj_height, proj_width, channels), dtype=np.float32)
    counter = np.zeros(proj_shape, dtype=np.uint32)

    valid = mapping.mask & (mapping.map_x >= 0) & (mapping.map_y >= 0)
    if np.count_nonzero(valid) == 0:
        raise RuntimeError("Decoded maps contain no valid pixels for projector POV computation.")

    u = mapping.map_x[valid]
    v = mapping.map_y[valid]
    samples = camera_img[valid].astype(np.float32)

    for c in range(channels):
        np.add.at(accum[..., c], (v, u), samples[:, c])
    np.add.at(counter, (v, u), 1)

    nonzero = counter > 0
    divisor = np.maximum(counter, 1)
    accum /= divisor[..., None]
    accum[~nonzero] = 0.0

    filtered = np.zeros_like(accum)
    for c in range(channels):
        filtered[..., c] = cv2.medianBlur(accum[..., c], ksize=3)

    result = np.clip(filtered, 0.0, 255.0).astype(np.uint8)

    scene_stem = Path(scene_image).stem if scene_image is not None else "albedo"
    if output_path is None:
        output_path = run_dir / f"projector_dual_{scene_stem}.png"
    else:
        output_path = Path(output_path)

    if counter_output is None:
        counter_output = output_path.with_name(output_path.stem + "_counter.npy")
    else:
        counter_output = Path(counter_output)

    cv2.imwrite(str(output_path), result)
    np.save(counter_output, counter)

    return output_path, counter_output


def _load_camera_image(
    scene_image: Optional[Union[Path, str]], mapping: MappingProducts
) -> np.ndarray:
    if scene_image is None:
        gray = mapping.i_white
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    scene_path = Path(scene_image)
    img = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read camera image: {scene_path}")
    return img


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the projector's point-of-view image via T^T c_scene."
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
        "--scene-image",
        type=Path,
        default=None,
        help="Camera-space image to back-project (defaults to stored I_white).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the projector-view image.",
    )
    parser.add_argument(
        "--counter-output",
        type=Path,
        default=None,
        help="Optional output path for the Counter map (stored as .npy).",
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
    output_path, counter_path = run_projector_pov(
        run_dir=run_dir,
        scene_image=args.scene_image,
        output_path=args.output,
        counter_output=args.counter_output,
        mapping_file=args.mapping_file,
        decoder=args.decoder,
    )

    print(f"Projector POV image saved to {output_path}")
    print(f"Accumulation counter saved to {counter_path}")


if __name__ == "__main__":
    main()
