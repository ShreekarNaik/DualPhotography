from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class MappingProducts:
    """Decoded data products derived from a Gray-code acquisition."""

    map_x: np.ndarray  # (H_c, W_c) int32, projector column per camera pixel
    map_y: np.ndarray  # (H_c, W_c) int32, projector row per camera pixel
    mask: np.ndarray  # (H_c, W_c) bool, confidence mask
    signal_strength: np.ndarray  # (H_c, W_c) float32, I_white - I_black
    i_white: np.ndarray  # (H_c, W_c) uint8
    i_black: np.ndarray  # (H_c, W_c) uint8
    noise_threshold: float
    projector_size: Tuple[int, int]

    @property
    def camera_shape(self) -> Tuple[int, int]:
        return self.map_x.shape


def build_gray_code_products(
    white_img: np.ndarray,
    black_img: np.ndarray,
    bit_planes: Dict[int, np.ndarray],
    projector_size: Tuple[int, int],
    noise_threshold: float,
) -> MappingProducts:
    mask, signal_strength = _compute_confidence_mask(
        white_img, black_img, noise_threshold
    )
    indices = _decode_gray_code_indices(bit_planes, black_img, signal_strength)
    map_x, map_y = _build_coordinate_maps(indices, mask, projector_size)

    return MappingProducts(
        map_x=map_x,
        map_y=map_y,
        mask=mask,
        signal_strength=signal_strength,
        i_white=white_img.copy(),
        i_black=black_img.copy(),
        noise_threshold=float(noise_threshold),
        projector_size=projector_size,
    )


def save_mapping_products(
    run_dir: Path,
    products: MappingProducts,
    filename: str = "mapping_products.npz",
) -> Path:
    output_path = run_dir / filename
    np.savez_compressed(
        output_path,
        map_x=products.map_x.astype(np.int32),
        map_y=products.map_y.astype(np.int32),
        mask=products.mask.astype(np.uint8),
        signal_strength=products.signal_strength.astype(np.float32),
        i_white=products.i_white.astype(np.uint8),
        i_black=products.i_black.astype(np.uint8),
        noise_threshold=np.array([products.noise_threshold], dtype=np.float32),
        projector_width=np.array([products.projector_size[0]], dtype=np.int32),
        projector_height=np.array([products.projector_size[1]], dtype=np.int32),
    )
    return output_path


def load_mapping_products(
    run_dir: Path,
    filename: str = "mapping_products.npz",
) -> MappingProducts:
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Mapping products file not found at {path}. Run the acquisition first."
        )

    with np.load(path) as data:
        map_x = data["map_x"].astype(np.int32)
        map_y = data["map_y"].astype(np.int32)
        mask = data["mask"].astype(bool)
        signal_strength = data["signal_strength"].astype(np.float32)
        i_white = data["i_white"].astype(np.uint8)
        i_black = data["i_black"].astype(np.uint8)
        noise_threshold = float(data["noise_threshold"][0])
        projector_width = int(data["projector_width"][0])
        projector_height = int(data["projector_height"][0])

    return MappingProducts(
        map_x=map_x,
        map_y=map_y,
        mask=mask,
        signal_strength=signal_strength,
        i_white=i_white,
        i_black=i_black,
        noise_threshold=noise_threshold,
        projector_size=(projector_width, projector_height),
    )


def _compute_confidence_mask(
    white_img: np.ndarray,
    black_img: np.ndarray,
    noise_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    white = white_img.astype(np.float32)
    black = black_img.astype(np.float32)
    signal_strength = np.clip(white - black, a_min=0.0, a_max=None)
    mask = signal_strength > float(noise_threshold)
    return mask, signal_strength


def _decode_gray_code_indices(
    bit_planes: Dict[int, np.ndarray],
    black_img: np.ndarray,
    signal_strength: np.ndarray,
) -> np.ndarray:
    if not bit_planes:
        raise ValueError("No Gray-code measurements provided for decoding.")

    max_plane = max(bit_planes.keys())
    expected_planes = set(range(max_plane + 1))
    missing = expected_planes.difference(bit_planes.keys())
    if missing:
        missing_str = ", ".join(str(idx) for idx in sorted(missing))
        raise ValueError(f"Missing Gray-code measurements for planes: {missing_str}")

    height, width = next(iter(bit_planes.values())).shape
    gray_values = np.zeros((height, width), dtype=np.uint32)

    black = black_img.astype(np.float32)
    safe_signal = np.where(signal_strength > 1e-6, signal_strength, 1.0)

    for plane in range(max_plane + 1):
        image = bit_planes[plane].astype(np.float32)
        normalized = np.clip((image - black) / safe_signal, 0.0, 1.0)
        bit = (normalized > 0.5).astype(np.uint32)
        gray_values |= bit << plane

    return _gray_to_binary(gray_values)


def _gray_to_binary(gray_values: np.ndarray) -> np.ndarray:
    binary = gray_values.copy()
    mask = binary >> 1
    while np.any(mask):
        binary ^= mask
        mask >>= 1
    return binary.astype(np.int64)


def _build_coordinate_maps(
    indices: np.ndarray,
    mask: np.ndarray,
    projector_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    proj_width, proj_height = projector_size
    num_pixels = proj_width * proj_height

    valid = (indices >= 0) & (indices < num_pixels) & mask

    map_x = np.full((height, width), -1, dtype=np.int32)
    map_y = np.full((height, width), -1, dtype=np.int32)

    flat_indices = indices.astype(np.int64)
    map_x[valid] = (flat_indices[valid] % proj_width).astype(np.int32)
    map_y[valid] = (flat_indices[valid] // proj_width).astype(np.int32)

    return map_x, map_y
