# Structured-Light Workflow

This project now separates data acquisition from downstream rendering to make it
easy to reuse the same decoded maps for relighting or projector-point-of-view
simulations.

## 1. Acquisition (Basis Scan)

```bash
./run.sh basis \
  --run-name <experiment_name> \
  --noise-threshold 8.0          # optional, default 5.0
```

- The scanner now captures reference frames (`I_black`, `I_white`) before the
  Gray-code patterns. These provide the signal strength map
  `S = I_white - I_black` and the albedo image used for relighting.
- After capture, the decoder builds `Map_X`, `Map_Y`, the confidence mask, and
  saves everything in `mapping_products.npz` inside the run directory. This
  file is the only dependency for later stages.

## 2. Relighting (Camera POV)

```bash
./run.sh relight \
  --run-name <experiment_name> \
  --pattern path/to/projector_image.png \
  [--output runs/<experiment_name>/relit_custom.png]
```

- The pattern resolution must match the projector resolution used during the
  scan (defaults: 1024×768). The stored `Map_X/Map_Y` perform the lookup, and
  pixels excluded by the confidence mask remain black.
- Output defaults to `runs/<run>/relit_<pattern_stem>.png`.

## 3. Projector POV (Dual Image)

```bash
./run.sh projector-pov \
  --run-name <experiment_name> \
  [--scene-image path/to/camera.png] \
  [--output runs/<experiment_name>/projector_dual.png]
```

- If `--scene-image` is omitted the script uses the stored `I_white` as the
  camera intensity. The algorithm accumulates values into projector space,
  averages using the visit counter, and applies a 3×3 median filter to suppress
  holes/Moiré artifacts.
- The command also writes a counter map (`*_counter.npy`) so you can inspect
  how many camera pixels contributed to each projector pixel.

## Stored Mapping Data

`mapping_products.npz` contains:

- `map_x`, `map_y` – projector column/row per camera pixel (invalid pixels are
  `-1`).
- `mask` – confidence mask (`S > noise_threshold`).
- `signal_strength` – the raw `S` map for debugging/visualization.
- `i_white`, `i_black` – the reference captures (albedo and shadow baselines).
- `projector_width`, `projector_height`, `noise_threshold` – metadata needed by
  the relighting and projector-pov utilities.

Any relighting or projector-pov invocation simply loads this file, so you can
run multiple lighting experiments without re-running the acquisition.

## 4. Acquisition (Adaptive Hierarchical Scan)

```bash
./run.sh adaptive \
  --run-name <experiment_name> \
  --noise-threshold 1e5        # default
```

- The adaptive scanner first captures `I_black` and `I_white`, then performs a
  hierarchical region scan, subdividing only regions whose energy exceeds the
  threshold.
- For each terminal region (typically a single projector pixel), it records the
  per-pixel camera response and post-processes all terminals into the same
  `mapping_products.npz` structure as the basis scan.
- This means you can pass an adaptive run to `run.sh relight` or
  `run.sh projector-pov` exactly as with a Gray-code basis run, while often
  using fewer measurements for sparse scenes.
