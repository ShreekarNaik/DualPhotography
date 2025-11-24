from __future__ import annotations

"""
Template scan algorithm for experimenting with new control strategies.

This file is deliberately verbose and heavily commented so that you can
quickly adapt it into a concrete algorithm (e.g., information‑gain
policy, bounded hierarchical scan, etc.).

Typical usage pattern (from the command line):

    python -m template_scan_algorithm \\
        --projector-width 1024 \\
        --projector-height 768 \\
        --num-steps 10

The key pieces you will usually customize are:

  * TemplateControllerState           – what internal state / belief
                                        your controller maintains.
  * _initialize_controller()         – how you initialize that state.
  * _compute_next_pattern()          – the control law: choose u_k.
  * _update_controller_state_from_measurement()
                                     – update belief from y_k.

This mirrors the structure of run_decomposition_basis_scan() and
run_adaptive_scan(), but keeps the control logic factored into small,
easy‑to‑edit functions.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from experiment import ExperimentRun, create_experiment_run


@dataclass
class TemplateControllerState:
    """
    Minimal example of controller / belief state for a scan algorithm.

    You are expected to extend this with whatever your control
    strategy needs: queues of regions, priors over T, covariance
    matrices, candidate pattern pools, etc.
    """

    # Generic step counter so you can see how many control decisions
    # were taken. You can safely keep or remove this.
    step: int = 0

    # Example of where to put algorithm‑specific state:
    # pending_regions: list[dict] = field(default_factory=list)
    # covariance: Optional[np.ndarray] = None
    # prior_mean: Optional[np.ndarray] = None
    # Feel free to delete these and add your own fields.
    metadata: Dict[str, object] = field(default_factory=dict)


def _capture_reference_frame(
    experiment: ExperimentRun,
    value: int,
    label: str,
) -> np.ndarray:
    """
    Capture a reference frame (all‑ON or all‑OFF) and log it.

    Many algorithms (Gray codes, adaptive scans, information‑gain
    policies) want I_white and I_black so that they can compute a
    confidence mask and signal strength. This helper mirrors the
    versions used in the existing implementations.
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


def _initialize_controller(experiment: ExperimentRun) -> TemplateControllerState:
    """
    Initialize the controller state before the first measurement.

    This is the right place to:
      * precompute candidate patterns,
      * allocate belief / covariance matrices,
      * build a queue of regions (for hierarchical scans),
      * store any hyper‑parameters into state.metadata.
    """
    state = TemplateControllerState()
    state.metadata["note"] = "EDIT ME: describe your algorithm here"
    # Example: if you wanted a BFS queue of regions, you might put:
    # state.pending_regions = [initial_region]
    return state


def _compute_next_pattern(
    state: TemplateControllerState,
    experiment: ExperimentRun,
) -> Tuple[np.ndarray, TemplateControllerState, bool]:
    """
    Core control law: choose the next projector pattern u_k.

    Parameters
    ----------
    state:
        Current controller / belief state (will typically be mutated
        in‑place).
    experiment:
        Shared ExperimentRun context (projector size, run_dir, etc.).

    Returns
    -------
    pattern:
        Grayscale pattern to project, shape (H_proj, W_proj),
        dtype=uint8 with values in [0, 255].
    new_state:
        Updated state *after* choosing the pattern (before seeing y_k).
        In the simple template we just mutate `state` and return it.
    stop:
        If True, the scan terminates and the caller should not call
        experiment.capture() again.

    How to customize
    ----------------
    Replace the body of this function with your policy, e.g.:
      * choose the region with highest expected information gain,
      * pop a region from a queue and build a block‑indicator pattern,
      * select a Gray‑code bit plane or a random multiplexing pattern.
    """
    # --- Example default policy: single all‑ON frame, then stop. ---
    if state.step >= 1:
        # No more measurements – terminate the scan.
        return np.zeros(
            (experiment.projector_height, experiment.projector_width), dtype=np.uint8
        ), state, True

    # For the very first step, show an all‑ON pattern. This is useful
    # as a sanity check and as I_white for later decoding.
    pattern = np.full(
        (experiment.projector_height, experiment.projector_width),
        255,
        dtype=np.uint8,
    )

    # Update controller state (mutate in‑place so extra fields survive).
    state.step += 1

    return pattern, state, False


def _update_controller_state_from_measurement(
    state: TemplateControllerState,
    captured_img: np.ndarray,
    measurement_index: int,
) -> TemplateControllerState:
    """
    Update the controller / belief state given the new measurement y_k.

    This is where you implement:
      * Bayesian update of (mu_k, Sigma_k) for an information‑gain policy,
      * decisions to subdivide or prune regions based on total energy,
      * any stopping criteria derived from uncertainty or budget.

    The default implementation is a no‑op so that the template runs
    without modification; you almost certainly want to replace this.
    """
    # Example of how you might stash simple diagnostics:
    state.metadata.setdefault("energies", []).append(float(captured_img.sum()))
    state.metadata.setdefault("mean_intensities", []).append(
        float(captured_img.mean())
    )
    return state


def run_template_scan(
    num_steps: int = 1,
    projector_width: int = 1024,
    projector_height: int = 768,
    blender_dir: Union[Path, str] = Path("blender-virtual-experiment"),
    runs_root: Union[Path, str] = Path("runs"),
    run_name: Optional[str] = None,
) -> Path:
    """
    High‑level driver for a new scan algorithm.

    This function:
      * creates a new ExperimentRun (run directory + metadata),
      * initializes the controller state,
      * loops over control decisions, calling:
          - _compute_next_pattern()  to choose u_k,
          - ExperimentRun.capture()  to talk to Blender,
          - _update_controller_state_from_measurement() to apply feedback,
      * logs all measurements and final summary via ExperimentRun.finalize().

    By editing only the three small helpers above you can implement a
    wide range of control strategies without touching the boilerplate.
    """
    experiment: ExperimentRun = create_experiment_run(
        algorithm="template_scan_algorithm",
        blender_dir=blender_dir,
        runs_root=runs_root,
        run_name=run_name,
        projector_width=projector_width,
        projector_height=projector_height,
        extra_metadata={
            "num_steps_requested": num_steps,
            "template_file": __file__,
        },
    )

    # In case the user changed projector sizes in Blender, honor the
    # values populated into the ExperimentRun.
    projector_width = experiment.projector_width
    projector_height = experiment.projector_height

    # Optional: capture reference frames (comment out if not needed).
    reference_frames: Dict[str, np.ndarray] = {}
    reference_frames["black"] = _capture_reference_frame(experiment, 0, "black")
    reference_frames["white"] = _capture_reference_frame(experiment, 255, "white")

    state = _initialize_controller(experiment)

    # Main control loop. Replace/extend this if your algorithm uses a
    # different stopping rule (e.g., based on posterior covariance).
    for _ in range(num_steps):
        pattern, state, stop = _compute_next_pattern(state, experiment)
        if stop:
            break

        captured_img, rel_proj, rel_cap, measurement_index = experiment.capture(
            pattern
        )

        # Generic per‑measurement log. Add any algorithm‑specific
        # fields you find useful for analysis.
        experiment.log_measurement(
            {
                "measurement_index": measurement_index,
                "controller_step": state.step,
                "type": "template_measurement",
                "projection_pattern": rel_proj,
                "captured_image": rel_cap,
                "mean_intensity": float(captured_img.mean()),
                "energy_sum": float(captured_img.sum()),
            }
        )

        state = _update_controller_state_from_measurement(
            state=state,
            captured_img=captured_img,
            measurement_index=measurement_index,
        )

    # At this point you could optionally decode MappingProducts and
    # save them, similar to decomposition_basis_scan.py or
    # adaptive_scan.py. The template keeps things generic and only
    # writes metadata and the raw measurement log.
    experiment.finalize(
        summary={
            "controller_steps_taken": state.step,
            "num_reference_frames": len(reference_frames),
        },
        extra_outputs={"controller_state": state.metadata},
    )

    return experiment.run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Template scan algorithm for designing new closed‑loop "
            "or open‑loop control strategies on top of the shared "
            "ExperimentRun / Blender interface."
        )
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1,
        help="Maximum number of control decisions / measurements.",
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
        help="Root directory where per‑run folders are created.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit name for the run folder.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    run_dir = run_template_scan(
        num_steps=args.num_steps,
        projector_width=args.projector_width,
        projector_height=args.projector_height,
        blender_dir=args.blender_dir,
        runs_root=args.runs_root,
        run_name=args.run_name,
    )

    print(f"Template scan completed. Results stored in: {run_dir}")


if __name__ == "__main__":
    main()

