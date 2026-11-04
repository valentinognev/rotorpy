"""A deliberately cheaper camera-homing config for the offline reliability harness
(plan Phase 9).

The full `examples/camera_homing.jsonc` scene renders 2236-by-2236 images with
dense (5 cm spacing) surface features -- realistic for one recorded flight, far too
slow for hundreds of closed-loop trials. `harness_config` loads that same file (the
one source of truth for the target/vehicle/envelope geometry) and overrides only the
rendering resolution and feature spacing. Detection (`pixels.DetectorConfig`, a 4-
pixel minimum blob) and every guidance/estimator/command-limit module still run on
real rendered pixels through the real pinhole projection -- just fewer of them.
"""

import copy
from pathlib import Path
from typing import Optional

from rotorpy.guidance.homing_config import load_homing_config

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"
DEFAULT_CAMERA_HOMING_JSONC = _EXAMPLES_DIR / "camera_homing.jsonc"

# Proposed, harness-only (not used by production guidance/tracker): higher than a
# minimal thumbnail so the production target (`gazebo_target_radius_m`, 0.75 m) stays
# above `pixels.DetectorConfig.min_area` (4 px) across the scene's full ~110 m
# documented approach range -- a first attempt at a small (96 px) image plus a
# synthetically enlarged target instead broke the forward/upward camera handoff near
# closest approach (the upward camera's geometry assumes the production target
# size), surfacing as spurious `TEMPORARY_LOSS`/crashes that were a harness artifact,
# not a guidance defect (see `task-9-report.md`'s self-review). Rendering cost here
# is dominated by feature count (`feature_spacing_m`), not pixel count, so this stays
# inexpensive despite the larger frame.
PROPOSED_HARNESS_IMAGE_SIZE_PX = 600
# Proposed, harness-only: wider than the production 0.05 m so the translated-feature
# template (`target_world.MovingTargetWorldBuilder`) stays a few hundred points, not
# several thousand, without changing the target's apparent color coverage much at
# this resolution.
PROPOSED_HARNESS_FEATURE_SPACING_M = 0.1


def harness_config(
    base_path: Optional[Path] = None,
    image_size_px: int = PROPOSED_HARNESS_IMAGE_SIZE_PX,
    feature_spacing_m: float = PROPOSED_HARNESS_FEATURE_SPACING_M,
) -> dict:
    """A `load_homing_config` dict, with the rendering resolution and feature
    spacing overridden for the closed-loop harness. Every other field (target/
    vehicle/envelope geometry, thresholds) is the authoritative scene config:
    the target keeps its true production size (see the `image_size_px` docstring
    above for why a larger synthetic target was tried and reverted)."""
    path = DEFAULT_CAMERA_HOMING_JSONC if base_path is None else Path(base_path)
    config = copy.deepcopy(load_homing_config(path))
    config["image_width_px"] = int(image_size_px)
    config["image_height_px"] = int(image_size_px)
    config["feature_spacing_m"] = float(feature_spacing_m)
    return config
