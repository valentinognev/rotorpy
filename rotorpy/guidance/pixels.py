"""Target-pixel detection for camera homing."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage


def _target_mask(image: np.ndarray) -> np.ndarray:
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    return (red > 0.5) & (red > green) & (red > blue)


def target_pixel_count(image: np.ndarray) -> int:
    """Count pixels with red > 0.5 and red strictly greater than green and blue."""
    return int(np.count_nonzero(_target_mask(image)))


def target_centroid(image: np.ndarray) -> Optional[np.ndarray]:
    """Mean (column, row) of target pixels, or None when the count is 0."""
    rows, columns = np.nonzero(_target_mask(image))
    if rows.size == 0:
        return None
    return np.array([np.mean(columns), np.mean(rows)])


def target_square(image: np.ndarray) -> Optional[Tuple[float, float, int]]:
    """Bounding square (center_u, center_v, side), or None when the count is 0."""
    rows, columns = np.nonzero(_target_mask(image))
    if rows.size == 0:
        return None
    col_min = int(columns.min())
    col_max = int(columns.max())
    row_min = int(rows.min())
    row_max = int(rows.max())
    center_u = (col_min + col_max) / 2
    center_v = (row_min + row_max) / 2
    side = max(col_max - col_min + 1, row_max - row_min + 1)
    return (center_u, center_v, side)


class DetectionStatus(Enum):
    NO_FRAME = "no_frame"
    NO_DETECTION = "no_detection"
    AMBIGUOUS = "ambiguous"
    DETECTED = "detected"


@dataclass(frozen=True)
class DetectorConfig:
    min_red: float = 0.4
    min_dominance: float = 0.25  # red minus the larger of green and blue
    min_saturation: float = 0.5  # (red - min channel) / red
    min_area: int = 4
    max_area_fraction: float = 0.7
    min_fill: float = 0.35  # component area / bounding-box area
    max_aspect: float = 8.0  # unclipped bounding-box long side / short side
    clipped_confidence: float = 0.5
    clipped_score: float = 0.5
    dominance_ratio: float = 1.5  # best score must beat the runner-up by this factor
    base_uncertainty_px: float = 0.5
    extent_uncertainty: float = 0.1  # per pixel of the longer bounding-box side
    clipped_extent_uncertainty: float = 0.5
    # Saturating cap on the bounding-box side fed into the extent-based uncertainty terms
    # above. The extent penalty models centroid-localization ambiguity, which grows with a
    # blob's linear size but must not grow without bound: a large, near-field target that
    # fills (and clips) the frame on final approach is a *well*-localized, confident lock,
    # not a highly uncertain one. Without this cap its raw extent (300+ px on a 2236 px
    # frame) alone produced `uncertainty_px > _MAX_TRACK_UNCERTAINTY_PX`, aborting the
    # track the instant one frame was missed. Capping keeps "clipped > unclipped" ordering
    # (every existing test uses far smaller blobs) while bounding the worst-case penalty.
    max_uncertainty_extent_px: float = 120.0
    gate_sigmas: float = 3.0
    gate_ambiguity_ratio: float = 2.0
    gate_ambiguity_margin_px: float = 1.0


DEFAULT_CONFIG = DetectorConfig()


@dataclass(frozen=True)
class TrackPrediction:
    """Predicted target location (u, v) and one-sigma position uncertainty."""

    u: float
    v: float
    sigma_px: float


@dataclass(frozen=True)
class TargetObservation:
    status: DetectionStatus
    timestamp: float
    camera: str
    bounds: Optional[Tuple[int, int, int, int]] = None  # inclusive (u_min, v_min, u_max, v_max)
    reference: Optional[Tuple[float, float]] = None  # (u, v), component centroid
    confidence: float = 0.0
    uncertainty_px: float = float("inf")
    area: int = 0
    clipped: bool = False
    num_candidates: int = 0


@dataclass(frozen=True)
class _Component:
    bounds: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    fill: float
    clipped: bool

    @property
    def extent(self) -> int:
        return max(self.bounds[2] - self.bounds[0], self.bounds[3] - self.bounds[1]) + 1

    def score(self, config: DetectorConfig) -> float:
        return self.area * self.fill * (config.clipped_score if self.clipped else 1.0)


def _dominance_candidate_mask(image: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Cheap red-dominance prefilter mask, in the image's own dtype (int16 for integer
    input) with a margin so float32 rounding cannot drop a borderline pixel. A strict
    superset of `_red_mask`'s exact test: no pixel outside it can ever be part of a
    detected component, by construction."""
    integer = np.issubdtype(image.dtype, np.integer)
    scale = 255.0 if integer else 1.0
    margin = 1e-6
    red = image[:, :, 0]
    if integer:
        red = red.astype(np.int16)
    dominance = red - np.maximum(image[:, :, 1], image[:, :, 2])
    return dominance >= (config.min_dominance - margin) * scale


def _red_mask(image: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Boolean red mask. Only pixels passing a cheap red-dominance prefilter in the image's
    own dtype (with a margin so float32 rounding cannot drop a borderline pixel) get the
    exact float64 test."""
    integer = np.issubdtype(image.dtype, np.integer)
    scale = 255.0 if integer else 1.0
    mask = np.zeros(image.shape[:2], dtype=bool)
    rows, columns = np.nonzero(_dominance_candidate_mask(image, config))
    pixels = image[rows, columns].astype(float) / scale
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    dominance = red - np.maximum(green, blue)
    saturation = (red - np.minimum(green, blue)) / np.maximum(red, 1e-6)
    mask[rows, columns] = (
        (red >= config.min_red)
        & (dominance >= config.min_dominance)
        & (saturation >= config.min_saturation)
    )
    return mask


# Padding (px) added around the tight candidate bounding box before cropping. Not load-bearing
# for correctness (the box is already exact), just a safety margin.
_CANDIDATE_CROP_PAD_PX = 4


def _candidate_crop(image: np.ndarray, config: DetectorConfig):
    """`(crop, (row_offset, col_offset))` tightly bounding every pixel that could possibly
    pass `_red_mask`'s exact test, or `(None, None)` when there is none.

    `_components` only needs `ndimage.label`/`find_objects` to see whatever region could
    contain a detected component; everywhere outside this crop is guaranteed all-background
    (`_dominance_candidate_mask` is a strict superset of the exact test, so nothing out
    there could ever pass it either). Labeling the crop instead of the whole frame is
    therefore mathematically identical to labeling the whole frame -- just far cheaper
    when, as in the normal hot path, a compact target sits in a much larger frame (a real
    2236x2236 Gazebo capture made per-tick detection take 0.2-0.9s, well past the 0.5s
    PX4 offboard setpoint expiry -- see UPDATES.md 4.15.2).
    """
    height, width = image.shape[:2]
    candidate = _dominance_candidate_mask(image, config)
    rows_any = candidate.any(axis=1)
    if not rows_any.any():
        return None, None
    cols_any = candidate.any(axis=0)
    row_indices = np.flatnonzero(rows_any)
    col_indices = np.flatnonzero(cols_any)
    row_min = max(0, int(row_indices[0]) - _CANDIDATE_CROP_PAD_PX)
    row_max = min(height, int(row_indices[-1]) + 1 + _CANDIDATE_CROP_PAD_PX)
    col_min = max(0, int(col_indices[0]) - _CANDIDATE_CROP_PAD_PX)
    col_max = min(width, int(col_indices[-1]) + 1 + _CANDIDATE_CROP_PAD_PX)
    return image[row_min:row_max, col_min:col_max], (row_min, col_min)


def _components(image: np.ndarray, config: DetectorConfig) -> List[_Component]:
    height, width = image.shape[:2]
    crop, origin = _candidate_crop(image, config)
    if crop is None:
        return []
    row_offset, col_offset = origin
    labels, count = ndimage.label(_red_mask(crop, config), structure=np.ones((3, 3)))
    components = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        rows, columns = np.nonzero(labels[box] == index)
        area = rows.size
        if area < config.min_area or area > config.max_area_fraction * height * width:
            continue
        v_min, u_min = box[0].start + row_offset, box[1].start + col_offset
        v_max, u_max = box[0].stop - 1 + row_offset, box[1].stop - 1 + col_offset
        box_height, box_width = v_max - v_min + 1, u_max - u_min + 1
        fill = area / (box_height * box_width)
        clipped = u_min == 0 or v_min == 0 or u_max == width - 1 or v_max == height - 1
        if fill < config.min_fill:
            continue
        if not clipped and max(box_height, box_width) > config.max_aspect * min(box_height, box_width):
            continue
        components.append(
            _Component(
                bounds=(u_min, v_min, u_max, v_max),
                centroid=(float(columns.mean() + u_min), float(rows.mean() + v_min)),
                area=area,
                fill=fill,
                clipped=clipped,
            )
        )
    return components


def _observation(component: _Component, timestamp: float, camera: str,
                 count: int, config: DetectorConfig) -> TargetObservation:
    extent = component.extent
    bounded_extent = min(extent, config.max_uncertainty_extent_px)
    uncertainty = config.base_uncertainty_px + config.extent_uncertainty * bounded_extent
    confidence = component.fill * min(1.0, np.sqrt(component.area) / 5.0)
    if component.clipped:
        uncertainty += config.clipped_extent_uncertainty * bounded_extent
        confidence *= config.clipped_confidence
    return TargetObservation(
        status=DetectionStatus.DETECTED,
        timestamp=timestamp,
        camera=camera,
        bounds=component.bounds,
        reference=component.centroid,
        confidence=float(confidence),
        uncertainty_px=float(uncertainty),
        area=component.area,
        clipped=component.clipped,
        num_candidates=count,
    )


def _select(components: List[_Component], prediction: Optional[TrackPrediction],
            config: DetectorConfig) -> Tuple[DetectionStatus, Optional[_Component]]:
    if prediction is None:
        ranked = sorted(components, key=lambda c: c.score(config), reverse=True)
        if len(ranked) > 1 and ranked[0].score(config) < config.dominance_ratio * ranked[1].score(config):
            return DetectionStatus.AMBIGUOUS, None
        return DetectionStatus.DETECTED, ranked[0]

    gated = []
    for component in components:
        distance = float(np.hypot(component.centroid[0] - prediction.u,
                                  component.centroid[1] - prediction.v))
        if distance <= config.gate_sigmas * prediction.sigma_px + 0.5 * component.extent:
            gated.append((distance, component))
    if not gated:
        return DetectionStatus.NO_DETECTION, None
    gated.sort(key=lambda item: item[0])
    if len(gated) > 1 and gated[1][0] < (config.gate_ambiguity_ratio * gated[0][0]
                                         + config.gate_ambiguity_margin_px):
        return DetectionStatus.AMBIGUOUS, None
    return DetectionStatus.DETECTED, gated[0][1]


def detect_target(image: Optional[np.ndarray], timestamp: float = 0.0, camera: str = "",
                  prediction: Optional[TrackPrediction] = None,
                  config: DetectorConfig = DEFAULT_CONFIG) -> TargetObservation:
    """Detect the red target as one connected component.

    With a prediction, only components inside its gate are candidates, so a
    distractor elsewhere cannot move the reference. Without one, the best
    component must clearly dominate the others or the result is AMBIGUOUS.
    """
    if image is None:
        return TargetObservation(DetectionStatus.NO_FRAME, timestamp, camera)
    components = _components(np.asarray(image), config)
    if not components:
        return TargetObservation(DetectionStatus.NO_DETECTION, timestamp, camera)
    status, chosen = _select(components, prediction, config)
    if chosen is None:
        return TargetObservation(status, timestamp, camera, num_candidates=len(components))
    return _observation(chosen, timestamp, camera, len(components), config)


@dataclass(frozen=True)
class LabeledFrame:
    """One evaluation frame: its image and whether a target is actually present.

    `reference` is the labeled ground-truth (u, v) of the target when `has_target`,
    used to reject a detection that locked onto the wrong thing (still a miss, even
    though `detect_target` reported DETECTED).
    """

    image: np.ndarray
    has_target: bool
    reference: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class DetectionRates:
    """False-positive / missed-detection rates of a held-out labeled set.

    These describe only the labeled set they were computed over (synthetic, by
    construction, unless real recorded frames are labeled and passed in); they are not
    a verified real-world performance claim.
    """

    num_frames: int
    num_positive: int
    num_negative: int
    false_positive_rate: float
    missed_detection_rate: float


def split_tuning_and_held_out(frames: List[LabeledFrame], held_out_fraction: float = 0.5,
                              rng: Optional[np.random.Generator] = None
                              ) -> Tuple[List[LabeledFrame], List[LabeledFrame]]:
    """Shuffle `frames` with `rng` and split into (tuning, held_out), with no overlap.

    Calibration must never be tuned on the frames it is scored against: pass a fixed
    `rng` (e.g. `np.random.default_rng(seed)`) for a reproducible split.
    """
    rng = np.random.default_rng() if rng is None else rng
    order = rng.permutation(len(frames))
    cut = int(round(len(frames) * (1.0 - held_out_fraction)))
    tuning = [frames[index] for index in order[:cut]]
    held_out = [frames[index] for index in order[cut:]]
    return tuning, held_out


def evaluate_detection_rates(frames: List[LabeledFrame], config: DetectorConfig = DEFAULT_CONFIG,
                             match_radius_px: float = 15.0) -> DetectionRates:
    """False-positive and missed-detection rates of bare per-frame `detect_target` calls.

    Each frame is scored independently (no track/prediction), matching a single still
    frame with no temporal context. A positive frame only counts as a hit when a
    reference was labeled and the detection falls within `match_radius_px` of it;
    detecting something elsewhere in the frame is still a miss, not a hit.
    """
    num_positive = num_negative = 0
    false_positives = missed = 0
    for frame in frames:
        obs = detect_target(frame.image, config=config)
        detected = obs.status is DetectionStatus.DETECTED
        if frame.has_target:
            num_positive += 1
            hit = detected
            if hit and frame.reference is not None and obs.reference is not None:
                distance = float(np.hypot(obs.reference[0] - frame.reference[0],
                                          obs.reference[1] - frame.reference[1]))
                hit = distance <= match_radius_px
            if not hit:
                missed += 1
        else:
            num_negative += 1
            if detected:
                false_positives += 1
    return DetectionRates(
        num_frames=len(frames),
        num_positive=num_positive,
        num_negative=num_negative,
        false_positive_rate=0.0 if num_negative == 0 else false_positives / num_negative,
        missed_detection_rate=0.0 if num_positive == 0 else missed / num_positive,
    )


class TargetAssociator:
    """Keeps one image-plane track and gates each new frame with it."""

    def __init__(self, config: DetectorConfig = DEFAULT_CONFIG, min_sigma_px: float = 4.0,
                 growth_px_per_s: float = 20.0, max_sigma_px: float = 200.0):
        self.config = config
        self.min_sigma_px = min_sigma_px
        self.growth_px_per_s = growth_px_per_s
        self.max_sigma_px = max_sigma_px
        self.track: Optional[TrackPrediction] = None
        self._camera = ""
        self._timestamp = 0.0

    def reset(self) -> None:
        self.track = None

    def update(self, image: Optional[np.ndarray], timestamp: float,
               camera: str, allow_reacquire: bool = False) -> TargetObservation:
        """Gate against the existing track (same camera), else start a fresh one.

        `allow_reacquire=True` retries an ungated search when the gate rejects every
        candidate: a legitimate large jump since the last detection (not a same-frame
        distractor, which the gate already preferred over) must not stay lost just
        because the gate has not grown wide enough yet. Default False keeps every other
        caller's existing gate-only behavior unchanged.
        """
        same_camera = self.track is not None and camera == self._camera
        prediction = self.track if same_camera else None
        obs = detect_target(image, timestamp, camera, prediction, self.config)
        if obs.status is DetectionStatus.NO_DETECTION and prediction is not None and allow_reacquire:
            obs = detect_target(image, timestamp, camera, None, self.config)
        if obs.status is DetectionStatus.DETECTED:
            sigma = max(self.min_sigma_px, obs.uncertainty_px)
            self.track = TrackPrediction(obs.reference[0], obs.reference[1], sigma)
            self._camera = camera
        elif same_camera:
            grown = self.track.sigma_px + self.growth_px_per_s * max(0.0, timestamp - self._timestamp)
            self.track = replace(self.track, sigma_px=min(self.max_sigma_px, grown))
        if obs.status is DetectionStatus.DETECTED or same_camera:
            self._timestamp = timestamp
        return obs

