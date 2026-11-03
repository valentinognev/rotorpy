# Gazebo Homing Reliability Update

## Status and scope

Implementation plan, not an implemented or validated solution.

Read the root [README](../../../README.md) and [UPDATES](../../../UPDATES.md)
before implementation. Preserve unrelated worktree changes.

The update is a closed-loop redesign rather than a gain-tuning exercise.
It covers stationary and moving targets in Gazebo Classic with PX4 SITL.

### Required behavior

- Approach the target at the maximum speed the vehicle can reliably sustain.
- Do not introduce a planned terminal hover or stopping maneuver.
- Observe the target through cameras and obtain vehicle state from PX4.
- Use Gazebo ground truth only for independent validation and scoring.
- End the simulation at ground contact, timeout, or an explicit failure.
- Distinguish target reached from simulation ended: ground contact alone is
  not successful homing.

### Decisions still to commission

- Define the supported moving-target speed and acceleration envelope.
  The initial proposal is bounded ground motion; unrestricted 3D motion is
  not assumed.
- Define the target reference point and geometric success criterion,
  including how target contact and ground-projected miss are reported.
- Establish achievable vehicle limits through PX4 parameter inspection and
  measured command-response tests.
- Establish observation-age, uncertainty, and failure-response thresholds.

These decisions must be recorded before the corresponding acceptance tests
are finalized. Proposed thresholds are not verified performance claims.

## Evidence and baseline failures

The existing `Results/camera_homing_gazebo.html` recording showed:

- Forward-camera steering initially.
- An upward-camera transition at approximately 2.14 seconds.
- No further steering after approximately 2.30 seconds, despite later
  forward-camera target visibility.
- Closest recorded horizontal approach of approximately 18.3 m.
- Timeout at approximately 20 seconds.
- Median recorded image-sample interval of approximately 0.122 seconds,
  compared with the configured 0.05 seconds. Instrumentation is needed to
  attribute this discrepancy; recording intervals alone do not identify
  camera or processing latency.

Code inspection and focused checks identified:

1. **Permanent loss after upward-camera handoff.** Once upward visibility
   latches, a subsequent missing target permanently disables steering.
   The existing test explicitly expects this behavior.
2. **Contaminated detection.** The red-pixel predicate accepts weak channel
   differences, and the target square bounds all matching pixels. The
   opening recorded square spans almost the entire preview although the
   target is small.
3. **Off-axis frame inconsistency.** For a north-facing vehicle, an
   image-right observation should produce an eastward bearing. The current
   Gazebo-to-homing path produces a westward bearing. A synthetic check
   using a pixel 223.6 pixels right of the principal point produced a
   22.62-degree bearing discrepancy; a centered observation passed.
4. **Unused PN acceleration in Gazebo.** MAVLink setpoints mask out all
   acceleration fields. Changing the navigation constant does not change
   the transmitted velocity-pursuit request.
5. **Direction-dependent speed limits.** The current speed law maximizes
   individual velocity components rather than enforcing a measured
   total-speed envelope.
6. **Coupled processing and transmission.** Images, state queries, guidance,
   recording, and setpoint sending share the tracker execution path.
7. **Incomplete outcome verification.** Ground contact and timeout do not
   establish that the target was reached.

Eleven focused existing tests passed during diagnosis. They confirmed
current behavior, not end-to-end homing reliability. No new Gazebo flight
was run during planning.

## Design constraints

- Use standard NED world and FRD body coordinates in Gazebo guidance.
- Reuse existing helpers in `rotorpy/frames.py`; keep legacy RotorPy
  conventions behind explicit adapters.
- Fix geometry, perception, and stale-data handling before tuning guidance.
- Preserve the Python example through regression tests and explicit
  compatibility boundaries.
- Do not feed target coordinates into guidance, including indirectly through
  initial-course generation.
- Keep truth-based scoring separate from the estimator and controller.
- Bound queues, input ages, retries, and failure transitions.
- Never keep a stale velocity command valid indefinitely.
- Do not simply enable acceleration feedforward: project history records
  instability in that path.
- Treat reduced speed for maneuverability, observation quality, or failure
  response separately from a terminal braking maneuver.

## Phase 1: Reproducible baseline and diagnostics

### Work

- Preserve the failed recording and its available scenario settings.
- Assign a unique run ID and avoid overwriting the only diagnostic record.
- Save machine-readable telemetry alongside HTML replay:
  - Image capture, receipt, and processing timestamps.
  - PX4 attitude, position, velocity, and telemetry ages.
  - Candidate components, selected target, confidence, and camera.
  - Target motion estimate and uncertainty.
  - Requested, limited, and transmitted setpoints.
  - Measured vehicle response and limiting reasons.
  - State transitions, PX4 mode, and failure reasons.
- Save original camera frames around acquisition, handoff, and loss.
  JPEG previews are not sufficient for detector calibration.
- Record firmware identity, effective PX4 parameters, configuration,
  rendering mode, and simulation timing.
- Define distinct outcomes for target reached, ground miss, timeout, target
  lost, estimator failure, offboard failure, and simulator failure.
- Provide recorded-input replay for perception and guidance.

### Acceptance

- Recorded inputs reproduce perception and guidance decisions without
  launching Gazebo.
- Failed runs retain sufficient evidence to locate the failing stage.

## Phase 2: Geometry and consistent configuration

Primary surfaces:

- `rotorpy/frames.py`
- `rotorpy/guidance/gazebo_bridge.py`
- `rotorpy/guidance/camera_geometry.py`
- `rotorpy/guidance/homing_config.py`
- `gazeboModels/hummingbird/hummingbird.sdf.jinja`
- `Dockerfiles/sitl_hummingbird_run.sh`

### Work

- Correct the off-axis camera-ray conversion using explicit coordinate
  transforms.
- Include each camera's translation and orientation. The forward camera is
  0.36 m ahead of the body and 0.05 m above it.
- Read or validate intrinsics against ROS `CameraInfo`.
- Define the tracked target reference and its relationship to the visible
  component. A cylinder center is not its ground contact point.
- Align PX4 local coordinates, altitude reference, and the configured ground
  plane without assuming Gazebo world coordinates equal estimator
  coordinates.
- Consolidate duplicated scene and camera settings, or generate/validate
  dependent model and launch values from one authoritative configuration.
- Make spawned target geometry and replay geometry agree.
- Remove target-position-based initial-course leakage into guidance.

### Tests

- Project and reconstruct known points through both cameras.
- Cover left/right, above/below center, arbitrary attitudes, and camera
  translation.
- Verify estimator-origin and simulator-origin alignment.
- Validate analytic projections against rendered Gazebo images.
- Retain legacy RotorPy behavior at the intended compatibility boundary.

### Acceptance

- Analytic and rendered-image geometry checks pass.
- Tests expose the original lateral sign error; quaternion round trips alone
  are insufficient.

## Phase 3: Timestamp-aware processing and independent control streaming

Primary surface: `Dockerfiles/camera_homing_tracker.py`.

### Work

Separate responsibilities into:

1. Lightweight image and PX4 telemetry receivers.
2. Bounded latest-observation processing.
3. Timestamp-aware state estimation and guidance.
4. Independent periodic MAVLink setpoint publishing.
5. Recording that cannot block control.

Additional requirements:

- Preserve image timestamps and use vehicle attitude at capture time.
- Handle each camera independently. A fresh image from one camera must not
  make the other camera's cached image appear fresh.
- Handle supported encodings and row stride explicitly.
- Drop obsolete queued work instead of accumulating latency.
- Establish clock alignment and distinguish simulation time from wall time.
- Handle simulation pauses, resets, and timestamp discontinuities.
- Use appropriate wall-time watchdogs for communications.
- Invalidate expired state and commands even if the last heartbeat reported
  armed/offboard mode.
- Measure end-to-end observation age and setpoint publication intervals.

### Acceptance

- Slow perception or recording does not interrupt setpoint streaming.
- Stale observations and telemetry cannot remain valid indefinitely.
- Input age, queue behavior, and clock discontinuities are testable.

## Phase 4: Robust detection and target association

Primary surface: `rotorpy/guidance/pixels.py`.

### Work

- Require meaningful saturation/red dominance.
- Detect connected components rather than bounding all red-like pixels.
- Reject implausible components using area, shape, clipping, and temporal
  consistency.
- Associate detections with the existing track using predicted image
  location and uncertainty.
- Return structured observations containing timestamp, camera, bounds,
  reference point, confidence, and measurement uncertainty.
- Distinguish no frame, no detection, and ambiguous detection.
- Preserve small distant targets; avoid filtering that removes them.
- Calibrate on raw frames with lighting changes, textured backgrounds,
  shadows, red distractors, and varying target distances.
- Keep held-out recordings separate from tuning inputs.

### Acceptance

- The opening failure frame selects the cylinder instead of an image-wide
  region.
- Report false-positive and missed-detection rates on held-out frames.
- Distractors cannot silently move the target reference across the image.

## Phase 5: Recoverable tracking state machine

Primary surface: `rotorpy/guidance/homing.py`.

### Work

Use explicit states:

`STARTUP -> ACQUIRE -> TRACK -> TEMPORARY_LOSS -> REACQUIRE`

with separate `COMPLETE` and `ABORT` outcomes.

- Require sustained valid observations for acquisition.
- Select cameras using confidence and visibility, with switching hysteresis.
- Maintain one target track across camera changes.
- Fall back to forward observations when upward visibility disappears.
- Permit bounded prediction-only gaps based on age and uncertainty.
- Transition to an explicit failure response after prolonged loss.
- Reset derivative history when observations are discontinuous.
- Do not accumulate permanent loss during startup.
- Replace the test that expects permanent upward loss with a regression
  proving recovery.

### Acceptance

- The recorded early handoff no longer permanently disables guidance.
- Camera switching does not create artificial derivative spikes.
- Prolonged loss does not leave the vehicle in normal stale-command homing.

## Phase 6: Target-motion estimation

Primary surfaces: guidance observation/estimation logic and
`rotorpy/guidance/navigation.py`.

### Work

- Begin with a timestamped constant-velocity target estimator with explicit
  uncertainty and outlier rejection.
- Separate drone-induced image motion from actual target motion.
- Estimate range only where the observation geometry supports it.
- For ground targets, use a valid ground-plane and target-reference model;
  do not interpret a cylinder-center ray's ground intersection as the
  cylinder's position.
- Reject ill-conditioned estimates near the horizon or under severe
  clipping.
- Predict observations to control time to account for processing delay.
- Estimate relative velocity and closing speed rather than assuming a
  stationary target.
- Keep Gazebo truth inaccessible to the estimator.
- Establish the bounded target-motion envelope before acceptance testing.

### Acceptance

- A stationary target does not acquire false motion during drone maneuvers.
- Moving-target errors are measured against independent truth.
- Unobservable or uncertain estimates are identified rather than treated as
  exact positions.

## Phase 7: Maximum-feasible-speed guidance and PX4 integration

Primary surfaces:

- `rotorpy/guidance/speed.py`
- `rotorpy/guidance/navigation.py`
- `rotorpy/guidance/homing.py`
- `Dockerfiles/camera_homing_tracker.py`
- `Dockerfiles/airframes/1100_gazebo-classic_hummingbird`

### Work

- Inspect actual firmware and effective speed, acceleration, tilt, thrust,
  and vertical-rate limits.
- Characterize velocity tracking and turning with progressively stronger
  commands.
- Define a total-speed objective inside the verified vehicle envelope.
- Apply acceleration, jerk, yaw-rate, and command-continuity constraints.
- Account for visibility constraints needed to keep observations valid.
- Use estimated target motion rather than pursuing delayed pixel positions.
- Do not add a terminal hover or distance-based stopping phase.
- Reduce speed only when required by maneuverability, estimation quality,
  or failure handling.

### PN integration

1. Establish a stable velocity-controlled baseline.
2. Integrate PN into one coherent command-generation path.
3. Ensure velocity and acceleration requests describe the same desired
   motion; avoid double-counted corrections.
4. Validate PX4 feedforward semantics and frame conventions before enabling
   acceleration fields.
5. Verify that changing the navigation constant changes the intended
   command and measured behavior.

### Acceptance

- Increased requested speed gives measured improvement without
  unacceptable saturation, visibility loss, or miss growth.
- Commands remain finite, continuous, and within commissioned limits.
- PN behavior is observable if PN is enabled.
- No fix consists solely of removing the acceleration-ignore bits.

## Phase 8: Startup, outcomes, shutdown, and replay

Primary surfaces:

- `Dockerfiles/camera_homing_tracker.py`
- `Dockerfiles/sitl_hummingbird_run.sh`
- `rotorpy/guidance/gazebo_record.py`
- `rotorpy/guidance/homing_page.py`

### Work

- Require fresh estimator data, valid attitude, stable horizontal and
  vertical motion, and confirmed offboard state before homing.
- Bound service waits, arming attempts, motor startup, and stabilization.
- Replace broad catch-and-continue handling with explicit recoverable and
  fatal outcomes.
- Prevent normal stale-command transmission after fatal failures.
- Evaluate normal takeoff as the preferred initialization path. Keep the
  gravity-disabled airborne spawn only as an explicitly tested simulation
  mode if still needed.
- Detect target contact separately from ground contact.
- Score moving-target proximity at matching simulation timestamps.
- Use collision events or sufficiently fine sampling so high-speed contact
  cannot disappear between recording samples.
- Make termination independent of recorder presence and camera availability.
- Flush diagnostic output before orderly shutdown, including failure paths.
- Show actual target and camera geometry in replay.

### Acceptance

- Every run has an unambiguous outcome, including failed startup.
- Ground contact is not automatically scored as target reached.
- Recording failure cannot disable mission termination or block control.

## Phase 9: Progressive validation and reliability campaign

| Stage | Coverage |
| --- | --- |
| Unit tests | Frames, projection, decoding, detection, state transitions, estimation, command bounds |
| Recorded-input replay | Original failure, distractors, handoff, loss, stale and out-of-order observations |
| PX4 integration | Setpoint masks, telemetry freshness, tracking, saturation, offboard loss |
| Stationary-target Gazebo | Headings, offsets, heights, speeds, lighting |
| Moving-target Gazebo | Crossing, approaching, receding, turns, acceleration, temporary occlusion |
| Fault injection | Camera dropout, processing stalls, stale telemetry, pause/reset, recording failure |
| Repeated trials | Fixed reproducible seeds and held-out scenarios |

### Proposed release gates

- No permanent loss after recoverable camera dropout.
- No normal homing on expired observations.
- No target-truth dependency in guidance.
- No unexplained offboard exits or non-finite commands.
- Accuracy assessed against actual target/vehicle geometry.
- Report speed, tracking error, visibility loss, latency, saturation, and
  success rate together.
- Run at least 100 nominal trials plus separate stress trials.
- Report confidence intervals and failure categories; do not equate a small
  successful batch with proven reliability.
- Freeze and document only the operating envelope supported by results.

## Dependencies and delivery order

Recommended implementation sequence:

1. Diagnostics and baseline preservation.
2. Geometry and authoritative configuration.
3. Timing and PX4 vehicle-state integration.
4. Perception and target association.
5. Recoverable camera/track state machine.
6. Target-motion estimation.
7. Speed-envelope characterization and coherent guidance integration.
8. Startup, outcome, and shutdown hardening.
9. Stationary and moving-target reliability campaign.

Tests and diagnostics accompany each phase rather than being deferred to the
end. Guidance tuning depends on corrected geometry, valid observations,
timestamp handling, and a measured vehicle envelope.

## Documentation and completion

- Preserve intended Python-example behavior with adapters and regression
  tests; explicitly document intentional compatibility changes.
- Update root `README.md` when architecture changes.
- Add a versioned entry to root `UPDATES.md` after meaningful implementation
  changes, using the project's versioning convention.
- Document the commissioned target-motion envelope, vehicle envelope,
  failure behavior, configuration, and reproducible validation commands.
- Do not mark this plan complete based on one visually convincing flight.

Completion means a persistent, reproducible implementation with a measured
operating envelope and independently verified outcomes.
