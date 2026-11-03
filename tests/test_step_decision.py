"""The tracker's command/setpoint-stamp decision, extracted as a pure helper.

Covers the bug where `step()` restamped the setpoint stream's held command with the
current wall time on every tick, including the hold-last-command branch (no fresh image,
no PX4 estimate): the setpoint expiry then never fires, so a dead camera or dead
telemetry never stops the stream.
"""

import pytest

from rotorpy.guidance.step_decision import command_and_stamp


def test_a_fresh_estimate_and_image_computes_a_new_command_and_stamps_its_wall_time():
    calls = []

    def compute():
        calls.append(1)
        return "new-command"

    command, stamp = command_and_stamp(
        have_estimate=True, fresh=True, previous_command="old-command",
        placeholder_command="placeholder", compute_command=compute, observation_wall_s=5.0,
    )
    assert command == "new-command"
    assert stamp == 5.0
    assert calls == [1]


def test_holding_last_command_without_a_fresh_image_does_not_restamp():
    # No fresh camera frame this tick (dead or merely slow camera), but a PX4 estimate is
    # available and a command already exists: hold it, and do NOT restamp -- the
    # previously stamped time must keep aging so the stream can expire if the camera
    # never produces another frame.
    command, stamp = command_and_stamp(
        have_estimate=True, fresh=False, previous_command="old-command",
        placeholder_command="placeholder", compute_command=lambda: pytest.fail("must not compute"),
        observation_wall_s=5.0,
    )
    assert command == "old-command"
    assert stamp is None


def test_holding_last_command_without_a_px4_estimate_does_not_restamp():
    # Dead PX4 telemetry: no estimate, even though the camera is fresh. Hold the last
    # command and do not restamp it either.
    command, stamp = command_and_stamp(
        have_estimate=False, fresh=True, previous_command="old-command",
        placeholder_command="placeholder", compute_command=lambda: pytest.fail("must not compute"),
        observation_wall_s=5.0,
    )
    assert command == "old-command"
    assert stamp is None


def test_first_ever_tick_without_an_estimate_seeds_the_placeholder_and_stamps_it():
    command, stamp = command_and_stamp(
        have_estimate=False, fresh=False, previous_command=None,
        placeholder_command="placeholder", compute_command=lambda: pytest.fail("must not compute"),
        observation_wall_s=5.0,
    )
    assert command == "placeholder"
    assert stamp == 5.0


def test_first_px4_estimate_before_any_image_still_computes_the_initial_course():
    # An estimate arriving before the first camera frame must still let guidance adopt its
    # initial course from the vehicle's own heading (same as the original behaviour),
    # rather than being stuck on the placeholder until a camera frame eventually arrives.
    calls = []

    def compute():
        calls.append(1)
        return "initial-course"

    command, stamp = command_and_stamp(
        have_estimate=True, fresh=False, previous_command=None,
        placeholder_command="placeholder", compute_command=compute, observation_wall_s=5.0,
    )
    assert command == "initial-course"
    assert stamp == 5.0
    assert calls == [1]
