import os

import numpy as np
import pytest

import rotorpy.guidance.gazebo_record as gazebo_record
from rotorpy.guidance.gazebo_record import GazeboHomingRecorder, gazebo_page_path
from rotorpy.guidance.homing import HomingCommand
from rotorpy.guidance.homing_config import load_homing_config


def _config():
    return load_homing_config(os.path.join(
        os.path.dirname(__file__), "..", "examples", "camera_homing.jsonc"
    ))


def _command():
    return HomingCommand(
        np.array([1.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
        0.0,
        True,
        True,
        "forward",
        np.array([10.0, 12.0]),
        4.0,
    )


def test_page_path_ends_with_gazebo():
    path = gazebo_page_path("/opt/rotorpy/examples/camera_homing.jsonc")
    assert path.name == "camera_homing_gazebo.html"
    assert path.parent.name == "Results"


def test_recorder_writes_a_homing_page(tmp_path):
    recorder = GazeboHomingRecorder(_config())
    image = np.zeros((16, 16, 3), dtype=np.float32)
    image[:, :, 0] = 1.0
    status = recorder.observe(1.0, [50.0, 100.0, 30.0], [0, 0, 0, 1], image, image, _command())
    assert status is None
    assert np.allclose(recorder.positions[0], [100.0, 50.0, 30.0])
    destination = tmp_path / "camera_homing_gazebo.html"
    written = recorder.write(destination)
    assert written == destination
    text = destination.read_text()
    assert 'id="homing-slider"' in text
    assert recorder.write(destination) == destination


def test_recorder_can_retry_after_write_failure(tmp_path, monkeypatch):
    recorder = GazeboHomingRecorder(_config())
    image = np.zeros((16, 16, 3), dtype=np.float32)
    recorder.observe(1.0, [0.0, 0.0, 30.0], [0, 0, 0, 1], image, image, _command())
    destination = tmp_path / "flight.html"

    def fail_write(record, path):
        raise OSError("disk error")

    monkeypatch.setattr(gazebo_record, "write_homing_page", fail_write)
    with pytest.raises(OSError, match="disk error"):
        recorder.write(destination)
    monkeypatch.setattr(gazebo_record, "write_homing_page", lambda record, path: path)
    assert recorder.write(destination) == destination


@pytest.mark.parametrize(
    ("stamp", "height", "expected"),
    [(29.99, 30.0, None), (30.0, 30.0, "timeout"), (11.0, 0.25, "ground")],
)
def test_recorder_reports_ground_or_timeout(stamp, height, expected):
    recorder = GazeboHomingRecorder(_config())
    recorder.observe(10.0, [0.0, 0.0, 30.0], [0, 0, 0, 1], None, None, _command())
    status = recorder.observe(stamp, [0.0, 0.0, height], [0, 0, 0, 1], None, None, _command())
    assert status == expected


def test_recorder_builds_replay_geometry_from_the_authoritative_config():
    config = _config()
    recorder = GazeboHomingRecorder(config)
    for t in (0.0, 0.1):
        recorder.observe(
            t, np.array([0.0, 0.0, 30.0]), np.array([0.0, 0.0, 0.0, 1.0]),
            np.zeros((4, 4, 3)), np.zeros((4, 4, 3)), _command(),
            target_ground_ned=np.array([100.0, 50.0, 0.0]) + np.array([0.0, 2.0 * t, 0.0]),
        )
    record = recorder.record()
    geometry = record.geometry
    assert geometry.target_radius_m == config["gazebo_target_radius_m"]
    assert geometry.target_height_m == config["gazebo_target_height_m"]
    # NED (100, 50, 0) -> simulator north, east, up.
    assert geometry.target_ground_sim == pytest.approx([100.0, 50.0, 0.0])
    assert geometry.target_path_sim[-1] == pytest.approx([100.0, 50.2, 0.0])
    forward = geometry.cameras["forward"]
    assert forward["position_flu_m"] == list(config["forward_camera_position_flu_m"])
    assert forward["pitch_deg"] == config["forward_pitch_deg"]
    assert forward["fov_x_deg"] == config["fov_x_deg"]
    assert geometry.cameras["upward"]["pitch_deg"] == config["upward_pitch_deg"]


def test_recorder_without_target_samples_uses_the_configured_static_target():
    config = _config()
    recorder = GazeboHomingRecorder(config)
    recorder.observe(
        0.0, np.array([0.0, 0.0, 30.0]), np.array([0.0, 0.0, 0.0, 1.0]),
        np.zeros((4, 4, 3)), np.zeros((4, 4, 3)), _command(),
    )
    geometry = recorder.record().geometry
    assert geometry.target_ground_sim == pytest.approx([100.0, 50.0, 0.0])
    assert geometry.target_path_sim is None
