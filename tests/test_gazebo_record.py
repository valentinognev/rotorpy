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
