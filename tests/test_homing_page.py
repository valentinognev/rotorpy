import base64
import io
import json
import re

import numpy as np
import pytest
from PIL import Image

from rotorpy.guidance.camera_geometry import pitched_extrinsics
from rotorpy.guidance.homing_page import homing_page, write_homing_page
from rotorpy.guidance.runner import HomingRecord
from rotorpy.sensors.camera import PinholeCamera
from rotorpy.world import World


def _camera(extrinsics):
    return PinholeCamera(
        intrinsics={
            "fx": 16.0, "fy": 16.0, "width": 32, "height": 32,
            "cx": 16.0, "cy": 16.0, "dist_coeffs": np.zeros(5),
        },
        extrinsics=extrinsics,
        splat_radius=1,
        feature_output="rgb",
    )


def _record():
    world = World({
        "bounds": {"extents": [-5, 5, -5, 5, -5, 5]},
        "blocks": [{"extents": [1, 2, 3, 4, 0, 1], "color": [1, 0, 0]}],
    })
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    blue = np.zeros((8, 8, 3), dtype=np.uint8)
    blue[:, :, 2] = 255
    return HomingRecord(
        time=np.array([0.0, 0.01, 0.05]),
        position=np.array([[1.0, 2.0, 3.0], [1.1, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        quaternion=np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
        wind=np.zeros((3, 3)),
        image_time=np.array([0.0, 0.05]),
        forward_image=np.stack([red, blue]),
        upward_image=np.stack([blue, red]),
        source=["forward", None],
        tracker_center=np.array([[4.0, 4.0], [np.nan, np.nan]]),
        tracker_side=np.array([2.0, np.nan]),
        world=world,
        forward_camera=_camera({
            "position": np.zeros(3),
            "orientation": np.array([0.0, 0.0, 0.0, 1.0]),
        }),
        upward_camera=_camera(pitched_extrinsics(-70)),
        preview_step=1,
    )


def _jpeg(data_url):
    encoded = data_url.split(",", 1)[1]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))


def _payload(html):
    match = re.search(
        r'<script type="application/json" id="homing-data">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_slider_snapshots_follow_the_camera_frames():
    page = homing_page(_record())
    payload = _payload(page)
    assert "Plotly.newPlot" in page or "plotly" in page.lower()
    assert 'id="homing-slider"' in page
    assert payload["times"] == [0.0, 0.05]
    assert payload["snapshots"][0]["marker"] == pytest.approx([1.0, 2.0, 3.0])
    assert payload["snapshots"][1]["marker"] == pytest.approx([4.0, 5.0, 6.0])
    length = payload["axis_length"]
    body_x = payload["snapshots"][0]["body"][0]
    body_y = payload["snapshots"][0]["body"][1]
    body_z = payload["snapshots"][0]["body"][2]
    assert body_x["x"] == pytest.approx([1.0, 1.0 + length])
    assert body_x["y"] == pytest.approx([2.0, 2.0])
    assert body_x["z"] == pytest.approx([3.0, 3.0])
    assert body_y["y"] == pytest.approx([2.0, 2.0 - length])
    assert body_z["z"] == pytest.approx([3.0, 3.0 - length])
    assert payload["snapshots"][1]["body"][0]["x"][0] == pytest.approx(4.0)
    assert payload["snapshots"][0]["square"] == {"u": 4.0, "v": 4.0, "side": 2.0}
    assert payload["snapshots"][1]["square"] is None
    assert payload["snapshots"][0]["source"] == "forward"
    assert payload["snapshots"][1]["source"] is None
    assert payload["snapshots"][0]["frustum"]["x"][0] == pytest.approx(1.0)
    assert payload["snapshots"][1]["frustum"]["x"][0] == pytest.approx(4.0)
    assert payload["snapshots"][0]["upward_frustum"]["x"][0] == pytest.approx(1.0)
    assert payload["snapshots"][1]["upward_frustum"]["x"][0] == pytest.approx(4.0)
    assert payload["snapshots"][0]["upward_frustum"]["z"] != pytest.approx(
        payload["snapshots"][0]["frustum"]["z"]
    )
    assert "triad" not in payload["snapshots"][0]
    assert "[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]" in page
    assert page.index('id="forward-camera"') < page.index('id="upward-camera"')
    assert "#ffd23f" in page
    assert "#7ec8ff" in page
    forward_red = _jpeg(payload["snapshots"][0]["forward_image"])
    forward_blue = _jpeg(payload["snapshots"][1]["forward_image"])
    upward_blue = _jpeg(payload["snapshots"][0]["upward_image"])
    upward_red = _jpeg(payload["snapshots"][1]["upward_image"])
    assert forward_red[:, :, 0].mean() > 200
    assert forward_blue[:, :, 2].mean() > 200
    assert upward_blue[:, :, 2].mean() > 200
    assert upward_red[:, :, 0].mean() > 200


def _pieces(line):
    groups = []
    current = []
    for x, y, z in zip(line["x"], line["y"], line["z"]):
        if x is None:
            if current:
                groups.append(current)
                current = []
        else:
            current.append((x, y, z))
    if current:
        groups.append(current)
    return groups


def test_drone_is_an_x_with_props_above_the_arm_tips():
    page = homing_page(_record())
    page_data = _payload(page)
    snap = page_data["snapshots"][0]
    front = _pieces(snap["front"])
    back = _pieces(snap["back"])
    assert len(front) == 2 and len(back) == 2
    front_tips = []
    for arm in front:
        assert arm[0] == pytest.approx((1.0, 2.0, 3.0))
        assert arm[1][0] > 1.0
        assert arm[1][2] == pytest.approx(3.0)
        assert np.linalg.norm(np.subtract(arm[1], arm[0])) == pytest.approx(0.11 * page_data["axis_length"])
        front_tips.append(arm[1])
    assert front_tips[0][1] != pytest.approx(front_tips[1][1])
    for arm in back:
        assert arm[0] == pytest.approx((1.0, 2.0, 3.0))
        assert arm[1][0] < 1.0
        assert arm[1][2] == pytest.approx(3.0)
    assert "snap.front" in page and "snap.back" in page
    for tip, circle in zip(front_tips, _pieces(snap["front_props"])):
        center = np.mean(circle[:-1], axis=0)
        assert center[0] == pytest.approx(tip[0])
        assert center[1] == pytest.approx(tip[1])
        assert center[2] > tip[2]
        assert center[2] - tip[2] < 0.3 * abs(tip[0] - 1.0) * np.sqrt(2)


def test_page_writes_html(tmp_path):
    path = tmp_path / "camera_homing.html"
    written = write_homing_page(_record(), path)
    assert written == path
    text = path.read_text()
    assert _payload(text)["times"] == [0.0, 0.05]
    assert "Results" not in str(path)
