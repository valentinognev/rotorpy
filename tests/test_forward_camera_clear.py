"""The Gazebo forward camera must not see the Hummingbird airframe."""

import os
import xml.etree.ElementTree as ET

import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), "..", "gazeboModels", "hummingbird")
_SDF = os.path.join(_ROOT, "hummingbird.sdf.jinja")
_HALF_FOV = np.deg2rad(45.0)
_NEAR = 0.05


def _positions(path):
    root = ET.parse(path).getroot()
    points = []
    for element in root.iter():
        if element.tag.endswith("float_array") and (element.get("id") or "").endswith("positions-array"):
            values = np.fromstring(element.text, sep=" ")
            points.append(values.reshape(-1, 3))
    return np.vstack(points)


def _forward_pose():
    text = open(_SDF).read()
    start = text.index("<sensor name='forward_camera'")
    pose = text[start:text.index("</sensor>", start)]
    numbers = pose[pose.index("<pose>") + 6:pose.index("</pose>")].split()
    return np.array(numbers[:3], dtype=float), float(numbers[4])


def _in_picture(points, origin, pitch):
    """True when a point is inside the 90° pinhole view, looking 10° down."""
    look = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    up = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    right = np.cross(look, up)
    delta = np.asarray(points, dtype=float) - origin
    depth = delta @ look
    ahead = depth > _NEAR
    limit = depth * np.tan(_HALF_FOV)
    lateral = np.abs(delta @ right) <= limit
    vertical = np.abs(delta @ up) <= limit
    return bool(np.any(ahead & lateral & vertical))


def _y_up_to_z_up(points):
    """COLLADA Y-up to Gazebo Z-up: x stays, y becomes z, z becomes -y."""
    transformed = np.empty_like(points)
    transformed[:, 0] = points[:, 0]
    transformed[:, 1] = points[:, 2]
    transformed[:, 2] = -points[:, 1]
    return transformed


def _spun(points, hub):
    samples = []
    for turn in range(8):
        angle = turn * np.pi / 4.0
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated = np.column_stack([
            cosine * points[:, 0] - sine * points[:, 1],
            sine * points[:, 0] + cosine * points[:, 1],
            points[:, 2],
        ])
        samples.append(rotated + hub)
    return np.vstack(samples)


def test_forward_camera_excludes_the_airframe():
    origin, pitch = _forward_pose()
    assert origin[0] > 0.3
    body = _positions(os.path.join(_ROOT, "meshes", "hummingbird.dae"))
    assert not _in_picture(body, origin, pitch)
    assert not _in_picture(_y_up_to_z_up(body), origin, pitch)

    hubs = np.array([
        [0.17, 0.0, 0.01],
        [0.0, 0.17, 0.01],
        [-0.17, 0.0, 0.01],
        [0.0, -0.17, 0.01],
    ])
    for name in ("propeller_cw.dae", "propeller_ccw.dae"):
        mesh = 0.1 * _positions(os.path.join(_ROOT, "meshes", name))
        for frame in (mesh, _y_up_to_z_up(mesh)):
            for hub in hubs:
                assert not _in_picture(_spun(frame, hub), origin, pitch)
