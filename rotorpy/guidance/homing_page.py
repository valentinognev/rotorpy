"""Plotly page for one recorded camera-homing flight.

The page is a 3D scene beside the drone's camera image. A slider selects a
camera snapshot and moves the drone, its FRD body axes, and the camera
frustum to that pose. The drone is an X: forward arms and their props
are green, rear arms and their props are red, and each prop is a circle a
short distance above the arm tip.
"""

import base64
import io
import json
from pathlib import Path
from typing import Union

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from scipy.spatial.transform import Rotation

from rotorpy.utils.camera_plotter import frustum_ray_directions

_PATH_COLOR = "#1f77b4"
_FRD_FROM_FLU = np.diag([1.0, -1.0, -1.0])
_AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")


def _scenario_extents(position: np.ndarray, world, margin: float = 15.0) -> list:
    """Cube around the flown path and the target box, in simulator meters."""
    block = world.world["blocks"][0]["extents"]
    corners = np.array([
        [block[0], block[2], block[4]],
        [block[1], block[3], block[5]],
    ])
    points = np.vstack([np.asarray(position, dtype=float), corners])
    low = points.min(axis=0) - margin
    high = points.max(axis=0) + margin
    span = float(np.max(high - low))
    center = 0.5 * (low + high)
    half = 0.5 * span
    return list(center - half) + list(center + half)


def _box_segments(extents) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = extents
    corners = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.array([[corners[a], corners[b]] for a, b in edges])


def _segments_xyz(segments: np.ndarray) -> dict:
    xs, ys, zs = [], [], []
    for start, end in segments:
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])
    return {"x": xs, "y": ys, "z": zs}


def _polylines(chains) -> dict:
    """Join point chains with a gap so one scatter trace draws all of them."""
    xs, ys, zs = [], [], []
    for chain in chains:
        for point in chain:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
            zs.append(float(point[2]))
        xs.append(None)
        ys.append(None)
        zs.append(None)
    return {"x": xs, "y": ys, "z": zs}


def _frd_axes(rotation):
    """FRD body axes in the simulator world. Columns are forward, right, down."""
    return rotation @ _FRD_FROM_FLU


def _airframe(position, rotation, arm_length):
    """X in the FRD forward-right plane. Props sit opposite FRD down, on the top."""
    lift = 0.12 * arm_length
    radius = 0.34 * arm_length
    directions = np.array([
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
    ], dtype=float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    frd = _frd_axes(rotation)
    body_x = frd[:, 0]
    body_y = frd[:, 1]
    body_up = -frd[:, 2]
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=True)

    def arms_and_props(subset):
        arm_chains = []
        prop_chains = []
        for direction in subset:
            tip = position + frd @ (direction * arm_length)
            arm_chains.append((position, tip))
            center = tip + body_up * lift
            prop_chains.append([
                center + radius * (np.cos(angle) * body_x + np.sin(angle) * body_y)
                for angle in angles
            ])
        return _polylines(arm_chains), _polylines(prop_chains)

    front, front_props = arms_and_props(directions[:2])
    back, back_props = arms_and_props(directions[2:])
    return front, back, front_props, back_props


def _axis_xyz(origin: np.ndarray, direction: np.ndarray, length: float) -> dict:
    end = origin + direction * length
    return {
        "x": [float(origin[0]), float(end[0])],
        "y": [float(origin[1]), float(end[1])],
        "z": [float(origin[2]), float(end[2])],
    }


def _camera_frustum(position, rotation, camera, frustum_scale):
    """Wireframe of the camera view, in the simulator world."""
    extrinsics = camera.extrinsics
    p_bc = np.asarray(extrinsics["position"], dtype=float)
    body_to_camera = Rotation.from_quat(
        np.asarray(extrinsics["orientation"], dtype=float)
    ).as_matrix()
    ray_dirs = frustum_ray_directions(camera.intrinsics, frustum_scale) @ body_to_camera
    origin = position + rotation @ p_bc
    corners = origin + ray_dirs @ rotation.T
    segments = [[origin, corner] for corner in corners]
    for index in range(4):
        segments.append([corners[index], corners[(index + 1) % 4]])
    return _segments_xyz(np.asarray(segments))


def _jpeg_data_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        buffer, format="JPEG", quality=85
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _snapshots(record):
    rotations = Rotation.from_quat(record.quaternion).as_matrix()
    state_index = np.searchsorted(record.time, record.image_time, side="left")
    low_high = _scenario_extents(record.position, record.world)
    span = float(low_high[3] - low_high[0])
    frustum_scale = 0.1 * span
    axis_length = 0.5 * frustum_scale
    snapshots = []
    for frame, row in enumerate(state_index):
        position = np.asarray(record.position[row], dtype=float)
        rotation = rotations[row]
        frd = _frd_axes(rotation)
        frustum = _camera_frustum(position, rotation, record.camera, frustum_scale)
        front, back, front_props, back_props = _airframe(
            position, rotation, 0.11 * axis_length
        )
        snapshots.append({
            "marker": [float(position[0]), float(position[1]), float(position[2])],
            "body": [
                _axis_xyz(position, frd[:, axis], axis_length) for axis in range(3)
            ],
            "front": front,
            "back": back,
            "front_props": front_props,
            "back_props": back_props,
            "frustum": frustum,
            "image": _jpeg_data_url(record.image[frame]),
        })
    return snapshots, axis_length, low_high


def _line(xyz, color, name, width=4):
    return go.Scatter3d(
        x=xyz["x"], y=xyz["y"], z=xyz["z"],
        mode="lines",
        name=name,
        line={"color": color, "width": width},
        hoverinfo="skip",
        showlegend=False,
    )


def _figure(record, snapshots, low_high):
    first = snapshots[0]
    path = np.asarray(record.position, dtype=float)
    block = record.world.world["blocks"][0]
    color = block.get("color", [1, 0, 0])
    box_color = "rgb({},{},{})".format(
        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=path[:, 2],
        mode="lines",
        name="path",
        line={"color": _PATH_COLOR, "width": 4},
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(_line(_segments_xyz(_box_segments(block["extents"])), box_color, "target", width=6))
    fig.add_trace(go.Scatter3d(
        x=[first["marker"][0]], y=[first["marker"][1]], z=[first["marker"][2]],
        mode="markers",
        name="drone",
        marker={"color": "black", "size": 5},
        hoverinfo="skip",
        showlegend=False,
    ))
    for axis, tint in enumerate(_AXIS_COLORS):
        fig.add_trace(_line(first["body"][axis], tint, "body {}".format(axis), width=3))
    fig.add_trace(_line(first["front"], "#2ca02c", "front", width=4))
    fig.add_trace(_line(first["back"], "#d62728", "back", width=4))
    fig.add_trace(_line(first["front_props"], "#2ca02c", "front props", width=2))
    fig.add_trace(_line(first["back_props"], "#d62728", "back props", width=2))
    fig.add_trace(_line(first["frustum"], "black", "frustum", width=2))
    fig.update_layout(
        autosize=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={
            "aspectmode": "cube",
            "xaxis": {"title": "x", "range": [low_high[0], low_high[3]]},
            "yaxis": {"title": "y", "range": [low_high[1], low_high[4]]},
            "zaxis": {"title": "z", "range": [low_high[2], low_high[5]]},
        },
    )
    return fig


def homing_page(record) -> str:
    """HTML page for `record`, with one slider step per camera frame."""
    snapshots, axis_length, low_high = _snapshots(record)
    figure = _figure(record, snapshots, low_high)
    graph = figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id="homing-scene",
        config={"responsive": True, "displayModeBar": False},
    )
    payload = {
        "times": [float(t) for t in record.image_time],
        "axis_length": float(axis_length),
        "snapshots": snapshots,
    }
    return _html(graph, payload)


def _html(graph: str, payload: dict) -> str:
    data = json.dumps(payload)
    opened = "t = {:.2f} s".format(payload["times"][0])
    last = len(payload["times"]) - 1
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Camera homing</title>
<style>
  html, body { height: 100%; margin: 0; }
  body { display: flex; flex-direction: column; font-family: sans-serif; color: #222; }
  header { display: flex; gap: 16px; align-items: center; padding: 10px 16px; background: #f4f4f4; border-bottom: 1px solid #ddd; }
  #homing-time { min-width: 8em; font-size: 16px; font-weight: 600; font-variant-numeric: tabular-nums; }
  #homing-slider { flex: 1; }
  main { flex: 1; min-height: 0; display: grid; grid-template-columns: 3fr 2fr; }
  #scene-wrap { min-width: 0; min-height: 0; }
  #homing-scene { width: 100% !important; height: 100% !important; }
  #camera-wrap { min-width: 0; min-height: 0; margin: 0; background: #111; display: flex; flex-direction: column; }
  #homing-camera { width: 100%; height: 100%; object-fit: contain; }
  figcaption { padding: 4px 8px; color: #ddd; }
</style>
</head>
<body>
<header>
  <span id="homing-time">""" + opened + """</span>
  <input id="homing-slider" type="range" min="0" max=\"""" + str(last) + """\" value="0" step="1">
</header>
<main>
  <div id="scene-wrap">""" + graph + """</div>
  <figure id="camera-wrap">
    <img id="homing-camera" alt="Camera view">
    <figcaption>Camera view</figcaption>
  </figure>
</main>
<script type="application/json" id="homing-data">""" + data + """</script>
<script>
const data = JSON.parse(document.getElementById("homing-data").textContent);
const slider = document.getElementById("homing-slider");
const photo = document.getElementById("homing-camera");
const readout = document.getElementById("homing-time");
function show(index) {
  const snap = data.snapshots[index];
  readout.textContent = "t = " + data.times[index].toFixed(2) + " s";
  photo.src = snap.image;
  Plotly.restyle("homing-scene", {
    x: [[snap.marker[0]], snap.body[0].x, snap.body[1].x, snap.body[2].x,
        snap.front.x, snap.back.x, snap.front_props.x, snap.back_props.x,
        snap.frustum.x],
    y: [[snap.marker[1]], snap.body[0].y, snap.body[1].y, snap.body[2].y,
        snap.front.y, snap.back.y, snap.front_props.y, snap.back_props.y,
        snap.frustum.y],
    z: [[snap.marker[2]], snap.body[0].z, snap.body[1].z, snap.body[2].z,
        snap.front.z, snap.back.z, snap.front_props.z, snap.back_props.z,
        snap.frustum.z]
  }, [2, 3, 4, 5, 6, 7, 8, 9, 10]);
}
slider.addEventListener("input", () => show(Number(slider.value)));
function fit() {
  const wrap = document.getElementById("scene-wrap");
  Plotly.relayout("homing-scene", {height: wrap.clientHeight, width: wrap.clientWidth});
}
window.addEventListener("resize", fit);
show(0);
fit();
</script>
</body>
</html>
"""


def write_homing_page(record, path: Union[str, Path]) -> Path:
    """Write the homing page for `record` to `path` and return that path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(homing_page(record), encoding="utf-8")
    return destination
