import json

import pytest

from rotorpy.frames import FrameConvention
from rotorpy.world import World


def test_default_frame_and_file_override(tmp_path):
    world = World({"bounds": {"extents": [0, 1, 0, 1, 0, 1]}, "blocks": []})
    assert world.frame is FrameConvention.FLU_ENU
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"bounds": {"extents": [0, 1, 2, 3, 4, 5]}, "blocks": [], "frame": "frd_ned"}))
    loaded = World.from_file(str(path))
    assert loaded.frame is FrameConvention.FRD_NED
    assert list(loaded.world["bounds"]["extents"]) == [0, 1, 2, 3, 4, 5]
    forced = World.from_file(str(path), frame=FrameConvention.FLU_ENU)
    assert forced.frame is FrameConvention.FLU_ENU
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"bounds": {"extents": [0, 1, 0, 1, 0, 1]}, "blocks": [], "frame": "nwu"}))
    with pytest.raises(ValueError):
        World.from_file(str(bad))
    assert World.empty([0, 1, 0, 1, 0, 1], frame=FrameConvention.FRD_NED).frame is FrameConvention.FRD_NED
    assert World.grid_forest(1, 1, 0.2, 1.0, 1.0, frame=FrameConvention.FRD_NED).frame is FrameConvention.FRD_NED
