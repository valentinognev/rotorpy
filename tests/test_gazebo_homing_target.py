"""The Gazebo homing target has matching geometry and stands on the ground."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def test_homing_target_geometry_and_spawn_height():
    model = ET.parse(ROOT / "gazeboModels" / "homing" / "target_cylinder.sdf")
    for kind in ("visual", "collision"):
        cylinder = model.find(f"./model/link/{kind}/geometry/cylinder")
        assert cylinder is not None
        assert float(cylinder.findtext("radius")) == 0.75
        assert float(cylinder.findtext("length")) == 3.0

    launch = (ROOT / "Dockerfiles" / "sitl_hummingbird_run.sh").read_text()
    spawn = re.search(r"spawn_sdf[^\n]*--model-name=homing_target[^\n]*-z\s+([\d.]+)", launch)
    assert spawn is not None
    assert float(spawn.group(1)) == 1.5
