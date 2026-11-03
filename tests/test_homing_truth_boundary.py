"""Gazebo truth stays on the scoring side: guidance/estimator modules never import it."""

import ast
from pathlib import Path

import pytest

GUIDANCE_DIR = Path(__file__).resolve().parents[1] / "rotorpy" / "guidance"
ROOT = Path(__file__).resolve().parents[1]

TRUTH_MODULE = "rotorpy.guidance.scoring"
GUIDANCE_SIDE = (
    "homing", "tracking", "target_estimator", "command_limits", "pixels", "navigation", "speed",
    "sightline", "camera_geometry", "vehicle_state", "observation_buffer", "step_decision",
    "startup", "run_lifecycle", "setpoint_stream", "watchdog", "origin_alignment",
)


def _module_path(name):
    parts = name.split(".")
    base = ROOT.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _imports(path, package):
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            base = node.module or ""
            names.add(base)
            names.update("{}.{}".format(base, alias.name) for alias in node.names)
    return {name for name in names if name.startswith("rotorpy")}


def reachable(module_name):
    """Every `rotorpy` module transitively imported by `module_name` (by static import)."""
    seen = set()
    stack = [module_name]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_path(name)
        if path is None:
            continue
        stack.extend(_imports(path, name.rpartition(".")[0]))
    return seen


@pytest.mark.parametrize("module", GUIDANCE_SIDE)
def test_guidance_side_module_never_reaches_the_truth_scoring_module(module):
    name = "rotorpy.guidance." + module
    assert _module_path(name) is not None, "{} should exist".format(name)
    assert TRUTH_MODULE not in reachable(name)


def test_the_detector_sees_an_import_when_there_is_one(tmp_path):
    # Negative control: the same walk on a file that does import the truth module.
    module = tmp_path / "leaky.py"
    module.write_text("from rotorpy.guidance.scoring import TruthScorer\n")
    assert TRUTH_MODULE in _imports(module, "")
    module.write_text("from rotorpy.guidance import scoring\n")
    assert TRUTH_MODULE in _imports(module, "")


def test_scoring_itself_is_importable_and_truth_side():
    assert TRUTH_MODULE in reachable(TRUTH_MODULE)
    assert "rotorpy.guidance.homing" not in {
        name for name in _imports(GUIDANCE_DIR / "scoring.py", "rotorpy.guidance")
    }


def _tracker_methods():
    source = (ROOT / "Dockerfiles" / "camera_homing_tracker.py").read_text()
    tree = ast.parse(source)
    methods = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for node in cls.body:
                if isinstance(node, ast.FunctionDef):
                    methods[node.name] = node
    return source, methods


def test_only_the_truth_reader_calls_the_gazebo_truth_services():
    source, methods = _tracker_methods()
    callers = set()
    for name, node in methods.items():
        for call in ast.walk(node):
            if isinstance(call, ast.Attribute) and call.attr in ("_pose_truth", "_target_truth", "_model_state"):
                callers.add(name)
    # `_target_truth` itself reads the model-state service; `_read_truth` is the one
    # truth-side entry point the step uses for scoring and recording.
    assert callers <= {"_read_truth", "_target_truth", "_pose_truth", "_connect"}, callers


def test_the_tracker_reads_gazebo_truth_only_after_guidance_has_its_command():
    source, methods = _tracker_methods()
    body = ast.get_source_segment(source, methods["_step"])
    assert body.index("command_and_stamp(") < body.index("self._read_truth()")
    assert body.index("guidance.update(") < body.index("self._read_truth()")
    compute = body[body.index("def compute_command"):body.index("command, stamp_wall_s")]
    assert "truth" not in compute
    startup = ast.get_source_segment(source, methods["_advance_startup"])
    assert "truth" not in startup  # the startup gate and arming never see Gazebo truth
    # Nothing on the control side of the tracker consumes the truth reader.
    for name in ("_flight_state", "_poll_mavlink", "_publish_setpoint", "_send", "_engage", "_advance_startup"):
        assert "_pose_truth" not in ast.get_source_segment(source, methods[name])
        assert "_read_truth" not in ast.get_source_segment(source, methods[name])
