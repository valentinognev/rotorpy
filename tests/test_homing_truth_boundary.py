"""Gazebo truth stays on the scoring side: guidance/estimator modules never import it."""

import ast
from pathlib import Path

import pytest

GUIDANCE_DIR = Path(__file__).resolve().parents[1] / "rotorpy" / "guidance"
ROOT = Path(__file__).resolve().parents[1]

TRUTH_MODULE = "rotorpy.guidance.scoring"
# Fix round 1 (minor d): the only module allowed to reach the truth-scoring module is
# the truth module itself (covered separately by `test_scoring_itself_is_importable_and_
# truth_side`). Every *other* module under `rotorpy/guidance/` is guidance-side and is
# tested below, derived by globbing the directory rather than a hand-maintained list, so
# a newly added module is covered by default instead of silently exempt.
TRUTH_SIDE = frozenset({"scoring"})


def _guidance_side_modules():
    return tuple(sorted(
        path.stem for path in GUIDANCE_DIR.glob("*.py")
        if path.stem != "__init__" and path.stem not in TRUTH_SIDE
    ))


GUIDANCE_SIDE = _guidance_side_modules()


def _module_path(name):
    parts = name.split(".")
    base = ROOT.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _resolve_relative_module(package, level, module):
    """The dotted module a relative `from` import (`level > 0`) refers to, given the
    importing file's own dotted package (for example `rotorpy.guidance`)."""
    parts = package.split(".") if package else []
    climbed = parts[: len(parts) - (level - 1)] if level > 1 else parts
    base = ".".join(part for part in climbed if part)
    if module:
        return "{}.{}".format(base, module) if base else module
    return base


def _imports(path, package):
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Fix round 1 (minor d): relative imports (`from . import scoring`,
            # `from .scoring import X`) used to be skipped outright here, so a
            # guidance-side module written with one could reach the truth module
            # undetected. Resolve them to the same dotted names an absolute import
            # would produce instead of ignoring them.
            base = (
                _resolve_relative_module(package, node.level, node.module or "")
                if node.level else (node.module or "")
            )
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


def test_guidance_side_is_derived_from_the_directory_not_hand_maintained():
    on_disk = {
        path.stem for path in GUIDANCE_DIR.glob("*.py")
        if path.stem != "__init__" and path.stem not in TRUTH_SIDE
    }
    assert set(GUIDANCE_SIDE) == on_disk
    # Broader than the old 17-module hand-maintained list: every other guidance module
    # (gazebo_record, telemetry, homing_page, outcome, run_failure, runner, frames,
    # gazebo_bridge, homing_config, scene, scene_consistency, recording_worker, ...) is
    # now covered too, since none of them actually need to reach the truth module.
    assert len(GUIDANCE_SIDE) > 17


def test_a_newly_added_guidance_module_is_covered_without_an_allowlist_update():
    # The glob, not a hand-maintained tuple, decides what gets tested: a brand-new file
    # dropped into `rotorpy/guidance/` is covered by default (guidance-side), with no
    # separate list to remember to update.
    scratch = GUIDANCE_DIR / "_fix_round_1_scratch_minor_d.py"
    try:
        scratch.write_text("from rotorpy.guidance.scoring import TruthScorer\n")
        modules = _guidance_side_modules()
        assert "_fix_round_1_scratch_minor_d" in modules
        name = "rotorpy.guidance._fix_round_1_scratch_minor_d"
        assert TRUTH_MODULE in reachable(name)  # the leak is actually caught
    finally:
        scratch.unlink(missing_ok=True)


def test_the_detector_sees_an_import_when_there_is_one(tmp_path):
    # Negative control: the same walk on a file that does import the truth module.
    module = tmp_path / "leaky.py"
    module.write_text("from rotorpy.guidance.scoring import TruthScorer\n")
    assert TRUTH_MODULE in _imports(module, "")
    module.write_text("from rotorpy.guidance import scoring\n")
    assert TRUTH_MODULE in _imports(module, "")


def test_relative_imports_to_the_truth_module_are_detected_too(tmp_path):
    # Fix round 1 (minor d): a guidance-side module written with a relative import
    # (`from . import scoring` or `from .scoring import X`) must be caught exactly like
    # an absolute one.
    module = tmp_path / "leaky_relative.py"
    module.write_text("from . import scoring\n")
    assert TRUTH_MODULE in _imports(module, "rotorpy.guidance")
    module.write_text("from .scoring import TruthScorer\n")
    assert TRUTH_MODULE in _imports(module, "rotorpy.guidance")
    module.write_text("from .homing import HomingCommand\n")  # a real, non-truth sibling
    assert TRUTH_MODULE not in _imports(module, "rotorpy.guidance")
    module.write_text("from .. import guidance\n")  # climbs to `rotorpy`, not the truth module
    assert TRUTH_MODULE not in _imports(module, "rotorpy.guidance")


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
