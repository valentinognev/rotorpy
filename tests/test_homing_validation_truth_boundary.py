"""The offline reliability harness (`rotorpy/guidance/validation/`) is truth-side:
it is deliberately outside `tests/test_homing_truth_boundary.py`'s non-recursive
`rotorpy/guidance/*.py` glob (see that test's `GUIDANCE_DIR.glob("*.py")`), not
merely incidentally so -- this test pins that choice down and proves the harness
actually does import the truth-side scoring module, the thing a guidance-side
module is forbidden from doing.
"""

import ast
from pathlib import Path

import pytest

GUIDANCE_DIR = Path(__file__).resolve().parents[1] / "rotorpy" / "guidance"
VALIDATION_DIR = GUIDANCE_DIR / "validation"


def test_validation_package_is_outside_the_guidance_side_glob():
    assert VALIDATION_DIR.is_dir()
    # The real truth-boundary test's own glob pattern: single-segment "*.py" never
    # matches a path whose parent is the `validation/` subdirectory.
    matched = set(GUIDANCE_DIR.glob("*.py"))
    assert not any(path.parent == VALIDATION_DIR for path in matched)
    assert "validation" not in {path.stem for path in matched}


def test_harness_module_imports_the_truth_side_scoring_module():
    harness_path = VALIDATION_DIR / "harness.py"
    tree = ast.parse(harness_path.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "rotorpy.guidance.scoring" in imported


@pytest.mark.parametrize("name", ["vehicle_model", "target_world", "faults", "harness_config", "harness",
                                   "scenarios", "stats", "campaign"])
def test_every_validation_module_exists(name):
    assert (VALIDATION_DIR / "{}.py".format(name)).is_file()


def test_real_truth_boundary_test_still_passes_with_the_validation_package_present():
    """Reuses `test_homing_truth_boundary`'s own discovery helper directly (rather
    than re-deriving it) to pin down that adding this whole package did not grow
    `GUIDANCE_SIDE` by even one module."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import test_homing_truth_boundary as boundary
    finally:
        sys.path.pop(0)
    assert "validation" not in boundary.GUIDANCE_SIDE
    for module in boundary.GUIDANCE_SIDE:
        assert "validation" not in module
