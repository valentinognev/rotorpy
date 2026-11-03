"""Verify Gazebo's GPU and software rendering launch configuration."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


def _launch(tmp_path, software=None, headless=False):
    commands = tmp_path / "bin"
    commands.mkdir()
    trace = tmp_path / "docker-args"
    _executable(
        commands / "docker",
        '#!/bin/sh\nif [ "$1" = run ]; then printf "%s\\n" "$@" > "$DOCKER_ARGS"; fi\n',
    )
    _executable(commands / "xhost", "#!/bin/sh\nexit 0\n")
    env = os.environ.copy()
    env.update(
        PATH=str(commands) + os.pathsep + env["PATH"],
        DOCKER_ARGS=str(trace),
        DISPLAY=":test",
        XAUTHORITY=str(tmp_path / "no-xauthority"),
    )
    env.pop("LIBGL_ALWAYS_SOFTWARE", None)
    env.pop("HEADLESS", None)
    if software is not None:
        env["LIBGL_ALWAYS_SOFTWARE"] = software
    if headless:
        env["HEADLESS"] = "1"
    result = subprocess.run(
        ["bash", str(ROOT / "run_camera_homing_sitl.sh")],
        env=env, text=True, capture_output=True, check=False,
    )
    return result, trace.read_text().splitlines() if trace.exists() else []


def test_default_launch_requests_nvidia_gpu(tmp_path):
    result, args = _launch(tmp_path)
    assert result.returncode == 0, result.stderr
    assert args[args.index("--gpus") + 1] == "all"
    assert "--env=LIBGL_ALWAYS_SOFTWARE=0" in args
    assert "--env=NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility,compute" in args
    assert "--env=__NV_PRIME_RENDER_OFFLOAD=1" in args
    assert "--env=__GLX_VENDOR_LIBRARY_NAME=nvidia" in args
    assert "--env=GAZEBO_NVIDIA_GPU=1" in args
    assert "--env=HOMING=1" in args


@pytest.mark.parametrize("headless", [False, True])
def test_software_fallback_and_headless_display(tmp_path, headless):
    result, args = _launch(tmp_path, software="1", headless=headless)
    assert result.returncode == 0, result.stderr
    assert "--gpus" not in args
    assert "--env=LIBGL_ALWAYS_SOFTWARE=1" in args
    assert "--env=GAZEBO_NVIDIA_GPU=1" not in args
    assert "--env=DISPLAY=:test" in args
    assert ("--env=HEADLESS=1" in args) is headless


def test_headless_gpu_retains_display(tmp_path):
    result, args = _launch(tmp_path, headless=True)
    assert result.returncode == 0, result.stderr
    assert "--gpus" in args
    assert "--env=HEADLESS=1" in args
    assert "--env=DISPLAY=:test" in args


def test_invalid_software_setting_is_rejected(tmp_path):
    result, args = _launch(tmp_path, software="invalid")
    assert result.returncode != 0
    assert "LIBGL_ALWAYS_SOFTWARE must be 0 or 1" in result.stderr
    assert not args


def test_gpu_renderer_check_rejects_non_nvidia(tmp_path):
    _executable(tmp_path / "glxinfo", '#!/bin/sh\necho "OpenGL vendor string: Intel"\n')
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    env["GAZEBO_NVIDIA_GPU"] = "1"
    result = subprocess.run(
        ["bash", str(ROOT / "Dockerfiles" / "sitl_hummingbird_run.sh")],
        env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "did not select the NVIDIA OpenGL renderer" in result.stderr
