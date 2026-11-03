#!/bin/bash
# Interactive shell in px4-noetic-sim-ros with the Hummingbird model mounted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="px4-noetic-sim-ros"
CONTAINER_NAME="rotorpy-hummingbird-bash"
PX4_ROOT="/home/valentin/PX4-Autopilot"

if docker ps --format '{{.Names}}' | grep -qx 'px4-noetic-sim-ros'; then
	echo "px4-noetic-sim-ros is already running. Stop that container before this shell; both use Gazebo port 11345."
	exit 1
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
xhost + >/dev/null 2>&1 || true
xhost +local:docker >/dev/null 2>&1 || true

docker run -it --rm --net=host \
	--cap-drop=all \
	--privileged \
	--env="DISPLAY=${DISPLAY:-:0}" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume="${ROOT}/gazeboModels/hummingbird:${PX4_ROOT}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/hummingbird:ro" \
	--volume="${SCRIPT_DIR}/sitl_hummingbird_run.sh:${PX4_ROOT}/Tools/simulation/gazebo-classic/sitl_hummingbird_run.sh:ro" \
	--volume="${SCRIPT_DIR}/airframes/1100_gazebo-classic_hummingbird:${PX4_ROOT}/build/px4_sitl_default/etc/init.d-posix/airframes/1100_gazebo-classic_hummingbird:ro" \
	--name="${CONTAINER_NAME}" \
	"${IMAGE}"

xhost -local:docker >/dev/null 2>&1 || true
xhost - >/dev/null 2>&1 || true
