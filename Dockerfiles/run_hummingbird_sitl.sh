#!/bin/bash
# Start one Hummingbird in the existing px4-noetic-sim-ros image.
# The model and the in-container runner are bind-mounted from this repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="px4-noetic-sim-ros"
CONTAINER_NAME="rotorpy-hummingbird-sitl"
PX4_ROOT="/home/valentin/PX4-Autopilot"
WORLD="${WORLD:-empty}"
NUM="${NUM:-1}"

# Stop a previous sim before argument checks, xhost, or a new container.
if [ "${1:-}" != "-h" ] && [ "${1:-}" != "--help" ]; then
	echo "stopping ${CONTAINER_NAME}"
	docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

remove_container() {
	docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

cleanup_on_exit() {
	remove_container
	xhost -local:docker >/dev/null 2>&1 || true
	xhost - >/dev/null 2>&1 || true
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--help|-h)
			echo "Usage: $0 [--kill] [--num N] [--world NAME]"
			echo "Image: ${IMAGE}. Container name: ${CONTAINER_NAME}."
			echo "NVIDIA GPU rendering is the default; LIBGL_ALWAYS_SOFTWARE=1 selects software rendering."
			echo "HEADLESS=1 runs gzserver without a Gazebo window; camera rendering still needs a display."
			exit 0
			;;
		--kill)
			cleanup_on_exit
			exit 0
			;;
		--num)
			NUM="$2"
			shift 2
			;;
		--num=*)
			NUM="${1#*=}"
			shift
			;;
		--world)
			WORLD="$2"
			shift 2
			;;
		--world=*)
			WORLD="${1#*=}"
			shift
			;;
		*)
			echo "Unknown option: $1"
			exit 1
			;;
	esac
done

if ! [[ "${NUM}" =~ ^[0-9]+$ ]]; then
	echo "--num must be an integer"
	exit 1
fi
if ! [[ "${WORLD}" =~ ^[A-Za-z0-9_-]+$ ]]; then
	echo "--world must be a single name"
	exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx 'px4-noetic-sim-ros'; then
	echo "px4-noetic-sim-ros is already running. Stop that container before this one; both use Gazebo port 11345."
	exit 1
fi

trap cleanup_on_exit EXIT INT TERM
remove_container

xhost + >/dev/null 2>&1 || true
xhost +local:docker >/dev/null 2>&1 || true
if [ -z "${DISPLAY:-}" ]; then
	export DISPLAY=:0
fi
if [ -z "${XAUTHORITY:-}" ] && [ -f "${HOME}/.Xauthority" ]; then
	export XAUTHORITY="${HOME}/.Xauthority"
fi
XAUTH_FILE="${XAUTHORITY:-${HOME}/.Xauthority}"

SIM_DISPLAY="${DISPLAY}"
EXTRA_ENV=()
GPU_ARGS=()
if [ "${HEADLESS:-0}" = "1" ]; then
	EXTRA_ENV+=(--env="HEADLESS=1")
fi
case "${LIBGL_ALWAYS_SOFTWARE:-0}" in
	0)
		GPU_ARGS+=(--gpus all)
		EXTRA_ENV+=(
			--env="LIBGL_ALWAYS_SOFTWARE=0"
			--env="NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility,compute"
			--env="__NV_PRIME_RENDER_OFFLOAD=1"
			--env="__GLX_VENDOR_LIBRARY_NAME=nvidia"
			--env="GAZEBO_NVIDIA_GPU=1"
		)
		;;
	1)
		EXTRA_ENV+=(--env="LIBGL_ALWAYS_SOFTWARE=1")
		;;
	*)
		echo "LIBGL_ALWAYS_SOFTWARE must be 0 or 1" >&2
		exit 1
		;;
esac

DOCKER_VOLUMES=(
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"
	--volume="${ROOT}/gazeboModels/hummingbird:${PX4_ROOT}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/hummingbird:ro"
	--volume="${SCRIPT_DIR}/sitl_hummingbird_run.sh:${PX4_ROOT}/Tools/simulation/gazebo-classic/sitl_hummingbird_run.sh:ro"
	--volume="${SCRIPT_DIR}/airframes/1100_gazebo-classic_hummingbird:${PX4_ROOT}/build/px4_sitl_default/etc/init.d-posix/airframes/1100_gazebo-classic_hummingbird:ro"
)
if [ -f "${XAUTH_FILE}" ]; then
	DOCKER_VOLUMES+=(--volume="${XAUTH_FILE}:${XAUTH_FILE}:ro")
fi
if [ "${HOMING:-0}" = "1" ]; then
	DOCKER_VOLUMES+=(
		--volume="${ROOT}:/opt/rotorpy:ro"
		--volume="${ROOT}/Results:/opt/rotorpy/Results:rw"
		--volume="${ROOT}/gazeboModels/homing/target_cylinder.sdf:${PX4_ROOT}/Tools/simulation/gazebo-classic/target_cylinder.sdf:ro"
	)
	EXTRA_ENV+=(--env="HOMING=1" --env="PYTHONPATH=/opt/rotorpy")
fi

# A docker TTY makes gzserver abort on this host: "Header is empty" then
# std::length_error. The Gazebo window uses DISPLAY, not the container TTY.
docker run --rm --net=host \
	--cap-drop=all \
	--privileged \
	"${GPU_ARGS[@]}" \
	--env="DISPLAY=${SIM_DISPLAY}" \
	--env="QT_X11_NO_MITSHM=1" \
	--env="GAZEBO_IP=127.0.0.1" \
	--env="GAZEBO_MASTER_URI=http://127.0.0.1:11345" \
	--env="XAUTHORITY=${XAUTH_FILE}" \
	"${EXTRA_ENV[@]}" \
	"${DOCKER_VOLUMES[@]}" \
	--name="${CONTAINER_NAME}" \
	"${IMAGE}" \
	/bin/bash -c "./Tools/simulation/gazebo-classic/sitl_hummingbird_run.sh -n ${NUM} -w ${WORLD}"
