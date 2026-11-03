#!/bin/bash
# Spawn the Hummingbird in Gazebo Classic and start PX4 SITL.
# Mounted into px4-noetic-sim-ros at Tools/simulation/gazebo-classic/.

set -euo pipefail

if [ "${GAZEBO_NVIDIA_GPU:-0}" = "1" ]; then
	if ! command -v glxinfo >/dev/null 2>&1; then
		echo "GPU rendering requires glxinfo in the simulation image" >&2
		exit 1
	fi
	if ! gl_info="$(glxinfo -B)"; then
		echo "Unable to initialize the NVIDIA OpenGL display" >&2
		exit 1
	fi
	if ! grep -q '^OpenGL vendor string: NVIDIA Corporation' <<<"${gl_info}"; then
		echo "Gazebo did not select the NVIDIA OpenGL renderer:" >&2
		printf '%s\n' "${gl_info}" >&2
		exit 1
	fi
	printf '%s\n' "${gl_info}"
fi

cleanup() {
	local pid
	for pid in "${tracker_pid:-}" "${gzclient_pid:-}" "${px4_pids[@]}" "${gz_pid:-}" "${roscore_pid:-}"; do
		if [ -n "${pid}" ]; then
			kill "${pid}" 2>/dev/null || true
		fi
	done
}

render_model() {
	local n="$1"
	python3 \
		"${src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/scripts/jinja_gen.py" \
		"${src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/hummingbird/hummingbird.sdf.jinja" \
		"${src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic" \
		--mavlink_tcp_port "$((4560 + n))" \
		--mavlink_udp_port "$((14560 + n))" \
		--mavlink_id "$((1 + n))" \
		--output-file "/tmp/hummingbird_${n}.sdf"
}

start_px4() {
	local n="$1"
	local working_dir="${build_path}/rootfs/${n}"
	mkdir -p "${working_dir}"
	echo "starting instance ${n} in ${working_dir}"
	(cd "${working_dir}" && exec "${build_path}/bin/px4" -i "${n}" -d "${build_path}/etc" >out.log 2>err.log) &
	px4_pids+=("$!")
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
	echo "Usage: $0 [-n <num_vehicles>] [-w <world>]"
	exit 0
fi

NUM_VEHICLES=1
WORLD=empty
while getopts n:w: option; do
	case "${option}" in
		n) NUM_VEHICLES=${OPTARG} ;;
		w) WORLD=${OPTARG} ;;
		*) exit 1 ;;
	esac
done

if [ "${NUM_VEHICLES}" -lt 1 ] || [ "${NUM_VEHICLES}" -gt 255 ]; then
	echo "num_vehicles must be from 1 to 255"
	exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_path="${SCRIPT_DIR}/../../.."
target=px4_sitl_default
export PX4_SIM_MODEL=gazebo-classic_hummingbird
build_path="${src_path}/build/${target}"

gz_pid=""
gzclient_pid=""
tracker_pid=""
roscore_pid=""
px4_pids=()
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# setup_gazebo.bash appends to these. This script uses nounset, and a fresh
# container does not have them yet.
export GAZEBO_PLUGIN_PATH="${GAZEBO_PLUGIN_PATH:-}"
export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# shellcheck disable=SC1090
source "${src_path}/Tools/simulation/gazebo-classic/setup_gazebo.bash" \
	"${src_path}" "${build_path}"

ros_args=""
if [[ -n "${ROS_VERSION:-}" ]] && [ "${ROS_VERSION}" = "2" ]; then
	ros_args="-s libgazebo_ros_init.so -s libgazebo_ros_factory.so"
else
	ros_args="-s libgazebo_ros_api_plugin.so"
	if command -v roscore >/dev/null 2>&1 && ! rosnode list >/dev/null 2>&1; then
		echo "Starting roscore"
		roscore &
		roscore_pid=$!
		_i=0
		while [ "${_i}" -lt 50 ]; do
			rosnode list >/dev/null 2>&1 && break
			sleep 0.1
			_i=$((_i + 1))
		done
	fi
fi

GAZEBO11_PLUGINS="/usr/lib/x86_64-linux-gnu/gazebo-11/plugins"
if [ -d "${GAZEBO11_PLUGINS}" ]; then
	export LD_LIBRARY_PATH="${GAZEBO11_PLUGINS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

gz_server_alive() {
	local stat
	stat=$(sed -n 's/.*) //p' /proc/"$1"/stat 2>/dev/null | awk '{print $1}') || return 1
	[ -n "${stat}" ] && [ "${stat}" != "Z" ]
}

# Gazebo transport sometimes reads an empty header while models are spawning
# and aborts (std::length_error). Wait until the server is serving, and start
# it over if it dies. Do not start PX4 against a dead server.
wait_for_gz() {
	local i
	for i in $(seq 1 40); do
		gz_server_alive "${gz_pid}" || return 1
		if rosservice list 2>/dev/null | grep -q /gazebo/spawn_sdf_model; then
			sleep 2
			gz_server_alive "${gz_pid}" || return 1
			return 0
		fi
		sleep 0.5
	done
	return 1
}

spawn_sdf() {
	gz model "$@" || return 1
	gz_server_alive "${gz_pid}"
}

stop_sim_processes() {
	if [ -n "${gz_pid}" ]; then
		kill "${gz_pid}" 2>/dev/null || true
		wait "${gz_pid}" 2>/dev/null || true
		gz_pid=""
	fi
}

if [ "${HOMING:-0}" = "1" ] && ! python3 -c "import PIL.Image" >/dev/null 2>&1; then
	echo "installing Pillow"
	python3 -m pip install --disable-pip-version-check --quiet Pillow
fi

gz_up=0
for gz_attempt in 1 2 3; do
	if [ "${gz_attempt}" -gt 1 ]; then
		echo "gazebo aborted on attempt $((gz_attempt - 1)); starting again"
		stop_sim_processes
	fi
	echo "Starting gazebo"
	gzserver "${src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/${WORLD}.world" --verbose ${ros_args} &
	gz_pid=$!
	if ! wait_for_gz; then
		wait "${gz_pid}" 2>/dev/null || true
		continue
	fi
	spawn_ok=1
	if [ "${HOMING:-0}" = "1" ]; then
		# Gazebo is east-north-up. The homing start is north 0, east 0, 30 m up.
		# The red cylinder stands on north 100, east 50. Nose points at that ground point.
		TARGET_SDF="${TARGET_SDF:-${src_path}/Tools/simulation/gazebo-classic/target_cylinder.sdf}"
		if ! rosservice call /gazebo/pause_physics; then
			spawn_ok=0
		fi
		if [ "${spawn_ok}" = "1" ]; then
			if ! spawn_sdf --spawn-file="${TARGET_SDF}" --model-name=homing_target -x 50 -y 100 -z 1.5; then
				spawn_ok=0
			fi
		fi
		if [ "${spawn_ok}" = "1" ]; then
			YAW="$(python3 -c 'import math; print(math.atan2(100.0, 50.0))')"
			render_model 0
			echo "Spawning hummingbird_0 at 0 0 30 yaw ${YAW}"
			if ! spawn_sdf --spawn-file="/tmp/hummingbird_0.sdf" --model-name="hummingbird_0" -x 0 -y 0 -z 30 -Y "${YAW}"; then
				spawn_ok=0
			fi
		fi
	else
		n=0
		while [ "${n}" -lt "${NUM_VEHICLES}" ]; do
			render_model "${n}"
			echo "Spawning hummingbird_${n} at 0 $((n * 3)) 0.2"
			if ! spawn_sdf --spawn-file="/tmp/hummingbird_${n}.sdf" --model-name="hummingbird_${n}" -x 0 -y "$((n * 3))" -z 0.2; then
				spawn_ok=0
				break
			fi
			n=$((n + 1))
		done
	fi
	if [ "${spawn_ok}" = "1" ]; then
		gz_up=1
		break
	fi
	echo "gazebo died while spawning"
done
if [ "${gz_up}" != 1 ]; then
	echo "gzserver failed to stay up"
	exit 1
fi

if [ "${HOMING:-0}" = "1" ]; then
	start_px4 0
	python3 /opt/rotorpy/Dockerfiles/camera_homing_tracker.py \
		--config /opt/rotorpy/examples/camera_homing.jsonc \
		--model hummingbird_0 &
	tracker_pid=$!
else
	n=0
	while [ "${n}" -lt "${NUM_VEHICLES}" ]; do
		start_px4 "${n}"
		n=$((n + 1))
	done
fi

if [ "${HEADLESS:-0}" = "1" ]; then
	echo "HEADLESS=1 — leaving gzserver up without gzclient"
	if [ "${HOMING:-0}" = "1" ]; then
		wait -n "${tracker_pid}" "${gz_pid}"
	else
		wait
	fi
else
	echo "Starting gazebo client"
	if [ "${HOMING:-0}" = "1" ]; then
		gzclient &
		gzclient_pid=$!
		wait -n "${tracker_pid}" "${gz_pid}" "${gzclient_pid}"
	else
		gzclient
	fi
fi
