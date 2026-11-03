#!/bin/bash
# Camera-homing flight in px4-noetic-sim-ros: two cameras, red cylinder, homing tracker.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HOMING=1
exec "${SCRIPT_DIR}/Dockerfiles/run_hummingbird_sitl.sh" --num 1 --world empty "$@"
