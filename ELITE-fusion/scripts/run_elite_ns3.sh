#!/usr/bin/env bash
# One-click launcher for ELITE controller (Python, ns3-ai) + NS-3 ELTI scenario (with Poisson flows).
# - Starts the controller (as non-root TARGET_USER), waits for junction_legend.csv, then runs NS-3 (same user).
# - Adds safety: clean stale ns3-ai SysV shared memory, unify ownership, optional pre-kill, wall-time timeout.
#
# Env/Args you may override:
#   NS3_DIR            default: /home/Lyh/VaN3Twin/ns-3-dev
#   CTRL_PY            default: /home/Lyh/ELITE-zg-main/ELITE-fusion/controller_server.py
#   NET_XML            default: /home/Lyh/ELITE-zg-main/grid_network/grid.net.xml
#   OUT_DIR            default: /home/Lyh/ELITE-zg-main/dtn_out
#   BANDWIDTH_BPS      default: 6000000
#   COMM_RADIUS        default: 300
#   MEMPOOL_KEY        default: 1234  (0x000004d2)
#   MEM_SIZE           default: 4096
#   MEMBLOCK_KEY       default: 2333  (0x0000091d)
#   SUMO_WAIT          default: 20    (seconds to wait before TraCI connect)
#   POISSON_ENABLE     default: 1
#   POISSON_LAMBDA     default: 0.5
#   SIM_TIME           default: 100   (ns-3 simulation time in seconds)
#   TARGET_USER        default: Lyh   (both controller and ns-3 run as this non-root user)
#   NS_LOG             optional, e.g. 'EliteForwarderApp=level_info:TraciClient=level_info|prefix_time'
#   SUMO_STEP_LOG      default: 0; set 1 to print 'Step #' lines from SUMO
#   KILL_OLD           default: 1; set 0 to skip killing old sumo/ns3/controller
#   CLEAN_SHM          default: 1; clear stale SysV shared memory for keys 1234/2333 before run
#   CTRL_READY_TIMEOUT default: 60    (seconds to wait for junction_legend.csv)
#   WALL_TIMEOUT       default: SIM_TIME+120 (seconds; kill ns-3 if it exceeds wall-clock timeout)
#
# Usage:
#   bash ELITE-fusion/scripts/run_elite_ns3.sh
#   ELITE_DEBUG=1 SUMO_STEP_LOG=1 SIM_TIME=60 bash ELITE-fusion/scripts/run_elite_ns3.sh

set -euo pipefail

NS3_DIR="${NS3_DIR:-/home/Lyh/VaN3Twin/ns-3-dev}"
CTRL_PY="${CTRL_PY:-/home/Lyh/ELITE-zg-main/ELITE-fusion/controller_server.py}"
NET_XML="${NET_XML:-/home/Lyh/ELITE-zg-main/grid_network/grid.net.xml}"
OUT_DIR="${OUT_DIR:-/home/Lyh/ELITE-zg-main/dtn_out}"
BANDWIDTH_BPS="${BANDWIDTH_BPS:-6000000}"
COMM_RADIUS="${COMM_RADIUS:-300}"
MEMPOOL_KEY="${MEMPOOL_KEY:-1234}"
MEM_SIZE="${MEM_SIZE:-65536}"
MEMBLOCK_KEY="${MEMBLOCK_KEY:-2333}"
SUMO_WAIT="${SUMO_WAIT:-20}"
POISSON_ENABLE="${POISSON_ENABLE:-1}"
POISSON_LAMBDA="${POISSON_LAMBDA:-0.5}"
SIM_TIME="${SIM_TIME:-100}"
TARGET_USER="${TARGET_USER:-Lyh}"
KILL_OLD="${KILL_OLD:-1}"
CLEAN_SHM="${CLEAN_SHM:-1}"
CTRL_READY_TIMEOUT="${CTRL_READY_TIMEOUT:-60}"
SUMO_CONFIG="${SUMO_CONFIG:-${NS3_DIR}/src/automotive/examples/grid_network/grid.sumocfg}"
STATIONS_XML="${STATIONS_XML:-${NS3_DIR}/src/automotive/examples/grid_network/stations.xml}"

# Wall-clock timeout: default SIM_TIME + 120s slack
WALL_TIMEOUT="${WALL_TIMEOUT:-$(( SIM_TIME + 120 ))}"
if [[ "${WALL_TIMEOUT}" -lt 1 ]]; then WALL_TIMEOUT=$(( SIM_TIME + 120 )); fi

JUNCTION_LEGEND="${OUT_DIR}/junction_legend.csv"
CTRL_LOG="${OUT_DIR}/controller.log"
NS3_LOG="${NS3_DIR}/src/automotive/examples/ELTI/run_ELTI_auto.log"
NS3_LOG_OUT="${OUT_DIR}/ns3_run.log"
# Keep ns3-ai SysV pool parameters consistent across controller/ns-3.
NS_GLOBAL_VALUE_STR="SharedMemoryKey=${MEMPOOL_KEY};SharedMemoryPoolSize=${MEM_SIZE};"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

sanitize_nslog() {
  # Convert commas to colons if user mistakenly used commas between components.
  local s="${1:-}"
  if [[ "$s" == *","* ]]; then
    s="${s//,/:}"
  fi
  echo "$s"
}

run_as_target() {
  # Run given command as TARGET_USER if current user is root; otherwise run directly.
  if [[ "${EUID}" -eq 0 ]]; then
    # Preserve relevant ELITE_* env vars when switching user
    sudo -E -u "${TARGET_USER}" -H "$@"
  else
    "$@"
  fi
}

# Basic checks
[[ -f "${CTRL_PY}" ]] || die "Controller not found: ${CTRL_PY}"
[[ -f "${NET_XML}" ]] || die "SUMO net.xml not found: ${NET_XML}"
[[ -x "${NS3_DIR}/ns3" ]] || die "ns3 launcher not found/executable: ${NS3_DIR}/ns3"
mkdir -p "${OUT_DIR}"

# Ensure ns3-ai py_interface can be imported by controller
export PYTHONPATH="${NS3_DIR}/contrib/ns3-ai/py_interface:${PYTHONPATH:-}"
export ELITE_DEBUG="${ELITE_DEBUG:-0}"

# Optional: stop old processes to avoid port/shm conflicts
if [[ "${KILL_OLD}" == "1" ]]; then
  log "Killing any leftover ns3/sumo/controller processes (best effort)..."
  # Try to kill only matching processes; ignore failures
  pkill -f ns3-dev-ELTI-optimized 2>/dev/null || true
  pkill -f "sumo -c" 2>/dev/null || true
  pkill -f controller_server.py 2>/dev/null || true
  sleep 0.5
fi

# Optional: clear stale ns3-ai shared memory
if [[ "${CLEAN_SHM}" == "1" ]]; then
  log "Cleaning stale SysV shared memory for ns3-ai (keys ${MEMPOOL_KEY}/${MEMBLOCK_KEY})..."
  # hex keys used by ipcrm -M
  printf -v MEMPOOL_HEX "0x%08x" "${MEMPOOL_KEY}"
  printf -v MEMBLOCK_HEX "0x%08x" "${MEMBLOCK_KEY}"
  ipcrm -M "${MEMPOOL_HEX}" 2>/dev/null || true
  ipcrm -M "${MEMBLOCK_HEX}" 2>/dev/null || true
fi

# Unify ownership for output directory & logs
if id -u "${TARGET_USER}" >/dev/null 2>&1; then
  chown -R "${TARGET_USER}:${TARGET_USER}" "${OUT_DIR}" 2>/dev/null || true
fi
touch "${CTRL_LOG}" "${NS3_LOG_OUT}" 2>/dev/null || true
if id -u "${TARGET_USER}" >/dev/null 2>&1; then
  chown "${TARGET_USER}:${TARGET_USER}" "${CTRL_LOG}" "${NS3_LOG_OUT}" 2>/dev/null || true
fi

# Clear old logs (optional)
: > "${CTRL_LOG}" || true
: > "${NS3_LOG}" || true
: > "${NS3_LOG_OUT}" || true
: > "${NS3_DIR}/src/automotive/examples/grid_network/SumoMsg.log" || true
: > "${NS3_DIR}/src/automotive/examples/grid_network/SumoError.log" || true

# Start controller in background as TARGET_USER
log "Starting ELITE controller (ns3-ai) as ${TARGET_USER} ..."
set +e
run_as_target env PYTHONUNBUFFERED=1 python3 "${CTRL_PY}" \
  "${NET_XML}" "${OUT_DIR}" "${BANDWIDTH_BPS}" "${COMM_RADIUS}" \
  "${MEMPOOL_KEY}" "${MEM_SIZE}" "${MEMBLOCK_KEY}" \
  > "${CTRL_LOG}" 2>&1 &
CTRL_PID=$!
set -e
log "Controller PID=${CTRL_PID}, logging to ${CTRL_LOG}"

cleanup() {
  # Try to stop controller (if still running)
  if ps -p "${CTRL_PID}" >/dev/null 2>&1; then
    log "Stopping controller (PID=${CTRL_PID})"
    kill "${CTRL_PID}" >/dev/null 2>&1 || true
    wait "${CTRL_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Wait for controller to initialize and write junction legend
log "Waiting for controller to initialize (junction legend)..."
SECONDS_WAITED=0
while [[ ! -s "${JUNCTION_LEGEND}" && "${SECONDS_WAITED}" -lt "${CTRL_READY_TIMEOUT}" ]]; do
  sleep 1
  SECONDS_WAITED=$((SECONDS_WAITED+1))
done
if [[ ! -s "${JUNCTION_LEGEND}" ]]; then
  log "ERROR: ${JUNCTION_LEGEND} not created by controller within ${CTRL_READY_TIMEOUT}s. See ${CTRL_LOG}"
  exit 2
fi
log "Found ${JUNCTION_LEGEND}"

# Compose ns-3 command
NS3_PROGRAM="ELTI"
NS3_CMD_ARGS=(
  --junction-legend="${JUNCTION_LEGEND}"
  --poisson-enable="${POISSON_ENABLE}"
  --poisson-lambda="${POISSON_LAMBDA}"
  --sumo-wait="${SUMO_WAIT}"
  --sim-time="${SIM_TIME}"
)
if [[ -n "${SUMO_CONFIG:-}" ]]; then
  NS3_CMD_ARGS+=( --sumo-config="${SUMO_CONFIG}" )
fi
if [[ -n "${STATIONS_XML:-}" ]]; then
  NS3_CMD_ARGS+=( --rsu-stations="${STATIONS_XML}" )
fi
# SUMO route file is read from .sumocfg; expose env only if scenario supports the arg.
# Optional: SUMO per-step stdout logging
if [[ "${SUMO_STEP_LOG:-0}" == "1" ]]; then
  NS3_CMD_ARGS+=( --sumo-step-log=1 )
fi
NS3_ARGS=( "${NS3_PROGRAM}" -- "${NS3_CMD_ARGS[@]}" )

# Sanitize NS_LOG if needed (handle accidental commas)
SAFE_NS_LOG="$(sanitize_nslog "${NS_LOG:-}")"

# Optional: wrap ns-3 inside gdb (set NS3_GDB=1); handy when ns-3 crashes and you need backtraces.
NS3_FRONTEND="${NS3_DIR}/ns3"
NS3_LAUNCH=( "${NS3_FRONTEND}" run "${NS3_ARGS[@]}" )
if [[ "${NS3_GDB:-0}" == "1" ]]; then
  GDB_BIN="${NS3_GDB_BIN:-gdb}"
  NS3_GDB_TARGET_DEFAULT="${NS3_DIR}/build/src/automotive/examples/ns3-dev-${NS3_PROGRAM}-optimized"
  NS3_GDB_TARGET="${NS3_GDB_TARGET:-${NS3_GDB_TARGET_DEFAULT}}"
  if [[ ! -x "${NS3_GDB_TARGET}" ]]; then
    log "WARNING: NS3_GDB_TARGET ${NS3_GDB_TARGET} missing or not executable; adjust NS3_GDB_TARGET if needed."
  fi
  GDB_EXTRA_CMDS=()
  if [[ -n "${NS3_GDB_EXE:-}" ]]; then
    # allow passing a whole command string (e.g., "-ex 'run' -ex 'bt full'")
    # shellcheck disable=SC2206
    GDB_EXTRA_CMDS=( ${NS3_GDB_EXE} )
  else
    GDB_EXTRA_CMDS=( -ex "set pagination off" -ex "handle SIGPIPE nostop noprint pass" -ex run -ex "bt" -ex "quit" )
  fi
  NS3_LAUNCH=( "${GDB_BIN}" -q "${GDB_EXTRA_CMDS[@]}" --args "${NS3_GDB_TARGET}" "${NS3_CMD_ARGS[@]}" )
  log "NS3_GDB=1 -> running ns-3 under gdb (${GDB_BIN})"
fi

# Run ns-3 as TARGET_USER with a wall-clock timeout; tee logs into both ns-3 tree and OUT_DIR
log "Running NS-3 (Poisson=${POISSON_ENABLE}, lambda=${POISSON_LAMBDA}, sim-time=${SIM_TIME}s, wall-time timeout=${WALL_TIMEOUT}s) ..."
set +e
if command -v timeout >/dev/null 2>&1; then
  # Send SIGTERM after WALL_TIMEOUT, then SIGKILL 10s later if still running
  run_as_target env NS_LOG="${SAFE_NS_LOG}" ELITE_NS3_TRACE="${ELITE_NS3_TRACE:-}" \
    NS_GLOBAL_VALUE="${NS_GLOBAL_VALUE_STR}" timeout -k 10s "${WALL_TIMEOUT}s" \
    "${NS3_LAUNCH[@]}" 2>&1 | tee -a "${NS3_LOG}" | tee -a "${NS3_LOG_OUT}"
  NS3_EC=${PIPESTATUS[0]}
else
  run_as_target env NS_LOG="${SAFE_NS_LOG}" ELITE_NS3_TRACE="${ELITE_NS3_TRACE:-}" \
    NS_GLOBAL_VALUE="${NS_GLOBAL_VALUE_STR}" \
    "${NS3_LAUNCH[@]}" 2>&1 | tee -a "${NS3_LOG}" | tee -a "${NS3_LOG_OUT}"
  NS3_EC=${PIPESTATUS[0]}
fi
set -e

if [[ "${NS3_EC}" -ne 0 ]]; then
  log "NS-3 exited non-zero (code=${NS3_EC}). See ${NS3_LOG_OUT} and ${NS3_LOG}"
else
  log "NS-3 finished successfully."
fi

log "NS-3 logs: ${NS3_LOG} and ${NS3_LOG_OUT}"
log "Controller log: ${CTRL_LOG}"
