#!/usr/bin/env bash
# One-click launcher for ELITE controller (Python, ns3-ai) + NS-3 ELTI scenario (with Poisson flows).
# - Starts the controller, waits for junction_legend.csv, then runs NS-3 as non-root user.
# - Defaults match the repository layout on this machine; override via env or CLI args if needed.
#
# Env/Args you may override:
#   NS3_DIR           (default: /home/Lyh/VaN3Twin/ns-3-dev)
#   CTRL_PY           (default: /home/Lyh/ELITE-zg-main/ELITE-fusion/controller_server.py)
#   NET_XML           (default: /home/Lyh/ELITE-zg-main/grid_network/grid.net.xml)
#   OUT_DIR           (default: /home/Lyh/ELITE-zg-main/dtn_out)
#   BANDWIDTH_BPS     (default: 6000000)
#   COMM_RADIUS       (default: 300)
#   MEMPOOL_KEY       (default: 1234)
#   MEM_SIZE          (default: 4096)
#   MEMBLOCK_KEY      (default: 2333)
#   SUMO_WAIT         (default: 20)
#   POISSON_ENABLE    (default: 1)
#   POISSON_LAMBDA    (default: 0.5)
#   SIM_TIME          (default: 100)
#   TARGET_USER       (default: Lyh)  # ns-3 must not run as root
#   NS_LOG            (optional)      # e.g. 'EliteForwarderApp=level_info'
#
# Usage:
#   bash ELITE-fusion/scripts/run_elite_ns3.sh
#   NS_LOG="EliteForwarderApp=level_info" SIM_TIME=60 bash ELITE-fusion/scripts/run_elite_ns3.sh

set -euo pipefail

NS3_DIR="${NS3_DIR:-/home/Lyh/VaN3Twin/ns-3-dev}"
CTRL_PY="${CTRL_PY:-/home/Lyh/ELITE-zg-main/ELITE-fusion/controller_server.py}"
NET_XML="${NET_XML:-/home/Lyh/ELITE-zg-main/grid_network/grid.net.xml}"
OUT_DIR="${OUT_DIR:-/home/Lyh/ELITE-zg-main/dtn_out}"
BANDWIDTH_BPS="${BANDWIDTH_BPS:-6000000}"
COMM_RADIUS="${COMM_RADIUS:-300}"
MEMPOOL_KEY="${MEMPOOL_KEY:-1234}"
MEM_SIZE="${MEM_SIZE:-4096}"
MEMBLOCK_KEY="${MEMBLOCK_KEY:-2333}"
SUMO_WAIT="${SUMO_WAIT:-20}"
POISSON_ENABLE="${POISSON_ENABLE:-1}"
POISSON_LAMBDA="${POISSON_LAMBDA:-0.5}"
SIM_TIME="${SIM_TIME:-100}"
TARGET_USER="${TARGET_USER:-Lyh}"

JUNCTION_LEGEND="${OUT_DIR}/junction_legend.csv"
CTRL_LOG="${OUT_DIR}/controller.log"
NS3_LOG="${NS3_DIR}/src/automotive/examples/ELTI/run_ELTI_auto.log"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Basic checks
[[ -f "${CTRL_PY}" ]] || { echo "Controller not found: ${CTRL_PY}"; exit 1; }
[[ -f "${NET_XML}" ]] || { echo "SUMO net.xml not found: ${NET_XML}"; exit 1; }
[[ -x "${NS3_DIR}/ns3" ]] || { echo "ns3 launcher not found/executable: ${NS3_DIR}/ns3"; exit 1; }
mkdir -p "${OUT_DIR}"

# Ensure ns3-ai py_interface can be imported by controller
export PYTHONPATH="${NS3_DIR}/contrib/ns3-ai/py_interface:${PYTHONPATH:-}"

# Clean any stale mempool (optional; disabled by default)
# ipcs -m | awk '/ 666 .*'"${MEMPOOL_KEY}"'/{print $2}' | xargs -r ipcrm -m || true

# Start controller in background
log "Starting ELITE controller (ns3-ai) ..."
python3 "${CTRL_PY}" \
  "${NET_XML}" "${OUT_DIR}" "${BANDWIDTH_BPS}" "${COMM_RADIUS}" \
  "${MEMPOOL_KEY}" "${MEM_SIZE}" "${MEMBLOCK_KEY}" \
  > "${CTRL_LOG}" 2>&1 &
CTRL_PID=$!
log "Controller PID=${CTRL_PID}, logging to ${CTRL_LOG}"

cleanup() {
  if ps -p "${CTRL_PID}" >/dev/null 2>&1; then
    log "Stopping controller (PID=${CTRL_PID})"
    kill "${CTRL_PID}" >/dev/null 2>&1 || true
    wait "${CTRL_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Wait for controller to initialize and write junction legend
log "Waiting for controller to initialize (junction legend)..."
for i in $(seq 1 60); do
  if [[ -s "${JUNCTION_LEGEND}" ]]; then
    break
  fi
  sleep 0.5
done
if [[ ! -s "${JUNCTION_LEGEND}" ]]; then
  echo "ERROR: ${JUNCTION_LEGEND} not created by controller. See ${CTRL_LOG}"
  exit 2
fi
log "Found ${JUNCTION_LEGEND}"

# Compose ns-3 command
NS3_ARGS=( ELTI -- \
  --junction-legend="${JUNCTION_LEGEND}" \
  --poisson-enable="${POISSON_ENABLE}" \
  --poisson-lambda="${POISSON_LAMBDA}" \
  --sumo-wait="${SUMO_WAIT}" \
  --sim-time="${SIM_TIME}" )

# Run ns-3 as non-root user
log "Running NS-3 (Poisson=${POISSON_ENABLE}, lambda=${POISSON_LAMBDA}, sim-time=${SIM_TIME}s) ..."
if [[ "${EUID}" -eq 0 ]]; then
  # shellcheck disable=SC2024
  sudo -u "${TARGET_USER}" -H env NS_LOG="${NS_LOG:-}" "${NS3_DIR}/ns3" run "${NS3_ARGS[@]}" 2>&1 | tee -a "${NS3_LOG}"
else
  env NS_LOG="${NS_LOG:-}" "${NS3_DIR}/ns3" run "${NS3_ARGS[@]}" 2>&1 | tee -a "${NS3_LOG}"
fi

log "NS-3 finished. Logs at ${NS3_LOG}"
log "Controller log at ${CTRL_LOG}"
