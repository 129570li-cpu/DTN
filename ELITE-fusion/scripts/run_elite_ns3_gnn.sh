#!/usr/bin/env bash
set -euo pipefail

NS3_DIR="${NS3_DIR:-/home/Lyh/VaN3Twin/ns-3-dev}"
CTRL_PY="${CTRL_PY:-/home/Lyh/ELITE-zg-main/ELITE-fusion/ctrl_work/controller_server.py}"
NET_XML="${NET_XML:-/home/Lyh/VaN3Twin/ns-3-dev/src/automotive/examples/grid_network_large/grid_network_fed/grid.net.xml}"
OUT_DIR="${OUT_DIR:-/home/Lyh/ELITE-zg-main/dtn_out}"
BASH_CTRL_PYTHON="${CTRL_PYTHON:-/home/Lyh/GNN/.venv/bin/python}"
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
SUMO_CONFIG="${SUMO_CONFIG:-${NS3_DIR}/src/automotive/examples/grid_network_large/grid_network_fed/grid.sumocfg}"
STATIONS_XML="${STATIONS_XML:-${NS3_DIR}/src/automotive/examples/grid_network_large/grid_network_fed/stations.xml}"
GNN_MODEL_DIR="${GNN_MODEL_DIR:-${OUT_DIR}/gnn_dqn_edge_mp/ldf}"
GNN_DEVICE="${GNN_DEVICE:-cuda}"  # GPU 推理更快
GNN_HIDDEN="${GNN_HIDDEN:-128}"
GNN_K_PATHS="${GNN_K_PATHS:-5}"   # 保持 5 条候选路径
ONLINE_UPDATE="${ONLINE_UPDATE:-0}"  # 关闭在线微调（实验阶段）
NODE_FEATURES_JSON="${NODE_FEATURES_JSON:-${OUT_DIR}/large_node_features_multi_edge.json}"
EDGE_MAP_JSON="${EDGE_MAP_JSON:-${OUT_DIR}/edge_id_map.json}"
WALL_TIMEOUT="${WALL_TIMEOUT:-$((SIM_TIME+120))}"

JUNCTION_LEGEND="${OUT_DIR}/junction_legend.csv"
CTRL_LOG="${OUT_DIR}/controller.log"
NS3_LOG="${NS3_DIR}/src/automotive/examples/ELTI/run_ELTI_auto.log"
NS3_LOG_OUT="${OUT_DIR}/ns3_run.log"
NS_GLOBAL_VALUE_STR="SharedMemoryKey=${MEMPOOL_KEY};SharedMemoryPoolSize=${MEM_SIZE};"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ echo "[FATAL] $*" >&2; exit 1; }
sanitize_nslog(){ local s="${1:-}"; [[ "$s" == *,* ]] && s="${s//,/:}"; echo "$s"; }
run_as_target(){ if [[ "${EUID}" -eq 0 ]]; then sudo -E -u "${TARGET_USER}" -H "$@"; else "$@"; fi }

[[ -f "${CTRL_PY}" ]] || die "Controller not found"
[[ -f "${NET_XML}" ]] || die "net.xml not found"
[[ -x "${NS3_DIR}/ns3" ]] || die "ns3 launcher missing"
mkdir -p "${OUT_DIR}"

ELITE_ROOT="/home/Lyh/ELITE-zg-main/ELITE-fusion"
export PYTHONPATH="${NS3_DIR}/contrib/ns3-ai/py_interface:${ELITE_ROOT}:${PYTHONPATH:-}"
export ELITE_DEBUG="${ELITE_DEBUG:-0}"
export ELITE_USE_GNN=1
export ELITE_GNN_MODEL_DIR="${GNN_MODEL_DIR}"
export ELITE_GNN_DEVICE="${GNN_DEVICE}"
export ELITE_GNN_HIDDEN="${GNN_HIDDEN}"
export ELITE_GNN_K_PATHS="${GNN_K_PATHS}"
export ELITE_ONLINE_UPDATE="${ONLINE_UPDATE}"
# 默认开启 ns-3 转发轨迹，便于每次实验留痕
export ELITE_NS3_TRACE="${ELITE_NS3_TRACE:-1}"
if [[ -n "${NODE_FEATURES_JSON}" && -f "${NODE_FEATURES_JSON}" ]]; then
  export ELITE_GNN_NODE_FEATS="${NODE_FEATURES_JSON}"
else
  unset ELITE_GNN_NODE_FEATS
fi
if [[ -n "${EDGE_MAP_JSON}" && -f "${EDGE_MAP_JSON}" ]]; then
  export ELITE_GNN_EDGE_MAP="${EDGE_MAP_JSON}"
else
  unset ELITE_GNN_EDGE_MAP
fi
if [[ -z "${SUMO_HOME:-}" ]]; then
  export SUMO_HOME="/usr/local/share/sumo"
fi

if [[ "${KILL_OLD}" == "1" ]]; then
  log "Killing leftover processes"
  pkill -f ns3-dev-ELTI-optimized 2>/dev/null || true
  pkill -f "sumo -c" 2>/dev/null || true
  pkill -f controller_server.py 2>/dev/null || true
  sleep 0.5
fi

if [[ "${CLEAN_SHM}" == "1" ]]; then
  printf -v MEMPOOL_HEX "0x%08x" "${MEMPOOL_KEY}"
  printf -v MEMBLOCK_HEX "0x%08x" "${MEMBLOCK_KEY}"
  ipcrm -M "${MEMPOOL_HEX}" 2>/dev/null || true
  ipcrm -M "${MEMBLOCK_HEX}" 2>/dev/null || true
fi

if id -u "${TARGET_USER}" >/dev/null 2>&1; then
  chown -R "${TARGET_USER}:${TARGET_USER}" "${OUT_DIR}" 2>/dev/null || true
fi
: > "${CTRL_LOG}" || true
: > "${NS3_LOG}" || true
: > "${NS3_LOG_OUT}" || true

log "Starting controller..."
set +e
run_as_target env PYTHONUNBUFFERED=1 "${BASH_CTRL_PYTHON}" "${CTRL_PY}" \
  "${NET_XML}" "${OUT_DIR}" "${BANDWIDTH_BPS}" "${COMM_RADIUS}" \
  "${MEMPOOL_KEY}" "${MEM_SIZE}" "${MEMBLOCK_KEY}" \
  > "${CTRL_LOG}" 2>&1 &
CTRL_PID=$!
set -e
trap 'if ps -p "${CTRL_PID}" >/dev/null; then kill "${CTRL_PID}" 2>/dev/null || true; wait "${CTRL_PID}" 2>/dev/null || true; fi' EXIT

log "Waiting for ${JUNCTION_LEGEND}"
SECONDS_WAITED=0
while [[ ! -s "${JUNCTION_LEGEND}" && "${SECONDS_WAITED}" -lt "${CTRL_READY_TIMEOUT}" ]]; do
  sleep 1
  SECONDS_WAITED=$((SECONDS_WAITED+1))
done
[[ -s "${JUNCTION_LEGEND}" ]] || die "controller failed to init"

NS3_CMD_ARGS=( --junction-legend="${JUNCTION_LEGEND}" --poisson-enable="${POISSON_ENABLE}" \
  --poisson-lambda="${POISSON_LAMBDA}" --sumo-wait="${SUMO_WAIT}" --sim-time="${SIM_TIME}" )
[[ -n "${SUMO_CONFIG:-}" ]] && NS3_CMD_ARGS+=( --sumo-config="${SUMO_CONFIG}" )
[[ -n "${STATIONS_XML:-}" ]] && NS3_CMD_ARGS+=( --rsu-stations="${STATIONS_XML}" )
[[ "${SUMO_STEP_LOG:-0}" == "1" ]] && NS3_CMD_ARGS+=( --sumo-step-log=1 )
SAFE_NS_LOG="$(sanitize_nslog "${NS_LOG:-}")"
NS3_LAUNCH=( "${NS3_DIR}/ns3" run ELTI -- "${NS3_CMD_ARGS[@]}" )

log "Running ns-3..."
set +e
if command -v timeout >/dev/null 2>&1 && [[ "${WALL_TIMEOUT}" -gt 0 ]]; then
  run_as_target env NS_LOG="${SAFE_NS_LOG}" NS_GLOBAL_VALUE="${NS_GLOBAL_VALUE_STR}" \
    timeout -k 10s "${WALL_TIMEOUT}"s \
    "${NS3_LAUNCH[@]}" 2>&1 | tee -a "${NS3_LOG}" | tee -a "${NS3_LOG_OUT}"
  NS3_EC=${PIPESTATUS[0]}
else
  run_as_target env NS_LOG="${SAFE_NS_LOG}" NS_GLOBAL_VALUE="${NS_GLOBAL_VALUE_STR}" \
    "${NS3_LAUNCH[@]}" 2>&1 | tee -a "${NS3_LOG}" | tee -a "${NS3_LOG_OUT}"
  NS3_EC=${PIPESTATUS[0]}
fi
set -e
if [[ "${NS3_EC}" -ne 0 ]]; then log "NS-3 exited with ${NS3_EC}"; else log "NS-3 finished"; fi
log "Controller log: ${CTRL_LOG}"
log "NS-3 logs: ${NS3_LOG}, ${NS3_LOG_OUT}"
