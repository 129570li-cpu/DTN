#!/usr/bin/env python3
"""
NS3-AI controller server for ELITE.
This module connects to ns-3 via ns3-ai (RL interface) and serves:
 - route requests: returns area_path and selected policyId (HRF/LDF/LBF)
 - path feedback: updates APN state-action table using paper's R0
It assumes:
 - SUMO net.xml is available to build topology (junction positions and adjacency)
 - Four single-target Q tables are trained (or will be trained on first run) and fusion is ready
"""
from __future__ import annotations
import os
import sys
import time
import signal
import atexit
from typing import List, Dict, Tuple, Optional
import threading
import math
import copy
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

# Use ns3-ai Python interface (ctypes-based)
try:
    from ctypes import Structure, c_double, c_int, sizeof
    # Import Init so we can initialize shm pool when not spawning ns-3 via Experiment
    from py_interface import Ns3AIRL, Experiment, EmptyInfo  # type: ignore
    from py_interface import Init as Ns3AIInit  # type: ignore
except Exception:
    Ns3AIRL = None
    Ns3AIInit = None

import Global_Par as Gp
from topology.sumo_net import parse_sumo_net
from topology.fcd_speed import build_past_present_future_speed_maps, build_past_present_future_density_maps
from Routing_table import Routing_Table
from SDVN_Controller import SDVNController
from apn import aggregate_reward, compute_load_level
from dtn.train import train_and_export

# Constants for observation/action shapes (fix-sized arrays for ns3-ai shared memory)
MAX_NEIGHBORS = 64
MAX_PATH_LEN = 32

# Debug printing gate (set ELITE_DEBUG=1 to enable verbose controller logs)
DEBUG = os.environ.get("ELITE_DEBUG", "0") == "1"

def _ctrl_log(msg: str) -> None:
    """Emit controller-specific log lines (always flushed)."""
    try:
        print(f"[CTRL] {msg}", flush=True)
    except Exception:
        pass

def dlog(msg: str) -> None:
    if DEBUG:
        try:
            print(msg, flush=True)
        except Exception:
            pass

def _install_signal_handlers() -> None:
    """Log when the controller receives termination signals (helps diagnose abrupt exits)."""
    def _handler(signum, _frame):
        _ctrl_log(f"Received signal {signum}. Preparing to shut down.")
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass
    atexit.register(lambda: _ctrl_log("Controller exiting."))

# Online Q update defaults (enable timer-based online update every 5s by default)
ONLINE_Q_UPDATE_PERIOD_S = float(os.environ.get("ELITE_ONLINE_Q_S", "5"))
ONLINE_Q_ALPHA = float(os.environ.get("ELITE_ONLINE_Q_ALPHA", "0.05"))
ONLINE_Q_GAMMA = 0.10
# Enable periodic SPL retraining by default (seconds). Set to 0 to disable.
SPL_PERIOD_S = float(os.environ.get("SPL_PERIOD_S", "600"))

def prepare_topology_and_policies(net_xml_path: str,
                                  qtable_dir: str,
                                  retrain: bool = True,
                                  comm_radius: float = 300.0,
                                  veh_speed: float = 23.0) -> Tuple[Routing_Table, SDVNController]:
    # Build topology from SUMO net (with lane speeds if available)
    it_pos, adj, road_len, speed_map = parse_sumo_net(net_xml_path, skip_internal=True)
    Gp.it_pos = it_pos
    Gp.adjacents_comb = adj
    Gp.adjacency_dis = road_len
    try:
        Gp.speed_map = speed_map
    except Exception:
        pass
    # Export junction legend for ns-3 (CSV: id,x,y)
    try:
        os.makedirs(qtable_dir, exist_ok=True)
        legend_path = os.path.join(qtable_dir, "junction_legend.csv")
        with open(legend_path, "w") as f:
            for jid, (x, y) in it_pos.items():
                f.write(f"{jid},{x},{y}\n")
    except Exception as e:
        print(f"[WARN] Failed to write junction legend: {e}", file=sys.stderr)

    # Train single-target Q tables (optional, or reuse cached)
    if retrain:
        paths = train_and_export(qtable_dir, episodes=8000, seed=1, it_pos=it_pos, adj=adj,
                                 speed_map=speed_map, comm_radius=comm_radius, veh_speed=veh_speed)
        if os.path.exists(paths["pdr"]):
            Gp.file_pdr = paths["pdr"]
            Gp.file_ad  = paths["ad"]
            Gp.file_hc  = paths["hc"]
            Gp.file_rc  = paths["rc"]

    # Helper to load routing table from current Gp.file_* pointers
    def build_rt() -> Routing_Table:
        rt = Routing_Table()
        rt.preprocessing()
        rt.fusion_weight()
        rt.fusion_fuzzy()
        return rt
    rt_present = build_rt()
    # Controller: node_num placeholder 0, intersection it_pos injected
    ctrl = SDVNController(node_num=0, intersection=it_pos)
    # Keep a temporal set holder on controller (present-only by default)
    ctrl.routing_tables = {"present": rt_present}
    # Default active table
    return rt_present, ctrl
def _matrix_to_Q(matrix_path: str, it_pos: Dict[int, Tuple[float, float]], adj: Dict[int, List[int]]) -> Dict[int, Dict[int, Dict[int, float]]]:
    """
    Load a CSV matrix (rows=(cur,neib) flattened, cols=dest ids in it_pos order)
    into Q dict: Q[dest][current][neighbor] = value
    """
    rows: List[List[float]] = []
    try:
        with open(matrix_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if not parts or parts == [""]:
                    continue
                try:
                    rows.append([float(x) for x in parts])
                except Exception:
                    rows.append([0.0 for _ in parts])
    except Exception:
        pass
    dests = list(it_pos.keys())
    Q: Dict[int, Dict[int, Dict[int, float]]] = {}
    ridx = 0
    for cur, neibs in adj.items():
        for neib in neibs:
            if ridx >= len(rows):
                break
            row = rows[ridx] if ridx < len(rows) else []
            for j, d in enumerate(dests):
                v = float(row[j]) if j < len(row) else 0.0
                Q.setdefault(d, {}).setdefault(cur, {})[neib] = v
            ridx += 1
    return Q

def _Q_to_matrix_csv(Q: Dict[int, Dict[int, Dict[int, float]]],
                     it_pos: Dict[int, Tuple[float, float]],
                     adj: Dict[int, List[int]],
                     out_path: str) -> None:
    """
    Write Q dict back to CSV with the same shape/order as train_and_export.
    """
    dests = list(it_pos.keys())
    with open(out_path, "w") as f:
        for cur, neibs in adj.items():
            for neib in neibs:
                vals: List[str] = []
                for d in dests:
                    v = float(Q.get(d, {}).get(cur, {}).get(neib, 0.0))
                    vals.append(str(v))
                f.write(",".join(vals) + "\n")

class LoadEstimator:
    def __init__(self):
        self.lock = threading.Lock()
        self.hist: Dict[Tuple[int,int], float] = {}  # (src,dst)->Lavg_bps (EMA)

    def update(self, src: int, dst: int, lavg_bps: float, alpha: float = 0.5):
        if lavg_bps <= 0:
            return
        with self.lock:
            prev = self.hist.get((src,dst), lavg_bps)
            self.hist[(src,dst)] = alpha * lavg_bps + (1.0 - alpha) * prev

    def get(self, src: int, dst: int) -> Optional[float]:
        with self.lock:
            return self.hist.get((src,dst))

def run_server(net_xml_path: str, qtable_dir: str, bandwidth_bps: float, comm_radius: float,
               mempool_key: int, mem_size: int, memblock_key: int, ns3_cmd: str = None,
               spl_period_s: int = 0):
    """
    Run ELITE controller with ns3-ai shared memory (RL).
    If ns3_cmd is provided, will spawn ns-3 via Experiment; otherwise assumes ns-3 already running and registered the memory block.
    """
    if Ns3AIRL is None:
        print("py_interface (ns3-ai) module not found. Please ensure contrib/ns3-ai/py_interface is installed. Exiting.")
        sys.exit(1)
    rt, ctrl = prepare_topology_and_policies(net_xml_path, qtable_dir, retrain=True, comm_radius=comm_radius, veh_speed=23.0)
    # Set constants in Global_Par
    Gp.comm_radius = comm_radius
    Gp.bandwidth_bps = bandwidth_bps

    # Define ctypes structures matching the layout
    class Env(Structure):
        _pack_ = 1
        _fields_ = [
            ('simTime', c_double),
            ('vehicleId', c_int),
            ('srcId', c_int),
            ('dstId', c_int),
            ('msgTypeId', c_int),
            ('policyIdUsed', c_int),
            ('queueLen', c_double),
            ('bufferBytes', c_double),
            ('neighborCount', c_int),
            ('neighbors', c_int * MAX_NEIGHBORS),
            ('distToNext', c_double * MAX_NEIGHBORS),
            ('routeRequestFlag', c_int),
            ('feedbackFlag', c_int),
            ('success', c_int),
            ('ADsd', c_double),
            ('HCsd', c_double),
            ('RCsd', c_double),
            ('hopCount', c_double),
            ('PL', c_double),
            ('RO', c_double),
            ('nsd', c_int),
            ('seg_ADca', c_double * MAX_PATH_LEN),
            ('seg_HCca', c_double * MAX_PATH_LEN),
            ('seg_lca', c_double * MAX_PATH_LEN),
            ('seg_RCca', c_double * MAX_PATH_LEN),
            ('path_len', c_int),
            ('path_ids', c_int * MAX_PATH_LEN),
        ]
    class Act(Structure):
        _pack_ = 1
        _fields_ = [
            ('policyId', c_int),
            ('path_len', c_int),
            ('path_ids', c_int * MAX_PATH_LEN),
        ]
    # Launch ns-3 (optional)
    exp = None
    if ns3_cmd:
        exp = Experiment(mempool_key, mem_size, ns3_cmd, os.getcwd())
        exp.reset()
        pro = exp.run()
    def _init_ns3ai_rl():
        if Ns3AIInit is None or Ns3AIRL is None:
            print("py_interface (ns3-ai) module not found. Cannot initialize shared memory.", file=sys.stderr)
            sys.exit(1)
        try:
            Ns3AIInit(mempool_key, int(mem_size))
        except Exception as e:
            print(f"Failed to initialize ns3-ai shared memory pool (key={mempool_key}, size={mem_size}): {e}",
                  file=sys.stderr)
            sys.exit(1)
        try:
            rl_obj = Ns3AIRL(memblock_key, Env, Act)
        except RuntimeError as e:
            print(f"Failed to register ns3-ai memblock (key={memblock_key}). "
                  f"Hint: cleanup via 'ipcrm -M 0x{mempool_key:08x}; ipcrm -M 0x{memblock_key:08x}'. "
                  f"Reason: {e}", file=sys.stderr)
            sys.exit(1)
        rl_obj.SetCond(2, 1)  # python: odd phase, ns-3: even
        dlog(f"[NS3-AI] mempool_key={mempool_key} mem_size={mem_size} "
             f"memblock_key={memblock_key} env_bytes={sizeof(Env)} act_bytes={sizeof(Act)}")
        return rl_obj

    # Connect RL shared memory (always initialize from controller side to guarantee memblock exists)
    rl = _init_ns3ai_rl()
    print("ELITE controller server started (ns3-ai). Waiting for requests...")
    # Policy/RT shared objects and lock
    policy_lock = threading.Lock()
    load_est = LoadEstimator()

    # Optional background SPL periodic retraining (including temporal DTNs)
    stop_flag = threading.Event()
    # Raw Q tables for online update; initialize from current CSVs
    q_lock = threading.Lock()
    raw_Q: Dict[str, Dict[int, Dict[int, Dict[int, float]]]] = {}
    try:
        raw_Q["pdr"] = _matrix_to_Q(Gp.file_pdr, Gp.it_pos, Gp.adjacents_comb)
        raw_Q["ad"]  = _matrix_to_Q(Gp.file_ad,  Gp.it_pos, Gp.adjacents_comb)
        raw_Q["hc"]  = _matrix_to_Q(Gp.file_hc,  Gp.it_pos, Gp.adjacents_comb)
        raw_Q["rc"]  = _matrix_to_Q(Gp.file_rc,  Gp.it_pos, Gp.adjacents_comb)
    except Exception as e:
        print(f"[WARN] Failed to load raw Q tables for online update: {e}", file=sys.stderr)
        raw_Q = {"pdr": {}, "ad": {}, "hc": {}, "rc": {}}
    def _scale_speed_map(spm: Dict[int, Dict[int, float]], factor: float) -> Dict[int, Dict[int, float]]:
        out: Dict[int, Dict[int, float]] = {}
        for u, m in (spm or {}).items():
            out[u] = {}
            for v, s in m.items():
                out[u][v] = max(0.1, float(s) * factor)
        return out
    def spl_worker():
        # Periodically retrain Q-tables and refresh routing table/policies
        # This simulates DTN 并行在线学习
        while not stop_flag.wait(timeout=max(1, spl_period_s)):
            try:
                # rebuild speed_map by parsing net again (static over time for most SUMO nets)
                try:
                    _, _, _, spm = parse_sumo_net(net_xml_path, skip_internal=True)
                except Exception:
                    spm = {}
                # If FCD exists next to net, use it to build past/present/future maps (perfect-forecast assumption)
                fcd_path_guess = os.environ.get("ELITE_FCD_PATH")
                if fcd_path_guess and not os.path.exists(fcd_path_guess):
                    print(f"[SPL] ELITE_FCD_PATH={fcd_path_guess} not found, ignoring", file=sys.stderr)
                    fcd_path_guess = None
                if not fcd_path_guess:
                    try:
                        # derive fcd path by replacing filename in same folder if net follows "*grid.net.xml" style
                        import os
                        folder = os.path.dirname(net_xml_path)
                        candidates = ["grid_fcd.out.xml", "fcd.out.xml", "netstate.xml"]
                        for nm in candidates:
                            p = os.path.join(folder, nm)
                            if os.path.exists(p) and p.endswith(".xml"):
                                fcd_path_guess = p
                                break
                    except Exception:
                        fcd_path_guess = None
                if fcd_path_guess:
                    try:
                        pa_spm, pr_spm, fu_spm = build_past_present_future_speed_maps(fcd_path_guess, net_xml_path)
                        pa_den, pr_den, fu_den = build_past_present_future_density_maps(fcd_path_guess, net_xml_path)
                        if pr_spm:
                            spm = pr_spm
                    except Exception as _:
                        pa_spm = {}; fu_spm = {}; pa_den = {}; pr_den = {}; fu_den = {}
                else:
                    pa_spm = {}; fu_spm = {}; pa_den = {}; pr_den = {}; fu_den = {}
                # Train present/past/future:
                # present: spm; past: pa_spm (if empty → scale 0.85); future: fu_spm (if empty → scale 1.15)
                seeds = int(time.time()) & 0xffff
                with q_lock:
                    warm_start_snapshot = copy.deepcopy(raw_Q)
                # present
                p_paths = train_and_export(qtable_dir, episodes=4000, seed=seeds,
                                           it_pos=Gp.it_pos, adj=Gp.adjacents_comb,
                                           speed_map=spm, density_map=pr_den if pr_den else None,
                                           comm_radius=Gp.comm_radius, veh_speed=23.0,
                                           init_Q=warm_start_snapshot)
                # Swap in to build RT
                if os.path.exists(p_paths.get("pdr","")):
                    Gp.file_pdr = p_paths["pdr"]; Gp.file_ad = p_paths["ad"]; Gp.file_hc = p_paths["hc"]; Gp.file_rc = p_paths["rc"]
                    # refresh raw Q so online updates continue from new baseline
                    new_q_present = {
                        "pdr": _matrix_to_Q(p_paths["pdr"], Gp.it_pos, Gp.adjacents_comb),
                        "ad":  _matrix_to_Q(p_paths["ad"],  Gp.it_pos, Gp.adjacents_comb),
                        "hc":  _matrix_to_Q(p_paths["hc"],  Gp.it_pos, Gp.adjacents_comb),
                        "rc":  _matrix_to_Q(p_paths["rc"],  Gp.it_pos, Gp.adjacents_comb),
                    }
                    with q_lock:
                        raw_Q.update(new_q_present)
                rt_present = Routing_Table(); rt_present.preprocessing(); rt_present.fusion_weight(); rt_present.fusion_fuzzy()
                # past
                past_spm = pa_spm if pa_spm else _scale_speed_map(spm, 0.85)
                pa_paths = train_and_export(qtable_dir, episodes=3000, seed=seeds ^ 0x1111,
                                            it_pos=Gp.it_pos, adj=Gp.adjacents_comb,
                                            speed_map=past_spm, density_map=pa_den if pa_den else None,
                                            comm_radius=Gp.comm_radius, veh_speed=23.0,
                                            init_Q=warm_start_snapshot)
                if os.path.exists(pa_paths.get("pdr","")):
                    Gp.file_pdr = pa_paths["pdr"]; Gp.file_ad = pa_paths["ad"]; Gp.file_hc = pa_paths["hc"]; Gp.file_rc = pa_paths["rc"]
                rt_past = Routing_Table(); rt_past.preprocessing(); rt_past.fusion_weight(); rt_past.fusion_fuzzy()
                # future
                fut_spm = fu_spm if fu_spm else _scale_speed_map(spm, 1.15)
                f_paths = train_and_export(qtable_dir, episodes=3000, seed=seeds ^ 0x2222,
                                           it_pos=Gp.it_pos, adj=Gp.adjacents_comb,
                                           speed_map=fut_spm, density_map=fu_den if fu_den else None,
                                           comm_radius=Gp.comm_radius, veh_speed=23.0,
                                           init_Q=warm_start_snapshot)
                if os.path.exists(f_paths.get("pdr","")):
                    Gp.file_pdr = f_paths["pdr"]; Gp.file_ad = f_paths["ad"]; Gp.file_hc = f_paths["hc"]; Gp.file_rc = f_paths["rc"]
                rt_future = Routing_Table(); rt_future.preprocessing(); rt_future.fusion_weight(); rt_future.fusion_fuzzy()
                dlog(f"[REQ] t={simTime:.2f} veh={vehicleId} src={srcId} dst={dstId} "
                     f"msgType={msgTypeId} queueLen={queueLen:.1f} bufferBytes={bufferBytes:.0f} nei={neighborCount}")
                with policy_lock:
                    # hot-swap routing tables inside controller
                    ctrl.routing_tables = {"present": rt_present, "past": rt_past, "future": rt_future}
                    ctrl.routing_table = rt_present  # keep default
            except Exception as ex:
                print(f"[SPL] retrain failed: {ex}", file=sys.stderr)

    worker = None
    if spl_period_s and spl_period_s > 0:
        worker = threading.Thread(target=spl_worker, daemon=True)
        worker.start()

    # Online Q update machinery: buffer feedbacks and update every ONLINE_Q_UPDATE_PERIOD_S
    feedback_buf_lock = threading.Lock()
    feedback_buf: List[Dict] = []
    mad_max_online: Dict[Tuple[int,int], float] = {}

    def queue_feedback_sample(sample: Dict) -> None:
        with feedback_buf_lock:
            feedback_buf.append(sample)

    def rebuild_and_hot_swap() -> None:
        try:
            with q_lock:
                _Q_to_matrix_csv(raw_Q.get("pdr", {}), Gp.it_pos, Gp.adjacents_comb, Gp.file_pdr)
                _Q_to_matrix_csv(raw_Q.get("ad",  {}), Gp.it_pos, Gp.adjacents_comb, Gp.file_ad)
                _Q_to_matrix_csv(raw_Q.get("hc",  {}), Gp.it_pos, Gp.adjacents_comb, Gp.file_hc)
                _Q_to_matrix_csv(raw_Q.get("rc",  {}), Gp.it_pos, Gp.adjacents_comb, Gp.file_rc)
            rt_new = Routing_Table(); rt_new.preprocessing(); rt_new.fusion_weight(); rt_new.fusion_fuzzy()
            with policy_lock:
                rts = getattr(ctrl, "routing_tables", {})
                rts["present"] = rt_new
                ctrl.routing_tables = rts
                ctrl.routing_table = rt_new
        except Exception as ex:
            print(f"[OnlineQ] rebuild failed: {ex}", file=sys.stderr)

    def online_q_worker():
        while not stop_flag.wait(timeout=max(0.1, ONLINE_Q_UPDATE_PERIOD_S)):
            try:
                with feedback_buf_lock:
                    batch = feedback_buf[:]
                    feedback_buf.clear()
                if not batch:
                    continue
                for s in batch:
                    if not s.get("success", 0):
                        continue
                    path_ids: List[int] = s.get("path_ids", [])
                    if len(path_ids) < 2:
                        continue
                    d = int(path_ids[-1])
                    srcJ = int(path_ids[0])
                    ADsd = float(s.get("ADsd", 0.0))
                    HCsd = float(s.get("HCsd", 0.0))
                    RCsd = float(s.get("RCsd", 0.0))
                    seg_ADca: List[float] = s.get("seg_ADca", [])
                    seg_HCca: List[float] = s.get("seg_HCca", [])
                    seg_lca:  List[float] = s.get("seg_lca",  [])
                    seg_RCca: List[float] = s.get("seg_RCca", [])
                    nsd = min(len(path_ids)-1, len(seg_ADca), len(seg_HCca), len(seg_lca), len(seg_RCca))
                    if nsd <= 0:
                        continue
                    key = (srcJ, d)
                    prev = mad_max_online.get(key, 0.0)
                    mADsd = ADsd if ADsd > prev else prev
                    if ADsd > prev:
                        mad_max_online[key] = ADsd
                    for i in range(nsd):
                        c = int(path_ids[i]); a = int(path_ids[i+1])
                        ADca = float(seg_ADca[i]); HCca = float(seg_HCca[i]); lca = max(float(seg_lca[i]), 1e-9); RCca = float(seg_RCca[i])
                        # Rewards per paper
                        R_pdr = 1.0 if i == nsd - 1 else 0.0
                        ADsd_eff = max(ADsd, 1e-9)
                        R_ad = 1.0 - (ADca / ADsd_eff) * (1.0 - min(ADsd_eff / max(mADsd, ADsd_eff), 1.0))
                        comm_r = max(1.0, Gp.comm_radius)
                        R_hc = math.exp(- HCca * comm_r / lca)
                        RCsd_eff = max(RCsd, 1e-9)
                        R_rc = 1.0 / (1.0 + RCca / RCsd_eff)
                        # next best at a
                        neibs_a = Gp.adjacents_comb.get(a, [])
                        def _next_best(Qd: Dict[int, Dict[int, float]]) -> float:
                            if not neibs_a:
                                return 0.0
                            best = None
                            for n in neibs_a:
                                v = float(Qd.get(a, {}).get(n, 0.0))
                                if best is None or v > best:
                                    best = v
                            return 0.0 if best is None else float(best)
                        with q_lock:
                            # ensure keys exist
                            for key_name in ("pdr","ad","hc","rc"):
                                raw_Q.setdefault(key_name, {}).setdefault(d, {}).setdefault(c, {})
                                raw_Q[key_name][d][c].setdefault(a, 0.0)
                            # PDR
                            q_old = raw_Q["pdr"][d][c][a]
                            nb = _next_best(raw_Q["pdr"].get(d, {}))
                            raw_Q["pdr"][d][c][a] = q_old + ONLINE_Q_ALPHA * (R_pdr + ONLINE_Q_GAMMA * nb - q_old)
                            # AD
                            q_old = raw_Q["ad"][d][c][a]
                            nb = _next_best(raw_Q["ad"].get(d, {}))
                            raw_Q["ad"][d][c][a] = q_old + ONLINE_Q_ALPHA * (R_ad + ONLINE_Q_GAMMA * nb - q_old)
                            # HC
                            q_old = raw_Q["hc"][d][c][a]
                            nb = _next_best(raw_Q["hc"].get(d, {}))
                            raw_Q["hc"][d][c][a] = q_old + ONLINE_Q_ALPHA * (R_hc + ONLINE_Q_GAMMA * nb - q_old)
                            # RC
                            q_old = raw_Q["rc"][d][c][a]
                            nb = _next_best(raw_Q["rc"].get(d, {}))
                            raw_Q["rc"][d][c][a] = q_old + ONLINE_Q_ALPHA * (R_rc + ONLINE_Q_GAMMA * nb - q_old)
                rebuild_and_hot_swap()
            except Exception as ex:
                print(f"[OnlineQ] update failed: {ex}", file=sys.stderr)

    online_worker = None
    if ONLINE_Q_UPDATE_PERIOD_S and ONLINE_Q_UPDATE_PERIOD_S > 0.0:
        online_worker = threading.Thread(target=online_q_worker, daemon=True)
        online_worker.start()
    try:
        # Track mADsd (moving max) per (src,dst)
        mad_max: Dict[Tuple[int,int], float] = {}
        # Policy weights for R0 (paper mapping: LDF emphasizes AD/HC; HRF emphasizes AD/RC; LBF emphasizes RC/AD/HC)
        policy_weights = {
            "HRF": (0.25, 0.0, 0.25),   # (a=AD, b=HC, gamma=RC)
            "LDF": (0.5,  0.25, 0.0),
            "LBF": (0.25, 0.25, 0.5),
        }
        # Prefer starting controller first: block-wait until ns-3 connects.
        handshake_done = False
        while True:
            # If ns-3 has already interacted at least once and then finished, exit.
            if handshake_done and rl.isFinish():
                break
            with rl as data:
                if data is None:
                    # ns-3 not yet connected or between phases; do not exit, just wait.
                    time.sleep(0.05)
                    continue
                # Mark that at least one successful handshake occurred
                handshake_done = True
                e = data.env
                # Unpack
                simTime = e.simTime
                vehicleId = int(e.vehicleId)
                srcId = int(e.srcId)
                dstId = int(e.dstId)
                msgTypeId = int(e.msgTypeId)
                policyIdUsed = int(e.policyIdUsed)
                queueLen = e.queueLen
                bufferBytes = e.bufferBytes
                neighborCount = int(e.neighborCount)
                neighbors = [int(e.neighbors[i]) for i in range(min(neighborCount, MAX_NEIGHBORS))]
                distToNext = [float(e.distToNext[i]) for i in range(min(neighborCount, MAX_NEIGHBORS))]
                routeRequestFlag = int(e.routeRequestFlag)
                feedbackFlag = int(e.feedbackFlag)
                successFlag = int(e.success)
                ADsd = float(e.ADsd)
                HCsd = float(e.HCsd)
                RCsd = float(e.RCsd)
                hopCount = float(e.hopCount)
                PL = float(e.PL)
                RO = float(e.RO)
                nsd = int(e.nsd)
                seg_ADca = [float(e.seg_ADca[i]) for i in range(min(nsd, MAX_PATH_LEN))]
                seg_HCca = [float(e.seg_HCca[i]) for i in range(min(nsd, MAX_PATH_LEN))]
                seg_lca  = [float(e.seg_lca[i]) for i in range(min(nsd, MAX_PATH_LEN))]
                seg_RCca = [float(e.seg_RCca[i]) for i in range(min(nsd, MAX_PATH_LEN))]
                path_len_in = int(e.path_len)
                path_ids_in = [int(e.path_ids[i]) for i in range(min(path_len_in, MAX_PATH_LEN))]

            # Map msgTypeId to names
            msg_map = {0: "security", 1: "efficiency", 2: "information", 3: "entertainment"}
            message_type = msg_map.get(msgTypeId, "efficiency")

            # Feedback handling → APN update
            if feedbackFlag == 1:
                dlog(f"[FB] t={simTime:.2f} veh={vehicleId} src={srcId} dst={dstId} "
                     f"succ={successFlag} ADsd={ADsd:.3f} HCsd={HCsd:.3f} RCsd={RCsd:.3f} nsd={nsd} "
                     f"queueLen={queueLen:.1f} bufferBytes={bufferBytes:.0f}")
                # Compute R0 with paper formula; determine weights by last policy used if available in input (not included → default per message type)
                # 优先使用实际下发的策略编号；否则按消息类型推断
                if policyIdUsed == 1:
                    chosen_policy = "HRF"
                elif policyIdUsed == 2:
                    chosen_policy = "LDF"
                elif policyIdUsed == 3:
                    chosen_policy = "LBF"
                else:
                    chosen_policy = "LDF" if message_type == "efficiency" else ("HRF" if message_type in ("security","information") else "LBF")
                a, b, gamma = policy_weights[chosen_policy]
                # Compute mADsd (moving max per src/dst)
                sd_key = (srcId, dstId)
                prev = mad_max.get(sd_key, 0.0)
                mADsd = ADsd if ADsd > prev else prev
                if ADsd > prev:
                    mad_max[sd_key] = ADsd
                R0 = aggregate_reward(ADsd, mADsd, comm_radius, seg_ADca[:nsd], seg_HCca[:nsd], seg_lca[:nsd], seg_RCca[:nsd], RCsd, a, b, gamma)
                # Compute Lavg over path (bits/sec) using per-seg metrics + path-level control overhead (3 control pkts * 256B)
                DATA_B = 1024.0; CTRL_B = 256.0
                total_bytes = 3.0 * CTRL_B  # req/reply/report
                total_time = max(ADsd, 1e-6)
                for i in range(nsd):
                    hops_i = seg_HCca[i] if i < len(seg_HCca) else 0.0
                    ctrl_i = seg_RCca[i] if i < len(seg_RCca) else 0.0
                    t_i = seg_ADca[i] if i < len(seg_ADca) else 0.0
                    if t_i <= 0.0:
                        continue
                    data_b = hops_i * DATA_B
                    ctrl_b = ctrl_i * CTRL_B
                    total_bytes += (data_b + ctrl_b)
                Lavg_bps = (total_bytes / total_time) * 8.0  # to bits/s
                load_est.update(srcId, dstId, Lavg_bps)
                # Load level estimation strictly per paper: 0:[0,0.3), 1:[0.3,0.7], 2:(0.7,1]
                _, load_level = compute_load_level([Lavg_bps], bandwidth_bps)
                # Update APN table
                with policy_lock:
                    ctrl.state_action.update((message_type, load_level), chosen_policy, R0)
                # Queue for online Q update (success only to match SPL)
                try:
                    sample = {
                        "success": int(successFlag),
                        "path_ids": path_ids_in[:max(0, path_len_in)],
                        "ADsd": ADsd, "HCsd": HCsd, "RCsd": RCsd,
                        "seg_ADca": seg_ADca[:nsd],
                        "seg_HCca": seg_HCca[:nsd],
                        "seg_lca":  seg_lca[:nsd],
                        "seg_RCca": seg_RCca[:nsd],
                    }
                    queue_feedback_sample(sample)
                except Exception:
                    pass

            # Route request handling
            if routeRequestFlag == 1:
                dlog(f"[REQ] t={simTime:.2f} veh={vehicleId} src={srcId} dst={dstId} "
                     f"msgType={msgTypeId} bufB={bufferBytes:.0f} nei={neighborCount} "
                     f"queueLen={queueLen:.1f} bufferBytes={bufferBytes:.0f}")
                # Strictly follow paper: first compute baseline path Lbas using Gbas (BP), then derive load level from Lavg/Bw
                try:
                    # Require it_cover to be set; if not, skip baseline estimation
                    # Here we assume srcId/dstId map to junction IDs directly; adapt mapping as needed in your environment.
                    with policy_lock:
                        # baseline always uses present tables for load estimation
                        active_rt = getattr(ctrl, "routing_tables", {}).get("present", ctrl.routing_table)
                        bp_table = active_rt.table_BP
                        # if ns-3 provided nearest junctions in path_ids[0..1], prefer them
                        srcJ = None; dstJ = None
                        if path_len_in >= 2:
                            srcJ = int(path_ids_in[0]); dstJ = int(path_ids_in[1])
                        # temporarily forge it_cover to use specified junctions
                        if srcJ is not None and dstJ is not None:
                            # populate minimal it_cover entries
                            ctrl.it_cover[srcId] = (srcJ,)
                            ctrl.it_cover[dstId] = (dstJ,)
                            area_path_bas = ctrl.calculate_area_path_with_table(srcId, dstId, bp_table)
                        else:
                            area_path_bas = ctrl.calculate_area_path_with_table(srcId, dstId, bp_table)
                    # Estimate Lavg on Lbas using segment-level approximations (like feedback path)
                    DATA_B = 1024.0; CTRL_B = 256.0
                    total_bytes = 3.0 * CTRL_B
                    total_time = 0.0
                    comm_r = max(1.0, Gp.comm_radius)
                    # speed map from SUMO if available
                    spm = getattr(Gp, "speed_map", {})
                    for i in range(1, len(area_path_bas)):
                        u = int(area_path_bas[i-1]); v = int(area_path_bas[i])
                        lca = float(Gp.adjacency_dis.get(u, {}).get(v, 0.0))
                        if lca <= 0.0:
                            # fallback euclid
                            try:
                                ux, uy = Gp.it_pos[u]; vx, vy = Gp.it_pos[v]
                                dx = ux - vx; dy = uy - vy
                                lca = (dx*dx + dy*dy) ** 0.5
                            except Exception:
                                lca = 1.0
                        vms = float(spm.get(u, {}).get(v, 0.0)) if spm else 0.0
                        if vms <= 0.0:
                            vms = 23.0
                        t_i = lca / max(1e-6, vms)
                        hops_i = max(1, int(lca / comm_r + 0.9999))
                        ctrl_i = float(hops_i)
                        total_time += t_i
                        total_bytes += (hops_i * DATA_B + ctrl_i * CTRL_B)
                    Lavg_bps_req = (total_bytes / max(total_time, 1e-6)) * 8.0
                    _, load_level = compute_load_level([Lavg_bps_req], bandwidth_bps)
                except Exception:
                    # fallback to historical estimator or current buffer
                    lv = load_est.get(srcId, dstId)
                    if lv is None:
                        _, load_level = compute_load_level([bufferBytes], bandwidth_bps)
                    else:
                        _, load_level = compute_load_level([lv], bandwidth_bps)
                # Calculate area path by APN-selected policy
                with policy_lock:
                    # Choose temporal RT set for final path: high-load→past, low-load→future, else present
                    rts = getattr(ctrl, "routing_tables", {})
                    if load_level >= 2 and "past" in rts:
                        ctrl.routing_table = rts["past"]
                    elif load_level == 0 and "future" in rts:
                        ctrl.routing_table = rts["future"]
                    else:
                        ctrl.routing_table = rts.get("present", ctrl.routing_table)
                    area_path = ctrl.calculate_area_path(srcId, dstId, load_level=load_level, message_type=message_type)
                    # Choose policy matching APN selection for policyId reporting
                    table = ctrl._select_policy_table(load_level=load_level, message_type=message_type)
                if table is ctrl.routing_table.table_HRF:
                    policyId = 1
                elif table is ctrl.routing_table.table_LDF:
                    policyId = 2
                elif table is ctrl.routing_table.table_LBF:
                    policyId = 3
                else:
                    policyId = 0
                dlog(f"[RSP] t={simTime:.2f} src={srcId} dst={dstId} loadLv={load_level} policy={policyId} pathLen={len(area_path)}")
                # Fill action back
                data.act.policyId = int(policyId)
                plen = min(len(area_path), MAX_PATH_LEN)
                data.act.path_len = int(plen)
                for i in range(plen):
                    data.act.path_ids[i] = int(area_path[i])
            # Do not call rl.Release() explicitly here; the context manager (with rl as data)
            # already releases the shared-memory lock. Double release would cause ns3-ai
            # to raise a Lock status error.
    finally:
        # Close ns-3 experiment if started
        if 'pro' in locals():
            pro.wait()
        if 'exp' in locals() and exp:
            del exp
        stop_flag.set()
        if 'worker' in locals() and worker:
            worker.join(timeout=1.0)
        if 'online_worker' in locals() and online_worker:
            online_worker.join(timeout=1.0)

if __name__ == "__main__":
    try:
        if len(sys.argv) < 8:
            print("Usage: controller_server.py <sumo_net.xml> <qtable_dir> <bandwidth_bps> <comm_radius> <mempool_key> <mem_size> <memblock_key> [ns3_cmd]")
            sys.exit(1)
        _install_signal_handlers()
        net_xml = sys.argv[1]
        qdir = sys.argv[2]
        bw = float(sys.argv[3])
        cr = float(sys.argv[4])
        mpk = int(sys.argv[5])
        msz = int(sys.argv[6])
        mbk = int(sys.argv[7])
        cmd = sys.argv[8] if len(sys.argv) >= 9 else None
        run_server(net_xml_path=net_xml, qtable_dir=qdir, bandwidth_bps=bw, comm_radius=cr,
                   mempool_key=mpk, mem_size=msz, memblock_key=mbk, ns3_cmd=cmd)
    except KeyboardInterrupt:
        # Graceful exit on Ctrl-C without traceback spam
        print("Ctrl-C received, stopping ELITE controller.")
        sys.exit(0)
    except Exception as exc:
        _ctrl_log(f"Fatal exception: {exc}")
        raise
