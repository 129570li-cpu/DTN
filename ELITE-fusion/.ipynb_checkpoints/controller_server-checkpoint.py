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
from typing import List, Dict, Tuple, Optional
import threading
import math

import numpy as np

# Use ns3-ai Python interface (ctypes-based)
try:
    from ctypes import Structure, c_double, c_int
    from py_interface import Ns3AIRL, Experiment, EmptyInfo  # type: ignore
except Exception:
    Ns3AIRL = None

import Global_Par as Gp
from topology.sumo_net import parse_sumo_net
from Routing_table import Routing_Table
from SDVN_Controller import SDVNController
from apn import aggregate_reward, compute_load_level
from dtn.train import train_and_export

# Constants for observation/action shapes (fix-sized arrays for ns3-ai shared memory)
MAX_NEIGHBORS = 64
MAX_PATH_LEN = 32

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
    rt, ctrl = prepare_topology_and_policies(net_xml_path, qtable_dir, retrain=False, comm_radius=comm_radius, veh_speed=23.0)
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
    # Connect RL shared memory
    rl = Ns3AIRL(memblock_key, Env, Act)
    rl.SetCond(2, 1)  # python: odd phase, ns-3: even
    print("ELITE controller server started (ns3-ai). Waiting for requests...")
    # Policy/RT shared objects and lock
    policy_lock = threading.Lock()
    load_est = LoadEstimator()

    # Optional background SPL periodic retraining (including temporal DTNs)
    stop_flag = threading.Event()
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
                # Train present/past/future with simple temporal surrogates:
                # present: raw spm; past: 0.85x (拥堵更严重)；future: 1.15x（通行改善）
                seeds = int(time.time()) & 0xffff
                # present
                p_paths = train_and_export(qtable_dir, episodes=4000, seed=seeds,
                                           it_pos=Gp.it_pos, adj=Gp.adjacents_comb,
                                           speed_map=spm, comm_radius=Gp.comm_radius, veh_speed=23.0)
                # Swap in to build RT
                if os.path.exists(p_paths.get("pdr","")):
                    Gp.file_pdr = p_paths["pdr"]; Gp.file_ad = p_paths["ad"]; Gp.file_hc = p_paths["hc"]; Gp.file_rc = p_paths["rc"]
                rt_present = Routing_Table(); rt_present.preprocessing(); rt_present.fusion_weight(); rt_present.fusion_fuzzy()
                # past
                past_spm = _scale_speed_map(spm, 0.85)
                pa_paths = train_and_export(qtable_dir, episodes=3000, seed=seeds ^ 0x1111,
                                            it_pos=Gp.it_pos, adj=Gp.adjacents_comb,
                                            speed_map=past_spm, comm_radius=Gp.comm_radius, veh_speed=23.0)
                if os.path.exists(pa_paths.get("pdr","")):
                    Gp.file_pdr = pa_paths["pdr"]; Gp.file_ad = pa_paths["ad"]; Gp.file_hc = pa_paths["hc"]; Gp.file_rc = pa_paths["rc"]
                rt_past = Routing_Table(); rt_past.preprocessing(); rt_past.fusion_weight(); rt_past.fusion_fuzzy()
                # future
                fut_spm = _scale_speed_map(spm, 1.15)
                f_paths = train_and_export(qtable_dir, episodes=3000, seed=seeds ^ 0x2222,
                                           it_pos=Gp.it_pos, adj=Gp.adjacents_comb,
                                           speed_map=fut_spm, comm_radius=Gp.comm_radius, veh_speed=23.0)
                if os.path.exists(f_paths.get("pdr","")):
                    Gp.file_pdr = f_paths["pdr"]; Gp.file_ad = f_paths["ad"]; Gp.file_hc = f_paths["hc"]; Gp.file_rc = f_paths["rc"]
                rt_future = Routing_Table(); rt_future.preprocessing(); rt_future.fusion_weight(); rt_future.fusion_fuzzy()
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

    try:
        # Track mADsd (moving max) per (src,dst)
        mad_max: Dict[Tuple[int,int], float] = {}
        # Policy weights for R0 (paper mapping: LDF emphasizes AD/HC; HRF emphasizes AD/RC; LBF emphasizes RC/AD/HC)
        policy_weights = {
            "HRF": (0.25, 0.0, 0.25),   # (a=AD, b=HC, gamma=RC)
            "LDF": (0.5,  0.25, 0.0),
            "LBF": (0.25, 0.25, 0.5),
        }
        while not rl.isFinish():
            with rl as data:
                if data is None:
                    break
                e = data.env
                # Unpack
                simTime = e.simTime
                vehicleId = int(e.vehicleId)
                srcId = int(e.srcId)
                dstId = int(e.dstId)
                msgTypeId = int(e.msgTypeId)
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
                # Compute R0 with paper formula; determine weights by last policy used if available in input (not included → default per message type)
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

            # Route request handling
            if routeRequestFlag == 1:
                # Strictly follow paper: first compute baseline path Lbas using Gbas (BP), then derive load level from Lavg/Bw
                try:
                    # Require it_cover to be set; if not, skip baseline estimation
                    # Here we assume srcId/dstId map to junction IDs directly; adapt mapping as needed in your environment.
                    with policy_lock:
                        # baseline always uses present tables for load estimation
                        active_rt = getattr(ctrl, "routing_tables", {}).get("present", ctrl.routing_table)
                        bp_table = active_rt.table_BP
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
                # Fill action back
                data.act.policyId = int(policyId)
                plen = min(len(area_path), MAX_PATH_LEN)
                data.act.path_len = int(plen)
                for i in range(plen):
                    data.act.path_ids[i] = int(area_path[i])
            rl.Release()
    finally:
        # Close ns-3 experiment if started
        if 'pro' in locals():
            pro.wait()
        if 'exp' in locals() and exp:
            del exp
        stop_flag.set()
        if 'worker' in locals() and worker:
            worker.join(timeout=1.0)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: controller_server.py <sumo_net.xml> <qtable_dir> <bandwidth_bps> <comm_radius> <mempool_key> <mem_size> <memblock_key> [ns3_cmd]")
        sys.exit(1)
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
