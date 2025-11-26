#!/usr/bin/env python3
"""
Evaluate trained Q-tables quality before connecting to ns-3.
Checks:
  - Basic stats (min/max/mean, nonzero ratio) for each metric table
  - Greedy success rate over random S-D pairs using exploitation-only policy
Usage:
  python3 eval_qtables.py --net /path/to/grid.net.xml --dir dtn_out --trials 200
"""
from __future__ import annotations
import argparse
import os
import random
from typing import Dict, List, Tuple
import math

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import Global_Par as Gp
from topology.sumo_net import parse_sumo_net
from Routing_table import Routing_Table
from dtn.env import DTNEnv, EnvConfig
import pandas as pd

def _load_rt(net_xml: str, qdir: str) -> Routing_Table:
    it_pos, adj, road_len, speed_map = parse_sumo_net(net_xml, skip_internal=True)
    Gp.it_pos = it_pos
    Gp.adjacents_comb = adj
    # point to files inside qdir
    Gp.file_pdr = os.path.join(qdir, "table_record_minidtn_pdr.csv")
    Gp.file_ad  = os.path.join(qdir, "table_record_minidtn_ad.csv")
    Gp.file_hc  = os.path.join(qdir, "table_record_minidtn_hc.csv")
    Gp.file_rc  = os.path.join(qdir, "table_record_minidtn_rc.csv")
    rt = Routing_Table()
    rt.preprocessing()
    return rt

def _stats(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    vals = df.to_numpy().ravel()
    nz = (vals != 0).sum()
    total = len(vals)
    nz_ratio = 0.0 if total == 0 else float(nz) / float(total)
    return float(vals.min(initial=0.0)), float(vals.max(initial=0.0)), float(vals.mean()), nz_ratio

def _build_env(net_xml: str) -> DTNEnv:
    it_pos, adj, _, speed_map = parse_sumo_net(net_xml, skip_internal=True)
    cfg = EnvConfig(seed=1, comm_radius=300.0, veh_speed=23.0)
    env = DTNEnv(it_pos, adj, cfg, speed_map=speed_map)
    return env

def _greedy_episode(env: DTNEnv,
                    Q: Dict[int, Dict[int, Dict[int, float]]],
                    s: int, d: int, max_steps: int = 128) -> Tuple[bool, List[Tuple[int,int]], Dict[str,float]]:
    cur = s
    visited = set([cur])
    traj: List[Tuple[int,int]] = []
    ad_sum = 0.0
    hc_sum = 0.0
    rc_sum = 0.0
    steps = 0
    while steps < max_steps and cur != d:
        steps += 1
        # Exploitation: argmax over neighbors
        neibs = env.adj.get(cur, [])
        if not neibs:
            break
        best_n = None
        best_q = -1e18
        for n in neibs:
            qv = Q.get(d, {}).get(cur, {}).get(n, 0.0)
            if qv > best_q:
                best_q = qv
                best_n = n
        nxt = best_n if best_n is not None else random.choice(neibs)
        traj.append((cur, nxt))
        ad_sum += env.seg_delay(cur, nxt)
        hc_sum += env.seg_hops(cur, nxt)
        rc_sum += env.seg_ctrl_overhead(cur, nxt)
        cur = nxt
        if cur in visited:
            break
        visited.add(cur)
    return (cur == d), traj, {"ADsd": ad_sum, "HCsd": float(hc_sum), "RCsd": rc_sum}

def _matrix_to_Q(df: pd.DataFrame, it_pos: Dict[int, Tuple[float,float]], adj: Dict[int, List[int]]) -> Dict[int, Dict[int, Dict[int, float]]]:
    # df has rows MultiIndex (cur, neib), columns dest
    Q: Dict[int, Dict[int, Dict[int, float]]] = {}
    dests = list(it_pos.keys())
    for d in dests:
        Q[d] = {}
        for cur, neibs in adj.items():
            Q[d][cur] = {}
            try:
                series = df[d][cur]
            except Exception:
                # fallback to loc
                try:
                    series = df.loc[(cur, slice(None)), d]
                except Exception:
                    series = None
            if series is None:
                for n in neibs:
                    Q[d][cur][n] = 0.0
                continue
            for n in neibs:
                try:
                    Q[d][cur][n] = float(series[n])
                except Exception:
                    Q[d][cur][n] = 0.0
    return Q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="SUMO net.xml path")
    ap.add_argument("--dir", required=True, help="Directory containing table_record_minidtn_*.csv")
    ap.add_argument("--trials", type=int, default=200, help="Number of random S-D pairs to evaluate")
    args = ap.parse_args()

    rt = _load_rt(args.net, args.dir)
    # Basic stats
    for name, df in [("PDR", rt.table_PRR), ("AD", rt.table_AD), ("HC", rt.table_HC), ("RC", rt.table_RC)]:
        mn, mx, mu, nz = _stats(df)
        print(f"{name}: min={mn:.4f} max={mx:.4f} mean={mu:.4f} nonzero={nz*100:.1f}%")
    # Greedy success on each metric's raw table
    env = _build_env(args.net)
    ids = list(env.it_pos.keys())
    random.seed(1)
    for name, df in [("PDR", rt.table_PRR), ("AD", rt.table_AD), ("HC", rt.table_HC), ("RC", rt.table_RC)]:
        Q = _matrix_to_Q(df, env.it_pos, env.adj)
        succ = 0
        steps = 0
        ad = 0.0; hc = 0.0; rc = 0.0
        trials = max(1, args.trials)
        for _ in range(trials):
            s = random.choice(ids); d = random.choice(ids)
            if s == d:
                d = ids[(ids.index(d)+1) % len(ids)]
            ok, traj, agg = _greedy_episode(env, Q, s, d)
            succ += 1 if ok else 0
            steps += len(traj)
            ad += agg["ADsd"]; hc += agg["HCsd"]; rc += agg["RCsd"]
        print(f"[Greedy {name}] success={succ}/{trials} ({succ*100.0/trials:.1f}%) "
              f"avg_steps={steps/float(trials):.2f} AD={ad/trials:.2f} HC={hc/trials:.2f} RC={rc/trials:.2f}")

if __name__ == "__main__":
    main()

