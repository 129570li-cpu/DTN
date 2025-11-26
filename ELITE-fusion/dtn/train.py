"""
Orchestrate DTN training for four single-target agents (PDR/AD/HC/RC),
then export Q-tables as CSV matrices matching the shape expected by
Routing_table.table_config():
  rows: for each current in adjacents_comb, for each neighbor in adjacents[current]
  cols: for each destination in it_pos
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import csv
import os

from .graph import make_grid
from .env import DTNEnv, EnvConfig
from .rl import train_single_metric

def _matrix_from_Q(Q: Dict[int, Dict[int, Dict[int, float]]],
                   it_pos: Dict[int, Tuple[float, float]],
                   adj: Dict[int, List[int]]) -> List[List[float]]:
    """
    Build matrix with the same row/col iteration order as Routing_table.table_config().
    Rows iterate adj.items() insertion order; within each, neighbor list order.
    Cols are it_pos keys insertion order.
    """
    matrix: List[List[float]] = []
    dest_ids = list(it_pos.keys())
    for cur, neibs in adj.items():
        for neib in neibs:
            row: List[float] = []
            for dest in dest_ids:
                row.append(Q.get(dest, {}).get(cur, {}).get(neib, 0.0))
            matrix.append(row)
    return matrix

def _write_csv(path: str, matrix: List[List[float]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(matrix)

def train_and_export(output_dir: str,
                     n_rows: int = 4,
                     n_cols: int = 4,
                     episodes: int = 4000,
                     seed: int = 1,
                     it_pos: Dict[int, Tuple[float, float]] = None,
                     adj: Dict[int, List[int]] = None,
                     speed_map: Dict[int, Dict[int, float]] = None,
                     density_map: Dict[int, Dict[int, float]] = None,
                     comm_radius: float = 300.0,
                     veh_speed: float = 23.0,
                     init_Q: Optional[Dict[str, Dict[int, Dict[int, Dict[int, float]]]]] = None) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    if it_pos is None or adj is None:
        it_pos, adj = make_grid(n_rows=n_rows, n_cols=n_cols, spacing=250.0)
    cfg = EnvConfig(seed=seed, comm_radius=comm_radius, veh_speed=veh_speed, alpha=0.90, gamma=0.10)
    env = DTNEnv(it_pos, adj, cfg, speed_map=speed_map, density_map=density_map)
    init_map = init_Q or {}
    Qpdr = train_single_metric(env, "PDR", episodes=episodes, init_Q=init_map.get("pdr"))
    Qad  = train_single_metric(env, "AD",  episodes=episodes, init_Q=init_map.get("ad"))
    Qhc  = train_single_metric(env, "HC",  episodes=episodes, init_Q=init_map.get("hc"))
    Qrc  = train_single_metric(env, "RC",  episodes=episodes, init_Q=init_map.get("rc"))
    Mpdr = _matrix_from_Q(Qpdr, it_pos, adj)
    Mad  = _matrix_from_Q(Qad,  it_pos, adj)
    Mhc  = _matrix_from_Q(Qhc,  it_pos, adj)
    Mrc  = _matrix_from_Q(Qrc,  it_pos, adj)
    paths = {
        "pdr": os.path.join(output_dir, "table_record_minidtn_pdr.csv"),
        "ad":  os.path.join(output_dir, "table_record_minidtn_ad.csv"),
        "hc":  os.path.join(output_dir, "table_record_minidtn_hc.csv"),
        "rc":  os.path.join(output_dir, "table_record_minidtn_rc.csv"),
    }
    _write_csv(paths["pdr"], Mpdr)
    _write_csv(paths["ad"],  Mad)
    _write_csv(paths["hc"],  Mhc)
    _write_csv(paths["rc"],  Mrc)
    # Return file paths and the topology used so caller can inject into Global_Par
    return paths
