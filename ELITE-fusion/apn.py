"""
APN reward and load estimation strictly following the paper:
- Load estimation:
    load_bas = min(Lavg / Bw, 1)
  where Lavg is average buffer usage along the baseline path Lbas computed by Gbas
  (This module exposes helper to compute load level from path buffers and bandwidth).
- Reward aggregation:
    R0 = a * g(ADsd) + b * q(HCsd) + gamma * l(RCsd)
  where
    g(ADsd) = (1/nsd) * sum_{(c,a) in Lsd} RAD(c,d,a)
    q(HCsd) = (1/nsd) * sum RHC(c,d,a)
    l(RCsd) = (1/nsd) * sum RRC(c,d,a)
  and per-segment rewards follow:
    RAD(c,d,a) = 1 - (ADca/ADsd) * (1 - ADsd/mADsd)
    RHC(c,d,a) = exp( - HCca * C / l_ca )
    RRC(c,d,a) = 1 / (1 + RCca / RCsd)
Inputs to functions below are plain numerics collected/aggregated by ns-3 side.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import math

def compute_load_level(buffers_on_path: List[float], bandwidth_bps: float) -> Tuple[float, int]:
    """
    Compute load_bas and discrete level:
      0: [0,0.3), 1: [0.3,0.7], 2: (0.7,1.0]
    buffers_on_path: list of buffer usage (bytes or normalized) for vehicles along Gbas path
    bandwidth_bps: channel bandwidth in bits/s (must be consistent with buffers_on_path units)
    """
    if not buffers_on_path or bandwidth_bps <= 0:
        return 0.0, 0
    Lavg = sum(buffers_on_path) / float(len(buffers_on_path))
    load_bas = min(Lavg / float(bandwidth_bps), 1.0)
    # map to level
    if load_bas < 0.3:
        level = 0
    elif load_bas <= 0.7:
        level = 1
    else:
        level = 2
    return load_bas, level

def per_segment_rewards(ADsd: float,
                        mADsd: float,
                        comm_radius: float,
                        seg_ADca: List[float],
                        seg_HCca: List[float],
                        seg_lca: List[float],
                        seg_RCca: List[float],
                        RCsd: float) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute per-segment RAD/RHC/RRC for all segments in Lsd.
    Arrays must have same length (nsd). l_ca must be >0; use small epsilon otherwise.
    """
    nsd = min(len(seg_ADca), len(seg_HCca), len(seg_lca), len(seg_RCca))
    if nsd <= 0:
        # No per-segment stats (ns-3 feedback empty): return zeros so controller stays alive.
        return [0.0], [0.0], [0.0]
    ADsd = max(ADsd, 1e-9)
    mADsd = max(mADsd, ADsd)
    RCsd = max(RCsd, 1e-9)
    RAD_list: List[float] = []
    RHC_list: List[float] = []
    RRC_list: List[float] = []
    for i in range(nsd):
        ADca = max(seg_ADca[i], 0.0)
        HCca = max(seg_HCca[i], 0.0)
        lca = max(seg_lca[i], 1e-9)
        RCca = max(seg_RCca[i], 0.0)
        RAD = 1.0 - (ADca / ADsd) * (1.0 - min(ADsd / mADsd, 1.0))
        RHC = math.exp( - HCca * comm_radius / lca )
        RRC = 1.0 / (1.0 + RCca / RCsd)
        RAD_list.append(RAD)
        RHC_list.append(RHC)
        RRC_list.append(RRC)
    return RAD_list, RHC_list, RRC_list

def aggregate_reward(ADsd: float,
                     mADsd: float,
                     comm_radius: float,
                     seg_ADca: List[float],
                     seg_HCca: List[float],
                     seg_lca: List[float],
                     seg_RCca: List[float],
                     RCsd: float,
                     a: float,
                     b: float,
                     gamma: float) -> float:
    """
    Compute R0 according to the paper.
    """
    RAD_list, RHC_list, RRC_list = per_segment_rewards(ADsd, mADsd, comm_radius, seg_ADca, seg_HCca, seg_lca, seg_RCca, RCsd)
    nsd = max(1, len(RAD_list))
    g = sum(RAD_list) / nsd
    q = sum(RHC_list) / nsd
    l = sum(RRC_list) / nsd
    R0 = a * g + b * q + gamma * l
    return R0
