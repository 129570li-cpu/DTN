from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import copy

from .env import DTNEnv, EnvConfig

# Q-table structure:
# Q[dest][current][neighbor] = value

def init_Q_table(env: DTNEnv) -> Dict[int, Dict[int, Dict[int, float]]]:
    Q: Dict[int, Dict[int, Dict[int, float]]] = {}
    ids = list(env.it_pos.keys())
    for d in ids:
        Q[d] = {}
        for c in ids:
            if c not in env.adj:
                continue
            Q[d][c] = {}
            for n in env.adj[c]:
                Q[d][c][n] = 0.0
    return Q

def train_single_metric(env: DTNEnv,
                        metric: str,
                        episodes: int = 4000,
                        alpha: float = None,
                        gamma: float = None,
                        init_Q: Optional[Dict[int, Dict[int, Dict[int, float]]]] = None) -> Dict[int, Dict[int, Dict[int, float]]]:
    """
    Train a single-target Q-table for the given metric.
    metric in {"PDR","AD","HC","RC"}
    """
    if alpha is None:
        alpha = env.cfg.alpha
    if gamma is None:
        gamma = env.cfg.gamma

    if init_Q is not None:
        Q = copy.deepcopy(init_Q)
        ids = list(env.it_pos.keys())
        for d in ids:
            Q.setdefault(d, {})
            for c in ids:
                if c not in env.adj:
                    continue
                Q[d].setdefault(c, {})
                for n in env.adj[c]:
                    Q[d][c].setdefault(n, 0.0)
    else:
        Q = init_Q_table(env)
    ids = list(env.it_pos.keys())
    # normalization baselines
    # moving maximum AD per (s,d) pair to approximate mADsd in paper
    mAD_map: Dict[Tuple[int,int], float] = {}

    for ep in range(episodes):
        # sample S-D pair
        s = ids[ep % len(ids)]
        d = ids[(ep * 7 + 3) % len(ids)]
        if s == d:
            d = ids[(d + 1) % len(ids)]
        # run one episode under tri-modal policy
        success, traj, agg = env.run_episode(Q, d, s)
        if not success or not traj:
            # only successful paths are reported/updated per paper (including PDR)
            continue
        # compute path-level normalizers
        ADsd = max(agg["ADsd"], 1e-6)
        HCsd = max(agg["HCsd"], 1.0)
        RCsd = max(agg["RCsd"], 1e-6)
        key = (s, d)
        prev = mAD_map.get(key, 0.0)
        mADsd = ADsd if ADsd > prev else prev
        if ADsd > prev:
            mAD_map[key] = ADsd
        # back-propagate along the path (reverse order)
        for i in range(len(traj)-1, -1, -1):
            c, a = traj[i]
            # immediate reward per metric
            if metric == "PDR":
                # Terminal-only reward for PDR: give 1.0 only at the final transition
                # when the episode successfully reaches the destination; all earlier
                # steps receive 0. This avoids biasing towards longer paths while
                # still propagating success backward through bootstrapping.
                R = 1.0 if i == len(traj) - 1 else 0.0
            elif metric == "AD":
                # RAD(c,d,a) = 1 - (ADca/ADsd) * (1 - ADsd/mADsd)
                ADca = env.seg_delay(c, a)
                R = 1.0 - (ADca / ADsd) * (1.0 - min(ADsd / max(mADsd, ADsd), 1.0))
            elif metric == "HC":
                # RHC(c,d,a) = exp( - HCca * C / l_ca )
                HCca = float(env.seg_hops(c, a))
                lca = max(env.seg_length(c, a), 1e-6)
                R = math.exp( - HCca * env.cfg.comm_radius / lca )
            elif metric == "RC":
                # RRC(c,d,a) = 1 / (1 + RCca / RCsd)
                RCca = env.seg_ctrl_overhead(c, a)
                R = 1.0 / (1.0 + RCca / RCsd)
            else:
                R = 0.0
            # bootstrap
            next_best = 0.0
            if i < len(traj) - 1:
                c_next, _ = traj[i+1]
                # choose best action value at c_next
                if env.adj.get(c_next):
                    next_best = max(Q[d][c_next].get(n, 0.0) for n in env.adj[c_next])
            # Q update
            Q[d][c][a] += alpha * (R + gamma * next_best - Q[d][c][a])
    return Q
