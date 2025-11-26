"""
Minimal DTN environment for SPL training as described in the ELITE paper.
We model:
- State: (current_junction, destination_junction)
- Action: choose an adjacent junction as the next hop
- Episode: iterate until reaching destination or step limit

We approximate per-segment metrics to compute rewards:
- ADca: segment delay proportional to road length / speed
- HCca: expected hops on the road ~ ceil(length / C)
- RCca: control overhead proxy ~ base + k * degree(current)
At episode end we can compute ADsd, mADsd etc. We follow paper's shapes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import random

from .graph import euclid

Coord = Tuple[float, float]

@dataclass
class EnvConfig:
    comm_radius: float = 300.0  # C in paper
    veh_speed: float = 23.0     # up to 23 m/s per paper Table 3
    base_ctrl_overhead: float = 1.0
    deg_weight: float = 0.5
    max_steps: int = 128
    gamma: float = 0.10         # discount factor per paper
    alpha: float = 0.90         # learning rate per paper
    seed: int = 1

class DTNEnv:
    def __init__(self,
                 it_pos: Dict[int, Coord],
                 adj: Dict[int, List[int]],
                 cfg: EnvConfig = EnvConfig(),
                 speed_map: Optional[Dict[int, Dict[int, float]]] = None,
                 density_map: Optional[Dict[int, Dict[int, float]]] = None) -> None:
        self.it_pos = it_pos
        self.adj = adj
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        # precompute degree
        self.deg: Dict[int, int] = {i: len(neibs) for i, neibs in adj.items()}
        # optional per-edge speed overrides from SUMO (m/s)
        self.speed_map: Dict[int, Dict[int, float]] = speed_map or {}
        # optional per-edge average vehicle count (vehicles per edge per timestep)
        self.density_map: Dict[int, Dict[int, float]] = density_map or {}
        # approximate max path delay baseline for normalization
        self.max_pair_delay = self._estimate_max_pair_delay()

    # Basic kinematics and proxy metrics
    def seg_length(self, c: int, a: int) -> float:
        return euclid(self.it_pos[c], self.it_pos[a])

    def seg_delay(self, c: int, a: int) -> float:
        # time = distance / speed; prefer SUMO lane speed if available
        v = self.speed_map.get(c, {}).get(a, self.cfg.veh_speed)
        return self.seg_length(c, a) / max(1e-6, v)

    def seg_hops(self, c: int, a: int) -> int:
        # expected hops if vehicles could multihop within comm_radius
        return max(1, int(math.ceil(self.seg_length(c, a) / max(1.0, self.cfg.comm_radius))))

    def seg_ctrl_overhead(self, c: int, a: int) -> float:
        # Estimate control messages on segment via average vehicle count * beacon_rate * segment_time
        beacon_rate = 1.0  # 1 Hz (Table 3 beacon interval 1s)
        dens = self.density_map.get(c, {}).get(a, 0.0)
        t = self.seg_delay(c, a)
        ctrl = beacon_rate * dens * t
        # fallback if density missing
        if ctrl <= 0.0:
            ctrl = self.cfg.base_ctrl_overhead + self.cfg.deg_weight * (self.deg.get(c, 0) + self.deg.get(a, 0)) / 2.0
        return ctrl

    def greedy_next(self, c: int, d: int) -> int:
        # choose neighbor closest to destination
        best = None
        best_dist = float("inf")
        dst = self.it_pos[d]
        for n in self.adj[c]:
            dist = euclid(self.it_pos[n], dst)
            if dist < best_dist:
                best_dist = dist
                best = n
        return best if best is not None else self.rng.choice(self.adj[c])

    def _estimate_max_pair_delay(self) -> float:
        # take the farthest pair straight-line delay as a rough upper bound
        ids = list(self.it_pos.keys())
        m = 0.0
        for i in ids:
            for j in ids:
                m = max(m, euclid(self.it_pos[i], self.it_pos[j]) / max(1e-6, self.cfg.veh_speed))
        return max(m, 1.0)

    # Run one episode under a tri-modal policy for junction selection
    def run_episode(self,
                    Q: Dict[int, Dict[int, Dict[int, float]]],
                    d: int,
                    start: int,
                    mode_probs=(0.5, 0.2, 0.3)
                    ) -> Tuple[bool, List[Tuple[int, int]], Dict[str, float]]:
        """
        Returns:
          success: whether destination reached
          traj: list of (current, next) edges taken
          agg: aggregated path metrics: ADsd, HCsd, RCsd
        """
        cur = start
        visited = set([cur])
        traj: List[Tuple[int, int]] = []
        steps = 0
        # accumulators for path-level metrics
        ad_sum = 0.0
        hc_sum = 0
        rc_sum = 0.0
        while steps < self.cfg.max_steps and cur != d:
            steps += 1
            r = self.rng.random()
            # mode selection: exploit / greedy / explore
            if r < mode_probs[0] and Q.get(d, {}).get(cur):
                # exploitation: argmax over Q[d][cur][n]
                best_n = None
                best_q = -1e18
                for n in self.adj[cur]:
                    qv = Q.get(d, {}).get(cur, {}).get(n, 0.0)
                    if qv > best_q:
                        best_q = qv
                        best_n = n
                nxt = best_n if best_n is not None else self.rng.choice(self.adj[cur])
            elif r < (mode_probs[0] + mode_probs[1]):
                # greedy
                nxt = self.greedy_next(cur, d)
            else:
                # exploration
                nxt = self.rng.choice(self.adj[cur])
            # apply step
            traj.append((cur, nxt))
            # accumulate per-segment proxies
            ad_sum += self.seg_delay(cur, nxt)
            hc_sum += self.seg_hops(cur, nxt)
            rc_sum += self.seg_ctrl_overhead(cur, nxt)
            cur = nxt
            # simple loop breaking
            if cur in visited and cur != d and steps > 2:
                # loop detected: fail early
                break
            visited.add(cur)
        success = (cur == d)
        agg = {"ADsd": ad_sum, "HCsd": float(hc_sum), "RCsd": rc_sum}
        return success, traj, agg
