#!/usr/bin/env python3
"""
Standalone trainer for the GraphSAGE+DQN router (non-federated, offline).
This copies the FD components so we can iterate without touching the original pipeline.
"""
from __future__ import annotations
import os
import sys
import math
import argparse
import json
import random
import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional, Callable, Any
import shutil

import torch
from torch.optim import Adam
from torch_geometric.data import Data

# Allow importing ELITE-fusion modules (topology, env, etc.)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from topology.sumo_net import parse_sumo_net  # type: ignore
from dtn.env import DTNEnv, EnvConfig  # type: ignore

from data_converter import GraphConverter
from model import FedG_DQN, MultiPolicyFedG
from dqn_utils import ReplayBuffer, Transition, multi_head_train_step
from node_features import NodeFeatureTimeline, load_node_feature_timeline


def _build_graph(converter: GraphConverter,
                 it_pos: Dict[int, Tuple[float, float]],
                 adj: Dict[int, List[int]],
                 node_features: Optional[Dict[int, List[float]]] = None,
                 edge_features: Optional[Dict[str, List[float]]] = None) -> Data:
    """Utility wrapper to build PyG Data and move tensors to device later."""
    if getattr(converter, "position_bounds", None) is None:
        converter.set_position_bounds(GraphConverter.infer_position_bounds(it_pos))
    data = converter.build_graph(it_pos, adj, node_features=node_features, edge_features=edge_features)
    return data


def _build_graph_sequence(converter: GraphConverter,
                          it_pos: Dict[int, Tuple[float, float]],
                          adj: Dict[int, List[int]],
                          node_timeline: NodeFeatureTimeline,
                          edge_frames: List[Dict[str, List[float]]],
                          edge_base: Dict[str, List[float]]) -> List[Data]:
    if node_timeline is None or node_timeline.num_frames() == 0:
        raise ValueError("Node feature timeline must contain at least one frame")
    frames: List[Data] = []
    num_frames = node_timeline.num_frames()
    for idx, frame in enumerate(node_timeline.iter_frames()):
        edge_dyn = edge_frames[idx] if idx < len(edge_frames) else {}
        merged_edge: Dict[str, List[float]] = {}
        keys = set(edge_base.keys()) | set(edge_dyn.keys())
        for k in keys:
            base = edge_base.get(k, [])
            dyn = edge_dyn.get(k, [])
            vals = list(base) + list(dyn)
            merged_edge[k] = vals
        frames.append(_build_graph(converter, it_pos, adj, node_features=frame, edge_features=merged_edge))
    return frames


def _load_feature_file(path: Optional[str], edge_map_path: Optional[str]) -> Tuple[Optional[NodeFeatureTimeline], List[Dict[str, List[float]]], Dict[str, List[float]], Dict[str, Any]]:
    if not path:
        return None, [], {}, {}
    with open(path, "r") as f:
        raw = json.load(f)
    node_tl = load_node_feature_timeline(path)
    edge_frames: List[Dict[str, List[float]]] = []
    meta = raw.get("meta", {})
    edges_raw = raw.get("edges", {})
    if edges_raw:
        # determine frame count
        first_key = next(iter(edges_raw))
        vals = edges_raw[first_key]
        if isinstance(vals, list) and vals and isinstance(vals[0], list):
            num_frames = len(vals)
        else:
            num_frames = 1
        edge_frames = [{} for _ in range(num_frames)]
        for k, v_list in edges_raw.items():
            if isinstance(v_list, list) and v_list and isinstance(v_list[0], list):
                limit = min(len(v_list), num_frames)
                for i in range(limit):
                    edge_frames[i][k] = [float(x) for x in v_list[i]]
            else:
                edge_frames[0][k] = [float(x) for x in v_list]
    # edge_base from edge_map
    edge_base: Dict[str, List[float]] = {}
    if edge_map_path and os.path.exists(edge_map_path):
        with open(edge_map_path, "r") as f:
            edge_map = json.load(f)
        edges = edge_map.get("edges", [])
        lengths = [float(e.get("length", 0.0)) for e in edges if e.get("length", None) is not None]
        max_len = max(lengths) if lengths else 0.0
        for e in edges:
            u = int(e.get("u", -1)); v = int(e.get("v", -1))
            if u < 0 or v < 0:
                continue
            key = f"{u}-{v}"
            length = float(e.get("length", 0.0))
            norm_len = math.log(length + 1.0) / math.log(max_len + 1.0) if max_len > 0 else 0.0
            edge_base[key] = [norm_len]
        # fill zero base for edges not in map but present in dynamic frames
    if not edge_base and edge_frames:
        # create zero base for any edge in dynamic frames
        keys = set()
        for fr in edge_frames:
            keys.update(fr.keys())
        for k in keys:
            edge_base[k] = [0.0]
    return node_tl, edge_frames, edge_base, meta


DEFAULT_POLICY_WEIGHTS: Dict[str, Tuple[float, float, float, float]] = {
    # weights correspond to (PDR, RAD, RHC, RRC)
    # Increase hop weight to discourage long paths
    "HRF": (0.35, 0.25, 0.30, 0.10),
    "LDF": (0.25, 0.20, 0.40, 0.15),
    "LBF": (0.25, 0.20, 0.35, 0.20),
}


def _parse_policy_specs(raw_specs: Optional[List[str]]) -> Dict[str, Tuple[float, float, float, float]]:
    if not raw_specs:
        return dict(DEFAULT_POLICY_WEIGHTS)
    policies: Dict[str, Tuple[float, float, float, float]] = {}
    for spec in raw_specs:
        if "=" in spec:
            name, rest = spec.split("=", 1)
        elif ":" in spec:
            name, rest = spec.split(":", 1)
        else:
            raise ValueError(f"Invalid policy spec '{spec}'. Expected NAME=wa,wb,wg.")
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError(f"Policy '{name}' must provide four weights (PDR,RAD,RHC,RRC).")
        try:
            weights = tuple(float(p) for p in parts)  # type: ignore
        except ValueError:
            raise ValueError(f"Policy '{name}' has non-numeric weights.") from None
        policies[name.strip()] = weights  # type: ignore
    return policies


class OfflineDQNTrainer:
    """Epsilon-greedy trainer that supports multiple time-varying graph snapshots."""

    def __init__(self,
                 env: DTNEnv,
                 graph_frames: List[Data],
                 converter: GraphConverter,
                 feature_timeline: NodeFeatureTimeline,
                 hidden_dim: int = 64,
                 lr: float = 1e-4,
                 gamma: float = 0.95,
                 buffer_size: int = 50000,
                 batch_size: int = 64,
                 device: str = "cpu",
                 policy_weights: Dict[str, Tuple[float, float, float, float]] | None = None,
                 failure_reward: float = -5.0,
                 dropout: float = 0.1,
                 sample_nodes: Optional[List[int]] = None,
                 baseline_sample_prob: float = 0.3,
                 bfs_sample_prob: float = 0.2,
                 random_walk_prob: float = 0.1,
                 baseline_modes: Optional[List[str]] = None,
                 step_penalty: float = -0.5,
                 goal_bonus: float = 6.0,
                 long_hop_threshold: int = 12,
                 long_hop_penalty: float = -0.75,
                 max_depth: int = 28):
        if not graph_frames:
            raise ValueError("graph_frames must contain at least one snapshot")
        self.env = env
        self.converter = converter
        self.feature_timeline = feature_timeline
        self.device = device
        self.graph_frames = [g.to(device) for g in graph_frames]
        self.active_frame_idx = 0
        self.last_frame_idx = len(self.graph_frames) - 1
        self.graph_data = self.graph_frames[0]
        in_dim = self.graph_data.x.size(1)
        edge_dim = self.graph_data.edge_attr.size(1) if hasattr(self.graph_data, "edge_attr") and self.graph_data.edge_attr is not None else 1
        self.policies = policy_weights or dict(DEFAULT_POLICY_WEIGHTS)
        self.policy_names = list(self.policies.keys())
        self.policy_net = MultiPolicyFedG(in_dim, hidden_dim, self.policy_names, dropout=dropout, edge_dim=edge_dim).to(device)
        self.target_net = MultiPolicyFedG(in_dim, hidden_dim, self.policy_names, dropout=dropout, edge_dim=edge_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.step_count = 0
        self.failure_reward = failure_reward
        self.step_penalty = step_penalty
        self.goal_bonus = goal_bonus
        self.long_hop_threshold = max(1, long_hop_threshold)
        self.long_hop_penalty = long_hop_penalty
        self.max_depth = max_depth
        self.history: Dict[str, List[Tuple[int, float, float]]] = {p: [] for p in self.policy_names}
        self.sample_nodes: List[int] = []
        self.baseline_sample_prob = max(0.0, min(1.0, baseline_sample_prob))
        self.bfs_sample_prob = max(0.0, min(1.0, bfs_sample_prob))
        self.random_walk_prob = max(0.0, min(1.0, random_walk_prob))
        if self.baseline_sample_prob + self.bfs_sample_prob + self.random_walk_prob >= 1.0:
            raise ValueError("Sum of baseline/bfs/random sampling probabilities must be < 1.0")
        self.baseline_generators: Dict[str, Callable[[int, int, Optional[int]], Optional[List[int]]]] = {
            "distance": self._shortest_path_baseline,
            "low_queue": self._low_queue_baseline,
            "low_delay": self._low_delay_baseline,
        }
        modes = baseline_modes or list(self.baseline_generators.keys())
        normalized: List[str] = []
        for mode in modes:
            key = mode.lower()
            if key not in self.baseline_generators:
                valid = ", ".join(sorted(self.baseline_generators.keys()))
                raise ValueError(f"Unknown baseline mode '{mode}'. Valid options: {valid}")
            normalized.append(key)
        self.baseline_modes = normalized or ["distance"]
        self.set_sample_nodes(sample_nodes)

    def set_sample_nodes(self, nodes: Optional[List[int]]):
        if not nodes:
            self.sample_nodes = list(self.env.it_pos.keys())
            return
        filtered = [nid for nid in nodes if nid in self.env.it_pos]
        if not filtered:
            raise ValueError("Sample node list has no valid entries for current topology.")
        self.sample_nodes = filtered

    def _node_idx(self, real_id: int) -> int:
        idx = self.converter.get_idx(real_id)
        if idx < 0:
            raise ValueError(f"Unknown node id {real_id}")
        return idx

    def _activate_frame(self, frame_idx: int):
        frame_idx = max(0, min(frame_idx, self.last_frame_idx))
        if frame_idx == self.active_frame_idx:
            return
        self.active_frame_idx = frame_idx
        self.graph_data = self.graph_frames[frame_idx]
        if self.feature_timeline:
            self.env.set_node_features(self.feature_timeline.frame(frame_idx))

    def _compute_path_reward(self, path: List[int]) -> Dict[str, List[float]]:
        """
        Compute per-step rewards for every policy. Returns dict:
        {policy: [reward_step_{t-1}, ..., reward_step_0]} reversed order for convenience.
        """
        if len(path) < 2:
            return {}
        seg_ADca, seg_HCca, seg_lca, seg_RCca = [], [], [], []
        total_ad = 0.0
        total_hc = 0.0
        total_rc = 0.0
        for i in range(1, len(path)):
            c, a = path[i-1], path[i]
            ADca = self.env.seg_delay(c, a)
            HCca = float(self.env.seg_hops(c, a))
            lca = max(self.env.seg_length(c, a), 1e-6)
            RCca = float(self.env.seg_ctrl_overhead(c, a))
            seg_ADca.append(ADca); seg_HCca.append(HCca); seg_lca.append(lca); seg_RCca.append(RCca)
            total_ad += ADca; total_hc += HCca; total_rc += RCca
        ADsd = max(total_ad, 1e-6)
        HCsd = max(total_hc, 1.0)
        RCsd = max(total_rc, 1e-6)
        # 用负向归一化惩罚时延：越大的单跳时延奖励越负，总和为 -ADsd/mADsd，不再随跳数正向增加
        mADsd = max(self.env.max_pair_delay, ADsd, 1e-6)
        RAD_list, RHC_list, RRC_list, PDR_list = [], [], [], []
        for idx, (ADca, HCca, lca, RCca) in enumerate(zip(seg_ADca, seg_HCca, seg_lca, seg_RCca)):
            # 负奖励：单跳时延占整体上界的比例，范围 [-1,0]
            RAD = - (ADca / mADsd)
            RAD = max(-1.0, min(0.0, RAD))
            RHC = math.exp(-HCca * self.env.cfg.comm_radius / lca)
            RRC = 1.0 / (1.0 + RCca / RCsd)
            c = path[idx]
            a = path[idx + 1]
            PDR = self.env.seg_delivery_prob(c, a)
            RAD_list.append(RAD); RHC_list.append(RHC); RRC_list.append(RRC); PDR_list.append(PDR)
        policy_rewards: Dict[str, List[float]] = {}
        for name, (wp, wa, wb, wg) in self.policies.items():
            rewards = []
            for PDR, RAD, RHC, RRC in zip(PDR_list, RAD_list, RHC_list, RRC_list):
                reward = wp * PDR + wa * RAD + wb * RHC + wg * RRC
                reward += self.step_penalty
                if len(rewards) >= self.long_hop_threshold:
                    reward += self.long_hop_penalty
                rewards.append(reward)
            rev = list(reversed(rewards))
            if rev:
                rev[0] += self.goal_bonus
            policy_rewards[name] = rev
        return policy_rewards

    def epsilon_greedy(self, curr: int, dest: int, eps: float) -> int:
        neighbors = self.env.adj.get(curr, [])
        if not neighbors:
            return curr
        if random.random() < eps:
            return random.choice(neighbors)
        curr_idx = self._node_idx(curr)
        dest_idx = self._node_idx(dest)
        neighbor_indices = [self._node_idx(n) for n in neighbors]
        edge_attr = self.graph_data.edge_attr if hasattr(self.graph_data, "edge_attr") and self.graph_data.edge_attr is not None else None
        if edge_attr is None:
            edge_attr = torch.zeros((self.graph_data.edge_index.size(1), 1), device=self.graph_data.x.device)
        with torch.no_grad():
            scores = self.policy_net(
                self.graph_data.x,
                self.graph_data.edge_index,
                curr_idx,
                dest_idx,
                neighbor_indices,
                edge_attr,
            )
            if scores.numel() == 0:
                return random.choice(neighbors)
            best = torch.argmax(scores).item()
            return neighbors[int(best)]

    def _q_guided_dfs_path(self,
                           src: int,
                           dst: int,
                           eps: float,
                           start_frame: int,
                           max_depth: Optional[int] = None
                           ) -> Tuple[bool, List[int], List[Tuple[int, int, int]], List[int]]:
        """Depth-first search that ranks neighbors by current Q scores.

        Returns (success, path, failed_edges_with_frame, edge_frames).
        """
        path: List[int] = [src]
        visited = {src}
        dead_ends: set[int] = set()
        failed_edges: List[Tuple[int, int, int]] = []
        edge_frames: List[int] = []
        limit = max_depth if max_depth is not None else len(self.env.it_pos) * 2
        last_frame = self.last_frame_idx

        def dfs(curr: int, depth: int, frame_idx: int) -> bool:
            self._activate_frame(frame_idx)
            if curr == dst:
                return True
            if depth >= limit:
                dead_ends.add(curr)
                return False
            neighbors = [
                n for n in self.env.adj.get(curr, [])
                if n not in visited and n not in dead_ends
            ]
            if not neighbors:
                dead_ends.add(curr)
                return False
            curr_idx = self._node_idx(curr)
            dest_idx = self._node_idx(dst)
            neighbor_indices = [self._node_idx(n) for n in neighbors]
            graph = self.graph_frames[frame_idx]
            edge_attr = graph.edge_attr if hasattr(graph, "edge_attr") and graph.edge_attr is not None else None
            if edge_attr is None:
                edge_attr = torch.zeros((graph.edge_index.size(1), 1), device=graph.x.device)
            with torch.no_grad():
                scores = self.policy_net(
                    graph.x,
                    graph.edge_index,
                    curr_idx,
                    dest_idx,
                    neighbor_indices,
                    edge_attr,
                )
                score_list = scores.detach().cpu().tolist()
            order = list(range(len(neighbors)))
            if random.random() < eps:
                random.shuffle(order)
            else:
                order.sort(key=lambda idx: score_list[idx], reverse=True)
            for idx in order:
                nbr = neighbors[idx]
                next_frame = min(frame_idx + 1, last_frame)
                path.append(nbr)
                visited.add(nbr)
                edge_frames.append(frame_idx)
                if dfs(nbr, depth + 1, next_frame):
                    return True
                visited.remove(nbr)
                path.pop()
                edge_frames.pop()
                dead_ends.add(nbr)
                failed_edges.append((curr, nbr, frame_idx))
            dead_ends.add(curr)
            return False

        success = dfs(src, 0, start_frame)
        return success, path.copy(), failed_edges.copy(), edge_frames.copy()

    def _shortest_path_baseline(self,
                                src: int,
                                dst: int,
                                max_depth: Optional[int] = None) -> Optional[List[int]]:
        limit = max_depth if max_depth is not None else len(self.env.it_pos) * 2
        queue = deque([(src, [src])])
        visited_depth = {src: 0}
        shortest_len: Optional[int] = None
        candidates: List[List[int]] = []
        while queue:
            node, path = queue.popleft()
            if shortest_len is not None and len(path) > shortest_len:
                continue
            if node == dst:
                shortest_len = len(path)
                candidates.append(path)
                continue
            if len(path) >= limit:
                continue
            for nbr in self.env.adj.get(node, []):
                if nbr in path:
                    continue
                depth = len(path)
                if shortest_len is not None and depth + 1 > shortest_len:
                    continue
                prev = visited_depth.get(nbr)
                if prev is not None and prev <= depth:
                    continue
                visited_depth[nbr] = depth
                queue.append((nbr, path + [nbr]))
        if candidates:
            return random.choice(candidates)
        return None

    def _weighted_path_baseline(self,
                                src: int,
                                dst: int,
                                max_depth: Optional[int],
                                edge_cost: Callable[[int, int], float]) -> Optional[List[int]]:
        limit = max_depth if max_depth is not None else len(self.env.it_pos) * 2
        heap: List[Tuple[float, int, List[int]]] = [(0.0, src, [src])]
        best_cost: Dict[int, float] = {src: 0.0}
        best_dst_cost: Optional[float] = None
        candidates: List[List[int]] = []
        while heap:
            cost, node, path = heapq.heappop(heap)
            if best_dst_cost is not None and cost > best_dst_cost:
                break
            if node == dst:
                best_dst_cost = cost
                candidates.append(path)
                continue
            if len(path) - 1 >= limit:
                continue
            for nbr in self.env.adj.get(node, []):
                if nbr in path:
                    continue
                step_cost = float(edge_cost(node, nbr))
                next_cost = cost + max(step_cost, 0.0)
                prev = best_cost.get(nbr)
                if prev is not None and next_cost >= prev:
                    continue
                best_cost[nbr] = next_cost
                heapq.heappush(heap, (next_cost, nbr, path + [nbr]))
        if candidates:
            return random.choice(candidates)
        return None

    def _low_queue_baseline(self,
                            src: int,
                            dst: int,
                            max_depth: Optional[int] = None) -> Optional[List[int]]:
        def cost(c: int, a: int) -> float:
            return 1e-3 + 0.5 * (self.env.node_queue.get(c, 0.0) + self.env.node_queue.get(a, 0.0))
        return self._weighted_path_baseline(src, dst, max_depth, cost)

    def _low_delay_baseline(self,
                            src: int,
                            dst: int,
                            max_depth: Optional[int] = None) -> Optional[List[int]]:
        def cost(c: int, a: int) -> float:
            return self.env.seg_delay(c, a)
        return self._weighted_path_baseline(src, dst, max_depth, cost)

    def _bfs_guided_path(self,
                         src: int,
                         dst: int,
                         eps: float,
                         start_frame: int,
                         max_depth: Optional[int] = None
                         ) -> Tuple[bool, List[int], List[Tuple[int, int, int]], List[int]]:
        limit = max_depth if max_depth is not None else len(self.env.it_pos) * 2
        queue = deque([(src, [src], start_frame, [])])
        while queue:
            curr, path, frame_idx, edge_frames = queue.popleft()
            if curr == dst:
                return True, path, [], edge_frames
            if len(path) - 1 >= limit:
                continue
            neighbors = [n for n in self.env.adj.get(curr, []) if n not in path]
            if not neighbors:
                continue
            curr_idx = self._node_idx(curr)
            dest_idx = self._node_idx(dst)
            neighbor_indices = [self._node_idx(n) for n in neighbors]
            graph = self.graph_frames[frame_idx]
            edge_attr = graph.edge_attr if hasattr(graph, "edge_attr") and graph.edge_attr is not None else None
            if edge_attr is None:
                edge_attr = torch.zeros((graph.edge_index.size(1), 1), device=graph.x.device)
            with torch.no_grad():
                scores = self.policy_net(
                    graph.x,
                    graph.edge_index,
                    curr_idx,
                    dest_idx,
                    neighbor_indices,
                    edge_attr,
                )
                score_list = scores.detach().cpu().tolist()
            order = list(range(len(neighbors)))
            if random.random() < eps:
                random.shuffle(order)
            else:
                order.sort(key=lambda idx: score_list[idx], reverse=True)
            for idx in order:
                nbr = neighbors[idx]
                next_frame = min(frame_idx + 1, self.last_frame_idx)
                queue.append((nbr, path + [nbr], next_frame, edge_frames + [frame_idx]))
        return False, [], [], []

    def _uniform_random_walk(self,
                             src: int,
                             dst: int,
                             start_frame: int,
                             max_depth: Optional[int] = None
                             ) -> Tuple[bool, List[int], List[Tuple[int, int, int]], List[int]]:
        limit = max_depth if max_depth is not None else self.env.cfg.max_steps
        path = [src]
        edge_frames: List[int] = []
        curr = src
        frame_idx = start_frame
        steps = 0
        while steps < limit and curr != dst:
            neighbors = self.env.adj.get(curr, [])
            if not neighbors:
                break
            nxt = random.choice(neighbors)
            path.append(nxt)
            edge_frames.append(frame_idx)
            curr = nxt
            frame_idx = min(frame_idx + 1, self.last_frame_idx)
            steps += 1
        if curr == dst:
            return True, path, [], edge_frames
        failed_edges: List[Tuple[int, int, int]] = []
        for i in range(len(path) - 1):
            frame = edge_frames[i] if i < len(edge_frames) else start_frame
            failed_edges.append((path[i], path[i + 1], frame))
        return False, path, failed_edges, edge_frames

    def collect_episode(self, eps: float, max_steps: Optional[int] = None):
        if len(self.sample_nodes) < 2:
            return
        src, dst = random.sample(self.sample_nodes, 2)
        depth_limit = max_steps if max_steps is not None else self.env.cfg.max_steps
        start_frame = random.randint(0, self.last_frame_idx)
        mode = random.random()
        baseline_thr = self.baseline_sample_prob
        bfs_thr = baseline_thr + self.bfs_sample_prob
        random_thr = bfs_thr + self.random_walk_prob
        if mode < baseline_thr:
            baseline_mode = random.choice(self.baseline_modes)
            baseline_fn = self.baseline_generators.get(baseline_mode, self._shortest_path_baseline)
            baseline_path = baseline_fn(src, dst, depth_limit)
            if baseline_path:
                success = True
                path = baseline_path
                failed_edges: List[Tuple[int, int, int]] = []
                edge_frames = [min(start_frame + i, self.last_frame_idx)
                               for i in range(len(path) - 1)]
            else:
                success = False
                path = []
                failed_edges = []
                edge_frames = []
        elif mode < bfs_thr:
            success, path, failed_edges, edge_frames = self._bfs_guided_path(
                src, dst, eps, start_frame, max_depth=depth_limit)
            if not success and not path:
                success, path, failed_edges, edge_frames = self._q_guided_dfs_path(
                    src, dst, eps, start_frame, max_depth=depth_limit)
        elif mode < random_thr:
            success, path, failed_edges, edge_frames = self._uniform_random_walk(
                src, dst, start_frame, max_depth=depth_limit)
        else:
            success, path, failed_edges, edge_frames = self._q_guided_dfs_path(
                src, dst, eps, start_frame, max_depth=depth_limit)

        if success and len(path) >= 2:
            policy_rev_rewards = self._compute_path_reward(path)
            if not policy_rev_rewards:
                return
            for i in range(len(path) - 1):
                c = path[i]
                a = path[i + 1]
                idx = len(path) - 2 - i
                reward_dict = {p: policy_rev_rewards[p][idx] for p in self.policy_names}
                done = (a == dst)
                frame_idx = edge_frames[i] if i < len(edge_frames) else start_frame
                frame_idx = max(0, min(frame_idx, self.last_frame_idx))
                next_frame_idx = min(frame_idx + 1, self.last_frame_idx)
                self.buffer.push(
                    self.graph_frames[frame_idx],
                    self._node_idx(c),
                    self._node_idx(dst),
                    self._node_idx(a),
                    reward_dict,
                    self.graph_frames[next_frame_idx],
                    done,
                )
        else:
            if not failed_edges:
                return
            reward_dict = {p: self.failure_reward for p in self.policy_names}
            for c, a, frame_idx in failed_edges:
                frame_idx = max(0, min(frame_idx, self.last_frame_idx))
                self.buffer.push(
                    self.graph_frames[frame_idx],
                    self._node_idx(c),
                    self._node_idx(dst),
                    self._node_idx(a),
                    reward_dict,
                    self.graph_frames[frame_idx],
                    True,
                )
            return
        if failed_edges:
            reward_dict = {p: self.failure_reward for p in self.policy_names}
            for c, a, frame_idx in failed_edges:
                frame_idx = max(0, min(frame_idx, self.last_frame_idx))
                self.buffer.push(
                    self.graph_frames[frame_idx],
                    self._node_idx(c),
                    self._node_idx(dst),
                    self._node_idx(a),
                    reward_dict,
                    self.graph_frames[frame_idx],
                    True,
                )

    def train_step(self):
        return multi_head_train_step(
            self.policy_net,
            self.target_net,
            self.optimizer,
            self.buffer,
            self.batch_size,
            gamma=self.gamma,
            device=self.device,
            policy_names=self.policy_names,
        )

    def soft_update(self, tau: float = 0.01):
        for t_param, p_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t_param.data.copy_(tau * p_param.data + (1.0 - tau) * t_param.data)

    def train(self,
              episodes: int,
              eps_start: float = 1.0,
              eps_end: float = 0.05,
              eps_decay: float = 0.995,
              target_sync_interval: int = 10,
              max_depth: Optional[int] = None):
        history = []
        eps = eps_start
        for ep in range(1, episodes + 1):
            depth_lim = max_depth if max_depth is not None else self.max_depth
            self.collect_episode(eps, max_depth=depth_lim)
            loss_dict = self.train_step()
            if loss_dict:
                history.append((ep, loss_dict, eps))
                for policy in self.policy_names:
                    if policy in loss_dict:
                        self.history[policy].append((ep, loss_dict[policy], eps))
            if ep % target_sync_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            eps = max(eps_end, eps * eps_decay)
        return history

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        mapping_path = os.path.join(out_dir, "id_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(self.converter.id_to_idx, f, indent=2)
        saved = {}
        for policy in self.policy_names:
            policy_dir = os.path.join(out_dir, policy.lower())
            os.makedirs(policy_dir, exist_ok=True)
            state_dict = self.policy_net.export_single_head_state(policy)
            model_path = os.path.join(policy_dir, "fedg_dqn.pt")
            torch.save(state_dict, model_path)
            shutil.copy(mapping_path, os.path.join(policy_dir, "id_mapping.json"))
            saved[policy] = (model_path, os.path.join(policy_dir, "id_mapping.json"))
        return saved

    def write_logs(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        for policy, entries in self.history.items():
            path = os.path.join(log_dir, f"{policy.lower()}_train_log.csv")
            with open(path, "w") as f:
                f.write("episode,loss,epsilon\n")
                for ep, loss, eps in entries:
                    f.write(f"{ep},{loss},{eps}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline GNN-DQN trainer for ELITE (non-federated).")
    parser.add_argument("--net-xml", required=True, help="Path to SUMO net.xml")
    parser.add_argument("--output-dir", default="dtn_out/gnn_dqn", help="Directory to save checkpoints")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--comm-radius", type=float, default=300.0)
    parser.add_argument("--veh-speed", type=float, default=23.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--node-features", default=None, help="Path to JSON with node/edge dynamic features.")
    parser.add_argument("--edge-map", default=None, help="Path to edge_id_map.json with static edge info.")
    parser.add_argument("--policy", action="append",
                        help="Define policy weights as NAME=wpdr,wrad,wrhc,wrrc (can be provided multiple times).")
    parser.add_argument("--failure-reward", type=float, default=-5.0,
                        help="Reward assigned to each step of failed episodes")
    parser.add_argument("--step-penalty", type=float, default=-0.5, help="Penalty per hop to discourage long paths")
    parser.add_argument("--goal-bonus", type=float, default=5.0, help="Bonus added when reaching destination")
    parser.add_argument("--log-dir", default=None, help="Directory to store per-policy training logs")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for GraphSAGE/MLP layers")
    parser.add_argument("--baseline-sample-prob", type=float, default=0.3,
                        help="Probability of sampling a policy-independent baseline path to reduce bias")
    parser.add_argument("--baseline-modes", default="distance,low_queue,low_delay",
                        help="Comma-separated list of baseline heuristics to mix (distance,low_queue,low_delay).")
    parser.add_argument("--bfs-sample-prob", type=float, default=0.2,
                        help="Probability of BFS-guided sampling per episode")
    parser.add_argument("--random-walk-prob", type=float, default=0.1,
                        help="Probability of uniform random-walk sampling per episode")
    parser.add_argument("--long-hop-threshold", type=int, default=16, help="Hop count after which extra penalty applies")
    parser.add_argument("--long-hop-penalty", type=float, default=-0.5, help="Additional penalty per hop after threshold")
    parser.add_argument("--max-depth", type=int, default=32, help="Maximum depth for DFS/BFS/random-walk paths during training")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_modes = None
    if args.baseline_modes:
        baseline_modes = [m.strip() for m in args.baseline_modes.split(",") if m.strip()]
    it_pos, adj, _, speed_map = parse_sumo_net(args.net_xml, skip_internal=True)
    cfg = EnvConfig(comm_radius=args.comm_radius, veh_speed=args.veh_speed)
    node_tl, edge_frames, edge_base, meta = _load_feature_file(args.node_features, args.edge_map)
    if node_tl is None:
        raise ValueError("--node-features is required to provide dynamic traffic data.")
    env = DTNEnv(it_pos, adj, cfg, speed_map=speed_map, node_features=node_tl.frame(0))
    converter = GraphConverter()
    graph_frames = _build_graph_sequence(converter, it_pos, adj, node_tl, edge_frames, edge_base)
    policy_weights = _parse_policy_specs(args.policy)
    trainer = OfflineDQNTrainer(
        env,
        graph_frames,
        converter,
        node_tl,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=args.device,
        policy_weights=policy_weights,
        failure_reward=args.failure_reward,
        dropout=args.dropout,
        baseline_sample_prob=args.baseline_sample_prob,
        bfs_sample_prob=args.bfs_sample_prob,
        random_walk_prob=args.random_walk_prob,
        baseline_modes=baseline_modes,
        step_penalty=args.step_penalty,
        goal_bonus=args.goal_bonus,
        long_hop_threshold=args.long_hop_threshold,
        long_hop_penalty=args.long_hop_penalty,
        max_depth=args.max_depth,
    )
    history = trainer.train(args.episodes, max_depth=args.max_depth)
    if args.log_dir:
        trainer.write_logs(args.log_dir)
    saved = trainer.save(args.output_dir)
    for policy, (model_path, mapping_path) in saved.items():
        print(f"[trainer] {policy} model saved to {model_path}")
        print(f"[trainer] {policy} id mapping saved to {mapping_path}")


if __name__ == "__main__":
    main()
