#!/usr/bin/env python3
"""
Multiprocess sampler + single GPU learner for GNN-DQN (ELITE).
 - Worker进程：CPU 推理 + 采样，发送 Transition 到主进程
 - 主进程：GPU 训练，多策略头共享 encoder

依赖：与 trainer.py 相同的环境/数据文件；workers 独立创建 env/graph。
"""
from __future__ import annotations

import argparse
import os
import queue
import random
import time
import math
import heapq
import json
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

import torch
from torch.optim import Adam
from torch_geometric.data import Data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from topology.sumo_net import parse_sumo_net  # type: ignore
from dtn.env import DTNEnv, EnvConfig  # type: ignore
from dtn.FD.model import MultiPolicyFedG  # type: ignore
from dtn.FD.dqn_utils import ReplayBuffer, Transition, multi_head_train_step  # type: ignore
from dtn.FD.data_converter import GraphConverter  # type: ignore
from dtn.FD.node_features import load_node_feature_timeline, NodeFeatureTimeline  # type: ignore

DEFAULT_POLICY_WEIGHTS: Dict[str, Tuple[float, float, float, float]] = {
    "HRF": (0.35, 0.25, 0.30, 0.10),
    "LDF": (0.25, 0.20, 0.40, 0.15),
    "LBF": (0.25, 0.20, 0.35, 0.20),
}

# ------------------------- Worker ------------------------- #
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
    if not edge_base and edge_frames:
        keys = set()
        for fr in edge_frames:
            keys.update(fr.keys())
        for k in keys:
            edge_base[k] = [0.0]
    return node_tl, edge_frames, edge_base, meta


def _build_graph_sequence(converter: GraphConverter,
                          it_pos: Dict[int, Tuple[float, float]],
                          adj: Dict[int, List[int]],
                          node_tl: NodeFeatureTimeline,
                          edge_frames: List[Dict[str, List[float]]],
                          edge_base: Dict[str, List[float]],
                          edge_feat_dim: int = 0) -> List[Data]:
    if node_tl is None or node_tl.num_frames() == 0:
        raise ValueError("Node feature timeline must contain frames.")
    frames: List[Data] = []
    base_dim = len(next(iter(edge_base.values()))) if edge_base else 0
    dyn_dim = edge_feat_dim
    # pre-set edge_dim to guarantee consistent padding across frames
    expected_dim = base_dim + dyn_dim if (base_dim or dyn_dim) else 0
    if expected_dim > 0:
        converter.edge_dim = expected_dim
    for idx, frame in enumerate(node_tl.iter_frames()):
        edge_dyn = edge_frames[idx] if idx < len(edge_frames) else {}
        merged: Dict[str, List[float]] = {}
        keys = set(edge_base.keys()) | set(edge_dyn.keys())
        for k in keys:
            base = edge_base.get(k, [])
            dyn = edge_dyn.get(k, [])
            if dyn_dim and len(dyn) < dyn_dim:
                dyn = list(dyn) + [0.0] * (dyn_dim - len(dyn))
            vals = list(base) + list(dyn)
            if expected_dim and len(vals) < expected_dim:
                vals = vals + [0.0] * (expected_dim - len(vals))
            merged[k] = vals
        frames.append(converter.build_graph(it_pos, adj, node_features=frame, edge_features=merged))
    return frames


def worker_proc(idx: int,
                args: argparse.Namespace,
                policy_weights: Dict[str, Tuple[float, float, float, float]],
                param_queue: mp.Queue,
                out_queue: mp.Queue,
                stop_event: mp.Event):
    random.seed(args.seed + idx)
    torch.manual_seed(args.seed + idx)

    it_pos, adj, _, speed_map = parse_sumo_net(args.net_xml, skip_internal=True)
    cfg = EnvConfig(comm_radius=args.comm_radius, veh_speed=args.veh_speed)
    node_feats, edge_frames, edge_base, meta = _load_feature_file(args.node_features, args.edge_map)
    if node_feats is None or node_feats.num_frames() == 0:
        raise RuntimeError("node feature timeline required for worker")
    env = DTNEnv(it_pos, adj, cfg, speed_map=speed_map, node_features=node_feats.frame(0))
    converter = GraphConverter()
    edge_feat_dim = int(meta.get("edge_feature_dim", 0)) if isinstance(meta, dict) else 0
    graph_frames = _build_graph_sequence(converter, it_pos, adj, node_feats, edge_frames, edge_base, edge_feat_dim=edge_feat_dim)

    in_dim = graph_frames[0].x.size(1)
    edge_dim = graph_frames[0].edge_attr.size(1) if hasattr(graph_frames[0], "edge_attr") and graph_frames[0].edge_attr is not None else 1
    policy_names = list(policy_weights.keys())
    policy_net = MultiPolicyFedG(in_dim, args.hidden_dim, policy_names, dropout=args.dropout, edge_dim=edge_dim).to("cpu")

    # 初始参数同步
    try:
        state_dict = param_queue.get(timeout=10)
        policy_net.load_state_dict(state_dict)
    except queue.Empty:
        pass

    eps = args.eps_start
    eps_min = args.eps_end
    eps_decay = args.eps_decay
    last_frame = len(graph_frames) - 1
    baseline_mode_list = [m.strip().lower() for m in (args.baseline_modes.split(",") if args.baseline_modes else []) if m.strip()]
    if not baseline_mode_list:
        baseline_mode_list = ["distance", "low_queue", "low_delay"]
    # ---------------- Path search helpers (参考 trainer.py) ---------------- #
    def _node_idx(real_id: int) -> int:
        idx = converter.get_idx(real_id)
        if idx < 0:
            raise ValueError(f"Unknown node id {real_id}")
        return idx

    def _compute_path_reward(path: List[int]) -> Dict[str, List[float]]:
        if len(path) < 2:
            return {}
        seg_ADca, seg_HCca, seg_lca, seg_RCca = [], [], [], []
        total_ad = total_hc = total_rc = 0.0
        for i in range(1, len(path)):
            c, a = path[i-1], path[i]
            ADca = env.seg_delay(c, a)
            HCca = float(env.seg_hops(c, a))
            lca = max(env.seg_length(c, a), 1e-6)
            RCca = float(env.seg_ctrl_overhead(c, a))
            seg_ADca.append(ADca); seg_HCca.append(HCca); seg_lca.append(lca); seg_RCca.append(RCca)
            total_ad += ADca; total_hc += HCca; total_rc += RCca
        ADsd = max(total_ad, 1e-6)
        HCsd = max(total_hc, 1.0)
        RCsd = max(total_rc, 1e-6)
        mADsd = max(env.max_pair_delay, ADsd, 1e-6)
        RAD_list, RHC_list, RRC_list, PDR_list = [], [], [], []
        for ADca, HCca, lca, RCca in zip(seg_ADca, seg_HCca, seg_lca, seg_RCca):
            RAD = - (ADca / mADsd)
            RAD = max(-1.0, min(0.0, RAD))
            RHC = math.exp(-HCca * env.cfg.comm_radius / lca)
            RRC = 1.0 / (1.0 + RCca / RCsd)
            RAD_list.append(RAD); RHC_list.append(RHC); RRC_list.append(RRC)
        for i in range(1, len(path)):
            c, a = path[i-1], path[i]
            PDR_list.append(env.seg_delivery_prob(c, a))
        policy_rewards: Dict[str, List[float]] = {}
        for name, (wp, wa, wb, wg) in policy_weights.items():
            rewards = []
            for PDR, RAD, RHC, RRC in zip(PDR_list, RAD_list, RHC_list, RRC_list):
                reward = wp * PDR + wa * RAD + wb * RHC + wg * RRC
                reward += args.step_penalty
                if len(rewards) >= args.long_hop_threshold:
                    reward += args.long_hop_penalty
                rewards.append(reward)
            rev = list(reversed(rewards))
            if rev:
                rev[0] += args.goal_bonus
            policy_rewards[name] = rev
        return policy_rewards

    def _score_neighbors(graph: Data, curr: int, dest: int, neighbors: List[int]) -> List[float]:
        curr_idx = _node_idx(curr)
        dest_idx = _node_idx(dest)
        neighbor_indices = [_node_idx(n) for n in neighbors]
        edge_attr = graph.edge_attr if hasattr(graph, "edge_attr") and graph.edge_attr is not None else None
        if edge_attr is None:
            edge_dim_local = getattr(policy_net, "edge_dim", 1)
            edge_attr = torch.zeros((graph.edge_index.size(1), edge_dim_local), device=graph.x.device)
        with torch.no_grad():
            scores = policy_net(graph.x, graph.edge_index, curr_idx, dest_idx, neighbor_indices, edge_attr)
        return scores.detach().cpu().tolist() if scores.numel() > 0 else []

    def _q_guided_dfs_path(src: int, dst: int, graph_idx: int, eps_local: float, max_depth: int):
        path = [src]
        visited = {src}
        dead_ends = set()
        failed_edges: List[Tuple[int,int,int]] = []

        def dfs(curr: int, depth: int, frame_idx: int) -> bool:
            if curr == dst:
                return True
            if depth >= max_depth:
                dead_ends.add(curr)
                return False
            neighbors = [n for n in env.adj.get(curr, []) if n not in visited and n not in dead_ends]
            if not neighbors:
                dead_ends.add(curr)
                return False
            graph = graph_frames[frame_idx]
            scores = _score_neighbors(graph, curr, dst, neighbors)
            order = list(range(len(neighbors)))
            if random.random() < eps_local:
                random.shuffle(order)
            else:
                order.sort(key=lambda i: scores[i] if scores else 0.0, reverse=True)
            for idx in order:
                nbr = neighbors[idx]
                next_frame = min(frame_idx + 1, last_frame)
                path.append(nbr)
                visited.add(nbr)
                if dfs(nbr, depth + 1, next_frame):
                    return True
                visited.remove(nbr)
                path.pop()
                failed_edges.append((curr, nbr, frame_idx))
            dead_ends.add(curr)
            return False

        success = dfs(src, 0, graph_idx)
        return success, path.copy(), failed_edges

    def _bfs_guided_path(src: int, dst: int, graph_idx: int, eps_local: float, max_depth: int):
        queue_ = deque([(src, [src], graph_idx)])
        failed_edges: List[Tuple[int,int,int]] = []
        while queue_:
            curr, path, frame_idx = queue_.popleft()
            if len(path) - 1 >= max_depth:
                continue
            if curr == dst:
                return True, path, failed_edges
            neighbors = [n for n in env.adj.get(curr, []) if n not in path]
            if not neighbors:
                continue
            scores = _score_neighbors(graph_frames[frame_idx], curr, dst, neighbors)
            order = list(range(len(neighbors)))
            if random.random() < eps_local:
                random.shuffle(order)
            else:
                order.sort(key=lambda i: scores[i] if scores else 0.0, reverse=True)
            for idx_ in order:
                nbr = neighbors[idx_]
                next_frame = min(frame_idx + 1, last_frame)
                queue_.append((nbr, path + [nbr], next_frame))
        return False, [], failed_edges

    def _uniform_random_walk(src: int, dst: int, graph_idx: int, max_depth: int):
        path = [src]
        frame_idx = graph_idx
        failed_edges: List[Tuple[int,int,int]] = []
        for _ in range(max_depth):
            curr = path[-1]
            if curr == dst:
                return True, path, failed_edges
            neighbors = env.adj.get(curr, [])
            if not neighbors:
                break
            nxt = random.choice(neighbors)
            failed_edges.append((curr, nxt, frame_idx))
            path.append(nxt)
            frame_idx = min(frame_idx + 1, last_frame)
        return path[-1] == dst, path, failed_edges

    # ---------------- Baseline heuristics ---------------- #
    def _shortest_path_baseline(src: int, dst: int, max_depth: int) -> Optional[List[int]]:
        queue_ = deque([(src, [src])])
        visited_depth = {src: 0}
        shortest_len: Optional[int] = None
        candidates: List[List[int]] = []
        while queue_:
            node, pth = queue_.popleft()
            if shortest_len is not None and len(pth) > shortest_len:
                continue
            if node == dst:
                shortest_len = len(pth)
                candidates.append(pth)
                continue
            if len(pth) >= max_depth:
                continue
            for nbr in env.adj.get(node, []):
                if nbr in pth:
                    continue
                depth = len(pth)
                if shortest_len is not None and depth + 1 > shortest_len:
                    continue
                prev = visited_depth.get(nbr)
                if prev is not None and prev <= depth:
                    continue
                visited_depth[nbr] = depth
                queue_.append((nbr, pth + [nbr]))
        if candidates:
            return random.choice(candidates)
        return None

    def _weighted_path_baseline(src: int, dst: int, max_depth: int, edge_cost) -> Optional[List[int]]:
        limit = max_depth
        heap: List[Tuple[float, int, List[int]]] = [(0.0, src, [src])]
        best_cost: Dict[int, float] = {src: 0.0}
        best_dst_cost: Optional[float] = None
        candidates: List[List[int]] = []
        while heap:
            cost, node, pth = heapq.heappop(heap)
            if best_dst_cost is not None and cost > best_dst_cost:
                break
            if node == dst:
                best_dst_cost = cost
                candidates.append(pth)
                continue
            if len(pth) - 1 >= limit:
                continue
            for nbr in env.adj.get(node, []):
                if nbr in pth:
                    continue
                step_cost = float(edge_cost(node, nbr))
                next_cost = cost + max(step_cost, 0.0)
                prev = best_cost.get(nbr)
                if prev is not None and next_cost >= prev:
                    continue
                best_cost[nbr] = next_cost
                heapq.heappush(heap, (next_cost, nbr, pth + [nbr]))
        if candidates:
            return random.choice(candidates)
        return None

    def _low_queue_baseline(src: int, dst: int, max_depth: int) -> Optional[List[int]]:
        def cost(c: int, a: int) -> float:
            return 1e-3 + 0.5 * (env.node_queue.get(c, 0.0) + env.node_queue.get(a, 0.0))
        return _weighted_path_baseline(src, dst, max_depth, cost)

    def _low_delay_baseline(src: int, dst: int, max_depth: int) -> Optional[List[int]]:
        def cost(c: int, a: int) -> float:
            return env.seg_delay(c, a)
        return _weighted_path_baseline(src, dst, max_depth, cost)

    while not stop_event.is_set():
        # 接收最新参数
        try:
            while True:
                state_dict = param_queue.get_nowait()
                policy_net.load_state_dict(state_dict)
        except queue.Empty:
            pass

        # 采样一条 episode（混合策略，参考 trainer 的模式）
        if len(env.it_pos) < 2:
            continue
        src, dst = random.sample(list(env.it_pos.keys()), 2)
        start_frame = random.randint(0, last_frame)
        env.set_node_features(node_feats.frame(start_frame))

        mode = random.random()
        baseline_thr = args.baseline_sample_prob
        bfs_thr = baseline_thr + args.bfs_sample_prob
        random_thr = bfs_thr + args.random_walk_prob
        depth_limit = args.max_steps
        success = False
        path: List[int] = []
        failed_edges: List[Tuple[int,int,int]] = []
        if mode < baseline_thr:
            baseline_mode = random.choice(baseline_mode_list)
            if baseline_mode == "distance":
                path = _shortest_path_baseline(src, dst, depth_limit) or []
            elif baseline_mode == "low_queue":
                path = _low_queue_baseline(src, dst, depth_limit) or []
            elif baseline_mode == "low_delay":
                path = _low_delay_baseline(src, dst, depth_limit) or []
            else:
                path = _shortest_path_baseline(src, dst, depth_limit) or []
            success = bool(path) and path[-1] == dst
        elif mode < bfs_thr:
            success, path, failed_edges = _bfs_guided_path(src, dst, start_frame, eps, depth_limit)
            if not success and not path:
                success, path, failed_edges = _q_guided_dfs_path(src, dst, start_frame, eps, depth_limit)
        elif mode < random_thr:
            success, path, failed_edges = _uniform_random_walk(src, dst, start_frame, depth_limit)
        else:
            success, path, failed_edges = _q_guided_dfs_path(src, dst, start_frame, eps, depth_limit)

        if success and len(path) >= 2:
            policy_rev_rewards = _compute_path_reward(path)
            if policy_rev_rewards:
                for i in range(len(path) - 1):
                    c = path[i]; a = path[i+1]
                    idx = len(path) - 2 - i
                    reward_dict = {p: policy_rev_rewards[p][idx] for p in policy_weights.keys()}
                    done = (a == dst)
                    frame_idx = min(start_frame + i, last_frame)
                    next_frame_idx = min(frame_idx + 1, last_frame)
                    try:
                        out_queue.put(Transition(
                            graph_frames[frame_idx],
                            _node_idx(c),
                            _node_idx(dst),
                            _node_idx(a),
                            reward_dict,
                            graph_frames[next_frame_idx],
                            done,
                        ), timeout=0.1)
                    except queue.Full:
                        pass
        else:
            if not failed_edges and len(path) >= 2:
                # 若没有显式 failed_edges，就把整条路径标记为失败
                for i in range(len(path) - 1):
                    failed_edges.append((path[i], path[i+1], start_frame))
            reward_dict = {p: args.failure_reward for p in policy_weights.keys()}
            for c, a, frame_idx in failed_edges:
                frame_idx = max(0, min(frame_idx, last_frame))
                try:
                    out_queue.put(Transition(
                        graph_frames[frame_idx],
                        _node_idx(c),
                        _node_idx(dst),
                        _node_idx(a),
                        reward_dict,
                        graph_frames[frame_idx],
                        True,
                    ), timeout=0.1)
                except queue.Full:
                    pass

        eps = max(eps_min, eps * eps_decay)

    # 退出时清理
    try:
        param_queue.close()
        out_queue.close()
    except Exception:
        pass


# ------------------------- Learner ------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Multiprocess sampler + GPU learner for GNN-DQN (ELITE)")
    p.add_argument("--net-xml", required=True)
    p.add_argument("--node-features", required=True)
    p.add_argument("--edge-map", default=None, help="edge_id_map.json with static edge info (length)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--comm-radius", type=float, default=300.0)
    p.add_argument("--veh-speed", type=float, default=23.0)
    p.add_argument("--policy", action="append",
                   help="NAME=wp,wa,wb,wg (can repeat)")
    p.add_argument("--failure-reward", type=float, default=-5.0)
    p.add_argument("--step-penalty", type=float, default=-0.5, help="Penalty per hop to discourage long paths")
    p.add_argument("--goal-bonus", type=float, default=5.0, help="Bonus added when reaching destination")
    p.add_argument("--long-hop-threshold", type=int, default=16, help="Hop count after which extra penalty applies")
    p.add_argument("--long-hop-penalty", type=float, default=-0.5, help="Penalty per hop beyond threshold")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--worker-buffer-size", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=28, help="per-episode step limit in sampling")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.995)
    p.add_argument("--baseline-sample-prob", type=float, default=0.3)
    p.add_argument("--bfs-sample-prob", type=float, default=0.2)
    p.add_argument("--random-walk-prob", type=float, default=0.05)
    p.add_argument("--baseline-modes", default="distance,low_queue,low_delay",
                   help="Comma-separated baseline modes: distance,low_queue,low_delay")
    p.add_argument("--log-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-sync", type=int, default=200, help="steps between target net sync + param broadcast")
    return p.parse_args()


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
        weights = tuple(float(p) for p in parts)  # type: ignore
        policies[name.strip()] = weights  # type: ignore
    return policies


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    def _cpu_state_dict(model) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 构建 learner 侧数据
    it_pos, adj, _, speed_map = parse_sumo_net(args.net_xml, skip_internal=True)
    cfg = EnvConfig(comm_radius=args.comm_radius, veh_speed=args.veh_speed)
    node_feats, edge_frames, edge_base, meta = _load_feature_file(args.node_features, args.edge_map)
    if node_feats is None or node_feats.num_frames() == 0:
        raise RuntimeError("--node-features required and must have frames")
    env = DTNEnv(it_pos, adj, cfg, speed_map=speed_map, node_features=node_feats.frame(0))
    converter = GraphConverter()
    edge_feat_dim = int(meta.get("edge_feature_dim", 0)) if isinstance(meta, dict) else 0
    graph_frames = _build_graph_sequence(converter, it_pos, adj, node_feats, edge_frames, edge_base, edge_feat_dim=edge_feat_dim)
    device = args.device
    in_dim = graph_frames[0].x.size(1)
    policy_weights = _parse_policy_specs(args.policy)
    policy_names = list(policy_weights.keys())
    edge_dim = graph_frames[0].edge_attr.size(1) if hasattr(graph_frames[0], "edge_attr") and graph_frames[0].edge_attr is not None else 1
    policy_net = MultiPolicyFedG(in_dim, args.hidden_dim, policy_names, dropout=args.dropout, edge_dim=edge_dim).to(device)
    target_net = MultiPolicyFedG(in_dim, args.hidden_dim, policy_names, dropout=args.dropout, edge_dim=edge_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    ctx = mp.get_context("spawn")
    param_queue: mp.Queue = ctx.Queue(maxsize=1)
    out_queue: mp.Queue = ctx.Queue(maxsize=args.buffer_size)
    stop_event = ctx.Event()

    # 广播初始参数
    param_queue.put(_cpu_state_dict(policy_net))

    workers = []
    for wid in range(args.num_workers):
        p = ctx.Process(
            target=worker_proc,
            args=(wid, args, policy_weights, param_queue, out_queue, stop_event),
            daemon=True,
        )
        p.start()
        workers.append(p)

    eps_logging_interval = max(1, args.episodes // 20)
    train_steps = 0
    start_time = time.time()
    try:
        while train_steps < args.episodes:
            # 拉取样本
            try:
                trans = out_queue.get(timeout=0.1)
                buffer.push(*trans)
            except queue.Empty:
                pass

            loss_dict = multi_head_train_step(
                policy_net,
                target_net,
                optimizer,
                buffer,
                args.batch_size,
                gamma=args.gamma,
                device=device,
                policy_names=policy_names,
            )
            if loss_dict is not None:
                train_steps += 1
                if train_steps % args.target_sync == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    # 发送最新参数给 worker
                    try:
                        if param_queue.empty():
                            param_queue.put(_cpu_state_dict(policy_net))
                    except queue.Full:
                        pass
                if args.log_dir and (train_steps % eps_logging_interval == 0):
                    for pname, loss in loss_dict.items():
                        with open(os.path.join(args.log_dir, f"{pname.lower()}_train_log.csv"), "a") as f:
                            f.write(f"{train_steps},{loss}\n")
                if train_steps % eps_logging_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"[learner] steps={train_steps}/{args.episodes} elapsed={elapsed:.1f}s loss={loss_dict}")
        # 保存模型
        mapping_path = os.path.join(args.output_dir, "id_mapping.json")
        with open(mapping_path, "w") as f:
            import json
            json.dump(converter.id_to_idx, f, indent=2)
        for pname in policy_names:
            out_dir = os.path.join(args.output_dir, pname.lower())
            os.makedirs(out_dir, exist_ok=True)
            torch.save(policy_net.export_single_head_state(pname), os.path.join(out_dir, "fedg_dqn.pt"))
            if not os.path.exists(os.path.join(out_dir, "id_mapping.json")):
                with open(os.path.join(out_dir, "id_mapping.json"), "w") as f:
                    import json
                    json.dump(converter.id_to_idx, f, indent=2)
        print(f"[learner] training complete. models saved to {args.output_dir}")
    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=5)
        try:
            param_queue.close(); out_queue.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
