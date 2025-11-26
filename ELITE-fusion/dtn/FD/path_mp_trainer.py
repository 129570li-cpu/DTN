#!/usr/bin/env python3
"""
路径候选集（K 最短路径）+ GNN-DQN 评分的多进程训练器。
- Worker：生成随机 (src,dst,frame) + K 条候选路径，计算每条路径的奖励与特征。
- Learner：GPU 侧评估/选路（epsilon-greedy），并根据即时奖励回传梯度。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import time
import multiprocessing as mp
from typing import Dict, List, Tuple

import torch
from torch.optim import Adam
from torch_geometric.data import Data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from topology.sumo_net import parse_sumo_net  # type: ignore
from dtn.env import DTNEnv, EnvConfig  # type: ignore
from dtn.FD.model import PathPolicyNet  # type: ignore
from dtn.FD.path_dqn_utils import ReplayBuffer, PathTransition, path_train_step  # type: ignore
from dtn.FD.data_converter import GraphConverter  # type: ignore
from dtn.FD.node_features import load_node_feature_timeline  # type: ignore
from dtn.FD.path_utils import (
    load_edge_costs_from_map,
    k_shortest_simple_paths,
    aggregate_path_features,
    path_total_cost,
)  # type: ignore

# (PDR, RAD, RHC, RRC)，各策略的偏好维度拉到 0.5
DEFAULT_POLICY_WEIGHTS: Dict[str, Tuple[float, float, float, float]] = {
    "HRF": (0.50, 0.20, 0.20, 0.10),  # 可靠性/时延偏好最高
    "LDF": (0.20, 0.10, 0.50, 0.20),  # 跳数最优偏好最高
    "LBF": (0.20, 0.10, 0.20, 0.50),  # 开销最优偏好最高
}


def _load_feature_file(path: str) -> Tuple:
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
    return node_tl, edge_frames, meta


def _build_graph_sequence(converter: GraphConverter,
                          it_pos,
                          adj,
                          node_tl,
                          edge_frames,
                          edge_base: Dict[str, List[float]],
                          edge_feat_dim: int = 0) -> List[Data]:
    frames: List[Data] = []
    base_dim = len(next(iter(edge_base.values()))) if edge_base else 0
    expected_dim = base_dim + edge_feat_dim if (base_dim or edge_feat_dim) else 0
    if expected_dim > 0:
        converter.edge_dim = expected_dim
    for idx, frame in enumerate(node_tl.iter_frames()):
        edge_dyn = edge_frames[idx] if idx < len(edge_frames) else {}
        merged: Dict[str, List[float]] = {}
        keys = set(edge_base.keys()) | set(edge_dyn.keys())
        for k in keys:
            base = edge_base.get(k, [])
            dyn = edge_dyn.get(k, [])
            if edge_feat_dim and len(dyn) < edge_feat_dim:
                dyn = list(dyn) + [0.0] * (edge_feat_dim - len(dyn))
            vals = list(base) + list(dyn)
            if expected_dim and len(vals) < expected_dim:
                vals = vals + [0.0] * (expected_dim - len(vals))
            merged[k] = vals
        frames.append(converter.build_graph(it_pos, adj, node_features=frame, edge_features=merged))
    return frames


def _compute_path_metrics(env: DTNEnv, path: List[int]) -> Dict[str, float]:
    ad = rc = length = 0.0
    hops = 0
    pdr = 1.0
    for i in range(1, len(path)):
        c, a = path[i - 1], path[i]
        ad += env.seg_delay(c, a)
        rc += env.seg_ctrl_overhead(c, a)
        length += env.seg_length(c, a)
        hops += env.seg_hops(c, a)
        pdr *= env.seg_delivery_prob(c, a)
    return {
        "ADsd": ad,
        "HCsd": float(hops),
        "RCsd": rc,
        "length": length,
        "PDR": pdr,
        "mADsd": max(env.max_pair_delay, ad, 1e-6),
    }


def _compute_policy_reward(metrics: Dict[str, float],
                           weights: Tuple[float, float, float, float],
                           step_penalty: float,
                           goal_bonus: float,
                           long_hop_threshold: int,
                           long_hop_penalty: float,
                           comm_radius: float) -> float:
    wp, wa, wb, wg = weights
    ADsd = metrics["ADsd"]
    HCsd = metrics["HCsd"]
    RCsd = metrics["RCsd"]
    length = max(metrics["length"], 1e-3)
    PDR = metrics["PDR"]
    mADsd = metrics["mADsd"]
    # 正向时延奖励：时延越小越接近 1，越大越接近 0
    RAD = 1.0 - (ADsd / max(mADsd, 1e-6))
    RAD = max(0.0, min(1.0, RAD))
    RHC = math.exp(-HCsd * comm_radius / length)
    RRC = 1.0 / (1.0 + RCsd)
    reward = wp * PDR + wa * RAD + wb * RHC + wg * RRC
    reward += step_penalty * HCsd
    if HCsd > long_hop_threshold:
        reward += long_hop_penalty * (HCsd - long_hop_threshold)
    reward += goal_bonus
    return reward


def worker_proc(idx: int,
                args: argparse.Namespace,
                policy_weights: Dict[str, Tuple[float, float, float, float]],
                out_queue: mp.Queue,
                stop_event: mp.Event):
    random.seed(args.seed + idx)
    torch.manual_seed(args.seed + idx)
    it_pos, adj, road_len, speed_map = parse_sumo_net(args.net_xml, skip_internal=True)
    node_tl, edge_frames, meta = _load_feature_file(args.node_features)

    edge_base: Dict[str, List[float]] = {}
    lengths = [val for inner in road_len.values() for val in inner.values()]
    max_len = max(lengths) if lengths else 0.0
    for u, nbrs in road_len.items():
        for v, l in nbrs.items():
            norm_len = math.log(l + 1.0) / math.log(max_len + 1.0) if max_len > 0 else 0.0
            edge_base[f"{u}-{v}"] = [norm_len]

    edge_cost = load_edge_costs_from_map(args.edge_map)
    if not edge_cost:
        edge_cost = {u: dict(vs) for u, vs in road_len.items()}
    max_cost = 0.0
    for u, vs in edge_cost.items():
        for v, c in vs.items():
            max_cost = max(max_cost, c)
    max_cost = max(max_cost, 1.0)

    converter = GraphConverter()
    edge_feat_dim = int(meta.get("edge_feature_dim", 0)) if isinstance(meta, dict) else 0
    graph_frames = _build_graph_sequence(converter, it_pos, adj, node_tl, edge_frames, edge_base, edge_feat_dim=edge_feat_dim)

    cfg = EnvConfig(comm_radius=args.comm_radius, veh_speed=args.veh_speed, max_steps=args.max_hops)
    env = DTNEnv(it_pos, adj, cfg, speed_map=speed_map, node_features=node_tl.frame(0))
    last_frame = node_tl.num_frames() - 1

    while not stop_event.is_set():
        frame_idx = random.randint(0, last_frame)
        env.set_node_features(node_tl.frame(frame_idx))
        src, dst = random.sample(list(adj.keys()), 2)
        candidates = k_shortest_simple_paths(
            adj,
            edge_cost,
            src,
            dst,
            k=args.k_paths,
            max_hops=args.max_hops,
            max_expansions=args.max_expansions,
        )
        if not candidates:
            continue
        path_idx, path_edge_tensor, path_scalar_tensor = aggregate_path_features(
            candidates, converter, graph_frames[frame_idx], edge_cost, max_cost, args.max_hops
        )
        if not path_idx:
            continue
        path_rewards: Dict[str, List[float]] = {p: [] for p in policy_weights.keys()}
        for cand in candidates:
            metrics = _compute_path_metrics(env, cand)
            for pname, w in policy_weights.items():
                r = _compute_policy_reward(
                    metrics,
                    w,
                    args.step_penalty,
                    args.goal_bonus,
                    args.long_hop_threshold,
                    args.long_hop_penalty,
                    args.comm_radius,
                )
                path_rewards[pname].append(r)
        out_queue.put({
            "frame_idx": frame_idx,
            "src_idx": converter.get_idx(src),
            "dst_idx": converter.get_idx(dst),
            "paths": path_idx,
            "path_edge_feats": path_edge_tensor.tolist(),
            "path_scalar_feats": path_scalar_tensor.tolist(),
            "path_rewards": path_rewards,
        })


def parse_args():
    p = argparse.ArgumentParser(description="GNN-DQN 路径选择训练器（优化参数版）")
    # === 必需参数 ===
    p.add_argument("--net-xml", type=str, required=True, help="SUMO net.xml 路径")
    p.add_argument("--node-features", type=str, required=True, help="节点/边特征 JSON 路径")
    p.add_argument("--edge-map", type=str, default=None, help="edge_id_map.json，用于长度/成本")
    p.add_argument("--output-dir", type=str, default="dtn_out/gnn_dqn_path_mp", help="模型输出目录")
    
    # === 训练规模参数（已优化）===
    p.add_argument("--episodes", type=int, default=10000, help="训练步数（原500→10000）")
    p.add_argument("--buffer-size", type=int, default=50000, help="经验池容量（原20000→50000）")
    p.add_argument("--batch-size", type=int, default=128, help="批量大小（原64→128）")
    
    # === 网络结构参数 ===
    p.add_argument("--hidden-dim", type=int, default=128, help="隐藏层维度")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率")
    
    # === 学习参数（已优化）===
    p.add_argument("--lr", type=float, default=3e-4, help="学习率（原1e-4→3e-4）")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子（路径级决策不使用）")
    p.add_argument("--eps-start", type=float, default=0.9, help="初始探索率（原0.6→0.9）")
    p.add_argument("--eps-end", type=float, default=0.05, help="最终探索率")
    p.add_argument("--eps-decay", type=float, default=0.9995, help="探索率衰减（原0.995→0.9995）")
    
    # === 路径搜索参数 ===
    p.add_argument("--k-paths", type=int, default=8, help="候选路径数（原5→8）")
    p.add_argument("--max-hops", type=int, default=32, help="最大跳数限制")
    p.add_argument("--max-expansions", type=int, default=200000, help="候选路径搜索的最大扩展次数")
    
    # === 并行与日志参数（已优化）===
    p.add_argument("--num-workers", type=int, default=4, help="工作进程数")
    p.add_argument("--train-every", type=int, default=4, help="每N步训练一次（原10→4）")
    p.add_argument("--log-every", type=int, default=100, help="每N步记录日志（原50→100）")
    
    # === 环境参数 ===
    p.add_argument("--comm-radius", type=float, default=300.0, help="通信半径(m)")
    p.add_argument("--veh-speed", type=float, default=23.0, help="默认车速(m/s)")
    
    # === 奖励塑形参数（已优化）===
    p.add_argument("--step-penalty", type=float, default=-0.01, help="每跳惩罚（原0→-0.01）")
    p.add_argument("--goal-bonus", type=float, default=2.0, help="到达奖励（原5.0→2.0）")
    p.add_argument("--long-hop-threshold", type=int, default=15, help="长路径阈值（原1M→15）")
    p.add_argument("--long-hop-penalty", type=float, default=-0.05, help="超长路径惩罚（原0→-0.05）")
    
    # === 策略配置 ===
    p.add_argument("--policy-weights", type=str, default=None, help="json {name:[wp,wa,wb,wg]}")
    p.add_argument("--behavior-policy", type=str, default="LDF", help="用于 epsilon-greedy 的策略名")
    
    # === 其他 ===
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--gpu", action="store_true", help="使用 GPU 训练")
    p.add_argument("--log-csv", type=str, default=None, help="训练日志 CSV 路径")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    policy_weights = DEFAULT_POLICY_WEIGHTS.copy()
    if args.policy_weights:
        with open(args.policy_weights, "r") as f:
            loaded = json.load(f)
            for k, v in loaded.items():
                if isinstance(v, (list, tuple)) and len(v) == 4:
                    policy_weights[k] = tuple(float(x) for x in v)  # type: ignore
    policy_names = list(policy_weights.keys())
    if args.behavior_policy not in policy_names:
        args.behavior_policy = policy_names[0]

    # 预加载图（用于训练前向），与 worker 独立构建但映射一致
    it_pos, adj, road_len, _ = parse_sumo_net(args.net_xml, skip_internal=True)
    node_tl, edge_frames, meta = _load_feature_file(args.node_features)
    edge_base: Dict[str, List[float]] = {}
    lengths = [val for inner in road_len.values() for val in inner.values()]
    max_len = max(lengths) if lengths else 0.0
    for u, nbrs in road_len.items():
        for v, l in nbrs.items():
            norm_len = math.log(l + 1.0) / math.log(max_len + 1.0) if max_len > 0 else 0.0
            edge_base[f"{u}-{v}"] = [norm_len]
    edge_feat_dim = int(meta.get("edge_feature_dim", 0)) if isinstance(meta, dict) else 0
    converter = GraphConverter()
    graph_frames = _build_graph_sequence(converter, it_pos, adj, node_tl, edge_frames, edge_base, edge_feat_dim=edge_feat_dim)
    in_dim = graph_frames[0].x.size(1)
    edge_dim = graph_frames[0].edge_attr.size(1) if hasattr(graph_frames[0], "edge_attr") and graph_frames[0].edge_attr is not None else 1

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    policy_net = PathPolicyNet(in_dim, args.hidden_dim, policy_names, path_feat_dim=2, dropout=args.dropout, edge_dim=edge_dim).to(device)
    optimizer = Adam(policy_net.parameters(), lr=args.lr)
    memory = ReplayBuffer(args.buffer_size)
    # 日志准备
    log_path = args.log_csv or os.path.join(args.output_dir, "logs", "train.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    last_loss: Dict[str, float] = {}
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("step,eps,num_paths")
            for pname in policy_names:
                f.write(f",action_{pname.lower()},hop_{pname.lower()},reward_{pname.lower()},loss_{pname.lower()}")
            f.write("\n")

    # 启动 worker
    out_queue: mp.Queue = mp.Queue(maxsize=args.num_workers * 2 + 8)
    stop_event = mp.Event()
    workers = []
    for i in range(args.num_workers):
        p = mp.Process(target=worker_proc, args=(i, args, policy_weights, out_queue, stop_event), daemon=True)
        p.start()
        workers.append(p)

    eps = args.eps_start
    total_steps = 0
    last_log = time.time()
    try:
        while total_steps < args.episodes:
            try:
                sample = out_queue.get(timeout=10)
            except queue.Empty:
                print("[Main] waiting for samples...")
                continue
            frame_idx = sample["frame_idx"]
            paths = sample["paths"]
            if not paths:
                continue
            data = graph_frames[frame_idx]
            path_edge_feats = torch.tensor(sample["path_edge_feats"], dtype=torch.float)
            path_scalar_feats = torch.tensor(sample["path_scalar_feats"], dtype=torch.float)
            src_idx = sample["src_idx"]
            dst_idx = sample["dst_idx"]
            path_rewards = sample["path_rewards"]

            # 为每个策略头单独 epsilon-greedy 选路
            with torch.no_grad():
                node_embs = policy_net.get_embeddings(
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None,
                )
                action_dict: Dict[str, int] = {}
                reward_dict: Dict[str, float] = {}
                for pname in policy_names:
                    q_vals = policy_net.forward_policy(
                        pname,
                        node_embs,
                        src_idx,
                        dst_idx,
                        paths,
                        path_edge_feats.to(device),
                        path_scalar_feats.to(device),
                    )
                    if random.random() < eps or q_vals.numel() == 0:
                        act = random.randint(0, len(paths) - 1)
                    else:
                        act = int(torch.argmax(q_vals).item())
                    action_dict[pname] = act
                    reward_dict[pname] = path_rewards.get(pname, [0.0])[act]
            eps = max(args.eps_end, eps * args.eps_decay)

            transition = PathTransition(
                data,
                src_idx,
                dst_idx,
                paths,
                path_edge_feats,
                path_scalar_feats,
                action_dict,
                reward_dict,
            )
            memory.push(*transition)
            total_steps += 1

            if total_steps % args.train_every == 0:
                loss = path_train_step(policy_net, optimizer, memory, args.batch_size, device=device, policy_names=policy_names)
                if loss:
                    last_loss = loss

            if total_steps % args.log_every == 0:
                now = time.time()
                print(f"[Main] step={total_steps}, eps={eps:.3f}, queue={out_queue.qsize()}, dt={now - last_log:.1f}s")
                last_log = now
                # 记录 CSV
                with open(log_path, "a") as f:
                    f.write(f"{total_steps},{eps:.4f},{len(paths)}")
                    for pname in policy_names:
                        act = action_dict.get(pname, -1)
                        hop = len(paths[act]) - 1 if act >= 0 and act < len(paths) else -1
                        rew = reward_dict.get(pname, 0.0)
                        los = last_loss.get(pname, float("nan"))
                        f.write(f",{act},{hop},{rew:.4f},{los:.6f}")
                    f.write("\n")

    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=2)
        # 模型落盘
        for name in policy_names:
            save_path = os.path.join(args.output_dir, name.lower())
            os.makedirs(save_path, exist_ok=True)
            torch.save({"model_state_dict": policy_net.export_single_head_state(name)}, os.path.join(save_path, "fedg_dqn.pt"))
        print(f"[Main] finished training {total_steps} steps. Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
