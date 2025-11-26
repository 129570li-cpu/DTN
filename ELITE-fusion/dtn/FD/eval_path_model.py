#!/usr/bin/env python3
"""
离线评估：K 最短候选路径 + PathPolicyNet 选路质量。
输出连通率、平均跳数、相对最短路比例等指标，可写入日志文件。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple

import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from topology.sumo_net import parse_sumo_net  # type: ignore
from dtn.FD.model import PathPolicyNet  # type: ignore
from dtn.FD.data_converter import GraphConverter  # type: ignore
from dtn.FD.node_features import load_node_feature_timeline  # type: ignore
from dtn.FD.path_utils import (
    load_edge_costs_from_map,
    k_shortest_simple_paths,
    aggregate_path_features,
    path_total_cost,
)  # type: ignore
from dtn.env import DTNEnv, EnvConfig  # type: ignore


def _load_feature_file(path: str):
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
                          edge_feat_dim: int = 0):
    frames = []
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--net-xml", type=str, required=True)
    p.add_argument("--node-features", type=str, required=True)
    p.add_argument("--edge-map", type=str, default=None)
    p.add_argument("--model-dir", type=str, required=True, help="包含 fedg_dqn.pt 的目录")
    p.add_argument("--policy", type=str, default="LDF")
    p.add_argument("--k-paths", type=int, default=5)
    p.add_argument("--max-hops", type=int, default=32)
    p.add_argument("--max-expansions", type=int, default=200000, help="候选路径搜索的最大扩展次数")
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--frame-idx", type=int, default=-1, help="-1 随机，多帧采样")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device and torch.cuda.is_available() else "cpu")

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

    edge_cost = load_edge_costs_from_map(args.edge_map)
    if not edge_cost:
        edge_cost = {u: dict(vs) for u, vs in road_len.items()}
    max_cost = 0.0
    for u, vs in edge_cost.items():
        for _, c in vs.items():
            max_cost = max(max_cost, c)
    max_cost = max(max_cost, 1.0)

    in_dim = graph_frames[0].x.size(1)
    edge_dim = graph_frames[0].edge_attr.size(1) if hasattr(graph_frames[0], "edge_attr") and graph_frames[0].edge_attr is not None else 1
    model = PathPolicyNet(in_dim, args.hidden_dim, [args.policy], path_feat_dim=2, dropout=0.0, edge_dim=edge_dim).to(device)
    ckpt_path = os.path.join(args.model_dir, "fedg_dqn.pt")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    rng = random.Random(123)
    cfg = EnvConfig()
    env = DTNEnv(it_pos, adj, cfg, speed_map=None, node_features=node_tl.frame(0))
    valid = 0
    hop_sum = 0.0
    ratio_sum = 0.0
    cost_sum = 0.0
    no_path = 0
    delay_sum = 0.0
    pdr_sum = 0.0
    rc_sum = 0.0

    for i in range(args.samples):
        frame_idx = args.frame_idx if args.frame_idx >= 0 else rng.randint(0, node_tl.num_frames() - 1)
        data = graph_frames[frame_idx]
        env.set_node_features(node_tl.frame(frame_idx))
        src, dst = rng.sample(list(adj.keys()), 2)
        candidates = k_shortest_simple_paths(adj, edge_cost, src, dst, k=args.k_paths, max_hops=args.max_hops, max_expansions=args.max_expansions)
        if not candidates:
            no_path += 1
            continue
        path_idx, path_edge_feats, path_scalar_feats = aggregate_path_features(
            candidates, converter, data, edge_cost, max_cost, args.max_hops
        )
        if not path_idx:
            no_path += 1
            continue
        with torch.no_grad():
            node_embs = model.get_embeddings(
                data.x.to(device),
                data.edge_index.to(device),
                data.edge_attr.to(device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None,
            )
            q_vals = model.forward_policy(
                args.policy,
                node_embs,
                converter.get_idx(src),
                converter.get_idx(dst),
                path_idx,
                path_edge_feats.to(device),
                path_scalar_feats.to(device),
            )
            act = int(torch.argmax(q_vals).item()) if q_vals.numel() > 0 else 0
        chosen = candidates[act]
        hop = len(chosen) - 1
        shortest_hop = min(len(p) - 1 for p in candidates)
        cost = path_total_cost(chosen, edge_cost)
        hop_sum += hop
        ratio_sum += (float(hop) / float(shortest_hop)) if shortest_hop > 0 else 1.0
        cost_sum += cost
        valid += 1
        # 代理网络指标
        ad = rc = 0.0
        pdr_path = 1.0
        for j in range(1, len(chosen)):
            c, a = chosen[j - 1], chosen[j]
            ad += env.seg_delay(c, a)
            rc += env.seg_ctrl_overhead(c, a)
            pdr_path *= env.seg_delivery_prob(c, a)
        delay_sum += ad
        rc_sum += rc
        pdr_sum += pdr_path

    lines = []
    lines.append(f"Samples: {args.samples}, valid: {valid}, no_path: {no_path}")
    if valid > 0:
        lines.append(f"Avg hops: {hop_sum / valid:.2f}")
        lines.append(f"Avg hop ratio vs shortest: {ratio_sum / valid:.2f}")
        lines.append(f"Avg cost (sum edge length): {cost_sum / valid:.2f}")
        lines.append(f"Avg delay (proxy): {delay_sum / valid:.2f}")
        lines.append(f"Avg PDR (proxy): {pdr_sum / valid:.4f}")
        lines.append(f"Avg RC (proxy): {rc_sum / valid:.2f}")
    out = "\n".join(lines)
    print(out)
    if args.log_file:
        with open(args.log_file, "w") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    main()
