#!/usr/bin/env python3
"""
Offline evaluator for GNN router.
Loads a trained fedg_dqn.pt + id_mapping.json, builds a single-frame graph,
and samples random src/dst pairs to measure reachability, hop count, and loops.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import deque
from typing import Dict, List, Tuple, Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from topology.sumo_net import parse_sumo_net  # type: ignore
from dtn.FD.data_converter import GraphConverter  # type: ignore
from dtn.FD.model import FedG_DQN  # type: ignore
from dtn.FD.node_features import load_node_feature_timeline  # type: ignore


def _load_feature_file(path: str, edge_map_path: Optional[str]):
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
            norm_len = (0.0 if max_len <= 0 else (torch.log(torch.tensor(length + 1.0)) / torch.log(torch.tensor(max_len + 1.0))).item())
            edge_base[key] = [norm_len]
    if not edge_base and edge_frames:
        keys = set()
        for fr in edge_frames:
            keys.update(fr.keys())
        for k in keys:
            edge_base[k] = [0.0]
    return node_tl, edge_frames, edge_base, meta


def build_graph(converter: GraphConverter,
                it_pos: Dict[int, Tuple[float, float]],
                adj: Dict[int, List[int]],
                node_frame: Dict[int, List[float]],
                edge_frame: Dict[str, List[float]],
                edge_base: Dict[str, List[float]]) -> torch.Tensor:
    # merge edge features
    merged: Dict[str, List[float]] = {}
    keys = set(edge_base.keys()) | set(edge_frame.keys())
    base_dim = len(next(iter(edge_base.values()))) if edge_base else 0
    dyn_dim = len(next(iter(edge_frame.values()))) if edge_frame else 0
    expected = base_dim + dyn_dim
    if expected > 0:
        converter.edge_dim = expected
    for k in keys:
        base = edge_base.get(k, [])
        dyn = edge_frame.get(k, [])
        vals = list(base) + list(dyn)
        if expected and len(vals) < expected:
            vals.extend([0.0] * (expected - len(vals)))
        merged[k] = vals
    if getattr(converter, "position_bounds", None) is None:
        converter.set_position_bounds(GraphConverter.infer_position_bounds(it_pos))
    data = converter.build_graph(it_pos, adj, node_features=node_frame, edge_features=merged)
    return data


def shortest_hop(adj: Dict[int, List[int]], src: int, dst: int) -> Optional[int]:
    if src == dst:
        return 0
    q = deque([(src, 0)])
    seen = {src}
    while q:
        node, d = q.popleft()
        for nbr in adj.get(node, []):
            if nbr == dst:
                return d + 1
            if nbr in seen:
                continue
            seen.add(nbr)
            q.append((nbr, d + 1))
    return None


def greedy_path(model: FedG_DQN,
                graph,
                converter: GraphConverter,
                adj: Dict[int, List[int]],
                src: int,
                dst: int,
                max_hops: int) -> List[int]:
    path = [src]
    visited = {src}
    cur = src
    for _ in range(max_hops):
        if cur == dst:
            break
        neighbors = [n for n in adj.get(cur, []) if n not in visited]
        if not neighbors:
            break
        curr_idx = converter.get_idx(cur)
        dest_idx = converter.get_idx(dst)
        neighbor_idx = [converter.get_idx(n) for n in neighbors]
        with torch.no_grad():
            scores = model(
                graph.x,
                graph.edge_index,
                curr_idx,
                dest_idx,
                neighbor_idx,
                graph.edge_attr,
            )
        if scores.numel() == 0:
            break
        best = torch.argmax(scores).item()
        nxt = neighbors[int(best)]
        if nxt in visited:
            path.append(nxt)
            break
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    return path


def evaluate(args):
    torch.set_num_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    it_pos, adj, _, _ = parse_sumo_net(args.net_xml, skip_internal=True)
    node_tl, edge_frames, edge_base, meta = _load_feature_file(args.node_features, args.edge_map)
    if node_tl is None or node_tl.num_frames() == 0:
        raise RuntimeError("Node feature timeline missing frames.")
    frame_idx = min(max(args.frame_index, 0), node_tl.num_frames() - 1)
    node_frame = node_tl.frame(frame_idx)
    edge_frame = edge_frames[frame_idx] if edge_frames else {}

    converter = GraphConverter()
    mapping_path = os.path.join(args.model_dir, "id_mapping.json")
    if not os.path.exists(mapping_path):
        raise RuntimeError(f"id_mapping.json not found in {args.model_dir}")
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    converter.id_to_idx = {int(k): int(v) for k, v in raw.items()}
    converter.idx_to_id = {int(v): int(k) for k, v in raw.items()}
    converter.num_nodes = len(converter.id_to_idx)
    converter.set_position_bounds(GraphConverter.infer_position_bounds(it_pos))

    graph = build_graph(converter, it_pos, adj, node_frame, edge_frame, edge_base).to(args.device)
    in_dim = graph.x.size(1)
    edge_dim = graph.edge_attr.size(1) if hasattr(graph, "edge_attr") and graph.edge_attr is not None else 1

    ckpt = os.path.join(args.model_dir, "fedg_dqn.pt")
    if not os.path.exists(ckpt):
        raise RuntimeError(f"fedg_dqn.pt not found in {args.model_dir}")
    model = FedG_DQN(in_dim, args.hidden_dim, edge_dim=edge_dim).to(args.device)
    state = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(state, strict=True)
    model.eval()

    nodes = list(converter.id_to_idx.keys())
    if len(nodes) < 2:
        raise RuntimeError("Not enough nodes for sampling.")

    reach = 0
    loops = 0
    hop_sum = 0.0
    hop_cnt = 0
    sh_sum = 0.0
    ratio_sum = 0.0
    long_examples: List[Tuple[int, int, int, int]] = []

    for _ in range(args.num_samples):
        src, dst = random.sample(nodes, 2)
        path = greedy_path(model, graph, converter, adj, src, dst, args.max_hops)
        if path and path[-1] == dst:
            reach += 1
            hops = len(path) - 1
            hop_sum += hops
            hop_cnt += 1
            sh = shortest_hop(adj, src, dst)
            if sh is not None and sh > 0:
                sh_sum += sh
                ratio_sum += hops / sh
            if len(set(path)) != len(path):
                loops += 1
            if hops >= args.long_path_threshold and len(long_examples) < 5:
                long_examples.append((src, dst, hops, sh or -1))
        else:
            if path and len(set(path)) != len(path):
                loops += 1

    lines: List[str] = []
    lines.append(f"Samples: {args.num_samples}")
    lines.append(f"Reachability: {reach}/{args.num_samples} = {reach/args.num_samples*100:.2f}%")
    if hop_cnt > 0:
        lines.append(f"Avg hops (reachable): {hop_sum/hop_cnt:.2f}")
    if sh_sum > 0:
        lines.append(f"Avg shortest hops: {sh_sum/hop_cnt:.2f}")
        lines.append(f"Avg hop ratio (model/shortest): {ratio_sum/hop_cnt:.2f}")
    lines.append(f"Loop rate: {loops/args.num_samples*100:.2f}%")
    if long_examples:
        lines.append("Examples of long paths (src,dst,hops,shortest):")
        for src, dst, h, sh in long_examples:
            lines.append(f"  {src}->{dst} hops={h} shortest={sh}")
    output = "\n".join(lines)
    print(output)
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        with open(args.log_file, "w") as f:
            f.write(output + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Offline GNN router evaluator")
    ap.add_argument("--model-dir", required=True, help="Directory containing fedg_dqn.pt and id_mapping.json")
    ap.add_argument("--net-xml", required=True)
    ap.add_argument("--node-features", required=True, help="JSON with node/edge timelines")
    ap.add_argument("--edge-map", required=True, help="edge_id_map.json")
    ap.add_argument("--frame-index", type=int, default=0, help="Which frame to evaluate (default 0)")
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--max-hops", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--long-path-threshold", type=int, default=16)
    ap.add_argument("--log-file", default=None, help="Optional path to write evaluation summary")
    return ap.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
