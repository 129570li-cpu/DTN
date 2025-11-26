import json
import heapq
from typing import Dict, List, Tuple, Optional, Set


def load_edge_costs_from_map(edge_map_path: Optional[str], default_cost: float = 1.0) -> Dict[int, Dict[int, float]]:
    """
    Parse edge_id_map.json 风格的文件，返回 {u:{v:cost}}。
    """
    costs: Dict[int, Dict[int, float]] = {}
    if not edge_map_path:
        return costs
    with open(edge_map_path, "r") as f:
        raw = json.load(f)
    edges = raw.get("edges", [])
    for e in edges:
        try:
            u = int(e.get("u", -1))
            v = int(e.get("v", -1))
        except Exception:
            continue
        if u < 0 or v < 0:
            continue
        length = float(e.get("length", default_cost))
        costs.setdefault(u, {})[v] = length
    return costs


def k_shortest_simple_paths(adj: Dict[int, List[int]],
                            cost: Dict[int, Dict[int, float]],
                            src: int,
                            dst: int,
                            k: int = 5,
                            max_hops: int = 64,
                            max_expansions: int = 200000) -> List[List[int]]:
    """
    Yen 风格的 K 最短简单路径（基于 Dijkstra 的替代）。
    - 先跑一次 Dijkstra 拿最短路 P0
    - 后续迭代生成偏离路径，存入小根堆，弹出代价最小的未重复路径
    - 邻居去重避免多车道重复
    - max_expansions 仍做保险，防止极端爆炸
    """
    def dijkstra(start: int, banned_edges: Set[Tuple[int, int]]) -> Tuple[float, List[int]]:
        pq: List[Tuple[float, int, List[int]]] = [(0.0, start, [start])]
        visited_cost: Dict[int, float] = {}
        while pq:
            dist, u, path = heapq.heappop(pq)
            if u == dst:
                return dist, path
            if dist > visited_cost.get(u, float("inf")):
                continue
            for v in set(adj.get(u, [])):
                if (u, v) in banned_edges:
                    continue
                if v in path:
                    continue
                nd = dist + cost.get(u, {}).get(v, 1.0)
                if nd < visited_cost.get(v, float("inf")) and len(path) < max_hops + 1:
                    visited_cost[v] = nd
                    heapq.heappush(pq, (nd, v, path + [v]))
        return float("inf"), []

    k_paths: List[List[int]] = []
    expansions = 0
    base_cost, base_path = dijkstra(src, set())
    if not base_path:
        return k_paths
    k_paths.append(base_path)
    candidates: List[Tuple[float, List[int]]] = []

    for _ in range(1, k):
        last_path = k_paths[-1]
        if not last_path:
            break
        for i in range(len(last_path) - 1):
            spur_node = last_path[i]
            root_path = last_path[: i + 1]
            banned: Set[Tuple[int, int]] = set()
            # ban edges that would recreate same prefix
            for p in k_paths:
                if len(p) > i and p[: i + 1] == root_path:
                    banned.add((p[i], p[i + 1]))
            spur_cost, spur_path = dijkstra(spur_node, banned)
            expansions += 1
            if expansions > max_expansions:
                break
            if spur_path:
                # stitch root + spur (excluding spur duplicate)
                new_path = root_path[:-1] + spur_path
                if new_path not in k_paths:
                    new_cost = path_total_cost(new_path, cost)
                    heapq.heappush(candidates, (new_cost, new_path))
        if not candidates:
            break
        _, next_path = heapq.heappop(candidates)
        k_paths.append(next_path)
        if expansions > max_expansions:
            break
    return k_paths


def path_total_cost(path: List[int], cost: Dict[int, Dict[int, float]], default_cost: float = 1.0) -> float:
    total = 0.0
    for i in range(1, len(path)):
        u, v = path[i - 1], path[i]
        total += cost.get(u, {}).get(v, default_cost)
    return total


def aggregate_path_features(paths: List[List[int]],
                            converter,
                            graph_data,
                            edge_cost: Dict[int, Dict[int, float]],
                            max_cost: float,
                            max_hops: int) -> Tuple:
    """
    将路径转换为
      - idx 路径列表（供 GNN 索引）
      - 边特征均值张量 [num_paths, edge_dim]
      - 路径尺度特征张量 [num_paths, 2] = [hop_norm, cost_norm]
    """
    import torch  # 局部导入避免硬依赖

    edge_attr = graph_data.edge_attr if hasattr(graph_data, "edge_attr") else None
    edge_dim = edge_attr.size(1) if edge_attr is not None else 0
    path_edge_feats = []
    path_scalar_feats = []
    path_idx_list: List[List[int]] = []
    default_cost = 1.0

    for p in paths:
        idx_path: List[int] = []
        edge_feats = []
        invalid = False
        for nid in p:
            idx = converter.get_idx(nid)
            if idx < 0:
                invalid = True
                break
            idx_path.append(idx)
        if invalid:
            continue
        for i in range(1, len(p)):
            u_real, v_real = p[i - 1], p[i]
            key = f"{u_real}-{v_real}"
            ei = converter.edge_key_to_idx.get(key, None)
            if edge_attr is not None and ei is not None:
                edge_feats.append(edge_attr[ei])
        if edge_feats:
            edge_stack = torch.stack(edge_feats, dim=0)
            edge_mean = edge_stack.mean(dim=0)
        else:
            edge_mean = torch.zeros(edge_dim, dtype=torch.float)
        hops = max(len(p) - 1, 1)
        cost = path_total_cost(p, edge_cost, default_cost=default_cost)
        hop_norm = float(hops) / float(max_hops) if max_hops > 0 else float(hops)
        cost_norm = float(cost) / float(max_cost) if max_cost > 0 else float(cost)
        path_edge_feats.append(edge_mean)
        path_scalar_feats.append(torch.tensor([hop_norm, cost_norm], dtype=torch.float))
        path_idx_list.append(idx_path)

    if path_edge_feats:
        path_edge_tensor = torch.stack(path_edge_feats, dim=0)
    else:
        path_edge_tensor = torch.zeros((0, edge_dim), dtype=torch.float)
    if path_scalar_feats:
        path_scalar_tensor = torch.stack(path_scalar_feats, dim=0)
    else:
        path_scalar_tensor = torch.zeros((0, 2), dtype=torch.float)
    return path_idx_list, path_edge_tensor, path_scalar_tensor
