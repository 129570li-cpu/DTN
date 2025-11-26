#!/usr/bin/env python3
"""
ELITE Controller (ns3-ai) - 路径候选 + GNN-DQN 选路版
流程：
 1) 载入 PathPolicyNet（HRF/LDF/LBF 多策略头），构建图数据（节点+边特征）。
 2) 每次路由请求：生成 K 条最短候选路径（Yen），聚合路径特征，模型打分选一条。
 3) 返回整条路径给 ns-3，避免逐跳 DFS。
环境变量（与 run_elite_ns3_gnn.sh 对齐）：
  ELITE_GNN_MODEL_DIR   模型目录，需包含 fedg_dqn.pt (HRF/LDF/LBF 子目录)
  ELITE_GNN_DEVICE      cpu/cuda（默认 cpu）
  ELITE_GNN_HIDDEN      隐层宽度（默认 128）
  ELITE_GNN_NODE_FEATS  节点/边特征时间序列 JSON
  ELITE_GNN_EDGE_MAP    edge_id_map.json（提供边长）
  ELITE_GNN_K_PATHS     候选路径数，默认 5
  ELITE_GNN_MAX_HOPS    最大跳数，默认 32
"""
from __future__ import annotations
import os
import sys
import time
import signal
import atexit
from typing import List, Dict, Tuple, Optional
import math
import json
import threading
from collections import deque

import numpy as np

try:
    from ctypes import Structure, c_double, c_int, sizeof
    from py_interface import Ns3AIRL, Experiment, EmptyInfo  # type: ignore
    from py_interface import Init as Ns3AIInit  # type: ignore
except Exception:
    Ns3AIRL = None
    Ns3AIInit = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import Global_Par as Gp
from topology.sumo_net import parse_sumo_net
from SDVN_Controller import SDVNController

try:
    import torch  # type: ignore
    from dtn.FD.data_converter import GraphConverter  # type: ignore
    from dtn.FD.model import PathPolicyNet  # type: ignore
    from dtn.FD.node_features import load_node_feature_timeline, NodeFeatureTimeline  # type: ignore
    from dtn.FD.path_utils import (
        load_edge_costs_from_map,
        k_shortest_simple_paths,
        aggregate_path_features,
    )  # type: ignore
except Exception:
    torch = None
    GraphConverter = None
    PathPolicyNet = None
    load_node_feature_timeline = None
    NodeFeatureTimeline = None

# 环境变量配置
USE_GNN_ROUTER = os.environ.get("ELITE_USE_GNN", "1") == "1"
GNN_MODEL_DIR = os.environ.get("ELITE_GNN_MODEL_DIR", "")
GNN_DEVICE = os.environ.get("ELITE_GNN_DEVICE", "cpu")
GNN_HIDDEN_DIM = int(os.environ.get("ELITE_GNN_HIDDEN", "128"))
GNN_NODE_FEATS_PATH = os.environ.get("ELITE_GNN_NODE_FEATS")
GNN_EDGE_MAP_PATH = os.environ.get("ELITE_GNN_EDGE_MAP")
K_PATHS = int(os.environ.get("ELITE_GNN_K_PATHS", "5"))
MAX_HOPS = int(os.environ.get("ELITE_GNN_MAX_HOPS", "32"))
USE_ONLINE_UPDATE = os.environ.get("ELITE_ONLINE_UPDATE", "1") == "1"

# 日志与限制
DEBUG = os.environ.get("ELITE_DEBUG", "0") == "1"
MAX_PATH_LEN = 64  # 必须与 ns-3 侧 ELITE_MAX_PATH_LEN 保持一致
# 与 ns-3 侧保持一致的邻居上限
MAX_NEIGHBORS = 64
# 策略与 policyId 映射（沿用旧版）
POLICY_ID_MAP = {"LDF": 1, "HRF": 2, "LBF": 3, "DIRECT": 4}
ID_TO_POLICY = {v: k for k, v in POLICY_ID_MAP.items()}
DEFAULT_POLICY_WEIGHTS: Dict[str, Tuple[float, float, float, float]] = {
    "HRF": (0.50, 0.20, 0.20, 0.10),
    "LDF": (0.20, 0.10, 0.50, 0.20),
    "LBF": (0.20, 0.10, 0.20, 0.50),
}

# 消息类型常量（与 NS-3 侧对齐）
MSG_TYPE_NORMAL = 0      # 普通消息
MSG_TYPE_EMERGENCY = 1   # 紧急消息
MSG_TYPE_PERIODIC = 2    # 周期性消息

# 网络负载阈值
LOAD_THRESHOLD_HIGH = 0.7   # 高负载阈值
LOAD_THRESHOLD_LOW = 0.3    # 低负载阈值


def select_policy(msg_type: int, queue_len: float, neighbor_count: int, buffer_bytes: float) -> str:
    """
    根据消息类型和网络状态自动选择策略
    
    策略选择规则：
    - 紧急消息 → HRF（高可靠性优先）
    - 高负载（队列长/邻居少/缓冲满）→ LBF（负载均衡优先）
    - 其他情况 → LDF（低时延优先）
    
    Args:
        msg_type: 消息类型 (0=普通, 1=紧急, 2=周期性)
        queue_len: 当前队列长度
        neighbor_count: 邻居车辆数量
        buffer_bytes: 缓冲区使用量
    
    Returns:
        策略名称: "HRF", "LDF", 或 "LBF"
    """
    # 紧急消息始终使用高可靠性策略
    if msg_type == MSG_TYPE_EMERGENCY:
        return "HRF"
    
    # 估算网络负载（简单启发式）
    # 队列长度归一化（假设最大队列 100）
    queue_load = min(1.0, queue_len / 100.0)
    # 邻居稀少表示网络稀疏，可能需要负载均衡
    neighbor_factor = 1.0 if neighbor_count < 3 else 0.0
    # 缓冲区使用归一化（假设最大 1MB）
    buffer_load = min(1.0, buffer_bytes / (1024 * 1024))
    
    # 综合负载评估
    load_score = 0.5 * queue_load + 0.3 * neighbor_factor + 0.2 * buffer_load
    
    if load_score > LOAD_THRESHOLD_HIGH:
        return "LBF"  # 高负载时优先负载均衡
    elif msg_type == MSG_TYPE_PERIODIC:
        # 周期性消息在中等负载时也倾向负载均衡
        return "LBF" if load_score > LOAD_THRESHOLD_LOW else "LDF"
    else:
        return "LDF"  # 默认低时延优先


def _ctrl_log(msg: str) -> None:
    try:
        print(f"[CTRL] {msg}", flush=True)
    except Exception:
        pass

def dlog(msg: str) -> None:
    if DEBUG:
        _ctrl_log(msg)


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        _ctrl_log(f"Received signal {signum}. Exiting.")
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass
    atexit.register(lambda: _ctrl_log("Controller exiting."))


def _load_feature_timeline(path: Optional[str]):
    if not path or load_node_feature_timeline is None:
        return None, None, None, None
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception as exc:
        _ctrl_log(f"Failed to load feature file {path}: {exc}")
        return None, None, None, None
    node_tl = None
    edge_tl = None
    meta = raw.get("meta", {})
    if "nodes" in raw:
        node_tl = load_node_feature_timeline(path)
        if node_tl and node_tl.num_frames() > 0:
            _ctrl_log(f"Loaded node feature timeline: {node_tl.num_frames()} frames.")
    if NodeFeatureTimeline and "edges" in raw and isinstance(raw["edges"], dict):
        raw_edges = raw["edges"]
        first_key = next(iter(raw_edges)) if raw_edges else None
        if first_key:
            vals = raw_edges[first_key]
            num_frames = len(vals) if (isinstance(vals, list) and vals and isinstance(vals[0], list)) else 1
            frames = [{} for _ in range(num_frames)]
            for k, v_list in raw_edges.items():
                if isinstance(v_list, list) and v_list and isinstance(v_list[0], list):
                    limit = min(len(v_list), num_frames)
                    for i in range(limit):
                        frames[i][k] = [float(x) for x in v_list[i]]
                else:
                    frames[0][k] = [float(x) for x in v_list]
            edge_tl = NodeFeatureTimeline(frames, raw.get("meta", {}))
            _ctrl_log(f"Loaded edge feature timeline: {len(frames)} frames.")
    return node_tl, meta, edge_tl, meta

def _load_edge_map(path: Optional[str]):
    if not path:
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        edges = data.get("edges") or []
        return edges
    except Exception as exc:
        _ctrl_log(f"Failed to load edge map {path}: {exc}")
        return None


class PathGNNRouter:
    """
    路径级推理封装：生成 K 最短候选 -> 聚合路径特征 -> PathPolicyNet 打分选路
    """
    def __init__(self,
                 model_dir: str,
                 it_pos: Dict[int, Tuple[float, float]],
                 adj: Dict[int, List[int]],
                 edge_cost: Dict[int, Dict[int, float]],
                 node_tl: Optional["NodeFeatureTimeline"],
                 edge_tl: Optional["NodeFeatureTimeline"],
                 edge_base: Dict[str, List[float]],
                 timeline_step: Optional[float],
                 edge_step: Optional[float],
                 device: str = "cpu",
                 hidden_dim: int = 128):
        if torch is None or GraphConverter is None or PathPolicyNet is None:
            raise RuntimeError("PyTorch or GNN modules unavailable.")
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.it_pos = it_pos
        self.adj = adj
        self.edge_cost = edge_cost
        self.node_tl = node_tl
        self.edge_tl = edge_tl
        self.timeline_step = timeline_step
        self.edge_step = edge_step
        self.converter = GraphConverter()
        self.edge_base = edge_base
        # 构建初始图（frame0）
        init_node = node_tl.frame(0) if node_tl else None
        init_edge: Dict[str, List[float]] = {}
        if edge_tl and edge_tl.num_frames() > 0:
            frame0 = edge_tl.frame(0)
            for k, vals in frame0.items():
                base = edge_base.get(str(k), [])
                init_edge[str(k)] = base + list(vals)
        elif edge_base:
            init_edge = {k: v for k, v in edge_base.items()}
        self.graph = self.converter.build_graph(it_pos, adj, node_features=init_node, edge_features=init_edge)
        self.graph = self.graph.to(self.device)
        in_dim = self.graph.x.size(1)
        edge_dim = self.graph.edge_attr.size(1) if hasattr(self.graph, "edge_attr") and self.graph.edge_attr is not None else 0
        self.edge_dim = edge_dim
        # 发现策略模型
        self.policy_models = self._load_policies(model_dir, in_dim, edge_dim, hidden_dim)
        self.policy_names = list(self.policy_models.keys())
        self.default_policy = "LDF" if "LDF" in self.policy_models else self.policy_names[0]
        # 估算最大路径时延（用于奖励归一化）
        self.max_pair_delay = self._estimate_max_pair_delay()
        # 路径缓存：(src, dst, policy) -> (path, timestamp)
        self._path_cache: Dict[Tuple[int, int, str], Tuple[List[int], float]] = {}
        self._cache_ttl = 2.0  # 缓存有效期（秒）
        self._cache_hits = 0
        self._cache_misses = 0

    def has_node(self, node_id: int) -> bool:
        return node_id in self.adj

    def _estimate_max_pair_delay(self, veh_speed: float = 13.9) -> float:
        """估算最大路径时延（用于奖励归一化）"""
        max_delay = 0.0
        positions = list(self.it_pos.values())
        for i, (x1, y1) in enumerate(positions):
            for x2, y2 in positions[i+1:]:
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                delay = dist / max(veh_speed, 1.0)
                max_delay = max(max_delay, delay)
        return max(max_delay, 30.0)  # 至少 30 秒

    def compute_path_features(self, paths: List[List[int]], max_hops: int = MAX_HOPS):
        if self.edge_cost:
            try:
                dyn_max_cost = max(max(costs.values()) for costs in self.edge_cost.values() if costs)
            except ValueError:
                dyn_max_cost = 1.0
        else:
            dyn_max_cost = 1.0
        return aggregate_path_features(
            paths,
            self.converter,
            self.graph,
            self.edge_cost,
            max_cost=dyn_max_cost,
            max_hops=max_hops,
        )

    def _load_policies(self, model_dir: str, in_dim: int, edge_dim: int, hidden_dim: int):
        models: Dict[str, PathPolicyNet] = {}
        def load_one(path: str, name: str):
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            policy_name = name.upper()
            model = PathPolicyNet(in_dim, hidden_dim, [policy_name], path_feat_dim=2, dropout=0.0, edge_dim=edge_dim).to(self.device)
            model.load_state_dict(state, strict=True)
            model.eval()
            models[policy_name] = model
            # 校验 edge_dim 是否一致
            if hasattr(model, "edge_dim") and model.edge_dim != edge_dim:
                raise RuntimeError(f"Edge feature dim mismatch: model {model.edge_dim} vs graph {edge_dim}")
        if os.path.isfile(os.path.join(model_dir, "fedg_dqn.pt")):
            load_one(os.path.join(model_dir, "fedg_dqn.pt"), "LDF")
        else:
            for sub in os.listdir(model_dir):
                ckpt = os.path.join(model_dir, sub, "fedg_dqn.pt")
                if os.path.isfile(ckpt):
                    load_one(ckpt, sub)
        if not models:
            raise RuntimeError(f"No fedg_dqn.pt under {model_dir}")
        return models

    def _apply_timeline(self, sim_time: float):
        # 组装当前时刻的边特征（基础 + 时间帧）
        edge_feats: Optional[Dict[str, List[float]]] = None
        if self.edge_base:
            edge_feats = {k: list(v) for k, v in self.edge_base.items()}
        if self.edge_tl and self.edge_step and self.edge_base:
            idx = int(sim_time // self.edge_step)
            frame = self.edge_tl.frame(idx)
            if frame:
                if edge_feats is None:
                    edge_feats = {}
                for k, vals in frame.items():
                    base = self.edge_base.get(str(k), [])
                    merged = list(base) + list(vals)
                    edge_feats[str(k)] = merged
                dlog(f"[GNN] edge frame idx={idx} applied (sim={sim_time:.2f})")

        # node timeline
        if self.node_tl and self.timeline_step:
            idx = int(sim_time // self.timeline_step)
            frame = self.node_tl.frame(idx)
            if frame:
                self.graph = self.converter.build_graph(
                    self.it_pos, self.adj, node_features=frame, edge_features=edge_feats
                ).to(self.device)
                dlog(f"[GNN] node frame idx={idx} applied (sim={sim_time:.2f})")
            else:
                # 没有节点帧更新，只更新边特征（保持节点特征不变）
                if edge_feats is not None and hasattr(self.graph, "edge_attr") and self.graph.edge_attr is not None:
                    for k, vals in edge_feats.items():
                        eid = self.converter.edge_key_to_idx.get(str(k))
                        if eid is None:
                            continue
                        merged = vals
                        if len(merged) < self.graph.edge_attr.size(1):
                            merged = merged + [0.0] * (self.graph.edge_attr.size(1) - len(merged))
                        merged = merged[: self.graph.edge_attr.size(1)]
                        self.graph.edge_attr[eid] = torch.tensor(
                            merged,
                            device=self.graph.edge_attr.device,
                            dtype=self.graph.edge_attr.dtype,
                        )

    def compute_path(self, src: int, dst: int, policy: Optional[str], sim_time: float, k_paths: int = 5, max_hops: int = 32) -> List[int]:
        if torch is None:
            return []
        
        policy_key = (policy.upper() if policy else self.default_policy)
        if policy_key not in self.policy_models:
            policy_key = self.default_policy
        
        # === 缓存查找 ===
        cache_key = (src, dst, policy_key)
        if cache_key in self._path_cache:
            cached_path, cached_time = self._path_cache[cache_key]
            if sim_time - cached_time < self._cache_ttl:
                self._cache_hits += 1
                dlog(f"[CACHE] HIT src={src} dst={dst} policy={policy_key} (hits={self._cache_hits})")
                return cached_path
        
        # === 缓存未命中，计算路径 ===
        self._cache_misses += 1
        self._apply_timeline(sim_time)
        candidates = k_shortest_simple_paths(self.adj, self.edge_cost, src, dst, k=k_paths, max_hops=max_hops)
        if not candidates:
            _ctrl_log(f"[GNN] no candidate path for {src}->{dst} (k={k_paths}, max_hops={max_hops})")
            return []
        # 聚合路径特征
        # 与训练一致的归一化：使用当前图的最大边 cost（展开嵌套 dict）
        if self.edge_cost:
            try:
                dyn_max_cost = max(max(costs.values()) for costs in self.edge_cost.values() if costs)
            except ValueError:
                dyn_max_cost = 1.0
        else:
            dyn_max_cost = 1.0
        path_idx, path_edge_feats, path_scalar_feats = aggregate_path_features(
            candidates, self.converter, self.graph, self.edge_cost, max_cost=dyn_max_cost, max_hops=max_hops
        )
        if not path_idx:
            _ctrl_log(f"[GNN] candidate features empty for {src}->{dst}")
            return []
        model = self.policy_models[policy_key]
        with torch.no_grad():
            node_embs = model.get_embeddings(
                self.graph.x.to(self.device),
                self.graph.edge_index.to(self.device),
                self.graph.edge_attr.to(self.device) if hasattr(self.graph, "edge_attr") and self.graph.edge_attr is not None else None,
            )
            q_vals = model.forward_policy(
                policy_key,
                node_embs,
                self.converter.get_idx(src),
                self.converter.get_idx(dst),
                path_idx,
                path_edge_feats.to(self.device),
                path_scalar_feats.to(self.device),
            )
            if q_vals.numel() == 0:
                chosen = candidates[0]
            else:
                act = int(torch.argmax(q_vals).item())
                chosen = candidates[act]
        
        # === 存入缓存 ===
        self._path_cache[cache_key] = (chosen, sim_time)
        
        # 定期清理过期缓存（每 100 次未命中清理一次）
        if self._cache_misses % 100 == 0:
            self._cleanup_cache(sim_time)
        
        dlog(f"[GNN] policy={policy_key} src={src} dst={dst} k={len(candidates)} choose_len={len(chosen)-1} (misses={self._cache_misses})")
        return chosen
    
    def _cleanup_cache(self, current_time: float) -> None:
        """清理过期缓存"""
        expired = [k for k, (_, t) in self._path_cache.items() if current_time - t >= self._cache_ttl]
        for k in expired:
            del self._path_cache[k]
        if expired:
            dlog(f"[CACHE] cleaned {len(expired)} expired entries, remaining={len(self._path_cache)}")


def prepare_topology_and_router(net_xml_path: str) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, float]], PathGNNRouter]:
    it_pos, adj, road_len, _ = parse_sumo_net(net_xml_path, skip_internal=True)
    Gp.it_pos = it_pos
    Gp.adjacents_comb = adj
    Gp.adjacency_dis = road_len
    # edge cost
    edge_cost = {u: dict(vs) for u, vs in road_len.items()}
    edge_map = load_edge_costs_from_map(GNN_EDGE_MAP_PATH)
    if edge_map:
        edge_cost = edge_map
    # edge base for features
    edge_list = _load_edge_map(GNN_EDGE_MAP_PATH) or []
    edge_base: Dict[str, List[float]] = {}
    if edge_list:
        lengths = [float(e.get("length", 0.0)) for e in edge_list if e.get("length", None) is not None]
        max_len = max(lengths) if lengths else 0.0
        for e in edge_list:
            u = int(e.get("u", -1)); v = int(e.get("v", -1))
            if u < 0 or v < 0:
                continue
            key = f"{u}-{v}"
            length = float(e.get("length", 0.0))
            norm_len = math.log(length + 1.0) / math.log(max_len + 1.0) if max_len > 0 else 0.0
            edge_base[key] = [norm_len]
    node_tl, meta, edge_tl, edge_meta = _load_feature_timeline(GNN_NODE_FEATS_PATH)
    router = PathGNNRouter(
        GNN_MODEL_DIR,
        it_pos,
        adj,
        edge_cost,
        node_tl,
        edge_tl,
        edge_base,
        timeline_step=(float(meta.get("window_step", meta.get("window_size", 0.0))) if meta else None),
        edge_step=(float(edge_meta.get("window_step", edge_meta.get("window_size", 0.0))) if edge_meta else None),
        device=GNN_DEVICE,
        hidden_dim=GNN_HIDDEN_DIM,
    )
    _ctrl_log(f"[GNN] router ready policies={router.policy_names}, default={router.default_policy}, k_paths={K_PATHS}, max_hops={MAX_HOPS}, device={GNN_DEVICE}")
    return adj, edge_cost, router


class TrafficFeatureCache:
    """
    保存 queue/neighbor/buffer 的最新值（旧版 Env 里带的实时指标）。
    当前路径版未直接用这些实时指标做特征，但按旧版要求记录日志，后续可扩展为动态特征。
    """
    def __init__(self):
        self.state: Dict[int, Dict[str, float]] = {}
        self.lock = threading.Lock()
    def update(self, area_id: int, queue: float, neighbor: float, buffer_bytes: Optional[float] = None):
        with self.lock:
            entry = self.state.setdefault(int(area_id), {})
            entry["queue"] = max(0.0, queue)
            entry["neighbor"] = max(0.0, neighbor)
            if buffer_bytes is not None:
                entry["buffer"] = max(0.0, buffer_bytes)
    def snapshot(self):
        with self.lock:
            return self.state.copy()


def run_server(net_xml_path: str,
               out_dir: str,
               bandwidth_bps: float,
               comm_radius: float,
               mempool_key: int,
               mem_size: int,
               memblock_key: int):
    if Ns3AIRL is None:
        print("py_interface (ns3-ai) module not found.", file=sys.stderr)
        sys.exit(1)
    adj, edge_cost, gnn_router = prepare_topology_and_router(net_xml_path)
    ctrl = SDVNController(node_num=0, intersection=Gp.it_pos)
    traffic_cache = TrafficFeatureCache()
    feedback_cache: List[Dict[str, float]] = []
    feedback_log_path = os.path.join(out_dir, "online_feedback.csv")
    online_buffers: Dict[str, deque] = {name: deque(maxlen=512) for name in gnn_router.policy_names}
    # 针对在线微调的优化器，复用动量
    online_optimizers: Dict[str, torch.optim.Optimizer] = {
        name: torch.optim.Adam(gnn_router.policy_models[name].parameters(), lr=1e-4)
        for name in gnn_router.policy_names
    }

    # 对齐旧版 ns3-ai 结构（包含邻居等信息；未用字段可忽略但需对齐大小）
    class Env(Structure):
        _pack_ = 1
        _fields_ = [
            ("simTime", c_double),
            ("vehicleId", c_int),
            ("srcId", c_int),
            ("dstId", c_int),
            ("msgTypeId", c_int),
            ("policyIdUsed", c_int),
            ("queueLen", c_double),
            ("bufferBytes", c_double),
            ("neighborCount", c_int),
            ("neighbors", c_int * MAX_NEIGHBORS),
            ("distToNext", c_double * MAX_NEIGHBORS),
            ("routeRequestFlag", c_int),
            ("feedbackFlag", c_int),
            ("success", c_int),
            ("ADsd", c_double),
            ("HCsd", c_double),
            ("RCsd", c_double),
            ("hopCount", c_double),
            ("PL", c_double),
            ("RO", c_double),
            ("nsd", c_int),
            ("seg_ADca", c_double * MAX_PATH_LEN),
            ("seg_HCca", c_double * MAX_PATH_LEN),
            ("seg_lca", c_double * MAX_PATH_LEN),
            ("seg_RCca", c_double * MAX_PATH_LEN),
            ("path_len", c_int),
            ("path_ids", c_int * MAX_PATH_LEN),
        ]

    class Act(Structure):
        _pack_ = 1
        _fields_ = [
            ("policyId", c_int),
            ("path_len", c_int),
            ("path_ids", c_int * MAX_PATH_LEN),
        ]

    # 初始化共享内存并对齐相位（与旧版保持一致：python odd phase, ns-3 even phase）
    try:
        Ns3AIInit(mempool_key, int(mem_size))
    except Exception as e:
        _ctrl_log(f"Failed to init ns3-ai pool key={mempool_key} size={mem_size}: {e}")
        sys.exit(1)
    try:
        ns3ai = Ns3AIRL(memblock_key, Env, Act)
        ns3ai.SetCond(2, 1)
        dlog(f"[NS3-AI] mempool_key={mempool_key} memblock_key={memblock_key} env_bytes={sizeof(Env)} act_bytes={sizeof(Act)}")
    except RuntimeError as e:
        _ctrl_log(f"Failed to register ns3-ai memblock key={memblock_key}: {e}")
        _ctrl_log(f"Hint: ipcrm -M 0x{mempool_key:08x}; ipcrm -M 0x{memblock_key:08x}")
        sys.exit(1)
    _ctrl_log("Shared memory registered.")

    def handle_request(env: Env) -> Act:
        # 反馈模式：用于在线学习/日志，不返回动作
        if env.feedbackFlag != 0:
            pid_name = ID_TO_POLICY.get(int(env.policyIdUsed), None)
            feedback_cache.append({
                "simTime": float(env.simTime),
                "src": int(env.srcId),
                "dst": int(env.dstId),
                "policyId": int(env.policyIdUsed),
                "policyName": pid_name,
                "success": int(env.success),
                "ADsd": float(env.ADsd),
                "HCsd": float(env.HCsd),
                "RCsd": float(env.RCsd),
                "path": [int(env.path_ids[i]) for i in range(max(0, env.path_len))],
            })
            return Act(policyId=0, path_len=0, path_ids=(c_int * MAX_PATH_LEN)(*([0] * MAX_PATH_LEN)))
        if env.routeRequestFlag == 0:
            return Act(policyId=0, path_len=0, path_ids=(c_int * MAX_PATH_LEN)(*([0] * MAX_PATH_LEN)))
        src = int(env.srcId)
        dst = int(env.dstId)
        # ns-3 会在 path_ids[0,1] 里塞入最近路口作为提示；优先使用
        # 路网节点编号是 0..80，0 也是合法路口，不能过滤掉
        hint_src = int(env.path_ids[0]) if env.path_len >= 1 and env.path_ids[0] >= 0 else None
        hint_dst = int(env.path_ids[1]) if env.path_len >= 2 and env.path_ids[1] >= 0 else None
        # 注意 0 也是合法路口 ID，不能用 truthy 判断
        j_src = hint_src if hint_src is not None else src
        j_dst = hint_dst if hint_dst is not None else dst
        if j_src not in gnn_router.adj or j_dst not in gnn_router.adj:
            _ctrl_log(f"[REQ] invalid junction hint src={j_src} dst={j_dst} (orig srcId={src} dstId={dst})")
            return Act(policyId=0, path_len=0, path_ids=(c_int * MAX_PATH_LEN)(*([0] * MAX_PATH_LEN)))
        # 更新实时指标缓存（队列/邻居/缓冲）
        queue_len = float(env.queueLen)
        neighbor_count = int(env.neighborCount)
        buffer_bytes = float(env.bufferBytes)
        msg_type = int(env.msgTypeId)
        traffic_cache.update(src, queue_len, neighbor_count, buffer_bytes)
        
        # 策略选择：优先使用 NS-3 指定的策略，否则自动选择
        policy_name = None
        if int(env.policyIdUsed) > 0:
            policy_name = ID_TO_POLICY.get(int(env.policyIdUsed))
        
        # 如果没有指定策略或策略无效，使用自动选择
        if policy_name is None or policy_name not in gnn_router.policy_names:
            policy_name = select_policy(msg_type, queue_len, neighbor_count, buffer_bytes)
            dlog(f"[AUTO] select_policy -> {policy_name} (msgType={msg_type}, queue={queue_len:.1f}, neighbors={neighbor_count})")
        
        _ctrl_log(f"[REQ] t={env.simTime:.2f}s src={src}->{dst} j_hint={j_src}->{j_dst} policy={policy_name} msgType={msg_type} queue={queue_len:.2f} neighbor={neighbor_count}")
        # 生成候选路径并选路
        path = gnn_router.compute_path(j_src, j_dst, policy_name, sim_time=float(env.simTime), k_paths=K_PATHS, max_hops=MAX_HOPS)
        if not path:
            _ctrl_log(f"[REQ] no path for {src}->{dst}")
            return Act(policyId=0, path_len=0, path_ids=(c_int * MAX_PATH_LEN)(*([0] * MAX_PATH_LEN)))
        plen = min(len(path), MAX_PATH_LEN)
        arr = [int(x) for x in path[:plen]] + [0] * (MAX_PATH_LEN - plen)
        c_path = (c_int * MAX_PATH_LEN)(*arr)
        # policyId 必须用正向映射（字符串 -> 整数），不要用 ID_TO_POLICY
        pid = POLICY_ID_MAP.get(policy_name or gnn_router.default_policy, 1)
        _ctrl_log(f"[RESP] policy={policy_name or gnn_router.default_policy} len={plen} path={path[:min(plen,10)]}{'...' if plen>10 else ''}")
        return Act(policyId=pid, path_len=plen, path_ids=c_path)

    _install_signal_handlers()
    _ctrl_log("Controller ready.")
    try:
        handshake_done = False
        last_flush = time.time()
        online_batch = 8
        while True:
            if handshake_done and ns3ai.isFinish():
                break
            with ns3ai as data:
                if data is None:
                    # 缩短轮询等待，降低端到端延迟（从 0.5ms 减到 0.1ms）
                    time.sleep(0.0001)
                    # 周期性刷盘反馈，供后续离/在线微调使用
                    if feedback_cache and time.time() - last_flush > 0.5:
                        try:
                            header = "simTime,src,dst,policyId,success,ADsd,HCsd,RCsd\n"
                            exists = os.path.exists(feedback_log_path)
                            with open(feedback_log_path, "a") as f:
                                if not exists:
                                    f.write(header)
                                for fb in feedback_cache:
                                    f.write(f"{fb['simTime']},{fb['src']},{fb['dst']},{fb['policyId']},{fb['success']},{fb['ADsd']},{fb['HCsd']},{fb['RCsd']}\n")
                                    # 推入在线 buffer（按策略名分桶）
                                    pname = fb.get("policyName") or ID_TO_POLICY.get(int(fb.get("policyId", 0)), None)
                                    if pname in online_buffers and fb.get("path"):
                                        online_buffers[pname].append(fb)
                                feedback_cache.clear()
                            last_flush = time.time()
                        except Exception as e:
                            dlog(f"[ONLINE] flush error: {e}")
                    # 在线微调：定期从 buffer 取样本做监督更新（逐样本处理，避免 src/dst 混淆）
                    if USE_ONLINE_UPDATE:
                        for pname, buf in online_buffers.items():
                            if len(buf) >= online_batch and pname in gnn_router.policy_models:
                                batch = [buf.pop() for _ in range(online_batch)]
                                model = gnn_router.policy_models[pname]
                                model.train()
                                optim = online_optimizers[pname]
                                optim.zero_grad()
                                
                                # 预计算节点嵌入（整图只需做一次）
                                node_embs = model.get_embeddings(
                                    gnn_router.graph.x.to(gnn_router.device),
                                    gnn_router.graph.edge_index.to(gnn_router.device),
                                    gnn_router.graph.edge_attr.to(gnn_router.device) if hasattr(gnn_router.graph, "edge_attr") and gnn_router.graph.edge_attr is not None else None,
                                )
                                
                                # 逐样本处理，累积损失（每个样本有自己的 src/dst）
                                total_loss = 0.0
                                valid_samples = 0
                                for b in batch:
                                    path = b.get("path")
                                    if not path or len(path) < 2:
                                        continue
                                    
                                    # 计算该样本的奖励
                                    succ = float(b["success"])
                                    ad = b["ADsd"]; hc = b["HCsd"]; rc = b["RCsd"]
                                    wp, wa, wb, wg = DEFAULT_POLICY_WEIGHTS.get(pname, (0.5, 0.2, 0.2, 0.1))
                                    mADsd = max(ad, gnn_router.max_pair_delay)
                                    rad = max(0.0, min(1.0, 1.0 - ad / max(mADsd, 1e-6)))
                                    rrc = 1.0 / (1.0 + rc)
                                    rhc = math.exp(-hc / 10.0)
                                    reward = wp * succ + wa * rad + wb * rhc + wg * rrc
                                    
                                    # 该样本的 src/dst（每个样本独立）
                                    try:
                                        src_idx = gnn_router.converter.get_idx(path[0])
                                        dst_idx = gnn_router.converter.get_idx(path[-1])
                                        path_idx = [[gnn_router.converter.get_idx(node) for node in path]]
                                    except Exception:
                                        continue
                                    
                                    # 计算该路径的特征
                                    _, edge_feat, scalar_feat = gnn_router.compute_path_features([path], max_hops=MAX_HOPS)
                                    if edge_feat is None or scalar_feat is None or edge_feat.numel() == 0:
                                        continue
                                    
                                    # 前向传播：使用正确的 src/dst
                                    q_pred = model.forward_policy(
                                        pname,
                                        node_embs,
                                        src_idx,
                                        dst_idx,
                                        path_idx,
                                        edge_feat.to(gnn_router.device),
                                        scalar_feat.to(gnn_router.device),
                                    )
                                    if q_pred.numel() == 0:
                                        continue
                                    
                                    # 累积损失
                                    target = torch.tensor(reward, device=gnn_router.device, dtype=torch.float)
                                    loss = torch.nn.functional.mse_loss(q_pred.squeeze(), target)
                                    total_loss = total_loss + loss
                                    valid_samples += 1
                                
                                if valid_samples == 0:
                                    model.eval()
                                    continue
                                
                                # 平均损失并反向传播
                                avg_loss = total_loss / valid_samples
                                avg_loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                                optim.step()
                                model.eval()
                                dlog(f"[ONLINE] {pname} samples={valid_samples}/{len(batch)} loss={avg_loss.item():.4f}")
                    continue
                handshake_done = True
                env = data.env
                act = handle_request(env)
                # 填充共享内存中的 act 结构体
                data.act.policyId = int(act.policyId)
                data.act.path_len = int(act.path_len)
                for i in range(min(act.path_len, MAX_PATH_LEN)):
                    data.act.path_ids[i] = int(act.path_ids[i])
    except SystemExit:
        pass
    except Exception as exc:
        _ctrl_log(f"Controller exception: {exc}")
    finally:
        _ctrl_log("Controller stopped.")


if __name__ == "__main__":
    # expects args: net_xml out_dir bandwidth comm_radius mempool_key mem_size memblock_key
    if len(sys.argv) < 8:
        print("Usage: controller_server.py <net_xml> <out_dir> <bandwidth> <comm_radius> <mempool_key> <mem_size> <memblock_key>")
        sys.exit(1)
    net_xml = sys.argv[1]
    out_dir = sys.argv[2]
    bandwidth = float(sys.argv[3])
    comm_radius = float(sys.argv[4])
    mempool_key = int(sys.argv[5])
    mem_size = int(sys.argv[6])
    memblock_key = int(sys.argv[7])
    os.makedirs(out_dir, exist_ok=True)
    # export junction legend
    try:
        legend_path = os.path.join(out_dir, "junction_legend.csv")
        with open(legend_path, "w") as f:
            for jid, (x, y) in parse_sumo_net(net_xml, skip_internal=True)[0].items():
                f.write(f"{jid},{x},{y}\n")
    except Exception as e:
        print(f"[WARN] failed to write junction legend: {e}", file=sys.stderr)
    run_server(net_xml, out_dir, bandwidth, comm_radius, mempool_key, mem_size, memblock_key)
