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
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional

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
from dtn.FD.node_features import load_node_feature_timeline  # type: ignore

DEFAULT_POLICY_WEIGHTS: Dict[str, Tuple[float, float, float, float]] = {
    "HRF": (0.5, 0.25, 0.0, 0.25),
    "LDF": (0.25, 0.5, 0.25, 0.0),
    "LBF": (0.25, 0.25, 0.25, 0.25),
}

# ------------------------- Worker ------------------------- #
def _build_graph_sequence(converter: GraphConverter,
                          it_pos: Dict[int, Tuple[float, float]],
                          adj: Dict[int, List[int]],
                          timeline) -> List[Data]:
    frames: List[Data] = []
    for frame in timeline.iter_frames():
        frames.append(converter.build_graph(it_pos, adj, node_features=frame))
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
    node_feats = load_node_feature_timeline(args.node_features)
    if node_feats is None or node_feats.num_frames() == 0:
        raise RuntimeError("node feature timeline required for worker")
    env = DTNEnv(it_pos, adj, cfg, speed_map=speed_map, node_features=node_feats.frame(0))
    converter = GraphConverter()
    graph_frames = _build_graph_sequence(converter, it_pos, adj, timeline=node_feats)

    in_dim = graph_frames[0].x.size(1)
    policy_names = list(policy_weights.keys())
    policy_net = MultiPolicyFedG(in_dim, args.hidden_dim, policy_names, dropout=args.dropout).to("cpu")

    # 初始参数同步
    try:
        state_dict = param_queue.get(timeout=10)
        policy_net.load_state_dict(state_dict)
    except queue.Empty:
        pass

    buffer = ReplayBuffer(args.worker_buffer_size)
    eps = args.eps_start
    eps_min = args.eps_end
    eps_decay = args.eps_decay
    last_frame = len(graph_frames) - 1

    def epsilon_greedy(graph_idx: int, curr: int, dest: int, neighbors: List[int]) -> int:
        if not neighbors:
            return curr
        if random.random() < eps:
            return random.choice(neighbors)
        g = graph_frames[graph_idx]
        curr_idx = converter.get_idx(curr)
        dest_idx = converter.get_idx(dest)
        neighbor_indices = [converter.get_idx(n) for n in neighbors]
        with torch.no_grad():
            scores = policy_net(
                g.x,
                g.edge_index,
                curr_idx,
                dest_idx,
                neighbor_indices,
            )
        if scores.numel() == 0:
            return random.choice(neighbors)
        best = torch.argmax(scores).item()
        return neighbors[int(best)]

    while not stop_event.is_set():
        # 接收最新参数
        try:
            while True:
                state_dict = param_queue.get_nowait()
                policy_net.load_state_dict(state_dict)
        except queue.Empty:
            pass

        # 采样一条 episode（单帧推进）
        if len(env.it_pos) < 2:
            continue
        src, dst = random.sample(list(env.it_pos.keys()), 2)
        start_frame = random.randint(0, last_frame)
        g = graph_frames[start_frame]
        env.set_node_features(node_feats.frame(start_frame))
        path = [src]
        visited = {src}
        success = False
        depth_limit = args.max_steps
        for _ in range(depth_limit):
            curr = path[-1]
            if curr == dst:
                success = True
                break
            neighbors = [n for n in env.adj.get(curr, []) if n not in visited]
            if not neighbors:
                break
            act = epsilon_greedy(start_frame, curr, dst, neighbors)
            visited.add(act)
            path.append(act)
        # 构造 Transition
        if len(path) >= 2:
            policy_rev_rewards = {}
            # 简化：沿用 trainer 的 _compute_path_reward 逻辑（近似）
            total_ad = 0.0
            total_hc = 0.0
            total_rc = 0.0
            seg_ADca = []
            seg_HCca = []
            seg_lca = []
            seg_RCca = []
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
                RHC = torch.exp(torch.tensor(-HCca * env.cfg.comm_radius / lca)).item()
                RRC = 1.0 / (1.0 + RCca / RCsd)
                PDR = env.seg_delivery_prob(path[seg_ADca.index(ADca)], path[seg_ADca.index(ADca)+1])
                RAD_list.append(RAD); RHC_list.append(RHC); RRC_list.append(RRC); PDR_list.append(PDR)
            for name, (wp, wa, wb, wg) in policy_weights.items():
                rewards = []
                for PDR, RAD, RHC, RRC in zip(PDR_list, RAD_list, RHC_list, RRC_list):
                    rewards.append(wp * PDR + wa * RAD + wb * RHC + wg * RRC)
                policy_rev_rewards[name] = list(reversed(rewards))

            # 将路径各段推入 out_queue
            for i in range(len(path) - 1):
                c = path[i]
                a = path[i + 1]
                idx = len(path) - 2 - i
                reward_dict = {p: policy_rev_rewards[p][idx] for p in policy_weights.keys()}
                done = (a == dst) or (i == len(path) - 2)
                frame_idx = min(start_frame + i, last_frame)
                next_frame_idx = min(frame_idx + 1, last_frame)
                try:
                    out_queue.put(Transition(
                        graph_frames[frame_idx],
                        converter.get_idx(c),
                        converter.get_idx(dst),
                        converter.get_idx(a),
                        reward_dict,
                        graph_frames[next_frame_idx],
                        done,
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
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--comm-radius", type=float, default=300.0)
    p.add_argument("--veh-speed", type=float, default=23.0)
    p.add_argument("--policy", action="append",
                   help="NAME=wp,wa,wb,wg (can repeat)")
    p.add_argument("--failure-reward", type=float, default=-5.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--worker-buffer-size", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=40, help="per-episode step limit in sampling")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.995)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
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

    # 构建 learner 侧数据
    it_pos, adj, _, speed_map = parse_sumo_net(args.net_xml, skip_internal=True)
    cfg = EnvConfig(comm_radius=args.comm_radius, veh_speed=args.veh_speed)
    node_feats = load_node_feature_timeline(args.node_features)
    if node_feats is None or node_feats.num_frames() == 0:
        raise RuntimeError("--node-features required and must have frames")
    env = DTNEnv(it_pos, adj, cfg, speed_map=speed_map, node_features=node_feats.frame(0))
    converter = GraphConverter()
    graph_frames = []
    for frame in node_feats.iter_frames():
        graph_frames.append(converter.build_graph(it_pos, adj, node_features=frame))
    device = args.device
    in_dim = graph_frames[0].x.size(1)
    policy_weights = _parse_policy_specs(args.policy)
    policy_names = list(policy_weights.keys())
    policy_net = MultiPolicyFedG(in_dim, args.hidden_dim, policy_names, dropout=args.dropout).to(device)
    target_net = MultiPolicyFedG(in_dim, args.hidden_dim, policy_names, dropout=args.dropout).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    ctx = mp.get_context("spawn")
    param_queue: mp.Queue = ctx.Queue(maxsize=1)
    out_queue: mp.Queue = ctx.Queue(maxsize=args.buffer_size)
    stop_event = ctx.Event()

    # 广播初始参数
    param_queue.put(policy_net.state_dict())

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
                if train_steps % 10 == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    # 发送最新参数给 worker
                    try:
                        if param_queue.empty():
                            param_queue.put(policy_net.state_dict())
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
