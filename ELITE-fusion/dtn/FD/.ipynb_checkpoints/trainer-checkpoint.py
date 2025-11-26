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
from typing import Dict, List, Tuple, Optional

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
from model import FedG_DQN
from dqn_utils import ReplayBuffer, Transition


def _build_graph(converter: GraphConverter,
                 it_pos: Dict[int, Tuple[float, float]],
                 adj: Dict[int, List[int]],
                 node_features: Optional[Dict[int, List[float]]] = None) -> Data:
    """Utility wrapper to build PyG Data and move tensors to device later."""
    data = converter.build_graph(it_pos, adj, node_features=node_features)
    return data


class OfflineDQNTrainer:
    """
    Simple epsilon-greedy DQN trainer that uses DTNEnv to generate rollouts.
    Graph is static (road topology), so every transition shares the same PyG Data.
    """

    def __init__(self,
                 env: DTNEnv,
                 graph_data: Data,
                 converter: GraphConverter,
                 hidden_dim: int = 64,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 50000,
                 batch_size: int = 64,
                 device: str = "cpu"):
        self.env = env
        self.graph_data = graph_data.to(device)
        self.converter = converter
        self.device = device
        in_dim = self.graph_data.x.size(1)
        self.policy_net = FedG_DQN(in_dim, hidden_dim).to(device)
        self.target_net = FedG_DQN(in_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.step_count = 0

    def _node_idx(self, real_id: int) -> int:
        idx = self.converter.get_idx(real_id)
        if idx < 0:
            raise ValueError(f"Unknown node id {real_id}")
        return idx

    def _reward(self, c: int, a: int, dest: int) -> float:
        """
        Approximates immediate reward using per-segment metrics as in the paper.
        """
        ADca = self.env.seg_delay(c, a)
        HCca = float(self.env.seg_hops(c, a))
        lca = max(self.env.seg_length(c, a), 1e-6)
        RCca = float(self.env.seg_ctrl_overhead(c, a))
        # Normalize components
        ad_term = math.exp(-ADca / max(self.env.max_pair_delay, 1e-3))
        hc_term = math.exp(-HCca * self.env.cfg.comm_radius / lca)
        rc_term = 1.0 / (1.0 + RCca)
        # Encourage reduction of geometric distance to destination
        dist_now = self.env.seg_length(c, dest)
        dist_next = self.env.seg_length(a, dest)
        progress = (dist_now - dist_next) / max(abs(dist_now) + 1e-6, 1.0)
        return 0.4 * ad_term + 0.3 * hc_term + 0.2 * rc_term + 0.1 * progress

    def epsilon_greedy(self, curr: int, dest: int, eps: float) -> int:
        neighbors = self.env.adj.get(curr, [])
        if not neighbors:
            return curr
        if random.random() < eps:
            return random.choice(neighbors)
        curr_idx = self._node_idx(curr)
        dest_idx = self._node_idx(dest)
        neighbor_indices = [self._node_idx(n) for n in neighbors]
        with torch.no_grad():
            scores = self.policy_net(
                self.graph_data.x,
                self.graph_data.edge_index,
                curr_idx,
                dest_idx,
                neighbor_indices,
            )
            if scores.numel() == 0:
                return random.choice(neighbors)
            best = torch.argmax(scores).item()
            return neighbors[int(best)]

    def collect_episode(self, eps: float, max_steps: int = 64):
        ids = list(self.env.it_pos.keys())
        src = random.choice(ids)
        dst = random.choice(ids)
        if src == dst:
            dst = ids[(ids.index(src) + 1) % len(ids)]
        curr = src
        for _ in range(max_steps):
            neighbors = self.env.adj.get(curr, [])
            if not neighbors:
                break
            action = self.epsilon_greedy(curr, dst, eps)
            reward = self._reward(curr, action, dst)
            done = (action == dst)
            self.buffer.push(
                self.graph_data,
                self._node_idx(curr),
                self._node_idx(dst),
                self._node_idx(action),
                reward,
                self.graph_data,
                done,
            )
            curr = action
            if done:
                break

    def train_step(self):
        loss = dqn_train_step(
            self.policy_net,
            self.target_net,
            self.optimizer,
            self.buffer,
            self.batch_size,
            gamma=self.gamma,
            device=self.device,
        )
        return loss

    def soft_update(self, tau: float = 0.01):
        for t_param, p_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t_param.data.copy_(tau * p_param.data + (1.0 - tau) * t_param.data)

    def train(self,
              episodes: int,
              eps_start: float = 1.0,
              eps_end: float = 0.05,
              eps_decay: float = 0.995,
              target_sync_interval: int = 10):
        history = []
        eps = eps_start
        for ep in range(1, episodes + 1):
            self.collect_episode(eps)
            loss = self.train_step()
            if loss is not None:
                history.append((ep, loss, eps))
            if ep % target_sync_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            eps = max(eps_end, eps * eps_decay)
        return history

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "fedg_dqn.pt")
        torch.save(self.policy_net.state_dict(), model_path)
        mapping_path = os.path.join(out_dir, "id_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(self.converter.id_to_idx, f, indent=2)
        return model_path, mapping_path


def dqn_train_step(policy_net, target_net, optimizer, memory, batch_size, gamma=0.99, device="cpu"):
    """Thin wrapper so we can reuse the existing logic without modifying original module."""
    if len(memory) < batch_size:
        return None
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    optimizer.zero_grad()
    losses = []
    for i in range(batch_size):
        state = batch.state_data[i]
        next_state = batch.next_state_data[i]
        curr = batch.curr_idx[i]
        dest = batch.dest_idx[i]
        action = batch.action_nbr_idx[i]
        reward = batch.reward[i]
        done = batch.done[i]
        q_values = policy_net(state.x, state.edge_index, curr, dest, [action])
        current_q = q_values[0]
        with torch.no_grad():
            if done:
                target = torch.tensor(reward, dtype=current_q.dtype, device=device)
            else:
                src = next_state.edge_index[0]
                dst_idx = next_state.edge_index[1]
                mask = (src == action)
                next_neighbors = dst_idx[mask].tolist()
                if not next_neighbors:
                    target = torch.tensor(reward, dtype=current_q.dtype, device=device)
                else:
                    next_qs = target_net(next_state.x, next_state.edge_index, action, dest, next_neighbors)
                    target = reward + gamma * next_qs.max()
        loss = torch.nn.functional.mse_loss(current_q, target)
        losses.append(loss)
    total = torch.stack(losses).mean()
    total.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
    optimizer.step()
    return float(total.detach().cpu())


def parse_args():
    parser = argparse.ArgumentParser(description="Offline GNN-DQN trainer for ELITE (non-federated).")
    parser.add_argument("--net-xml", required=True, help="Path to SUMO net.xml")
    parser.add_argument("--output-dir", default="dtn_out/gnn_dqn", help="Directory to save checkpoints")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--comm-radius", type=float, default=300.0)
    parser.add_argument("--veh-speed", type=float, default=23.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-file", default=None, help="Optional CSV log to record loss/epsilon")
    return parser.parse_args()


def main():
    args = parse_args()
    it_pos, adj, _, speed_map = parse_sumo_net(args.net_xml, skip_internal=True)
    cfg = EnvConfig(comm_radius=args.comm_radius, veh_speed=args.veh_speed)
    env = DTNEnv(it_pos, adj, cfg, speed_map=speed_map)
    converter = GraphConverter()
    data = _build_graph(converter, it_pos, adj, node_features=None)
    trainer = OfflineDQNTrainer(
        env,
        data,
        converter,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    history = trainer.train(args.episodes)
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        with open(args.log_file, "w") as f:
            f.write("episode,loss,epsilon\n")
            for ep, loss, eps in history:
                f.write(f"{ep},{loss},{eps}\n")
    model_path, mapping_path = trainer.save(args.output_dir)
    print(f"[trainer] model saved to {model_path}")
    print(f"[trainer] id mapping saved to {mapping_path}")


if __name__ == "__main__":
    main()
