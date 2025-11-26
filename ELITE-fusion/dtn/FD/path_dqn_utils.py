import random
from collections import deque, namedtuple
from typing import Dict, List

import torch
import torch.nn.functional as F

# Transition for路径级选择
PathTransition = namedtuple(
    "PathTransition",
    (
        "state_data",
        "src_idx",
        "dst_idx",
        "paths",              # List[List[int]] (node indices)
        "path_edge_feats",    # torch.Tensor [P, edge_dim]
        "path_scalar_feats",  # torch.Tensor [P, path_feat_dim]
        "action_dict",        # {policy: action_idx}
        "reward_dict",        # {policy: reward_for_its_action}
    ),
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(PathTransition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def path_train_step(policy_net,
                    optimizer,
                    memory,
                    batch_size,
                    device="cpu",
                    policy_names=None):
    """
    单步路径选择的多头训练：target = immediate reward（终止）。
    """
    if len(memory) < batch_size:
        return None
    policy_names = policy_names or []
    transitions = memory.sample(batch_size)
    optimizer.zero_grad()
    total_loss = 0.0
    policy_loss = {p: 0.0 for p in policy_names}

    # cache embeddings per graph，避免同一图重复做 GNN 前向
    emb_cache: Dict[int, torch.Tensor] = {}

    def get_emb(data):
        key = id(data)
        if key not in emb_cache:
            emb_cache[key] = policy_net.get_embeddings(
                data.x,
                data.edge_index,
                data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None,
            )
        return emb_cache[key]

    for t in transitions:
        data = t.state_data.to(device)
        paths = t.paths
        path_edge = t.path_edge_feats.to(device)
        path_scalar = t.path_scalar_feats.to(device)
        src_idx = t.src_idx
        dst_idx = t.dst_idx
        actions = t.action_dict
        rewards = t.reward_dict

        node_embs = get_emb(data)

        for pname in policy_names:
            if pname not in actions:
                continue
            act = actions.get(pname, 0)
            r = rewards.get(pname, 0.0)
            q_vals = policy_net.forward_policy(pname, node_embs, src_idx, dst_idx, paths, path_edge, path_scalar)
            if q_vals.numel() == 0:
                continue
            pred = q_vals[act]
            target = torch.tensor(r, dtype=torch.float, device=device)
            loss = F.smooth_l1_loss(pred, target)
            total_loss = total_loss + loss
            policy_loss[pname] += loss.item()

    total_loss = total_loss / max(1, len(policy_names) * batch_size)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
    optimizer.step()
    avg_policy_loss = {p: (policy_loss[p] / batch_size) for p in policy_names}
    return avg_policy_loss
