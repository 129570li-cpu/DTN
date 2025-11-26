import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch_geometric.nn import GINEConv

class FedG_DQN(nn.Module):
    """
    Federated Graph DQN Model for ELITE (edge-aware).
    Uses GINEConv with edge_attr; edge_attr 必须提供。
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout: float = 0.1, edge_dim: int = 1):
        super(FedG_DQN, self).__init__()
        self.in_channels = in_channels
        self.edge_dim = edge_dim
        mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        mlp2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn=mlp1, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn=mlp2, edge_dim=edge_dim)
        self.lin1 = nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels) # Output Q-value (scalar)
        self.dropout = dropout

    def get_embeddings(self, x, edge_index, edge_attr):
        """
        Forward pass of encoder to get embeddings for ALL nodes.
        """
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        if self.dropout and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        if self.dropout and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, curr_idx, dest_idx, neighbor_indices, edge_attr):
        """
        Calculate Q-values for all neighbors of the current node.
        """
        node_embs = self.get_embeddings(x, edge_index, edge_attr)
        curr_emb = node_embs[curr_idx] # [H]
        dest_emb = node_embs[dest_idx] # [H]
        q_values = []
        for nbr_idx in neighbor_indices:
            nbr_emb = node_embs[nbr_idx] # [H]
            cat_input = torch.cat([curr_emb, dest_emb, nbr_emb], dim=0).unsqueeze(0) # [1,3H]
            h = F.relu(self.lin1(cat_input))
            if self.dropout and self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            q = self.lin2(h).squeeze(-1) # [1] -> scalar
            q_values.append(q.squeeze(0))
        if not q_values:
            return torch.empty(0, device=node_embs.device, dtype=node_embs.dtype)
        return torch.stack(q_values) # [Num_Neighbors]


class MultiPolicyFedG(nn.Module):
    """
    Shared GINE encoder with multiple DQN heads (one per policy).
    Each policy head has its own MLP while the topology encoder is shared.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 policy_names,
                 out_channels: int = 1,
                 dropout: float = 0.1,
                 edge_dim: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.policy_names = list(policy_names)
        self.edge_dim = edge_dim

        mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        mlp2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn=mlp1, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn=mlp2, edge_dim=edge_dim)

        self.heads = nn.ModuleDict()
        for name in self.policy_names:
            self.heads[name] = nn.ModuleDict({
                "lin1": nn.Linear(3 * hidden_channels, hidden_channels),
                "lin2": nn.Linear(hidden_channels, out_channels),
            })

    def get_embeddings(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        if self.dropout and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        if self.dropout and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, curr_idx, dest_idx, neighbor_indices, edge_attr):
        if not self.policy_names:
            raise RuntimeError("MultiPolicyFedG has no policy heads.")
        return self.forward_policy(self.policy_names[0], x, edge_index, curr_idx, dest_idx, neighbor_indices, edge_attr)

    def forward_policy(self, policy_name: str, x, edge_index, curr_idx, dest_idx, neighbor_indices, edge_attr):
        if policy_name not in self.heads:
            raise ValueError(f"Unknown policy head {policy_name}")
        node_embs = self.get_embeddings(x, edge_index, edge_attr)
        curr_emb = node_embs[curr_idx]
        dest_emb = node_embs[dest_idx]
        lin1 = self.heads[policy_name]["lin1"]
        lin2 = self.heads[policy_name]["lin2"]

        q_values = []
        for nbr_idx in neighbor_indices:
            nbr_emb = node_embs[nbr_idx]
            cat_input = torch.cat([curr_emb, dest_emb, nbr_emb], dim=0).unsqueeze(0)
            h = F.relu(lin1(cat_input))
            if self.dropout and self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            q = lin2(h).squeeze(-1)
            q_values.append(q.squeeze(0))
        if not q_values:
            return torch.empty(0, device=node_embs.device, dtype=node_embs.dtype)
        return torch.stack(q_values)

    def export_single_head_state(self, policy_name: str):
        """
        Export a state_dict compatible with FedG_DQN for a given policy head.
        """
        if policy_name not in self.heads:
            raise ValueError(f"Unknown policy head {policy_name}")
        single = FedG_DQN(self.in_channels, self.hidden_channels, dropout=self.dropout, edge_dim=self.edge_dim)
        single_state = single.state_dict()
        multi_state = self.state_dict()
        for name in single_state.keys():
            if name.startswith("lin"):
                multi_key = f"heads.{policy_name}.{name}"
            else:
                multi_key = name
            single_state[name] = multi_state[multi_key]
        return single_state


class PathPolicyNet(nn.Module):
    """
    路径级多策略 GNN-DQN：
    - 先对整图做 GINE 编码拿到节点嵌入
    - 对候选路径做节点均值聚合 + 边特征均值 + 额外路径特征
    - 每个策略一个 head 输出该路径的 Q
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 policy_names,
                 path_feat_dim: int,
                 out_channels: int = 1,
                 dropout: float = 0.1,
                 edge_dim: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.policy_names = list(policy_names)
        self.edge_dim = edge_dim
        self.path_feat_dim = path_feat_dim

        mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        mlp2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn=mlp1, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn=mlp2, edge_dim=edge_dim)

        # 输入维度：src_emb + dst_emb + path_emb + edge_agg + path_feats
        fused_dim = 3 * hidden_channels + edge_dim + path_feat_dim
        self.heads = nn.ModuleDict()
        for name in self.policy_names:
            self.heads[name] = nn.ModuleDict({
                "lin1": nn.Linear(fused_dim, hidden_channels),
                "lin2": nn.Linear(hidden_channels, out_channels),
            })

    def get_embeddings(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        if self.dropout and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        if self.dropout and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward_policy(self,
                       policy_name: str,
                       node_embs: torch.Tensor,
                       src_idx: int,
                       dst_idx: int,
                       paths: List[List[int]],
                       path_edge_feats: torch.Tensor,
                       path_scalar_feats: torch.Tensor):
        if policy_name not in self.heads:
            raise ValueError(f"Unknown policy head {policy_name}")
        lin1 = self.heads[policy_name]["lin1"]
        lin2 = self.heads[policy_name]["lin2"]
        src_emb = node_embs[src_idx]
        dst_emb = node_embs[dst_idx]
        q_list = []
        for i, p in enumerate(paths):
            if not p:
                q_list.append(torch.tensor(0.0, device=node_embs.device, dtype=node_embs.dtype))
                continue
            path_emb = node_embs[p].mean(dim=0)
            edge_feat = path_edge_feats[i] if path_edge_feats.numel() > 0 else torch.zeros(self.edge_dim, device=node_embs.device, dtype=node_embs.dtype)
            scalar_feat = path_scalar_feats[i]
            fused = torch.cat([src_emb, dst_emb, path_emb, edge_feat, scalar_feat], dim=0).unsqueeze(0)
            h = F.relu(lin1(fused))
            if self.dropout and self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            q = lin2(h).squeeze(-1)
            q_list.append(q.squeeze(0))
        if not q_list:
            return torch.empty(0, device=node_embs.device, dtype=node_embs.dtype)
        return torch.stack(q_list)

    def forward_all(self,
                    data,
                    paths: List[List[int]],
                    path_edge_feats: torch.Tensor,
                    path_scalar_feats: torch.Tensor,
                    src_idx: int,
                    dst_idx: int,
                    policy_name: str):
        node_embs = self.get_embeddings(
            data.x.to(path_edge_feats.device),
            data.edge_index.to(path_edge_feats.device),
            data.edge_attr.to(path_edge_feats.device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None,
        )
        return self.forward_policy(policy_name, node_embs, src_idx, dst_idx, paths, path_edge_feats, path_scalar_feats)

    def export_single_head_state(self, policy_name: str):
        if policy_name not in self.heads:
            raise ValueError(f"Unknown policy head {policy_name}")
        single = PathPolicyNet(
            self.in_channels,
            self.hidden_channels,
            [policy_name],
            path_feat_dim=self.path_feat_dim,
            dropout=self.dropout,
            edge_dim=self.edge_dim,
        )
        single_state = single.state_dict()
        multi_state = self.state_dict()
        for name in single_state.keys():
            if name in multi_state:
                single_state[name] = multi_state[name]
        return single_state
