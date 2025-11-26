import torch
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from typing import Dict, Tuple, List

# Define a Transition tuple
# state_data: PyG Data object (x, edge_index)
# curr_idx: int (where agent is)
# dest_idx: int (where agent wants to go)
# action_nbr_idx: int (the neighbor node index chosen)
# reward: float
# next_state_data: PyG Data object at t+1
# done: bool
Transition = namedtuple('Transition',
                        ('state_data', 'curr_idx', 'dest_idx', 'action_nbr_idx', 'reward_dict', 'next_state_data', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def multi_head_train_step(policy_net,
                          target_net,
                          optimizer,
                          memory,
                          batch_size,
                          gamma=0.99,
                          device='cpu',
                          policy_names=None):
    """
    Multi-head variant: updates every policy head using shared experiences.
    Optimized to avoid per-sample GraphSAGE重复计算：对 batch 中相同 graph 共享一次 embeddings。
    """
    if len(memory) < batch_size:
        return None
    policy_names = policy_names or []
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    optimizer.zero_grad()
    total_loss = 0.0
    policy_loss = {p: 0.0 for p in policy_names}

    # cache embeddings per graph to避免重复 GraphSAGE 前向
    emb_cache: Dict[int, torch.Tensor] = {}
    tgt_emb_cache: Dict[int, torch.Tensor] = {}

    def get_emb(cache: Dict[int, torch.Tensor], model, data):
        key = id(data)
        if key not in cache:
            cache[key] = model.get_embeddings(
                data.x.to(device),
                data.edge_index.to(device),
                data.edge_attr.to(device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None,
            )
        return cache[key]

    def compute_q(model, pname: str, embs: torch.Tensor, curr_idx: int, dest_idx: int, neighbor_indices: List[int]):
        # select policy head
        if hasattr(model, "heads"):
            head = model.heads[pname]
            lin1, lin2 = head["lin1"], head["lin2"]
            dropout_p = getattr(model, "dropout", 0.0)
        else:
            lin1, lin2 = model.lin1, model.lin2
            dropout_p = getattr(model, "dropout", 0.0)
        curr_emb = embs[curr_idx]
        dest_emb = embs[dest_idx]
        q_vals = []
        for nidx in neighbor_indices:
            nbr_emb = embs[nidx]
            cat = torch.cat([curr_emb, dest_emb, nbr_emb], dim=0).unsqueeze(0)
            h = F.relu(lin1(cat))
            if dropout_p and dropout_p > 0:
                h = F.dropout(h, p=dropout_p, training=model.training)
            q = lin2(h).squeeze(-1)
            q_vals.append(q.squeeze(0))
        if not q_vals:
            return torch.empty(0, device=embs.device, dtype=embs.dtype)
        return torch.stack(q_vals)

    for i in range(batch_size):
        state = batch.state_data[i]
        next_state = batch.next_state_data[i]
        curr = batch.curr_idx[i]
        dest = batch.dest_idx[i]
        action = batch.action_nbr_idx[i]
        rewards = batch.reward_dict[i]
        done = batch.done[i]

        state_embs = get_emb(emb_cache, policy_net, state)
        next_embs = get_emb(tgt_emb_cache, target_net, next_state)

        for pname in policy_names:
            reward = rewards.get(pname, 0.0)
            q_values = compute_q(policy_net, pname, state_embs, curr, dest, [action])
            current_q = q_values[0]
            with torch.no_grad():
                if done:
                    target_q = torch.tensor(reward, dtype=torch.float, device=device)
                else:
                    new_curr = action
                    src = next_state.edge_index[0]
                    dst_idx = next_state.edge_index[1]
                    mask = (src == new_curr)
                    masked = dst_idx[mask]
                    next_neighbors = masked.detach().cpu().tolist()
                    if not next_neighbors:
                        target_q = torch.tensor(reward, dtype=torch.float, device=device)
                    else:
                        next_qs = compute_q(target_net, pname, next_embs, new_curr, dest, next_neighbors)
                        max_next_q = next_qs.max()
                        target_q = reward + gamma * max_next_q
            loss = F.smooth_l1_loss(current_q, target_q)
            total_loss = total_loss + loss
            policy_loss[pname] += loss.item()

    total_loss = total_loss / (batch_size * max(1, len(policy_names)))
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
    optimizer.step()
    avg_policy_loss = {p: (policy_loss[p] / batch_size) for p in policy_names}
    return avg_policy_loss
