import torch
import torch.nn.functional as F
import random
from collections import deque, namedtuple

# Define a Transition tuple
# state_data: PyG Data object (x, edge_index)
# curr_idx: int (where agent is)
# dest_idx: int (where agent wants to go)
# action_nbr_idx: int (the neighbor node index chosen)
# reward: float
# next_state_data: PyG Data object at t+1
# done: bool
Transition = namedtuple('Transition',
                        ('state_data', 'curr_idx', 'dest_idx', 'action_nbr_idx', 'reward', 'next_state_data', 'done'))

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

def train_step(policy_net, target_net, optimizer, memory, batch_size, gamma=0.99, device='cpu'):
    """
    Performs one step of DQN training (Single Batch Update).
    Adapted for GNN: Processes samples one-by-one (or batched if graphs are same)
    to handle dynamic graph structures if necessary.
    """
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    total_loss = 0
    optimizer.zero_grad()

    # Loop through the batch (Simple approach for dynamic graphs)
    # In a highly optimized version, we would use Batch.from_data_list() from PyG
    for i in range(batch_size):
        state = batch.state_data[i].to(device)
        next_state = batch.next_state_data[i].to(device)
        curr = batch.curr_idx[i]
        dest = batch.dest_idx[i]
        action = batch.action_nbr_idx[i]
        reward = batch.reward[i]
        done = batch.done[i]

        # -------------------------------------------------
        # 1. Compute Q(s, a)
        # -------------------------------------------------
        # We pass [action] as the neighbor list to get the Q-value for the chosen action only
        q_values = policy_net(state.x, state.edge_index, curr, dest, [action])
        current_q = q_values[0] # Scalar tensor

        # -------------------------------------------------
        # 2. Compute Target = r + gamma * max Q(s', a')
        # -------------------------------------------------
        with torch.no_grad():
            if done:
                target_q = torch.tensor(reward, dtype=torch.float, device=device)
            else:
                # In the next state, the agent is at 'action' (the node it moved to)
                new_curr = action 
                
                # Find neighbors of new_curr in next_state
                # edge_index is [2, E]. Find indices where src == new_curr
                src = next_state.edge_index[0]
                dst = next_state.edge_index[1]
                mask = (src == new_curr)
                next_neighbors = dst[mask].tolist()
                
                if not next_neighbors:
                    # Dead end (shouldn't happen in connected road graph, but possible)
                    target_q = torch.tensor(reward, dtype=torch.float, device=device)
                else:
                    # Get Q-values for ALL neighbors in next state
                    next_qs = target_net(next_state.x, next_state.edge_index, new_curr, dest, next_neighbors)
                    max_next_q = next_qs.max()
                    target_q = reward + gamma * max_next_q

        # -------------------------------------------------
        # 3. Accumulate Loss
        # -------------------------------------------------
        # Huber loss or MSE
        loss = F.mse_loss(current_q, target_q)
        total_loss += loss

    # Average loss and Backprop
    avg_loss = total_loss / batch_size
    avg_loss.backward()
    
    # Gradient Clipping (Optional but recommended)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    
    optimizer.step()
    
    return avg_loss.item()
