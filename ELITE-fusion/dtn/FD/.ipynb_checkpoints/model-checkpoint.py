import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FedG_DQN(nn.Module):
    """
    Federated Graph DQN Model for ELITE.
    Combines GraphSAGE (for state representation) and DQN (for routing decision).
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(FedG_DQN, self).__init__()
        
        # -----------------------------------------------------------
        # 1. GraphSAGE Encoder (State Representation)
        # -----------------------------------------------------------
        # Input: Node Features (Speed, Density, Queue Length, etc.)
        # Output: Node Embedding (captures topology + features)
        # We use 2 layers of SAGEConv as recommended in the paper.
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # -----------------------------------------------------------
        # 2. DQN Head (Q-Value Prediction)
        # -----------------------------------------------------------
        # Input: Concatenation of [Current_Node_Emb, Dest_Node_Emb, Neighbor_Node_Emb]
        # Why? 
        # - Current + Dest = "Global State" (Where am I, Where do I go)
        # - Neighbor = "Action" (Which way to turn)
        # This "Link-based" approach allows us to handle variable numbers of neighbors.
        # Input Dim = 3 * hidden_channels
        self.lin1 = nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels) # Output Q-value (scalar)

    def get_embeddings(self, x, edge_index):
        """
        Forward pass of GraphSAGE to get embeddings for ALL nodes.
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Layer 2
        x = self.conv2(x, edge_index)
        # x is now [Num_Nodes, Hidden_Channels]
        return x

    def forward(self, x, edge_index, curr_idx, dest_idx, neighbor_indices):
        """
        Calculate Q-values for all neighbors of the current node.
        
        Args:
            x: Node features [N, F]
            edge_index: Graph connectivity [2, E]
            curr_idx: ID of current node (int)
            dest_idx: ID of destination node (int)
            neighbor_indices: List of neighbor IDs [n1, n2, ...]
            
        Returns:
            q_values: Tensor of shape [Num_Neighbors, 1]
        """
        # 1. Get Graph Embeddings for the whole graph
        node_embs = self.get_embeddings(x, edge_index)
        
        # 2. Extract relevant embeddings
        curr_emb = node_embs[curr_idx] # [H]
        dest_emb = node_embs[dest_idx] # [H]
        
        q_values = []
        for nbr_idx in neighbor_indices:
            nbr_emb = node_embs[nbr_idx] # [H]
            
            # 3. Concatenate: State + Action
            # Input = [Current || Dest || Neighbor]
            cat_input = torch.cat([curr_emb, dest_emb, nbr_emb], dim=0) # [3H]
            
            # 4. MLP Forward
            h = F.relu(self.lin1(cat_input))
            q = self.lin2(h) # [1]
            q_values.append(q)
            
        if not q_values:
            return torch.tensor([])
            
        return torch.stack(q_values) # [Num_Neighbors, 1]
