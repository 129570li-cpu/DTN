import torch
from torch_geometric.data import Data
import numpy as np

class GraphConverter:
    """
    Converts raw ELITE topology data (dicts) into PyTorch Geometric Data objects.
    Handles ID mapping (OSM ID -> 0..N-1 Index).
    """
    def __init__(self):
        self.id_to_idx = {} # Real ID -> 0..N-1
        self.idx_to_id = {} # 0..N-1 -> Real ID
        self.num_nodes = 0
        
    def build_graph(self, it_pos, adj, node_features=None):
        """
        Args:
            it_pos: dict {id: [x, y]}
            adj: dict {id: [neighbor_id, ...]}
            node_features: dict {id: [feat1, feat2...]} (Optional dynamic features)
            
        Returns:
            data: torch_geometric.data.Data
        """
        # 1. Build ID Mapping (if not already built or topology changed)
        # For simplicity, we rebuild if node count changes. 
        # In production, you might want to be smarter about updates.
        current_ids = sorted(list(it_pos.keys()))
        if len(current_ids) != self.num_nodes:
            self._update_mapping(current_ids)
            
        # 2. Build Edge Index
        src_list = []
        dst_list = []
        for u_id, neighbors in adj.items():
            if u_id not in self.id_to_idx: continue
            u_idx = self.id_to_idx[u_id]
            
            for v_id in neighbors:
                if v_id in self.id_to_idx:
                    v_idx = self.id_to_idx[v_id]
                    src_list.append(u_idx)
                    dst_list.append(v_idx)
                    
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        
        # 3. Build Node Features (x)
        # Base features: [Norm_X, Norm_Y, Degree]
        # Dynamic features: [Speed, Density, Queue] (from node_features dict)
        
        feature_list = []
        
        # Normalize positions (simple min-max normalization based on current map)
        # In real deployment, use fixed boundaries.
        pos_values = np.array(list(it_pos.values()))
        min_pos = pos_values.min(axis=0)
        max_pos = pos_values.max(axis=0)
        range_pos = max_pos - min_pos + 1e-6
        
        for i in range(self.num_nodes):
            real_id = self.idx_to_id[i]
            
            # Static Features: Pos, Degree
            pos = it_pos[real_id]
            norm_pos = (np.array(pos) - min_pos) / range_pos
            degree = len(adj.get(real_id, []))
            
            feat = list(norm_pos) + [degree]
            
            # Dynamic Features
            if node_features and real_id in node_features:
                # Assume node_features[id] is a list like [speed, density, queue]
                feat.extend(node_features[real_id])
            else:
                # Default dynamic features (0.0) if missing
                # Adjust dimension based on your actual dynamic features
                feat.extend([0.0, 0.0, 0.0]) 
                
            feature_list.append(feat)
            
        x = torch.tensor(feature_list, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index)

    def _update_mapping(self, sorted_ids):
        self.id_to_idx = {uid: i for i, uid in enumerate(sorted_ids)}
        self.idx_to_id = {i: uid for i, uid in enumerate(sorted_ids)}
        self.num_nodes = len(sorted_ids)
        print(f"[GraphConverter] Updated mapping for {self.num_nodes} nodes.")

    def get_idx(self, real_id):
        return self.id_to_idx.get(real_id, -1)

    def get_id(self, idx):
        return self.idx_to_id.get(idx, -1)
