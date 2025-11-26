from typing import Dict
import torch
from torch_geometric.data import Data
import numpy as np

class GraphConverter:
    """
    Converts raw ELITE topology data (dicts) into PyTorch Geometric Data objects.
    Handles ID mapping (OSM ID -> 0..N-1 Index) and allows optional global bounds so
    federated clients share the same coordinate normalization.
    """
    def __init__(self, position_bounds=None):
        self.id_to_idx = {} # Real ID -> 0..N-1
        self.idx_to_id = {} # 0..N-1 -> Real ID
        self.num_nodes = 0
        self.position_bounds = None
        self.dynamic_dim = 0
        self.edge_dim = 0
        self.edge_key_to_idx: Dict[str, int] = {}
        if position_bounds:
            self.set_position_bounds(position_bounds)
        
    def set_position_bounds(self, bounds):
        if bounds is None:
            self.position_bounds = None
            return
        if len(bounds) != 4:
            raise ValueError("Position bounds must be a tuple (min_x,min_y,max_x,max_y).")
        min_x, min_y, max_x, max_y = bounds
        self.position_bounds = (float(min_x), float(min_y), float(max_x), float(max_y))

    @staticmethod
    def infer_position_bounds(it_pos):
        pos_values = np.array(list(it_pos.values()), dtype=float)
        min_vals = pos_values.min(axis=0)
        max_vals = pos_values.max(axis=0)
        return (float(min_vals[0]), float(min_vals[1]), float(max_vals[0]), float(max_vals[1]))
        
    def _resolve_bounds(self, it_pos):
        if self.position_bounds is not None:
            min_x, min_y, max_x, max_y = self.position_bounds
            min_pos = np.array([min_x, min_y], dtype=float)
            max_pos = np.array([max_x, max_y], dtype=float)
        else:
            bounds = self.infer_position_bounds(it_pos)
            min_pos = np.array(bounds[:2], dtype=float)
            max_pos = np.array(bounds[2:], dtype=float)
        range_pos = max_pos - min_pos
        range_pos[range_pos == 0] = 1e-6
        return min_pos, range_pos
        
    def build_graph(self, it_pos, adj, node_features=None, edge_features=None):
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
            
        # 2. Build Edge Index (+edge_attr if provided)
        src_list = []
        dst_list = []
        edge_attr_list = []
        self.edge_key_to_idx = {}
        for u_id, neighbors in adj.items():
            if u_id not in self.id_to_idx: continue
            u_idx = self.id_to_idx[u_id]

            for v_id in neighbors:
                if v_id in self.id_to_idx:
                    v_idx = self.id_to_idx[v_id]
                    src_list.append(u_idx)
                    dst_list.append(v_idx)
                    key = f"{u_id}-{v_id}"
                    self.edge_key_to_idx[key] = len(edge_attr_list)
                    if edge_features is not None:
                        vals = edge_features.get(key, [])
                        if vals and self.edge_dim == 0:
                            self.edge_dim = len(vals)
                        if self.edge_dim:
                            if len(vals) < self.edge_dim:
                                vals = list(vals) + [0.0] * (self.edge_dim - len(vals))
                            edge_attr_list.append(vals)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        
        # 3. Build Node Features (x)
        # Base features: [Norm_X, Norm_Y, Degree]
        # Dynamic features: [Speed, Density, Queue] (from node_features dict)
        
        feature_list = []
        
        # Normalize positions using shared bounds so all clients map coordinates consistently.
        min_pos, range_pos = self._resolve_bounds(it_pos)
        
        dyn_dim = self.dynamic_dim
        if node_features:
            for val in node_features.values():
                if val:
                    dyn_dim = len(val)
                    self.dynamic_dim = dyn_dim
                    break

        for i in range(self.num_nodes):
            real_id = self.idx_to_id[i]
            
            # Static Features: Pos, Degree
            pos = it_pos[real_id]
            norm_pos = (np.array(pos) - min_pos) / range_pos
            degree = len(adj.get(real_id, []))
            
            feat = list(norm_pos) + [degree]
            
            # Dynamic Features
            if node_features and real_id in node_features:
                values = list(node_features[real_id])
                if dyn_dim and len(values) < dyn_dim:
                    values.extend([0.0] * (dyn_dim - len(values)))
                feat.extend(values)
            else:
                feat.extend([0.0] * dyn_dim) 
                
            feature_list.append(feat)
            
        x = torch.tensor(feature_list, dtype=torch.float)
        
        if edge_attr_list:
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
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
