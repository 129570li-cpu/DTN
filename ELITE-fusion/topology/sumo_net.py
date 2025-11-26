"""
SUMO net.xml parser to construct junction positions and adjacency for ELITE controller.
Produces:
  - it_pos:     Ordered dict {junction_id:int -> [x:float, y:float]}
  - adj:        Ordered dict {junction_id:int -> [neighbor_ids...]}
  - road_len:   dict-of-dict road_len[u][v] = Euclidean distance
  - speed_map:  dict-of-dict speed_map[u][v] = average lane speed (m/s) on edge u->v, if available
Notes:
  - Skips internal junctions (type="internal") and non-traffic nodes.
  - Uses <edge from="..." to="..."> to build adjacency (directed; reverse exists only if an opposite edge exists).
  - Lane speeds are parsed from <lane speed="..."> under <edge>; when absent, speed_map may not contain that edge.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from collections import OrderedDict, defaultdict
import xml.etree.ElementTree as ET
import math

def parse_sumo_net(net_path: str,
                   skip_internal: bool = True,
                   include_lane_map: bool = False):
    tree = ET.parse(net_path)
    root = tree.getroot()
    # SUMO net uses namespace sometimes; we ignore it by tag.endswith
    def tags(name: str):
        return root.findall(".//" + name) + root.findall(".//{*}" + name)
    # Junctions
    it_pos_tmp: Dict[str, Tuple[float, float]] = {}
    for j in tags("junction"):
        jid = j.attrib.get("id")
        jtype = j.attrib.get("type", "")
        if skip_internal and jtype == "internal":
            continue
        try:
            x = float(j.attrib.get("x", "0"))
            y = float(j.attrib.get("y", "0"))
        except Exception:
            continue
        it_pos_tmp[jid] = (x, y)
    # Edges
    adj_tmp: Dict[str, List[str]] = defaultdict(list)
    edge_speed_tmp: Dict[Tuple[str, str], float] = {}
    lane_edge_tmp: Dict[str, Tuple[str, str]] = {} if include_lane_map else {}
    for e in tags("edge"):
        # skip internal or function edges
        if e.attrib.get("function", "") in ("internal", "connector"):
            continue
        frm = e.attrib.get("from")
        to = e.attrib.get("to")
        if frm in it_pos_tmp and to in it_pos_tmp:
            adj_tmp[frm].append(to)
            # average lane speed for this directed edge if available
            spds: List[float] = []
            for ln in e.findall("./lane") + e.findall("./{*}lane"):
                try:
                    sp = float(ln.attrib.get("speed", "0"))
                    if sp > 0:
                        spds.append(sp)
                except Exception:
                    pass
                if include_lane_map:
                    lane_id = ln.attrib.get("id")
                    if lane_id:
                        lane_edge_tmp[lane_id] = (frm, to)
            if spds:
                edge_speed_tmp[(frm, to)] = sum(spds) / float(len(spds))
    # Make reverse if both directions exist
    for u, nbrs in list(adj_tmp.items()):
        for v in nbrs:
            if u in adj_tmp.get(v, []):
                # bidirectional
                continue
            # keep directed if reverse not present
    # Convert ids to integers where possible else hashable stable ints
    # Build mapping str->int
    id_map: Dict[str, int] = {}
    rev_map: Dict[int, str] = {}
    next_id = 0
    for jid in it_pos_tmp.keys():
        try:
            iid = int(jid)
        except Exception:
            iid = next_id
            next_id += 1
        # ensure unique
        while iid in rev_map:
            iid += 1
        id_map[jid] = iid
        rev_map[iid] = jid
    it_pos: "OrderedDict[int, List[float]]" = OrderedDict()
    for jid, (x, y) in it_pos_tmp.items():
        it_pos[id_map[jid]] = [x, y]
    adj: "OrderedDict[int, List[int]]" = OrderedDict()
    for jid in it_pos.keys():
        adj[jid] = []
    for u_str, nbrs in adj_tmp.items():
        u = id_map[u_str]
        for v_str in nbrs:
            if v_str in id_map:
                v = id_map[v_str]
                adj[u].append(v)
    # Remove isolated nodes and enforce symmetry by pruning 1-degree dead ends if needed (optional)
    # Compute road lengths
    def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)
    road_len: Dict[int, Dict[int, float]] = defaultdict(dict)
    speed_map: Dict[int, Dict[int, float]] = defaultdict(dict)
    for u, nbrs in adj.items():
        pu = (it_pos[u][0], it_pos[u][1])
        for v in nbrs:
            pv = (it_pos[v][0], it_pos[v][1])
            road_len[u][v] = dist(pu, pv)
            # fill speed map if lane speed exists
            su = rev_map.get(u, "")
            sv = rev_map.get(v, "")
            sp = edge_speed_tmp.get((su, sv), 0.0)
            if sp > 0.0:
                speed_map[u][v] = sp
    if include_lane_map:
        lane_map: Dict[str, Tuple[int, int]] = {}
        for lane_id, (frm, to) in lane_edge_tmp.items():
            if frm in id_map and to in id_map:
                lane_map[lane_id] = (id_map[frm], id_map[to])
        return it_pos, adj, road_len, speed_map, lane_map
    return it_pos, adj, road_len, speed_map
