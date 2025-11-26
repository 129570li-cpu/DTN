"""
Build per-edge speed maps from SUMO FCD output (grid_fcd.out.xml) and net.xml.
Outputs:
  - speed_map: dict[int][int] = avg speed (m/s) for directed edge u->v
  - three temporal maps (optional): past/present/future based on time windows
Notes:
  - FCD lane attribute is typically "<edge_id>_<laneIndex>"; we strip the suffix to obtain edge_id.
  - net.xml is used to map edge_id to (from_junction, to_junction) and convert to integer ids consistent with controller.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import xml.etree.ElementTree as ET
import re
from collections import defaultdict, OrderedDict

def _parse_net_edge_map(net_path: str) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, int], Dict[int, str], OrderedDict[int, Tuple[float, float]]]:
    """Parse net.xml to obtain edge_id -> (from_id_str,to_id_str), and str<->int maps, and it_pos."""
    tree = ET.parse(net_path)
    root = tree.getroot()
    def tags(name: str):
        return root.findall(".//" + name) + root.findall(".//{*}" + name)
    # Junctions
    it_pos_tmp: Dict[str, Tuple[float, float]] = {}
    for j in tags("junction"):
        jid = j.attrib.get("id")
        jtype = j.attrib.get("type", "")
        if jtype == "internal":
            continue
        try:
            x = float(j.attrib.get("x", "0")); y = float(j.attrib.get("y", "0"))
            it_pos_tmp[jid] = (x, y)
        except Exception:
            continue
    # Edges
    edge_map: Dict[str, Tuple[str, str]] = {}
    for e in tags("edge"):
        if e.attrib.get("function", "") in ("internal", "connector"):
            continue
        frm = e.attrib.get("from"); to = e.attrib.get("to")
        eid = e.attrib.get("id")
        if frm in it_pos_tmp and to in it_pos_tmp and eid:
            edge_map[eid] = (frm, to)
    # id maps
    id_map: Dict[str, int] = {}
    rev_map: Dict[int, str] = {}
    next_id = 0
    for jid in it_pos_tmp.keys():
        try:
            iid = int(jid)
        except Exception:
            iid = next_id; next_id += 1
        while iid in rev_map:
            iid += 1
        id_map[jid] = iid; rev_map[iid] = jid
    it_pos = OrderedDict((id_map[j], (xy[0], xy[1])) for j, xy in it_pos_tmp.items())
    return edge_map, id_map, rev_map, it_pos

_LANE_RE = re.compile(r"(.+)_\d+$")

def _edge_from_lane(lane_id: str) -> Optional[str]:
    m = _LANE_RE.match(lane_id.strip())
    return m.group(1) if m else None

def build_speed_map_from_fcd(fcd_path: str, net_path: str,
                             t_start: Optional[float] = None,
                             t_end: Optional[float] = None) -> Dict[int, Dict[int, float]]:
    """Compute avg speed (m/s) per directed edge u->v over [t_start,t_end] from FCD."""
    edge_map, id_map, rev_map, _ = _parse_net_edge_map(net_path)
    tree = ET.parse(fcd_path)
    root = tree.getroot()
    sp_acc: Dict[Tuple[int, int], Tuple[float, int]] = {}
    def add(u: int, v: int, s: float):
        key = (u, v)
        total, cnt = sp_acc.get(key, (0.0, 0))
        sp_acc[key] = (total + s, cnt + 1)
    for ts in root.findall(".//timestep") + root.findall(".//{*}timestep"):
        try:
            t = float(ts.attrib.get("time", "0"))
        except Exception:
            continue
        if t_start is not None and t < t_start:
            continue
        if t_end is not None and t > t_end:
            continue
        for veh in ts.findall("./vehicle") + ts.findall("./{*}vehicle"):
            lane = veh.attrib.get("lane", "")
            edge_id = _edge_from_lane(lane) or ""
            if edge_id not in edge_map:
                continue
            try:
                s = float(veh.attrib.get("speed", "0"))
            except Exception:
                s = 0.0
            frm_str, to_str = edge_map[edge_id]
            u = id_map.get(frm_str); v = id_map.get(to_str)
            if u is None or v is None:
                continue
            add(u, v, s)
    speed_map: Dict[int, Dict[int, float]] = defaultdict(dict)
    for (u, v), (total, cnt) in sp_acc.items():
        if cnt > 0:
            speed_map[u][v] = total / float(cnt)
    return speed_map

def build_past_present_future_speed_maps(fcd_path: str, net_path: str) -> Tuple[Dict[int, Dict[int, float]],
                                                                               Dict[int, Dict[int, float]],
                                                                               Dict[int, Dict[int, float]]]:
    """Split the whole timespan into 3 equal windows and compute speed maps."""
    # Parse timespan
    tree = ET.parse(fcd_path)
    root = tree.getroot()
    times: list[float] = []
    for ts in root.findall(".//timestep") + root.findall(".//{*}timestep"):
        try:
            times.append(float(ts.attrib.get("time", "0")))
        except Exception:
            continue
    if not times:
        return {}, {}, {}
    t0, t1 = min(times), max(times)
    T = max(1e-6, t1 - t0)
    a = t0 + T/3.0
    b = t0 + 2*T/3.0
    past = build_speed_map_from_fcd(fcd_path, net_path, t_start=t0, t_end=a)
    present = build_speed_map_from_fcd(fcd_path, net_path, t_start=a, t_end=b)
    future = build_speed_map_from_fcd(fcd_path, net_path, t_start=b, t_end=t1)
    return past, present, future

def build_past_present_future_density_maps(fcd_path: str, net_path: str) -> Tuple[Dict[int, Dict[int, float]],
                                                                                  Dict[int, Dict[int, float]],
                                                                                  Dict[int, Dict[int, float]]]:
    """Compute per-edge average vehicle count (vehicles per edge) over three equal time windows."""
    edge_map, id_map, rev_map, _ = _parse_net_edge_map(net_path)
    tree = ET.parse(fcd_path)
    root = tree.getroot()
    # collect timesteps
    timesteps = root.findall(".//timestep") + root.findall(".//{*}timestep")
    times = []
    for ts in timesteps:
        try:
            times.append(float(ts.attrib.get("time", "0")))
        except Exception:
            times.append(0.0)
    if not times:
        return {}, {}, {}
    t0, t1 = min(times), max(times)
    T = max(1e-6, t1 - t0)
    a = t0 + T/3.0
    b = t0 + 2*T/3.0
    # helper to count vehicles per edge in window
    def edge_density(t_start: float, t_end: float) -> Dict[int, Dict[int, float]]:
        counts: Dict[Tuple[int, int], int] = {}
        slots: int = 0
        for ts in timesteps:
            try:
                t = float(ts.attrib.get("time", "0"))
            except Exception:
                continue
            if t < t_start or t > t_end:
                continue
            slots += 1
            seen: Dict[Tuple[int, int], int] = {}
            for veh in ts.findall("./vehicle") + ts.findall("./{*}vehicle"):
                lane = veh.attrib.get("lane", "")
                eid = _edge_from_lane(lane) or ""
                if eid not in edge_map:
                    continue
                frm_str, to_str = edge_map[eid]
                u = id_map.get(frm_str); v = id_map.get(to_str)
                if u is None or v is None:
                    continue
                seen[(u, v)] = seen.get((u, v), 0) + 1
            # sum counts for this timestep
            for k, c in seen.items():
                counts[k] = counts.get(k, 0) + c
        dens: Dict[int, Dict[int, float]] = defaultdict(dict)
        for (u, v), c in counts.items():
            if slots > 0:
                dens[u][v] = float(c) / float(slots)
        return dens
    return edge_density(t0, a), edge_density(a, b), edge_density(b, t1)
