#!/usr/bin/env python3
"""Build multi-window dynamic node features from SUMO FCD output."""
from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from topology.sumo_net import parse_sumo_net  # type: ignore

Coord = Tuple[float, float]

FEATURE_NAMES = [
    "avg_speed",
    "avg_density",
    "avg_queue",
    "peak_density",
    "peak_queue",
    "var_speed",
    "var_density",
    "var_queue",
]
EDGE_FEATURE_NAMES = [
    "avg_speed",
    "avg_flow",
    "avg_queue",
    "peak_queue",
]


def nearest_junction(x: float, y: float, it_pos: Dict[int, Coord], max_dist: float) -> int | None:
    best = None
    best_dist = max_dist
    for jid, (jx, jy) in it_pos.items():
        dist = math.hypot(jx - x, jy - y)
        if dist <= best_dist:
            best_dist = dist
            best = jid
    return best


def _new_window():
    return {
        "steps": 0,
        "speed_sum": defaultdict(float),
        "speed_sq_sum": defaultdict(float),
        "speed_count": defaultdict(int),
        "density_sum": defaultdict(float),
        "density_sq_sum": defaultdict(float),
        "queue_sum": defaultdict(float),
        "queue_sq_sum": defaultdict(float),
        "max_density": defaultdict(float),
        "max_queue": defaultdict(float),
        # edge-level accumulators
        "edge_speed_sum": defaultdict(float),
        "edge_speed_sq_sum": defaultdict(float),
        "edge_speed_count": defaultdict(int),
        "edge_count": defaultdict(int),
        "edge_queue_sum": defaultdict(float),
        "edge_queue_sq_sum": defaultdict(float),
        "edge_max_queue": defaultdict(float),
    }


def _assign_window_index(sim_time: float, window_size: float, window_step: float, window_offset: float) -> int:
    rel = sim_time - window_offset
    if rel < 0:
        return -1
    if window_size <= 0:
        return 0
    return int(rel // window_step)


def build_features(
    net_xml: str,
    fcd_xml: str,
    assign_mode: str = "lane_to",
    assign_radius: float = 120.0,
    slow_threshold: float = 2.0,
    window_size: float = 0.0,
    window_step: float = 0.0,
    max_windows: int = 0,
    window_offset: float = 0.0,
) -> Tuple[List[Dict[int, Tuple[float, ...]]], List[Dict[str, float]], Dict[str, int], List[Dict[str, Any]], List[Dict[str, List[float]]]]:
    it_pos, edges, _, _, lane_map = parse_sumo_net(net_xml, skip_internal=True, include_lane_map=True)
    # edge_id_map + 静态边特征（几何长度）
    edge_id_map: Dict[str, int] = {}
    edge_static: List[Dict[str, Any]] = []
    for u, nbrs in edges.items():
        for v in nbrs:
            key = f"{u}-{v}"
            if key in edge_id_map:
                continue
            edge_id_map[key] = len(edge_id_map)
            ux, uy = it_pos[u]
            vx, vy = it_pos[v]
            edge_static.append({"u": u, "v": v, "length": math.hypot(ux - vx, uy - vy)})

    if assign_mode not in {"lane_to", "lane_from", "nearest"}:
        raise ValueError("assign_mode must be lane_to, lane_from, or nearest")

    if window_size > 0 and window_step <= 0:
        window_step = window_size
    if window_size > 0 and window_step > window_size:
        raise ValueError("window_step must be <= window_size")

    window_data: Dict[int, dict] = {}
    window_bounds: Dict[int, Dict[str, float]] = {}

    context = ET.iterparse(fcd_xml, events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event != "end" or elem.tag != "timestep":
            continue
        sim_time = float(elem.attrib.get("time", "0"))
        idx = _assign_window_index(sim_time, window_size, window_step or window_size or 1.0, window_offset)
        if idx < 0:
            elem.clear()
            continue
        if window_size > 0:
            start_time = window_offset + idx * window_step
            if sim_time >= start_time + window_size:
                elem.clear()
                continue
            end_time = start_time + window_size
        else:
            start_time = window_offset
            end_time = sim_time
        window = window_data.setdefault(idx, _new_window())
        bounds = window_bounds.setdefault(idx, {"index": idx, "start": start_time, "end": end_time})
        if window_size <= 0:
            bounds["end"] = sim_time

        counts = defaultdict(int)
        queue_counts = defaultdict(int)
        speed_totals = defaultdict(float)
        speed_counts = defaultdict(int)
        speed_sq_totals = defaultdict(float)

        for veh in elem:
            if veh.tag != "vehicle":
                continue
            target = None
            lane_id = veh.attrib.get("lane")
            if lane_id and lane_id in lane_map and assign_mode != "nearest":
                frm, to = lane_map[lane_id]
                target = to if assign_mode == "lane_to" else frm
            if target is None:
                x = float(veh.attrib.get("x", "0"))
                y = float(veh.attrib.get("y", "0"))
                target = nearest_junction(x, y, it_pos, assign_radius)
            if target is None:
                continue
            speed = float(veh.attrib.get("speed", "0"))
            counts[target] += 1
            speed_totals[target] += speed
            speed_sq_totals[target] += speed * speed
            speed_counts[target] += 1
            if speed < slow_threshold:
                queue_counts[target] += 1
            # Edge-level stats if lane info available
            if lane_id and lane_id in lane_map:
                frm, to = lane_map[lane_id]
                edge_key = f"{frm}-{to}"
                window["edge_count"][edge_key] += 1
                window["edge_speed_sum"][edge_key] += speed
                window["edge_speed_sq_sum"][edge_key] += speed * speed
                window["edge_speed_count"][edge_key] += 1
                if speed < slow_threshold:
                    window["edge_queue_sum"][edge_key] += 1
                    window["edge_queue_sq_sum"][edge_key] += 1
                    window["edge_max_queue"][edge_key] = max(1.0, window["edge_max_queue"].get(edge_key, 0.0))

        window["steps"] += 1
        for node, total in speed_totals.items():
            window["speed_sum"][node] += total
            window["speed_sq_sum"][node] += speed_sq_totals[node]
            window["speed_count"][node] += speed_counts[node]
        for node, cnt in counts.items():
            window["density_sum"][node] += cnt
            window["density_sq_sum"][node] += cnt * cnt
            window["max_density"][node] = max(cnt, window["max_density"].get(node, 0.0))
        for node, cnt in queue_counts.items():
            window["queue_sum"][node] += cnt
            window["queue_sq_sum"][node] += cnt * cnt
            window["max_queue"][node] = max(cnt, window["max_queue"].get(node, 0.0))
        elem.clear()
    root.clear()

    if not window_data:
        raise RuntimeError("No timestep data found in FCD file")

    sorted_indices = sorted(window_data.keys())
    if max_windows > 0:
        sorted_indices = sorted_indices[-max_windows:]

    frames: List[Dict[int, Tuple[float, ...]]] = []
    edge_frames: List[Dict[str, List[float]]] = []
    bounds: List[Dict[str, float]] = []
    for idx in sorted_indices:
        window = window_data[idx]
        steps = max(1, window["steps"])
        frame: Dict[int, Tuple[float, ...]] = {}
        for node in it_pos.keys():
            spd_count = window["speed_count"].get(node, 0)
            avg_speed = window["speed_sum"].get(node, 0.0) / spd_count if spd_count > 0 else 0.0
            speed_sq_avg = window["speed_sq_sum"].get(node, 0.0) / spd_count if spd_count > 0 else 0.0
            var_speed = max(0.0, speed_sq_avg - avg_speed * avg_speed)

            avg_density = window["density_sum"].get(node, 0.0) / steps
            density_sq_avg = window["density_sq_sum"].get(node, 0.0) / steps
            var_density = max(0.0, density_sq_avg - avg_density * avg_density)

            avg_queue = window["queue_sum"].get(node, 0.0) / steps
            queue_sq_avg = window["queue_sq_sum"].get(node, 0.0) / steps
            var_queue = max(0.0, queue_sq_avg - avg_queue * avg_queue)

            peak_density = window["max_density"].get(node, 0.0)
            peak_queue = window["max_queue"].get(node, 0.0)

            frame[node] = (
                avg_speed,
                avg_density,
                avg_queue,
                peak_density,
                peak_queue,
                var_speed,
                var_density,
                var_queue,
            )
        frames.append(frame)
        # edge dynamic features
        edge_frame: Dict[str, List[float]] = {}
        for edge_key in edge_id_map.keys():
            spd_count = window["edge_speed_count"].get(edge_key, 0)
            avg_speed = window["edge_speed_sum"].get(edge_key, 0.0) / spd_count if spd_count > 0 else 0.0
            flow = window["edge_count"].get(edge_key, 0.0) / steps
            avg_queue = window["edge_queue_sum"].get(edge_key, 0.0) / steps
            peak_queue = window["edge_max_queue"].get(edge_key, 0.0)
            edge_frame[edge_key] = [avg_speed, flow, avg_queue, peak_queue]
        edge_frames.append(edge_frame)
        bounds.append(window_bounds[idx])
    return frames, bounds, edge_id_map, edge_static, edge_frames


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SUMO FCD into time-varying node features")
    parser.add_argument("--net-xml", required=True)
    parser.add_argument("--fcd-xml", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--edge-map-output", default=None, help="Optional path to write edge_id_map.json")
    parser.add_argument("--assign-mode", choices=["lane_to", "lane_from", "nearest"], default="lane_to")
    parser.add_argument("--assign-radius", type=float, default=120.0)
    parser.add_argument("--slow-threshold", type=float, default=2.0)
    parser.add_argument("--window-size", type=float, default=0.0, help="Window size (seconds). 0 = full sim")
    parser.add_argument("--window-step", type=float, default=0.0, help="Stride between windows (seconds)")
    parser.add_argument("--max-windows", type=int, default=0, help="Keep only the latest N windows")
    parser.add_argument("--window-offset", type=float, default=0.0, help="Ignore data before this time")
    return parser.parse_args()


def main():
    args = parse_args()
    frames, bounds, edge_id_map, edge_static, edge_frames = build_features(
        args.net_xml,
        args.fcd_xml,
        assign_mode=args.assign_mode,
        assign_radius=args.assign_radius,
        slow_threshold=args.slow_threshold,
        window_size=args.window_size,
        window_step=args.window_step,
        max_windows=args.max_windows,
        window_offset=args.window_offset,
    )
    node_payload: Dict[str, List[List[float]]] = {}
    for frame in frames:
        for nid, feats in frame.items():
            node_payload.setdefault(str(nid), []).append([float(v) for v in feats])
    edge_payload: Dict[str, List[List[float]]] = {}
    for frame in edge_frames:
        for eid, feats in frame.items():
            edge_payload.setdefault(str(eid), []).append([float(v) for v in feats])
    output = {
        "meta": {
            "feature_names": FEATURE_NAMES,
            "feature_dim": len(FEATURE_NAMES),
            "window_count": len(frames),
            "window_size": args.window_size,
            "window_step": args.window_step or args.window_size or 0.0,
            "assign_mode": args.assign_mode,
            "slow_threshold": args.slow_threshold,
            "edge_count": len(edge_id_map),
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "edge_feature_dim": len(EDGE_FEATURE_NAMES),
        },
        "windows": bounds,
        "nodes": node_payload,
        "edges": edge_payload,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(frames)} windows for {len(node_payload)} nodes to {args.output_json}")

    if args.edge_map_output:
        edge_out = {
            "edge_id_map": edge_id_map,
            "edges": edge_static,
            "meta": {"count": len(edge_id_map)},
        }
        with open(args.edge_map_output, "w") as f:
            json.dump(edge_out, f, indent=2)
        print(f"Saved edge map ({len(edge_id_map)}) to {args.edge_map_output}")


if __name__ == "__main__":
    main()
