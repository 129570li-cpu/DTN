#!/usr/bin/env python3
"""
Convert SUMO FCD output into per-junction dynamic features (avg speed, density, queue).

Usage:
  python build_node_features.py \
      --net-xml /path/to/grid.net.xml \
      --fcd-xml /path/to/grid_fcd_new.xml \
      --output-json /path/to/node_features.json \
      [--assign-radius 50]

Features per junction j:
  [
    avg_speed_mps,        # average vehicle speed mapped to junction j
    avg_density,          # avg number of vehicles mapped to j per timestep
    avg_queue             # avg number of slow vehicles (speed<threshold) per timestep
  ]

Mapping rule: for each vehicle we snap its (x,y) to the nearest junction.
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, Tuple

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from topology.sumo_net import parse_sumo_net  # type: ignore


def nearest_junction(x: float, y: float, it_pos: Dict[int, Tuple[float, float]], max_dist: float) -> int | None:
    best = None
    best_dist = max_dist
    for jid, (jx, jy) in it_pos.items():
        dx = jx - x
        dy = jy - y
        dist = math.hypot(dx, dy)
        if dist <= best_dist:
            best_dist = dist
            best = jid
    return best


def build_features(net_xml: str,
                   fcd_xml: str,
                   assign_radius: float = 80.0,
                   slow_threshold: float = 2.0) -> Dict[int, Tuple[float, float, float]]:
    it_pos, _, _, _ = parse_sumo_net(net_xml, skip_internal=True)
    speed_sum = defaultdict(float)
    count_sum = defaultdict(float)
    slow_sum = defaultdict(float)
    total_steps = 0

    context = ET.iterparse(fcd_xml, events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event == "end" and elem.tag == "timestep":
            total_steps += 1
            for veh in elem:
                if veh.tag != "vehicle":
                    continue
                x = float(veh.attrib.get("x", "0"))
                y = float(veh.attrib.get("y", "0"))
                speed = float(veh.attrib.get("speed", "0"))
                jid = nearest_junction(x, y, it_pos, assign_radius)
                if jid is None:
                    continue
                speed_sum[jid] += speed
                count_sum[jid] += 1.0
                if speed < slow_threshold:
                    slow_sum[jid] += 1.0
            elem.clear()
    root.clear()

    if total_steps == 0:
        raise RuntimeError("No timesteps parsed from FCD file.")
    features: Dict[int, Tuple[float, float, float]] = {}
    for jid in it_pos.keys():
        total_count = count_sum[jid]
        avg_speed = speed_sum[jid] / total_count if total_count > 0 else 0.0
        avg_density = count_sum[jid] / total_steps
        avg_queue = slow_sum[jid] / total_steps
        features[jid] = (avg_speed, avg_density, avg_queue)
    return features


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-junction features from SUMO FCD output.")
    parser.add_argument("--net-xml", required=True)
    parser.add_argument("--fcd-xml", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--assign-radius", type=float, default=80.0, help="Max distance to snap vehicle to junction.")
    parser.add_argument("--slow-threshold", type=float, default=2.0, help="Speed threshold (m/s) to count as queued.")
    return parser.parse_args()


def main():
    args = parse_args()
    feats = build_features(args.net_xml, args.fcd_xml, args.assign_radius, args.slow_threshold)
    json_ready = {str(k): list(v) for k, v in feats.items()}
    with open(args.output_json, "w") as f:
        json.dump(json_ready, f, indent=2)
    print(f"Saved node features for {len(json_ready)} junctions to {args.output_json}")


if __name__ == "__main__":
    main()
