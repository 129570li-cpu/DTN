#!/usr/bin/env python3
"""
Measure Control Overhead (CO) per paper definition:
  CO = average number of network control messages generated per second
       for maintaining routing tables when no data routing is present.
Includes:
  - CAM beacons: 1 beacon per vehicle per second (Beacon interval = 1s)
    We estimate average vehicle count from SUMO FCD file (grid_fcd.out.xml).
  - DTN virtual training "packets": episodes per second in SPL retraining.
Usage:
  python3 measure_co.py --fcd /path/to/grid_fcd.out.xml \
                        [--episodes-present 4000 --episodes-past 3000 --episodes-future 3000 \
                         --spl-period 60]
Outputs:
  Prints CO_cam, CO_dtn, CO_total in messages/second.
"""
from __future__ import annotations
import argparse
import xml.etree.ElementTree as ET

def avg_vehicles_from_fcd(fcd_path: str) -> float:
    tree = ET.parse(fcd_path)
    root = tree.getroot()
    total = 0
    steps = 0
    for ts in root.findall(".//timestep") + root.findall(".//{*}timestep"):
        steps += 1
        cnt = 0
        for _ in ts.findall("./vehicle") + ts.findall("./{*}vehicle"):
            cnt += 1
        total += cnt
    if steps == 0:
        return 0.0
    return float(total) / float(steps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcd", required=True, help="SUMO FCD file (e.g., grid_fcd.out.xml)")
    ap.add_argument("--episodes-present", type=int, default=4000, help="Episodes per present DTN retraining")
    ap.add_argument("--episodes-past", type=int, default=3000, help="Episodes per past DTN retraining")
    ap.add_argument("--episodes-future", type=int, default=3000, help="Episodes per future DTN retraining")
    ap.add_argument("--spl-period", type=float, default=0.0, help="Retraining period seconds (0 to ignore DTN part)")
    args = ap.parse_args()
    avg_veh = avg_vehicles_from_fcd(args.fcd)
    # CO from CAM beacons: 1 beacon per vehicle per second
    co_cam = avg_veh  # msgs/s
    # CO from DTN virtual training "packets": episodes per second over all temporal DTNs
    if args.spl_period and args.spl_period > 0:
        total_eps = max(0, args.episodes_present) + max(0, args.episodes_past) + max(0, args.episodes_future)
        co_dtn = float(total_eps) / float(args.spl_period)
    else:
        co_dtn = 0.0
    co_total = co_cam + co_dtn
    print(f"CO_cam (beacons): {co_cam:.3f} msgs/s")
    print(f"CO_dtn (virtual): {co_dtn:.3f} msgs/s")
    print(f"CO_total: {co_total:.3f} msgs/s")

if __name__ == "__main__":
    main()

