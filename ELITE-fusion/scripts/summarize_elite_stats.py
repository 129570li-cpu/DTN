#!/usr/bin/env python3
import sys, statistics, csv
from typing import List

def pct(v: List[float], p: float) -> float:
    if not v:
        return 0.0
    v2 = sorted(v)
    k = int(round((len(v2)-1) * p))
    return v2[k]

def main():
    if len(sys.argv) < 2:
        print("Usage: summarize_elite_stats.py <elite_path_stats.csv>")
        sys.exit(1)
    path = sys.argv[1]
    AD, HC, RC, RO = [], [], [], []
    total = 0
    succ = 0
    with open(path, 'r') as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#'):
                continue
            # src,dst,success,ADsd,HCsd,RCsd,RO,nsd
            try:
                total += 1
                s = int(row[2])
                succ += s
                AD.append(float(row[3]))
                HC.append(float(row[4]))
                RC.append(float(row[5]))
                RO.append(float(row[6]))
            except Exception:
                continue
    sr = (succ/total*100.0) if total>0 else 0.0
    def brief(name, arr):
        if not arr:
            print(f"{name}: N=0")
            return
        print(f"{name}: N={len(arr)} mean={statistics.mean(arr):.3f} p50={pct(arr,0.5):.3f} p90={pct(arr,0.9):.3f} p99={pct(arr,0.99):.3f}")
    print(f"Flows: total={total} succ={succ} SR={sr:.2f}%")
    brief("ADsd", AD)
    brief("HCsd", HC)
    brief("RCsd", RC)
    brief("RO", RO)

if __name__ == "__main__":
    main()

