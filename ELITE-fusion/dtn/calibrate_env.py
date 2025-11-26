#!/usr/bin/env python3
"""
DTN 环境校准脚本：用 NS-3 反馈数据校准 dtn/env.py 的参数

使用方法:
1. 运行 NS-3 仿真收集反馈数据 (online_feedback.csv)
2. 运行此脚本分析数据并输出校准参数
3. 将校准参数更新到 dtn/env.py

python -m dtn.calibrate_env --feedback dtn_out/online_feedback.csv --net-xml grid.net.xml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from topology.sumo_net import parse_sumo_net


def load_feedback(csv_path: str) -> List[Dict]:
    """加载 NS-3 反馈数据"""
    data = []
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            row = {}
            for i, h in enumerate(header):
                try:
                    row[h] = float(parts[i])
                except ValueError:
                    row[h] = parts[i]
            data.append(row)
    return data


def compute_path_stats(
    feedback: List[Dict],
    it_pos: Dict[int, Tuple[float, float]],
    adj: Dict[int, List[int]]
) -> Dict:
    """计算路径统计信息"""
    
    # 按 (src, dst) 分组
    pairs = defaultdict(list)
    for row in feedback:
        src = int(row.get("src", 0))
        dst = int(row.get("dst", 0))
        pairs[(src, dst)].append(row)
    
    # 计算每对的统计
    stats = {
        "total_samples": len(feedback),
        "unique_pairs": len(pairs),
        "success_rate": 0.0,
        "avg_delay": 0.0,
        "avg_hops": 0.0,
        "pairs": []
    }
    
    success_count = 0
    delay_sum = 0.0
    hops_sum = 0.0
    valid_count = 0
    
    for (src, dst), rows in pairs.items():
        if src not in it_pos or dst not in it_pos:
            continue
        
        # 计算欧氏距离
        sx, sy = it_pos[src]
        dx, dy = it_pos[dst]
        distance = math.hypot(dx - sx, dy - sy)
        
        # 统计成功率和平均指标
        successes = sum(1 for r in rows if r.get("success", 0) > 0.5)
        delays = [r.get("ADsd", 0) for r in rows if r.get("success", 0) > 0.5]
        hops = [r.get("HCsd", 0) for r in rows if r.get("success", 0) > 0.5]
        
        pair_stat = {
            "src": src,
            "dst": dst,
            "distance": distance,
            "samples": len(rows),
            "successes": successes,
            "pdr": successes / len(rows) if rows else 0,
            "avg_delay": sum(delays) / len(delays) if delays else 0,
            "avg_hops": sum(hops) / len(hops) if hops else 0,
        }
        stats["pairs"].append(pair_stat)
        
        success_count += successes
        if delays:
            delay_sum += sum(delays)
            hops_sum += sum(hops)
            valid_count += len(delays)
    
    stats["success_rate"] = success_count / len(feedback) if feedback else 0
    stats["avg_delay"] = delay_sum / valid_count if valid_count else 0
    stats["avg_hops"] = hops_sum / valid_count if valid_count else 0
    
    return stats


def fit_pdr_model(pairs: List[Dict]) -> Dict:
    """
    拟合 PDR 与距离的关系
    假设模型: PDR = exp(-distance / alpha) * beta
    """
    if not pairs:
        return {"alpha": 300.0, "beta": 1.0, "r2": 0.0}
    
    # 提取有效数据点
    distances = []
    pdrs = []
    for p in pairs:
        if p["samples"] >= 3:  # 至少 3 个样本
            distances.append(p["distance"])
            pdrs.append(p["pdr"])
    
    if len(distances) < 5:
        print("[WARN] 样本不足，使用默认参数")
        return {"alpha": 300.0, "beta": 1.0, "r2": 0.0}
    
    distances = np.array(distances)
    pdrs = np.array(pdrs)
    
    # 简单拟合：log(PDR) = -distance/alpha + log(beta)
    # 使用最小二乘法
    valid_mask = pdrs > 0.01
    if valid_mask.sum() < 5:
        return {"alpha": 300.0, "beta": 1.0, "r2": 0.0}
    
    log_pdrs = np.log(pdrs[valid_mask] + 1e-6)
    dists = distances[valid_mask]
    
    # 线性回归: log(PDR) = a * distance + b
    # a = -1/alpha, b = log(beta)
    n = len(dists)
    sum_x = dists.sum()
    sum_y = log_pdrs.sum()
    sum_xy = (dists * log_pdrs).sum()
    sum_x2 = (dists ** 2).sum()
    
    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return {"alpha": 300.0, "beta": 1.0, "r2": 0.0}
    
    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n
    
    alpha = -1.0 / a if abs(a) > 1e-10 else 300.0
    beta = min(1.0, math.exp(b))
    
    # 计算 R²
    y_pred = a * dists + b
    ss_res = ((log_pdrs - y_pred) ** 2).sum()
    ss_tot = ((log_pdrs - log_pdrs.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    
    return {
        "alpha": max(50.0, min(1000.0, alpha)),  # 限制范围
        "beta": max(0.1, min(1.0, beta)),
        "r2": r2
    }


def fit_delay_model(pairs: List[Dict]) -> Dict:
    """
    拟合时延与距离/跳数的关系
    假设模型: delay = a * hops + b * distance + c
    """
    if not pairs:
        return {"hop_factor": 0.1, "dist_factor": 0.001, "base": 0.01, "r2": 0.0}
    
    # 提取有效数据点
    data = []
    for p in pairs:
        if p["samples"] >= 3 and p["successes"] > 0:
            data.append({
                "distance": p["distance"],
                "hops": p["avg_hops"],
                "delay": p["avg_delay"]
            })
    
    if len(data) < 5:
        return {"hop_factor": 0.1, "dist_factor": 0.001, "base": 0.01, "r2": 0.0}
    
    # 简单回归: delay = a * hops + c（假设主要与跳数相关）
    hops = np.array([d["hops"] for d in data])
    delays = np.array([d["delay"] for d in data])
    
    n = len(hops)
    sum_x = hops.sum()
    sum_y = delays.sum()
    sum_xy = (hops * delays).sum()
    sum_x2 = (hops ** 2).sum()
    
    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return {"hop_factor": 0.1, "dist_factor": 0.001, "base": 0.01, "r2": 0.0}
    
    a = (n * sum_xy - sum_x * sum_y) / denom
    c = (sum_y - a * sum_x) / n
    
    # 计算 R²
    y_pred = a * hops + c
    ss_res = ((delays - y_pred) ** 2).sum()
    ss_tot = ((delays - delays.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    
    return {
        "hop_factor": max(0.01, a),
        "dist_factor": 0.001,  # 简化
        "base": max(0.0, c),
        "r2": r2
    }


def generate_calibrated_config(pdr_params: Dict, delay_params: Dict) -> str:
    """生成校准后的配置"""
    config = f"""
# DTN 环境校准参数（由 NS-3 反馈数据拟合）
# 生成时间: {__import__('datetime').datetime.now().isoformat()}

# PDR 模型: PDR = beta * exp(-distance / alpha)
PDR_ALPHA = {pdr_params['alpha']:.1f}  # 距离衰减系数 (R² = {pdr_params['r2']:.3f})
PDR_BETA = {pdr_params['beta']:.3f}    # 基础成功率

# 时延模型: delay = hop_factor * hops + base
DELAY_HOP_FACTOR = {delay_params['hop_factor']:.4f}  # 每跳时延 (R² = {delay_params['r2']:.3f})
DELAY_BASE = {delay_params['base']:.4f}              # 基础时延
"""
    return config


def main():
    parser = argparse.ArgumentParser(description="校准 DTN 环境参数")
    parser.add_argument("--feedback", required=True, help="NS-3 反馈 CSV 文件路径")
    parser.add_argument("--net-xml", required=True, help="SUMO net.xml 路径")
    parser.add_argument("--output", default=None, help="输出校准配置文件")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    args = parser.parse_args()
    
    if not os.path.exists(args.feedback):
        print(f"[ERROR] 反馈文件不存在: {args.feedback}")
        print("\n请先运行 NS-3 仿真收集数据:")
        print("  ./scripts/run_elite_ns3_gnn.sh")
        print("\n反馈数据将保存到: dtn_out/online_feedback.csv")
        return 1
    
    # 加载数据
    print(f"[INFO] 加载反馈数据: {args.feedback}")
    feedback = load_feedback(args.feedback)
    print(f"[INFO] 共 {len(feedback)} 条记录")
    
    # 加载拓扑
    print(f"[INFO] 加载拓扑: {args.net_xml}")
    it_pos, adj, _, _ = parse_sumo_net(args.net_xml, skip_internal=True)
    print(f"[INFO] 共 {len(it_pos)} 个路口")
    
    # 计算统计
    print("\n[INFO] 计算路径统计...")
    stats = compute_path_stats(feedback, it_pos, adj)
    
    print(f"\n{'='*50}")
    print("NS-3 反馈数据统计")
    print(f"{'='*50}")
    print(f"总样本数: {stats['total_samples']}")
    print(f"唯一路径对: {stats['unique_pairs']}")
    print(f"总体成功率: {stats['success_rate']:.2%}")
    print(f"平均时延: {stats['avg_delay']:.3f} 秒")
    print(f"平均跳数: {stats['avg_hops']:.1f}")
    
    # 拟合模型
    print("\n[INFO] 拟合 PDR 模型...")
    pdr_params = fit_pdr_model(stats["pairs"])
    print(f"  PDR = {pdr_params['beta']:.3f} * exp(-d / {pdr_params['alpha']:.1f})")
    print(f"  R² = {pdr_params['r2']:.3f}")
    
    print("\n[INFO] 拟合时延模型...")
    delay_params = fit_delay_model(stats["pairs"])
    print(f"  Delay = {delay_params['hop_factor']:.4f} * hops + {delay_params['base']:.4f}")
    print(f"  R² = {delay_params['r2']:.3f}")
    
    # 生成配置
    config = generate_calibrated_config(pdr_params, delay_params)
    print(f"\n{'='*50}")
    print("校准后的参数")
    print(f"{'='*50}")
    print(config)
    
    # 保存配置
    if args.output:
        with open(args.output, "w") as f:
            f.write(config)
        print(f"\n[INFO] 配置已保存到: {args.output}")
    
    # 生成更新 env.py 的建议
    print(f"\n{'='*50}")
    print("更新 dtn/env.py 的建议")
    print(f"{'='*50}")
    print(f"""
在 dtn/env.py 的 EnvConfig 中更新:

class EnvConfig:
    comm_radius: float = {pdr_params['alpha']:.1f}  # 校准后的通信半径
    # ...

在 seg_delivery_prob 方法中使用:

def seg_delivery_prob(self, c: int, a: int) -> float:
    d = self.seg_length(c, a)
    # 校准后的 PDR 模型
    pdr = {pdr_params['beta']:.3f} * math.exp(-d / {pdr_params['alpha']:.1f})
    return max(0.0, min(1.0, pdr))
""")
    
    if args.verbose:
        print(f"\n{'='*50}")
        print("详细数据（按距离排序）")
        print(f"{'='*50}")
        sorted_pairs = sorted(stats["pairs"], key=lambda x: x["distance"])
        for p in sorted_pairs[:20]:
            print(f"  {p['src']:3d} -> {p['dst']:3d}: dist={p['distance']:6.0f}m, "
                  f"PDR={p['pdr']:.2%}, delay={p['avg_delay']:.3f}s, hops={p['avg_hops']:.1f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

