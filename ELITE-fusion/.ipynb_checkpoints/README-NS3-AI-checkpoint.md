# ELITE Controller + NS3‑AI Integration Guide

This document explains how to run the ELITE controller (Python) with ns‑3 (C++) via ns3‑ai, strictly following the paper’s SPL/EFG/APN workflow.

Pipeline
- SPL（DTN并行学习）
  - 解析 SUMO `.net.xml` → `it_pos`（路口坐标）、`adjacents_comb`（邻接）
  - 在 DTN 环境中并行训练四张单指标 Q 表（PDR/AD/HC/RC）→ 导出 CSV
- EFG（经验融合）
  - 对齐到 (0,1]；加权平均得到 BP
  - 模糊融合（按论文 R/D/H/S、三角隶属、Min–Max、质心解模糊）得到 HRF/LDF/LBF
- APN（物理侧部署与更新）
  - 状态 = 消息类型 × 负载档，负载估计 `loadbas=min(Lavg/Bw,1)`
  - 奖励 `R0=a*g(ADsd)+b*q(HCsd)+γ*l(RCsd)`，`g/q/l` 为分段 RAD/RHC/RRC 的平均
  - 选择策略 G∈{HRF,LDF,LBF}；按 area_path 路口序列下发；逐跳贪心转发（遇空洞回退）；完成后上报反馈，迭代 P(state,action)

Python 端（控制器）
- 入口：`ELITE-fusion/controller_server.py`（需要 ns3-ai Python 模块）
  ```bash
  python3 ELITE-fusion/controller_server.py <sumo_net.xml> dtn_out 6000000 300 5555
  ```
  - 解析 SUMO `.net.xml` → 构建拓扑
  - 训练/加载四张 Q 表 → 归一化 → BP/HRF/LDF/LBF（paper_fuzzy）
  - 监听 ns3‑ai，共享内存收发 Observation/Action：
    - 请求：按状态选择 HRF/LDF/LBF，计算 area_path 并返回
    - 反馈：按论文 R0 更新 APN 的 P(state,action)

ns‑3 端（C++，802.11p + SUMO）
- 逐跳应用与自定义头（在本仓库 `ns3-integration/` 提供）
  - `EliteHeader.h`: 自定义头，携带 `policyId, routeLen, route[32], nextIndex, msgType, ttl, pathId`
  - `EliteForwarderApp.h/.cc`: 
    - 从 CAM 更新邻居；逐跳选择更靠近“下一目标路口”的邻居；遇空洞回退
    - 累计 per‑segment（ADca/HCca/RCca/lca）与 per‑path（ADsd/HCsd/RCsd/PL/RO）指标
    - 与 ns3‑ai 交互：发起路由请求、上报路径反馈（Observation 布局见下）
  - 将上述文件复制到 ns‑3 工程适当位置（或作为模块包含），在场景 `setupNewWifiNode` 中创建并安装 App，桥接 CAM 回调到 `OnCamReceived`

ns3‑ai 接口（固定长度双精度数组）
- Observation（长度=307，顺序严格对齐）
  1) `simTime, vehicleId, srcId, dstId, msgTypeId(0..3), queueLen, bufferBytes, neighborCount`（8）
  2) `neighbors[64], distToNext[64]`（128）
  3) `routeRequestFlag, feedbackFlag, success`（3）
  4) `ADsd, HCsd, RCsd, hopCount, PL, RO, nsd`（7）
  5) `seg_ADca[32], seg_HCca[32], seg_lca[32], seg_RCca[32]`（128）
  6) `path_len, path_ids[32]`（33）
- Action（长度=34）
  - `policyId(0=BP,1=HRF,2=LDF,3=LBF), path_len, path_ids[32]`

校准与参数
- IEEE 802.11p（10 MHz，OfdmRate6MbpsBW10MHz），TxPower/传播模型校准可靠半径≈300 m
- 信标周期 1 s；数据 1024B、控制 256B；生成速率 1–5 pkt/s
- RO/CO：数据期控制消息计入 RO（邻居信标、路径请求/回复、上报等）；CO 为维护/训练期开销（DTN 训练在控制器侧统计）

注意
- `EliteForwarderApp::BuildObservation/GetAction` 内含 ns3‑ai 调用占位，请按你的 ns3‑ai C++ API 实现
- `controller_server.py` 内的 APN 权重映射按照论文意图：
  - HRF: (a,b,γ) = (0.25,0.0,0.25)；LDF: (0.5,0.25,0.0)；LBF: (0.25,0.25,0.5)
  - `mADsd` 按 (src,dst) 的路径级最大时延进行跟踪
- 负载判档建议：沿基准路径 Lbas 统计车辆平均缓冲 Lavg（可上报 per‑hop 缓冲给控制器），`loadbas=min(Lavg/Bw,1)` → 0/1/2 档

