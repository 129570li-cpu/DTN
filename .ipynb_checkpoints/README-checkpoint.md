Instructions:
1. The file ‘Get_Move’ is used to process vehicle information including positions and trajectory.
2. File ‘Global_Par’ is used to record global parameters in simulation.
3. File ‘Init’ is used to initialize vehicles and the SDVN controller and determine the source nodes and destination nodes.
4. Files ‘Node’, ‘Packet’, ‘SDVN_Controller‘ are separately used to formulate vehicles, data packets, and the controller in SDVN.
5. File ‘traffic‘ is used to process junctions and roads.
6. File ‘Routing_table’ is used to generate and record routing policies. It contains fuzzy fusion and weight-based fusion.

The purpose of this project is to realize a hierarchical routing scheme in SDVN towards various messages flowing in vehicular networks. This project uses the SDVN Controller to collect global network information and generate a routing policy set towards various packet transmission service requirements. 

项目结构概览（递进但可独立运行）
- **ELITE 论文原型（baseline）**：位于 `ELITE-fusion/` 及 `ns3-integration/`，通过 `ctrl_work/controller_server.py` + `EliteForwarderApp` 复现原文的 SDVN 控制器、模糊融合与 NS-3 场景，可直接使用 `ELITE-fusion/scripts/run_elite_ns3.sh` 启动。
- **GNN-DQN 改进**：在 `ELITE-fusion/dtn/` 中实现 GraphSAGE+DQN（含路网转换、训练脚本、推理接口），控制器侧新增 `ELITE_USE_GNN` 开关，通过 `run_elite_ns3_gnn.sh` 插拔式加载训练好的 `fedg_dqn.pt`，无需改 baseline 逻辑。
- **联邦学习扩展**：`fedgraph/elite_routing/` 自建 FedGraph 任务（ROUTING），含客户端分区脚本、Ray 训练器、示例配置；生成的模型落在 `dtn_out/gnn_dqn_federated_*`，同样通过 GNN 控制器脚本接入。三阶段具有层次递进关系，但彼此可单独运行，对应的控制器与启动脚本我们分别保留了一份，便于切换验证。

后续章节列出的文件清单沿用 baseline 目录结构，联邦学习部分的使用文档见 `fedgraph/elite_routing/README.md`。

Files Overview (当前工程文件概览)
- 顶层
  - how-to-use.txt — 使用指南（控制器 + NS‑3 联动、参数、统计）
  - ELITE-An Intelligent Digital Twin-Based Hierarchical Routing Scheme for Softwarized Vehicular Networks-代码.pdf — 论文
  - plan.md — 工作计划与记录
- ELITE‑fusion（控制器/DTN + APN）
  - ELITE-fusion/controller_server.py — ns3‑ai 控制器服务（SPL/EFG/APN、在线重训、负载估计、R0 更新）
  - ELITE-fusion/Routing_table.py — 路由表构建（加权 + 模糊融合接口）
  - ELITE-fusion/SDVN_Controller.py — APN 策略选择与下发
  - ELITE-fusion/apn.py — R0 奖励聚合、负载档位工具
  - ELITE-fusion/state_action.py — 状态‑动作表
  - ELITE-fusion/topology/sumo_net.py — 解析 SUMO .net.xml（路口坐标/邻接）
  - ELITE-fusion/fuzzy/paper_fuzzy.py — 论文模糊融合实现（HRF/LDF/LBF）
  - ELITE-fusion/dtn/env.py, dtn/rl.py, dtn/train.py — DTN 并行学习环境与训练导出
  - ELITE-fusion/scripts/summarize_elite_stats.py — 路径统计汇总脚本
  - ELITE-fusion/examples/run_minidtn_training.py — 训练示例
  - ELITE-fusion/table_record_*.csv — 示例 Q 表/记录
- ns3‑integration（NS‑3 侧应用/头）
  - ns3-integration/EliteHeader.h — 自定义头（policyId、area_path、nextIndex、src/dst/nextHop、ttl、pathId）
  - ns3-integration/EliteForwarderApp.h, ns3-integration/EliteForwarderApp.cc — 逐跳转发应用（邻居维护、目标路口推进、TTL、段/路径指标采集、ns3‑ai 交互）
  - ns3-integration/README.md — 集成说明
- grid_network（SUMO 场景示例）
  - grid_network/grid.net.xml, grid.sumocfg, grid.rou.xml 等 — 示例路网与车流

说明
- 运行控制器前设置 PYTHONPATH 指向 ns3‑ai py_interface；详细见 how-to-use.txt。
- NS‑3 场景源文件在 VaN3Twin 工程（ELTI 目录）内，本文档列出本仓库内与之对接的头/应用代码（可复制到 NS‑3 侧使用）。

NS‑3 侧（不在本目录，但属于本工程的文件）
- `/home/Lyh/VaN3Twin/ns-3-dev/src/automotive/examples/ELTI/ELTIE.cc` — ELTI 场景入口（SUMO+802.11p+Poisson 触发）
- `/home/Lyh/VaN3Twin/ns-3-dev/src/automotive/examples/ELTI/EliteForwarderApp.h` — 车端逐跳转发应用（声明）
- `/home/Lyh/VaN3Twin/ns-3-dev/src/automotive/examples/ELTI/EliteForwarderApp.cc` — 车端逐跳转发应用（实现）
- `/home/Lyh/VaN3Twin/ns-3-dev/src/automotive/examples/ELTI/EliteHeader.h` — ELITE 自定义头
- `/home/Lyh/VaN3Twin/ns-3-dev/src/automotive/examples/CMakeLists.txt` — 已添加示例目标 `ELTI`

Full Manifest (this repo)
（以下为当前仓库内文件清单，便于第三方审核；NS‑3 侧文件清单见上节路径）
```

```
