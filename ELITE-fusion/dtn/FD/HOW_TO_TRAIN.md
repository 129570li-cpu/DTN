# GNN-DQN（节点+边特征，K 最短路径版）训练指引

> 现在推荐的训练方式：先枚举 **K 条最短候选路径**，再用 GNN-DQN 评分选择，已提供多进程采样脚本 `path_mp_trainer.py`。旧的逐跳探索 `trainer.py/mp_trainer.py` 仍可用，但速度/稳定性较差。

## 1. 数据准备

1) **跑 SUMO 拿 FCD**
```bash
sumo -c /home/Lyh/ELITE-zg-main/grid_network_fed/grid.sumocfg --duration-log.statistics
# 例：900s 仿真 → /home/Lyh/ELITE-zg-main/grid_network_fed/grid_fcd.out.xml
```

2) **生成多帧节点/边特征时间序列**（10s 窗口，10s 步长，全量保留）
```bash
/home/Lyh/GNN/.venv/bin/python ELITE-fusion/dtn/FD/build_node_features.py \
  --net-xml /home/Lyh/ELITE-zg-main/grid_network_fed/grid.net.xml \
  --fcd-xml /home/Lyh/ELITE-zg-main/grid_network_fed/grid_fcd.out.xml \
  --output-json /home/Lyh/ELITE-zg-main/dtn_out/large_node_features_multi_edge.json \
  --edge-map-output /home/Lyh/ELITE-zg-main/dtn_out/edge_id_map.json \
  --window-size 10 --window-step 10 --max-windows 0
```
- **节点特征 8 维**：`[avg_speed, avg_density, avg_queue, peak_density, peak_queue, var_speed, var_density, var_queue]`
- **边特征 4 维（动态）**：`[avg_speed, avg_flow, avg_queue, peak_queue]`
- 训练时会自动拼上 **静态边长（对数归一化）**，形成 `edge_attr = [len, dyn...]`
- `windows` 数量 = 仿真时长 / window_step（900s → 90 帧，600s → 60 帧）

## 2. 路径候选多进程训练（推荐）

```bash
/home/Lyh/GNN/.venv/bin/python ELITE-fusion/dtn/FD/path_mp_trainer.py \
  --net-xml /home/Lyh/ELITE-zg-main/grid_network_fed/grid.net.xml \
  --node-features /home/Lyh/ELITE-zg-main/dtn_out/large_node_features_multi_edge.json \
  --edge-map /home/Lyh/ELITE-zg-main/dtn_out/edge_id_map.json \
  --output-dir /home/Lyh/ELITE-zg-main/dtn_out/gnn_dqn_path_mp \
  --episodes 800 --hidden-dim 128 --batch-size 64 --num-workers 4 \
  --k-paths 5 --max-hops 32 --eps-start 0.6 --eps-end 0.05 --eps-decay 0.995 \
  --step-penalty -0.5 --goal-bonus 5.0 --long-hop-threshold 12 --long-hop-penalty -0.75 \
  --gpu
```
- 每个样本：生成 K 条最短简单路径（按边长），用 GNN 评分后 epsilon-greedy 选 1 条，奖励是整条路径的 PDR/RAD/RHC/RRC 加权。
- `--behavior-policy` 控制 epsilon-greedy 使用哪一头（默认 LDF）；模型同时训练 HRF/LDF/LBF 三个策略头。
- 产物：`output-dir/{hrf,ldf,lbf}/fedg_dqn.pt`。

## 3. 旧版逐跳训练（备用）

仍可使用 `trainer.py`（单进程）或 `mp_trainer.py`（多进程 DFS/BFS 探索）。命令与历史一致，此处不再赘述，推荐优先尝试路径候选版。

## 4. 离线评估

1) **路径候选模型评估（单帧/多帧）**
```bash
/home/Lyh/GNN/.venv/bin/python ELITE-fusion/dtn/FD/eval_path_model.py \
  --net-xml /home/Lyh/ELITE-zg-main/grid_network_fed/grid.net.xml \
  --node-features /home/Lyh/ELITE-zg-main/dtn_out/large_node_features_multi_edge.json \
  --edge-map /home/Lyh/ELITE-zg-main/dtn_out/edge_id_map.json \
  --model-dir /home/Lyh/ELITE-zg-main/dtn_out/gnn_dqn_path_mp/ldf \
  --policy LDF --k-paths 5 --samples 500 --log-file /home/Lyh/ELITE-zg-main/dtn_out/eval_ldf_path.txt
```
- 指标：有效样本数、平均跳数、相对最短路比例、平均路径成本。
- `--frame-idx` 指定帧，否则随机多帧评估。

2) **旧版逐跳评估**：继续使用 `eval_model.py`。

## 5. 在控制器侧使用

```bash
export ELITE_GNN_MODEL_DIR=/home/Lyh/ELITE-zg-main/dtn_out/gnn_dqn_path_mp/ldf   # 选 HRF/LDF/LBF
export ELITE_GNN_HIDDEN=128
export ELITE_GNN_DEVICE=cuda
export ELITE_GNN_NODE_FEATS=/home/Lyh/ELITE-zg-main/dtn_out/large_node_features_multi_edge.json
export ELITE_GNN_EDGE_MAP=/home/Lyh/ELITE-zg-main/dtn_out/edge_id_map.json
```
控制器已支持多模型目录扫描；路径上限 64 跳。

## 6. 注意事项
- **必须提供 edge_map**：K 最短路依赖边长；缺失边长会回退到欧氏长度但效果会差。
- **模型不兼容旧的 GraphSAGE 检查点**：当前是 GINE + 边特征架构。
- **多帧输入**：控制器/评估会按仿真时间切换帧，无需手动更新。
- **多进程队列**：`path_mp_trainer.py` 使用进程队列采样；如 GPU 利用率低，可调大 `--num-workers` 或增大学习率/减小 eps。
