# GNN-DQN（GINE + 边特征）训练指引

> 适用于 `ELITE-fusion/dtn/FD/trainer.py` 的离线单机训练流程。  
> 当前模型使用 GINEConv，强依赖 **边特征 + 节点特征**；旧的 GraphSAGE 检查点不可兼容。

## 1. 数据准备

1) **跑 SUMO 拿 FCD**
```bash
sumo -c /home/Lyh/ELITE-zg-main/grid_network_fed/grid.sumocfg --duration-log.statistics
# 例：900s 仿真会生成 /home/Lyh/ELITE-zg-main/grid_network_fed/grid_fcd.out.xml
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
- 训练时会自动拼上 **静态边长（对数归一化）**，成为 `edge_attr = [len, dyn...]`
- 输出的 `windows` 数量 = 仿真时长 / window_step（900s → 90 帧，600s → 60 帧）

## 2. 训练命令示例

> 多进程版 `mp_trainer.py` 尚未接入边特征，请使用单进程 `trainer.py`。

```bash
/home/Lyh/GNN/.venv/bin/python ELITE-fusion/dtn/FD/trainer.py \
  --net-xml /home/Lyh/ELITE-zg-main/grid_network_fed/grid.net.xml \
  --node-features /home/Lyh/ELITE-zg-main/dtn_out/large_node_features_multi_edge.json \
  --edge-map /home/Lyh/ELITE-zg-main/dtn_out/edge_id_map.json \
  --output-dir /home/Lyh/ELITE-zg-main/dtn_out/gnn_dqn_edge \
  --episodes 1000 \
  --hidden-dim 128 \
  --batch-size 64 \
  --buffer-size 50000 \
  --lr 1e-5 \
  --gamma 0.95 \
  --device cuda \
  --policy HRF=0.5,0.25,0.0,0.25 \
  --policy LDF=0.25,0.5,0.25,0.0 \
  --policy LBF=0.25,0.25,0.25,0.25 \
  --failure-reward -5.0 \
  --log-dir /home/Lyh/ELITE-zg-main/dtn_out/gnn_dqn_edge/logs
```
- `--policy NAME=wp,wa,wb,wg` 可重复填写，权重分别对应 PDR/RAD/RHC/RRC。未指定则默认训练 HRF/LDF/LBF。
- 建议学习率从 **1e-5** 起步（GINE+边特征更敏感），如收敛慢可升到 3e-5。
- 如需更稳定，可把 `train()` 里的 `target_sync_interval` 调大（默认 10，可改 100/200）。


    /home/Lyh/GNN/.venv/bin/python ELITE-fusion/dtn/FD/mp_trainer.py \
      --net-xml grid_network_fed/grid.net.xml \
      --node-features dtn_out/large_node_features_multi_edge.json \
      --edge-map dtn_out/edge_id_map.json \
      --output-dir dtn_out/gnn_dqn_edge_mp_new \
      --episodes 1000 --hidden-dim 128 --batch-size 64 \
      --lr 1e-5 --gamma 0.95 --device cuda \
      --num-workers 4 --failure-reward -5.0 \
      --step-penalty -0.5 --goal-bonus 5.0

## 3. 训练产物

假设 `--output-dir dtn_out/gnn_dqn_edge/`：
- `hrf/ldf/lbf/`：各自的 `fedg_dqn.pt`（模型）与 `id_mapping.json`。
- `logs/*.csv`：若配置了 `--log-dir`，会写入每个策略的 loss/epsilon/奖励曲线。
- 顶层 `id_mapping.json`：便于检查节点索引映射。

## 4. 在控制器侧使用

运行 `ELITE-fusion/scripts/run_elite_ns3_gnn.sh` 前设置：
```bash
export ELITE_GNN_MODEL_DIR=/home/Lyh/ELITE-zg-main/dtn_out/gnn_dqn_edge/hrf     # 选 HRF/LDF/LBF 其一
export ELITE_GNN_HIDDEN=128
export ELITE_GNN_DEVICE=cuda
export ELITE_GNN_NODE_FEATS=/home/Lyh/ELITE-zg-main/dtn_out/large_node_features_multi_edge.json
export ELITE_GNN_EDGE_MAP=/home/Lyh/ELITE-zg-main/dtn_out/edge_id_map.json
```
脚本会自动把节点/边时间序列加载为推理输入；路径长度上限已提升至 64 跳。

## 5. 常见注意事项
- **旧检查点无法加载**：GINE + `edge_attr` 结构与旧 GraphSAGE 不兼容，需要重新训练。
- **特征文件要配套**：`--node-features` 与 `--edge-map` 必须成对使用；若换了路网或窗口长度，需要重新导出两者。
- **多帧输入**：`large_node_features_multi_edge.json` 存的是时间序列，训练/推理会按仿真时间自动切帧，无需手动切换。
- **单进程采样瓶颈**：当前采样仍是单进程，GPU 利用率由 Python 侧串行采样限制。若要提速需额外改 mp_trainer 以适配边特征。
