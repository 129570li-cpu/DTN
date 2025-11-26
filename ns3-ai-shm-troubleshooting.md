# ns3-ai 共享内存导致仿真不推进的问题排查与解决

本页记录 ELITE 控制器（Python/ns3‑ai）与 NS‑3（ELTI 场景）联动时，因共享内存（SysV SHM）导致仿真“不推进/卡住/控制器无请求”的常见原因、诊断方法、解决办法与预防建议，便于后续同学快速定位。

## 现象特征（Symptoms）
- SUMO 输出只到 “Simulation version … started with time: 0.00.”，不再出现 Step # 行；`SumoMsg.log` 为空；`grid_fcd.out.xml` 不再增长。
- 控制器日志 `dtn_out/controller.log` 长时间只有初始化信息，没有 `[REQ]`/`[RSP]`、`source area/area path` 等请求/下发记录。
- 偶发看到 ns‑3 直接退出，SUMO 报 “Error: tcpip::Socket::recvAndCheck @ recv: peer shutdown”。
- 设置 `NS_LOG` 时若不当（用逗号分隔），ns‑3 会立即 abort，进一步加重“看起来像没推进”的表象。

## 根因（Root Cause）
1) 残留的共享内存段（SysV SHM）没有被清理：
   - ns3‑ai 使用 SysV 共享内存作为环节（mempool key=1234，memblock key=2333）。异常退出/中断后，这些段仍留在内核里。
   - 新一轮启动会“附着到旧段”，ns‑3 与控制器在 ns3‑ai 条件变量上不同步，表现为“控制器一直等、SUMO 不步进”。
2) 进程属主不一致：
   - 控制器用 root 运行、ns‑3 用 `Lyh` 运行，导致共享内存段和输出目录属主是 root，`Lyh` 侧写不进去，同步/日志都异常。
3) NS_LOG 配置错误：
   - 组件之间用逗号会触发 ns‑3 直接报错退出（正确应使用冒号）。

## 快速诊断（Checklist）
- 查看共享内存：
  ```bash
  ipcs -m | rg '0x000004d2|0x0000091d| 1234 | 2333'
  # 看到 key=0x000004d2(=1234)/0x0000091d(=2333) 存在且 owner=root，多为问题点
  ```
- 查看正在运行的进程与属主：
  ```bash
  ps -ef | rg '(ns3-dev-ELTI|sumo -c|controller_server.py)'
  # 控制器/ns‑3/SUMO 是否统一为同一用户（建议 Lyh）
  ```
- 检查输出目录/日志文件属主：
  ```bash
  ls -l /home/Lyh/ELITE-zg-main/dtn_out
  # 若 controller.log 属主为 root，ns‑3 侧可能写不进去
  ```
- 确认 SUMO 是否真正步进：
  ```bash
  tail -n 200 /home/Lyh/VaN3Twin/ns-3-dev/src/automotive/examples/grid_network/SumoMsg.log
  # 有 Step # 行说明在推进；空则多数是 ns‑3 未驱动 TraCI
  ```
- 检查 NS_LOG 写法：
  ```text
  正确：NS_LOG="TraciClient=level_info:EliteForwarderApp=level_info|prefix_time"
  错误：NS_LOG="EliteForwarderApp=level_info,TraciClient=level_info"
  ```

## 解决步骤（Fix Recipe）
1) 停掉残留进程（容错）：
   ```bash
   pkill -f ns3-dev-ELTI-optimized || true
   pkill -f sumo || true
   pkill -f controller_server.py || true
   ```
2) 清理共享内存段（重点）：
   ```bash
   # 1234(十进制)=0x000004d2，2333=0x0000091d
   ipcrm -M 0x000004d2 || true
   ipcrm -M 0x0000091d || true
   ```
3) 修正输出目录属主：
   ```bash
   chown -R Lyh:Lyh /home/Lyh/ELITE-zg-main/dtn_out
   ```
4) 正确启动（统一用户 + 正确 NS_LOG）：
   - 推荐使用仓库自带一键脚本（已内置清理/统一用户/等待就绪/墙钟超时）：
     ```bash
     ELITE_DEBUG=1 SUMO_STEP_LOG=1 SIM_TIME=60 \
       NS_LOG="TraciClient=level_info:EliteForwarderApp=level_info|prefix_time" \
       bash ELITE-fusion/scripts/run_elite_ns3.sh
     ```
   - 脚本会：
     - 清理旧进程与共享内存（key 1234/2333）；
     - 以同一非 root 用户（默认 Lyh）启动控制器与 ns‑3；
     - 等待控制器生成 `dtn_out/junction_legend.csv` 再启动 ns‑3；
     - 可选输出 SUMO Step 日志；
     - 提供墙钟超时（默认 `SIM_TIME+120s`）防止“卡死”。

## 预防建议（Prevention）
- 始终使用同一用户运行控制器与 ns‑3（建议 `Lyh`），避免 root/非 root 混跑。
- 启动/多次迭代测试时，优先用 `ELITE-fusion/scripts/run_elite_ns3.sh`，让脚本代替手工操作。
- 每次切换测试（尤其上一轮是 root 跑的）前，执行共享内存清理和目录属主修正。
- 记得用冒号分隔 `NS_LOG` 组件，避免因语法错误导致 ns‑3 早退。
- 判断仿真“是否推进”，以 Step 日志、FCD 文件增长、控制器请求日志为准；“peer shutdown”多为 ns‑3 正常结束后的 SUMO 提示。

## 关键参数与映射
- ns3‑ai 共享内存：
  - mempool key（内存池）= 1234（十六进制 0x000004d2；默认大小 4096）
  - memblock key（内存块）= 2333（十六进制 0x0000091d；Env/Act 交换用）
- 定位源码：
  - NS‑3 侧共享内存实现：`contrib/ns3-ai/model/memory-pool.cc/h`，`contrib/ns3-ai/model/ns3-ai-rl.h`
  - 本场景设置 memblock 的位置：`src/automotive/examples/ELTI/ELTIE.cc`（`eliteApp->SetAiKey(2333)`）
  - 控制器（Python）使用的 key：`ELITE-fusion/controller_server.py`（参数中传 `1234/4096/2333`）

## 常见误会澄清（Notes）
- SUMO 结束时的 “peer shutdown” 多为 ns‑3 正常收尾后关闭 TraCI 连接，不是仿真失败。
- `ns3_run.log` 偶发看不到 Step 行，可能是输出缓冲；以 `SumoMsg.log` 或 `grid_fcd.out.xml` 的增长为准。
- 只在“带 Poisson”时暴露问题，是因为该路径会触发 ns3‑ai 读写共享内存；不带 Poisson 基本不触发 ns3‑ai，因此问题不明显。

## 附：最小复现/修复命令集
```bash
# 1) 停旧进程
pkill -f ns3-dev-ELTI-optimized || true
pkill -f sumo || true
pkill -f controller_server.py || true

# 2) 清理共享内存（关键）
ipcrm -M 0x000004d2 || true  # mempool 1234
ipcrm -M 0x0000091d || true  # memblock 2333

# 3) 统一目录属主
chown -R Lyh:Lyh /home/Lyh/ELITE-zg-main/dtn_out

# 4) 一键启动（Poisson=1，λ=0.5，60s）
cd /home/Lyh/ELITE-zg-main
ELITE_DEBUG=1 SUMO_STEP_LOG=1 SIM_TIME=60 \
  NS_LOG="TraciClient=level_info:EliteForwarderApp=level_info|prefix_time" \
  bash ELITE-fusion/scripts/run_elite_ns3.sh
```

