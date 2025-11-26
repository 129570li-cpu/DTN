 一体化集成方案（按论文全量）

  - 目标
      - 控制器侧（SDVN）：DTN并行Q学习产出四张单指标表→归一化→加权/模糊融合（HRF/LDF/LBF）→APN状态‑动作表自适应选策→路口级路径规划并下发。
      - 物理侧（NS‑3）：SUMO驱动移动→802.11p(WAVE)真实通信→携带路径头逐跳转发→上报AD/HC/RO/CO→控制器用R0奖励更新P(state,action)闭环。
      - 论文参数复现：1.5km×1.5km、12路口/26道路、IEEE 802.11p（10MHz、TxPower等）、300m通信范围、信标1s、数据1024B/控制256B、1–5 pkt/s、评价PDR/AD/PL/
        RO/CO。
  - 控制器侧（Python，ELITE）
      - DTN并行学习（SPL）
          - 解析你SUMO路网（.net.xml）→构建 it_pos（路口id→x,y）、adjacents_comb（路口→邻居列表）、road_len（邻接路口间直线长度）。
          - 按论文三模式选路（利用/几何贪心/探索），成功路径回溯更新四个Q表：QPDR/QAD/QHC/QRC。
          - 奖励严格按论文：
              - RPDR(c,d,a)=1（仅成功路径更新）
              - RAD(c,d,a)=1−(ADca/ADsd)×(1−ADsd/mADsd)
              - RHC(c,d,a)=exp(−HCca·C/l_ca)
              - RRC(c,d,a)=1/(1+RCca/RCsd)
          - 导出CSV（维度=∑度×|路口|）→Routing_table读取。
      - 融合（EFG）
          - 归一化到(0,1]（对每个des,current的邻居列按log比值归一）。
          - 加权平均→BP；模糊融合→HRF/LDF/LBF（变量/集合/推理/解模糊严格按论文：R,D,H,S三角隶属；Min–Max；质心法）。
      - APN策略部署（严格按论文）
          - 状态=消息类型（安全/效率/信息/娱乐）×负载档[0,0.3)、[0.3,0.7]、(0.7,1.0]。
          - 负载估计：先用Gbas计算Lbas，Lavg为路径上车辆缓冲平均（ns‑3侧上报队列或应用缓存）；loadbas=min(Lavg/Bw,1)判档。
          - R0=a·g(ADsd)+b·q(HCsd)+γ·l(RCsd)，其中g/q/l为沿路径各段 RAD/RHC/RRC 的平均（分母nsd）；权重按动作（如LDF：AD:0.5, HC:0.25,
            PDR:0.25→a=0.5,b=0.25,γ=0）。
          - 一步增量更新 P(state,action)+=R0；按最大偏好选 G∈{HRF,LDF,LBF}。
  - NS‑3侧（C++，你已有 v2v-simple-cam-exchange-80211p.cc）
      - 802.11p与SUMO联动保持现状（PhyMode=OfdmRate6MbpsBW10MHz 起；TxPower调至300m可靠范围；TraCI同步）。
      - 定制转发头部（Header）
          - 字段：routeLen、route[32]（路口id）、nextIndex、policyId（HRF/LDF/LBF）、msgType、ttl、pathId。
          - 每包携带area_path；到达某路口附近（阈值=路口半径）则nextIndex++，目标换成下一个路口。
      - 逐跳转发App（EliteForwarderApp）
          - 邻居维护：基于CAM定期更新neighbor表(id、pos、lastSeen)。
          - 下一跳选择：优先“更靠近下一目标路口”的邻居；若无更近则回退（离当前路口最近）。
          - 统计 per‑segment 与 per‑path指标：
              - ADca：该段从进入到离开路段的累计传输时延（包内携带段起始时间戳）
              - HCca：该段内转发次数（至下一路口）
              - RCca：该段内控制消息数（含信标/请求/回复/上报），可通过分类的应用端口/Tag计数
          - 完成后上报：success/fail、ADsd、HCsd、RCsd、hopCount、PL（相邻传输几何和）、RO（数据期控制消息数）、CO（维持开销/训练时开销，见下）
      - 控制消息分类与计数
          - RO：数据期的控制消息（邻居信标、路径请求/回复、路径上报）。
          - CO：不携带业务数据的网络维护/训练期消息（周期信标、DTN训练统计可在控制器侧单独记录）。
      - 路口映射
          - 用TraCI拿车位(x,y)匹配最近路口it_pos；或edge→junction映射；控制器与ns‑3共享一份路口id空间（保持一致）。
  - NS3‑AI桥接（强同步/高吞吐）
      - Observation（定长结构，建议64邻居，32路径）：
          - simTime、vehicleId、pos(x,y)、currentJunc、destJunc、msgType
          - neighborCount、neighbors[64]、distToNext[64]、queueLen、bufferBytes、channelBusyRatio（可选）
          - routeRequestFlag（是否发起路由请求）
          - pathFeedback：ADsd、HCsd、RCsd、hopCount、PL、RO、success（用于APN更新）
      - Action：
          - policyId（HRF/LDF/LBF）、routeLen、route[32]；可选备选路径
      - 交互时机
          - 发起路由请求：填Obs→AiClient.SetObs()→等待→取Act→写入包头。
          - 周期策略切换：周期性上送负载/队列→控制器返回新的policyId（不变更路径）。
          - 完成/失败时：发送反馈Obs（带success/AD/HC/RC等）。

  交付与验收标准

  - 功能：按消息类型与负载档，自适应选 HRF/LDF/LBF，控制器下发area_path，车辆逐跳转发，完成后按R0更新P(state,action)。
  - 指标：输出PDR/AD/PL/RO/CO与论文相符的趋势（无需对比基线）。
  - 参数：802.11p参数、消息大小、生成速率、区域尺度与论文一致；通信范围≈300m。
  - 代码：控制器（Python）+ NS‑3（C++）+ NS3‑AI（桥接）；路网解析与Q表训练与融合；日志与绘图脚本。

  实施清单（一次到位）

  1. 路网解析与DTN训练
      - 解析你SUMO .net.xml，构建 it_pos/adjacents/road_len。
      - 按论文SPL训练4张Q表→CSV→融合出BP/HRF/LDF/LBF。
  2. APN严格实现
      - 负载估计：用Gbas路径与 ns‑3 上报队列/缓存求 Lavg，Bw=10MHz下配置值；loadbas判档。
      - R0= a·g(ADsd)+b·q(HCsd)+γ·l(RCsd)，g/q/l基于 per‑segment RAD/RHC/RRC 平均；按动作设置权重。
      - 完整状态‑动作表维护与持久化。
  3. NS‑3侧改造
      - 自定义Header与EliteForwarderApp；集成到 v2v-simple-cam-exchange-80211p.cc（在 setupNewWifiNode lambda 创建App，CAM回调维护邻居，发送/接收接
        入App）。
      - 分类统计RO/CO，路径完成时打包反馈。
  4. NS3‑AI桥接
      - 定义Obs/Act结构（固定数组大小），注册AiClient，打通“请求→下发→完成反馈→更新”的严格同步循环。
  5. 校准与回归
      - 调TxPower/传播模型，验证通信半径；检查指标计算正确；跑小场景验收，再扩到论文规模。

  立即可做的两步

  - 我先落控制器侧“严格版”：
      1. net.xml解析→DTN训练4表（替换掉合成网格）；
      2. APN用论文R0公式与loadbas判档，替换resolve_report中的简化奖励。
  - 同步你改 ns‑3：
      1. 在 v2v-simple-cam-exchange-80211p.cc 中新增EliteForwarderApp与自定义Header，按上面接口接好 NS3‑AI；
      2. 把路口映射（车→junction）打通，并把邻居/队列等观测填入Obs。

  说明：不做“最小样例”，按论文全链路一步到位。你确认后，我就按这个方案开始提交控制器侧严格版实现，同时给你ns‑3端的C++结构体/类与插入点（文件路径与行号）清
  单，便于你直接改。