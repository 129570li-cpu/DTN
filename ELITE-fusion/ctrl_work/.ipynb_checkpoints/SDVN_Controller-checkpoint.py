import Global_Par as Gp
import Routing_table as RT
import math
from state_action import StateActionTable

# Paper-accurate APN reward/load helpers
try:
    from apn import aggregate_reward as apn_aggregate_reward
except Exception:
    apn_aggregate_reward = None

# 计算两节点之间的距离
def nndis(ax, ay, bx, by):
    temp_x = ax - bx
    temp_y = ay - by
    temp_x = temp_x * temp_x
    temp_y = temp_y * temp_y
    result = math.sqrt(temp_x+temp_y)
    return result

class SDVNController:
    def __init__(self, node_num, intersection):
        self.hello_list = []  # hello请求列表
        self.flow_request_list = []  # 路由请求列表
        self.flow_report_list = [] # 路由回复列表
        self.intersection = intersection # 所有路口 {it_0:[x,y], it_1:[x,y], ..., it_n:[x,y]}
        self.node_info_dict = {i: [] for i in range(node_num)}  # 所有节点信息
        self.all_node_neighbor = {i: [] for i in range(node_num)} # 所有节点的邻居
        self.it_cover = {i: -1 for i in range(node_num)} # 所有节点所属路口 {node:intersection, ...}
        self.routing_table = RT.Routing_Table() # 路由表实例
        self.road_veh_num = {} # 道路上车辆数 {jun1: {junc1: x,...},...}
        self.road_veh_num_ow = {} # 道路上单向行驶车辆数 {jun1: {junc1: x,...},...}
        self.junc_veh = {}  # 单个路口内车辆列表 {junc: [veh1,...],...}
        self.road_veh = {}  # 道路上车辆列表 {junc: {junc: [],...},...}
        # Optional: state-action table for adaptive policy deployment
        self.state_action = StateActionTable()
        self.gnn_router = None
        self.gnn_router_enabled = False

    def set_gnn_router(self, router):
        self.gnn_router = router

    def enable_gnn_router(self, enabled: bool):
        self.gnn_router_enabled = enabled

    # 统计每个车辆的邻居
    def cal_neib(self):
        for veh, info in self.node_info_dict.items():
            c_x = info[0][0]
            c_y = info[0][1]
            for nveh, ninfo in self.node_info_dict.items():
                if veh == nveh:
                    continue
                n_x = ninfo[0][0]
                n_y = ninfo[0][1]
                d_cn = nndis(c_x, c_y, n_x, n_y)
                if d_cn <= Gp.com_dis:
                    self.all_node_neighbor[veh].append(nveh)
        return

    # 根据hello列表中的条目更新控制器中的节点信息
    def predict_position(self):
        # 提取列表中的每一个hello包，将（id，地理位置，速度，加速度，当前剩余缓存）存储到node_info_dict字典中。
        # 记录并更新当前全部节点的信息
        for value in self.hello_list:
            self.node_info_dict[value.node_id] = [value.position, value.current_cache]
        self.hello_list.clear() # 清空hello列表，为下一时刻腾地方
        self.cal_neib() # 邻居
        return

    # 通知节点所属区域
    def send_area_info(self, node_list):
        # 向每个节点发送流包，告诉他们所属的交叉口
        for node in node_list:
            # 节点所属区域id 是一个列表[]
            area = self.it_cover[node.node_id]
            # 环境侧自行定义通知格式，这里尝试直接调用节点接口
            try:
                node.receive_notify(area)
            except Exception:
                pass
        # 时延处理
        return

    # # 选出具有最大Q值的下一跳区域
    # candidates = Gp.q_table.table[des_area][current_area]
    # next = candidates[candidates == max(candidates)].index
    # next_area = random.choice(next)

    # Select a policy table based on state (message type and load level).
    # If no state-action info available, fall back to Gp.tag / BP.
    def _select_policy_table(self, load_level=1, message_type="efficiency"):
        # Determine action via state-action table (APN-like)
        try:
            action = self.state_action.select((message_type, load_level))
        except Exception:
            action = None
        # Fallback to tag if no learned preference yet
        if action is None:
            if Gp.tag == 1:
                action = "HRF"
            elif Gp.tag == 2:
                action = "LDF"
            elif Gp.tag == 3:
                action = "LBF"
            else:
                action = "BP"
        if action == "HRF":
            return self.routing_table.table_HRF
        if action == "LDF":
            return self.routing_table.table_LDF
        if action == "LBF":
            return self.routing_table.table_LBF
        return self.routing_table.table_BP

    # 查表计算区域路径（根据当前状态选择策略表）
    def calculate_area_path(self, node_id, des_id, load_level=1, message_type="efficiency"):
        if self.gnn_router_enabled and self.gnn_router:
            gnn_path = self._calculate_area_path_gnn(node_id, des_id)
            if gnn_path:
                return gnn_path
        table = self._select_policy_table(load_level=load_level, message_type=message_type)
        # 复用通用实现
        return self.calculate_area_path_with_table(node_id, des_id, table)

    # 指定策略表的路径计算（用于 baseline Gbas）
    def calculate_area_path_with_table(self, node_id, des_id, table):
        area_path = []
        # 严格依赖 NS-3 端上传的 it_cover 映射，不做回退
        node_area = self.it_cover[node_id][0]
        des_area = self.it_cover[des_id][0]
        print("source area: %d  target area: %d " % (node_area, des_area))
        current_area = node_area
        area_path.append(current_area) # 把当前数据包所在的区域添加到路径中
        while current_area != des_area:
            #print("current area: ", current_area)
            # 如果目的区域为当前区域的邻居，直接选择其为下一跳
            if des_area in Gp.adjacents_comb[current_area]:
                area_path.append(des_area)
                break
            # 否则按照权重降序选择未访问过的邻居；若都已访问则判定为回环/死路（不做几何贪婪兜底）
            # 1) 权重候选（按 HRF/LDF/LBF 的表值降序）
            candidates = table[des_area][current_area]  # Series: neighbor -> weight
            try:
                candidates_sorted = list(candidates.sort_values(ascending=False).index)
            except Exception:
                try:
                    items = list(candidates.items())
                    items.sort(key=lambda kv: kv[1], reverse=True)
                    candidates_sorted = [k for k, _ in items]
                except Exception:
                    candidates_sorted = []
            visited = set(area_path)
            prev_area = area_path[-2] if len(area_path) >= 2 else None
            next_area = None
            for nb in candidates_sorted:
                if nb in visited:
                    continue
                if prev_area is not None and nb == prev_area:
                    continue
                next_area = nb
                break
            if next_area is None:
                # 无法前进，判定为回环/死路
                print("loop in area selection")
                Gp.loop_fail_time += 1
                return []
            # 前进一步
            area_path.append(next_area)
            current_area = next_area
            # 保护：步数上限防止极端情况下的无限循环
            try:
                if len(area_path) > len(Gp.it_pos) + 5:
                    print("exceed path step limit, abort selection")
                    return []
            except Exception:
                pass
        # 截取
        i = 0
        try:
            # 当缺少精确的“车辆所属路口”信息时，跳过截断逻辑，保持完整的 area_path
            if isinstance(self.junc_veh, dict) and len(self.junc_veh) > 0:
                for area in area_path:
                    try:
                        vehs = self.junc_veh.get(area, [])
                        if node_id in vehs and area != node_area:
                            i = area_path.index(area)
                            break
                    except Exception:
                        # 局部异常不终止路径计算
                        continue
        except Exception:
            # self.junc_veh 不可用时保持 i=0
            pass
        area_path = area_path[i:]
        # 打印
        print("area path: ", area_path)
        return area_path

    def _calculate_area_path_gnn(self, node_id, des_id):
        try:
            return self.gnn_router.compute_path(node_id, des_id, max_hops=len(self.intersection) + 5)
        except Exception:
            return []

    # 发送路由回复
    # @staticmethod
    def send_reply(self, requester_id, area_path, node_list):
        # 控制器直接调用目标节点接口交付路径，由外层仿真环境实现
        for node in node_list:
            if node.node_id == requester_id:
                try:
                    node.receive_flow(area_path)
                except Exception:
                    pass
        return

    # 处理请求表中的每个请求，计算路由，发送回复
    def resolve_request(self, node_list):
        # 遍历路由请求列表
        for request in self.flow_request_list: # flow_request_list由Node中产生路由请求的generate_request()生成
            # 查表计算路径
            # TODO: derive real-time load_level/message_type from environment; for now choose medium/efficiency
            area_path = self.calculate_area_path(request.node_id, request.des_id, load_level=1, message_type="efficiency")
            # 向发送请求节点回复
            self.send_reply(request.node_id, area_path, node_list)
        self.flow_request_list.clear() # 清空路由请求列表，下回还得使
        return

    # 根据路由结束后的路由回复，解析，更新路由表
    def resolve_report(self):
        # 使用状态-动作表进行一次步更新（与论文一致的 R0 口径），以便策略自适应选择
        # 当无法获取精细的逐段指标时，使用基于道路几何的近似值进行回退估计
        # R0 = a*g(ADsd) + b*q(HCsd) + gamma*l(RCsd)
        # 其中：
        #   RAD(c,d,a) = 1 - (ADca/ADsd)*(1 - ADsd/mADsd)
        #   RHC(c,d,a) = exp( - HCca * C / l_ca )
        #   RRC(c,d,a) = 1 / (1 + RCca / RCsd)
        # 近似：ADca = l_ca / v, HCca = ceil(l_ca / C), RCca ≈ HCca
        veh_speed = 13.9  # ~50 km/h，作为路径级近似
        comm_radius = getattr(Gp, "comm_radius", 300.0)
        # 与控制器服务保持一致的策略权重（R0 仅包含 AD/HC/RC 三项）
        policy_weights = {
            "HRF": (0.25, 0.0, 0.25),   # (a=AD, b=HC, gamma=RC)
            "LDF": (0.5,  0.25, 0.0),
            "LBF": (0.25, 0.25, 0.5),
        }
        for report in self.flow_report_list:
            try:
                loss = getattr(report, "loss", 0)
                area_path = getattr(report, "area_path", [])
                if not area_path:
                    continue
                # 估算负载等级（如缺少实时缓存，仍使用路径长度分档）
                load_level = 0 if len(area_path) <= 3 else (1 if len(area_path) <= 6 else 2)
                message_type = "efficiency"
                # 推断本次使用的策略（与 calculate_area_path 保持一致）
                action_table = self._select_policy_table(load_level=load_level, message_type=message_type)
                if action_table is self.routing_table.table_HRF:
                    action = "HRF"
                elif action_table is self.routing_table.table_LDF:
                    action = "LDF"
                elif action_table is self.routing_table.table_LBF:
                    action = "LBF"
                else:
                    action = "LDF"  # 将 BP 回退映射到 LDF 权重以便 R0 计算
                # 构造逐段几何近似指标
                seg_lca, seg_ADca, seg_HCca, seg_RCca = [], [], [], []
                for i in range(1, len(area_path)):
                    u = area_path[i-1]; v = area_path[i]
                    lca = 0.0
                    try:
                        lca = float(Gp.adjacency_dis.get(u, {}).get(v, 0.0))
                    except Exception:
                        lca = 0.0
                    if lca <= 0.0:
                        # 如果缺长，按路口间欧氏距离估计（如有 it_pos）
                        try:
                            ux, uy = Gp.it_pos[u]
                            vx, vy = Gp.it_pos[v]
                            dx = ux - vx; dy = uy - vy
                            lca = math.sqrt(dx*dx + dy*dy)
                        except Exception:
                            lca = 1.0
                    seg_lca.append(lca)
                    seg_ADca.append(lca / max(veh_speed, 1e-6))
                    hops = max(1, int(math.ceil(lca / max(comm_radius, 1.0))))
                    seg_HCca.append(hops)
                    seg_RCca.append(float(hops))  # 用 hop 计数近似控制消息条数
                ADsd = sum(seg_ADca)
                HCsd = float(sum(seg_HCca))
                RCsd = float(sum(seg_RCca))
                # mADsd 使用路径级上界（不低于 ADsd）
                mADsd = max(ADsd, 1.0)
                if apn_aggregate_reward is not None:
                    a, b, gamma = policy_weights[action]
                    R0 = apn_aggregate_reward(ADsd, mADsd, comm_radius,
                                              seg_ADca, seg_HCca, seg_lca, seg_RCca, RCsd,
                                              a, b, gamma)
                else:
                    # 回退：成功 + 与短路径正相关
                    base = 1.0 if loss == 0 else -1.0
                    R0 = base + (0.5 if len(area_path) <= 4 else (0.2 if len(area_path) <= 6 else 0.0))
                self.state_action.update((message_type, load_level), action, R0)
            except Exception:
                # 保底不抛异常，避免打断其它报告处理
                pass
        self.flow_report_list = []
        return

    # 路由表融合
    # 先预处理，然后通过两种融合方式输出新的路由表
    def table_fusion(self):
        self.routing_table.preprocessing()
        self.routing_table.fusion_weight()
        self.routing_table.fusion_fuzzy()
        return

    # --------------------------------------------------------------------------------------- #
    #                                       路网分析
    # --------------------------------------------------------------------------------------- #
    def analyze(self):
        # 计算得到道路-车辆，路口-车辆 分布情况
        veh_detail, node_area = rntf.intra_vehicles_num(self.node_info_dict, Gp.it_pos, Gp.adjacents_comb)
        veh_num, veh_, veh_num_ow = rntf.inter_vehicles_num(self.node_info_dict, Gp.it_pos, Gp.adjacents_comb)
        # 记录到控制器中
        self.junc_veh = veh_detail # 区域所包含车辆 it:[node,...],...
        self.it_cover = node_area # 车辆所属路口 node:[it,...],...
        self.road_veh_num = veh_num # 道路上车辆数 {jun1: {junc1: x,...},...}
        self.road_veh = veh_ # 道路上的车辆信息 {junc: {junc: [],...},...}
        self.road_veh_num_ow = veh_num_ow # 道路上单向车辆数 {jun1: {junc1: x, ...}, ...}
        return

    # --------------------------------------------------------------------------------------- #
    #                                       虚拟智能体并行训练
    # --------------------------------------------------------------------------------------- #
    # 实例化虚拟智能体对象
    def instantiate_virtual_agent(self):
        for i in range(0, self.virtual_agents_num):
            self.virtual_agents.append(VAs.Virtual_agent(i))
        return

    # 虚拟智能体学习路由策略
    def virtual_training(self):
        for agent in self.virtual_agents:
            agent.learning()
        return

































    # def calculate_area_path(self, node_id, des_id):
    #     area_path = []
    #     node_area = self.it_cover[node_id][0]
    #     des_area = self.it_cover[des_id][0]
    #     #print("source area: %d  target area: %d " % (node_area, des_area))
    #     current_area = node_area
    #     area_path.append(current_area) # 把当前数据包所在的区域添加到路径中
    #     while current_area != des_area:
    #         #print("current area: ", current_area)
    #         # 如果目的区域为当前区域的邻居，直接选择其为下一跳
    #         if des_area in Gp.adjacents_comb[current_area]:
    #             area_path.append(des_area)
    #             break
    #         # 否则遍历Q表，选择具有最大Q值的邻居作为下一跳
    #         candidates_dict = self.routing_table.table[current_area][des_area] # 当前和目的区域限定的邻居区域及权重集合
    #         candidates = sorted(candidates_dict.items(), key=lambda item:item[1], reverse=True) # 按value的大小从大到小排序
    #         if candidates[0][0] not in area_path:
    #             area_path.append(candidates[0][0])
    #             current_area = candidates[0][0]
    #         else:
    #             print("loop in area selection")
    #             Gp.loop_fail_time += 1
    #             area_path = []
    #             return area_path
    #     # 截取
    #     i = 0
    #     for area in area_path:
    #         if node_id in self.intra_vehicles_detail[area] and area != node_area:
    #             i = area_path.index(area)
    #     area_path = area_path[i:]
    #     # 打印
    #     print("area path: ", area_path)
    #     return area_path
