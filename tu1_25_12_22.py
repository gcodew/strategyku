import networkx as nx
import numpy as np
from typing import Optional, Tuple, Dict, Any
import json
from collections import defaultdict
# import heapq
# import pulp

LENGTH_RATIO = 1000  # 实际巷道长度与欧氏距离的比值，用于估算路径长度

class MineNetworkBuilder: #构建矿井巷道网络

    deployable_edges = []
    NODE_TYPES = {
        'ground_entrance',     # 地表入口
        'ramp',                # 斜坡道段
        'shaft',               # 竖井段
        'junction',            # 巷道交叉点
        'transport_roadway',   # 运输大巷
        'return_airway',       # 回风巷
        'crosscut',            # 联络巷/石门
        'working_face',        # 工作面（目标点）
        'relay_point',         # 中继/充电点
        'robot_nest',          # 机器人巢穴
        'air_gate',            # 风门（可选）
        'custom'               # 自定义类型
    }

    def __init__(self, length_ratio: float = LENGTH_RATIO):
        """初始化一个空的矿井巷道图"""
        self.G = nx.Graph()
        self._next_id = 0
        self._edge_sum = 0
        self.LENGTH_RATIO = length_ratio

    def _auto_node_id(self) -> int:  #开头带下划线 _ 通常表示这是一个内部使用的方法,外部代码不应该直接调用它
        """
        -> int
        这是类型注解type hint,表示这个方法预期返回一个整数int类型的值。
        注意 这只是提示 Python 在运行时不会强制检查返回值类型（除非使用额外的工具如 mypy。
        """
        nid = self._next_id
        self._next_id += 1
        return nid
    
    def _auto_edge_id(self) -> int:  #开头带下划线 _ 通常表示这是一个内部使用的方法,外部代码不应该直接调用它
        """
        -> int
        这是类型注解type hint,表示这个方法预期返回一个整数int类型的值。
        注意 这只是提示 Python 在运行时不会强制检查返回值类型（除非使用额外的工具如 mypy。
        """
        nid = self._edge_sum
        self._edge_sum += 1
        return nid
    
    #添加巷道节点
    def add_node(
        self,
        node_id: Optional[int] = None, #表示节点的唯一标识符，可以为整数类型或者为 None。如果调用时未提供，则默认为 None
        xy: Tuple[float, float] = (0.0, 0.0),# 表示节点的空间位置，默认是一个三维坐标 (0.0, 0.0, 0.0)，意味着它期望一个包含三个浮点数的元组作为输入。
        **attrs #允许传入额外的关键字参数，这些参数会被视为节点的自定义属性。在python函数定义中，*a收集位置参数变成元组tuple，**a收集关键字参数变成字典
    ) -> int:#返回值的类型提示，表示该函数将会返回一个整数值 ，为节点id
        """
        使用示例
        my_graph.add_node(node_id=1, xyz=(1.0, 2.0, 3.0), node_type='sensor', status='active')
        创建了一个 ID 为 1 的节点，它的位置是 (1.0, 2.0, 3.0)，类型是 'sensor'，并且还附带了一个自定义属性 status='active'。实际调用时应替换 my_graph 为实际使用的对象实例名。
        """
        # if node_type not in self.NODE_TYPES:
        #     raise ValueError(f"Invalid node_type '{node_type}'. Use one of: {self.NODE_TYPES}")# f-string（格式化字符串字面量），会自动把 {} 中的变量替换成实际值
        #raise 是 Python 的抛出异常的关键字，ValueError 是一种内置的异常类型，表示“传入了无效的值”。
        if node_id is None:
            node_id = self._auto_node_id()
        
        attrs.update({
            'xy': tuple(xy),# 确保即使传入的是 list（如 [1,2,3]），也会转为不可变的 tuple (1.0, 2.0, 3.0)，避免后续被意外修改
        })#把必须的属性（坐标 xyz 和类型 type）注入到用户传入的 **attrs 字典中。覆盖 attrs 中已有的 'xyz' 或 'type' 键（强制使用方法参数的值）。
        self.G.add_node(node_id, **attrs)#调用 NetworkX 的 add_node 方法。**attrs 将字典展开为关键字参数，等价于：add_node(node_id, xyz=(x,y,z), type='junction', color='red', ...),不再有一个叫'attrs'的字段了
        return node_id
        
    def add_edge(
        self,
        u: int,#端点节点
        v: int,#端点节点
        edge_id: Optional[int] = None,
        **attrs#额外属性，以字典形式传入，包括如巷道类型、支护强度、堵塞信息等
    ) -> None:
        """
        添加一条巷道边（连接两个节点）。
        
        参数:
            u, v: 节点ID
            length: 巷道长度（米）。若为None，则根据xyz自动计算欧氏距离
            **attrs: 其他边属性（如 wind_speed=5.0, width=3.0, is_blocked=False）
        """
        if u not in self.G or v not in self.G:
            raise ValueError(f"Nodes {u} or {v} do not exist in the graph.")
        if edge_id is None:
            edge_id = self._auto_edge_id()

        # if length is None:
        pos_u = self.G.nodes[u]['xy']#获得端点的地理位置
        pos_v = self.G.nodes[v]['xy']
        angle = attrs.get('angle',0.0)
        length = float(np.linalg.norm(np.array(pos_u) - np.array(pos_v)))/np.cos(np.radians(angle))#自动计算边长度
        #float（）强制类型转换，np.linalg.norm 是 NumPy 库中用于计算向量或矩阵的范数（norm） 的函数。

        """
        pos_u = (1.0, 2.0, 3.0)
        pos_v = (4.0, 6.0, 3.0)

        vec = np.array(pos_u) - np.array(pos_v)  # → [-3.0, -4.0, 0.0]
        distance = np.linalg.norm(vec)           # → √((-3)² + (-4)² + 0²) = √(9+16) = √25 = 5.0
        """
        attrs['id'] = edge_id#将 length 存入 attrs
        attrs['length'] = round(length * self.LENGTH_RATIO, 3)
        #round(length, 3) 保留3 位小数
        self.G.add_edge(u, v, **attrs)
        #后续可以通过边的字典直接赋值G[u][v]['weight'] = 5.0 或者 G.edges[u, v]['weight'] = 5.0

    # ======================
    # 便捷语义方法，快速添加节点，#######待修改
    # ======================
    
    def add_ground_entrance(self, node_id=None, xy=(0,0), **attrs):
        return self.add_node(node_id=node_id, xy=xy, node_type='ground_entrance', **attrs)

    def add_working_face(self, node_id=None, xy=(0,0), is_target=True, **attrs):
        attrs['is_target'] = is_target
        return self.add_node(node_id=node_id, xy=xy, node_type='working_face', **attrs)

    def add_relay_point(self, node_id=None, xy=(0,0), **attrs):
        return self.add_node(node_id=node_id, xy=xy, node_type='relay_point', **attrs)

    def add_robot_nest(self, node_id=None, xy=(0,0), **attrs):
        return self.add_node(node_id=node_id, xy=xy, node_type='robot_nest', **attrs)

    def add_junction(self, node_id=None, xy=(0,0), **attrs):
        return self.add_node(node_id=node_id, xy=xy, node_type='junction', **attrs)

    # ======================
    # 查询与工具方法（目前没啥用）
    # ======================

    def get_target_nodes(self) -> list:
        """返回所有标记为 is_target=True 的工作面节点"""
        return [n for n, d in self.G.nodes(data=True) if d.get('is_target', False)]

    def get_nodes_by_type(self, node_type: str) -> list:
        """按类型筛选节点"""
        return [n for n, d in self.G.nodes(data=True) if d.get('type') == node_type]

    def compute_edge_lengths(self) -> None:
        """为所有未设置 length 的边计算长度（基于xyz）"""
        for u, v, data in self.G.edges(data=True):
            if 'length' not in data:
                pos_u = self.G.nodes[u]['xy']
                pos_v = self.G.nodes[v]['xy']
                data['length'] = float(np.linalg.norm(np.array(pos_u) - np.array(pos_v)))

    def validate(self) -> bool:
        """简单校验：所有节点是否都有 xyz 和 type"""
        for n, d in self.G.nodes(data=True):
            if 'xy' not in d or 'type' not in d:
                print(f"Node {n} missing 'xy' or 'type'")
                return False
        return True

    def load_data(self, file_path): 
        """从json文件加载节点和边"""
        with open(file_path,'r', encoding='utf-8') as f:
            data = json.load(f)

        for element in data.get('elements',[]):
            if element['type'] == 'node':
                node_id = element['id']
                x = element['lon']# x坐标
                y = element['lat']# y坐标
                attrs = element.get('attrs',{})
                self.add_node(node_id=node_id,xy=(x,y),**attrs)

            elif element['type'] == 'way':
                node_list = element['nodes']
                attrs = element.get('attrs',{})
                for i in range(len(node_list)-1):#仅当node_list = [101, 102, 103, 104]有2个以上点才用
                    u = node_list[i]
                    v = node_list[i+1]
                edge_id = element['id']
                self.add_edge(u,v,edge_id=edge_id,**attrs)

        


    # ======================
    # 可视化（可选）
    # ======================

    def draw_2d(self, figsize=(12, 8)):#figsize=(12, 8)，设置图形窗口大小（宽 12 英寸，高 8 英寸）
        try:#避免因未安装 matplotlib 导致程序报错。
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Matplotlib not installed. Skip drawing.")
            return

        if len(self.G) == 0:#空图检查
            print("Graph is empty.")
            return

        fig = plt.figure(figsize=figsize)#1. 创建一个图形窗口（Figure）
        # 2. 在这个窗口中添加一个坐标系（Axes），这里是 3D
        ax = fig.add_subplot(111)#projection='3d' 是关键：告诉 Matplotlib 创建一个3D 坐标轴
        #   111是一种简写形式，表示：1 行 × 1 列 的子图布局，选择第 1 个子图，add_subplot(nrows, ncols, index) 的作用是：在图形中划分网格，并选择其中一个格子来绘图。
        color_map = {#节点颜色映射（按类型着色）
            'ground_entrance': 'green',
            'working_face': 'red',
            'relay_point': 'orange',
            'robot_nest': 'purple',
            'junction': 'lightblue',
            'ramp': 'cyan',
            'shaft': 'darkblue',
            'transport_roadway': 'gold',
            'return_airway': 'limegreen',
            'crosscut': 'magenta',
            'air_gate': 'black'
        } 

        """
        绘制节点
        """
        # for node, data in self.G.nodes(data=True):#遍历所有节点，同时获取节点属性字典 data。
        #     x, y= data['xy']
        #     type = data.get('type','unknown')
        #     color = color_map.get(type, 'gray')#如果节点类型在 color_map 中，用对应颜色；否则用默认色 'gray'
        #     # 3. 在 ax 上画东西（而不是直接用 plt.xxx）
        #     ax.scatter(x, y, c=color, s=10, edgecolor='k')
        #     #s=60：点的大小,edgecolor='k'：黑色边框，让点更清晰

        """
        绘制边
        """

        # ax.scatter(362.0, 597.0, c='black', marker='.', s=300, edgecolors='white', linewidths=1.5, zorder=3, label='Start Point')

        for u, v, d in self.G.edges(data=True):#u是起点节点，v是终点节点，d是该边属性字典，也就是**attrs传入的所有自定义字段
            x_vals = [self.G.nodes[u]['xy'][0], self.G.nodes[v]['xy'][0]]#提取两个端点的 x 坐标；
            y_vals = [self.G.nodes[u]['xy'][1], self.G.nodes[v]['xy'][1]]#提取两个端点的 y 坐标；
            color = "gray"
            if d.get('blocked',False):#边绘制颜色控制
                color = "red"
            ax.plot(x_vals, y_vals, c=color, linewidth=2, alpha=1) #alpha=0.7：70% 透明度，避免遮挡节点
            #style = 'solid' if data.get('is_main', False) else 'dashed' #主运输巷用粗线，通风巷用虚线
            #ax.plot(..., linestyle=style)

        #设置坐标轴标签和方向
        # ax.legend(loc='upper right')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.title("MINE NETWORK")#添加标题并显示
        plt.tight_layout()#自动调整边距，防止标签被裁剪。
        plt.show()#弹出图形窗口。

    # ======================
    # 导出与兼容
    # ======================

    def get_graph(self) -> nx.Graph:
        """返回底层 NetworkX 图对象（用于脆弱性分析等）"""
        return self.G

    # ======================
    # 节点和边属性获取和更新
    # ======================

    # 获取节点属性
    def get_node_attr(self,node_id:int, key:str, default = None):
        return self.G.nodes[node_id].get(key,default)
    
    # 获取边属性
    def get_edge_attr(self, u:int,v:int,key:str, default = None):
        if not self.G.has_edge(u,v):
            return default
        return self.G[u][v].get(key,default)
    
    # 更新节点属性
    def set_node_attr(self, node_id:int,need_print = False,**attrs):
        if node_id not in self.G:
            raise KeyError(f"Node {node_id} does not exist in the graph.")
        if need_print:#需要打印的话       
            for key in attrs:
                if key in self.G.nodes[node_id]:
                    old_val = self.G.nodes[node_id].get(key)
                    print(f"revise the existed attribute {key}:old {old_val}->new {attrs[key]}")
                else: print(f"add new attribute{key}: {attrs[key]}")
        self.G.nodes[node_id].update(attrs)
    
    # 更新边属性
    def set_edge_attr(self, u:int,v:int, need_print = False, **attrs):
        if not self.G.has_edge(u,v):
            raise KeyError(f"Edge ({u}, {v}) does not exist.")
        if need_print:#需要打印的话，设置need_print = True   
            for key in attrs:
                if key in self.G[u][v]:
                    old_val = self.G[u][v].get(key)
                    print(f"revise the existed attribute {key}:old {old_val}->new {attrs[key]}")
                else: print(f"add new attribute{key}: {attrs[key]}")
        self.G[u][v].update(attrs)

    # ======================
    # 起点选址模型构建相关函数
    # ====================== 

    # 定义可以部署的主干类巷道
    def get_deployable_edges(self,*deployable_edge_types,judge_by_type = False):
        deployable_edges = []
        if judge_by_type:
            for u,v,d in self.G.edges(data = True):
                if d.get('edge_type','') in deployable_edge_types:
                    deployable_edges.append((u,v,d.get('id')))
                    self.set_edge_attr(u,v,deployable=True)
        else:
            for u,v,d in self.G.edges(data = True):
                if d.get('deployable','') == True:
                    deployable_edges.append((u,v,d.get('id')))
                    self.set_edge_attr(u,v,deployable=True)
        return deployable_edges
    
    
"""
在可部署的巷道上生成离散候选点（从而近似连续选址）,默认间隔为20m
返回:候选点
并记录候选点到左右两端点的距离和位置信息
"""
def discretize_deployable_edges(G, deployable_edges, interval = 20.0):
    candidates = []
    for edge_data in deployable_edges:
        # 兼容可能传入 (u, v) 或 (u, v, id) 的情况
        u, v = edge_data[0], edge_data[1]
        pos_u = np.array(G.nodes[u]['xy'])
        pos_v = np.array(G.nodes[v]['xy'])
        edge_vec = pos_v - pos_u
        length = G.edges[u, v]['length']
        if length == 0:
            continue
        
        unit_vec = edge_vec / length # 单位方向向量
        # num_points = int(length // interval) + 1 #采样点个数，+1是为了包含起点

        # 计算需要的分段数：至少1段（即两个端点）
        num_intervals = max(1, int(np.ceil(length / interval)))
        # 生成 num_intervals + 1 个点（包括起点和终点）
        offsets = np.linspace(0, length, num_intervals + 1)

        for offset in offsets:
            pos = pos_u + unit_vec * offset
            dist_to_u = offset
            dist_to_v = length - offset
            
            # 检查是否靠近高危区（简化：若附近有 working_face，则跳过？或后续约束处理）
            candidates.append({
                'edge': (u, v),
                'offset': float(offset),
                'pos': tuple(pos.tolist()),
                'length_along_edge': float(offset),
                'total_length': float(length),
                'dist_to_u': float(dist_to_u),
                'dist_to_v': float(dist_to_v)
            })
    return candidates


"""
计算所有节点到每个target点的最短路径时间
返回:距离矩阵D
dict[t][node] = distance from node to target t
"""
def compute_all_node_to_targets_distances(G, target_nodes, weight='length'):
    dist_to_targets = {}
    for t in target_nodes:
        try:
            dists = nx.single_source_dijkstra_path_length(G, source=t, weight=weight)
            dist_to_targets[t] = dists
        except nx.NetworkXError:
            print(f"Warning: Target {t} is not in graph or disconnected.")
            dist_to_targets[t] = {}
            # dists = nx.shortest_path_length(G, source=t, weight=weight)
            # dist_to_targets[t] = dists
    return dist_to_targets


"""
计算每个候选点到所有目标点的最短时间，也就是，合成候选点到目标的距离
根据候选点信息和预计算的节点距离，得到候选点到各目标的距离。
    
    参数:
        candidates: 来自 discretize_deployable_edges
        dist_to_targets: 来自 compute_node_to_targets_distances
        target_nodes: 目标列表
        weight: 必须与预计算时一致
    
    返回:
        D: np.ndarray shape=(n_candidates, n_targets)
        valid_candidates: 过滤掉不可达的候选点
"""
def compute_distance_matrix(dist_to_targets, candidates, target_nodes, weight='length'):
    D = []  # D[i][j] = time from candidate i to target j
    valid_candidates = []

    for cand in candidates:
        u, v = cand['edge']
        offset = cand['offset']
        # L = cand['total_length']
        dist_to_u = cand['dist_to_u']
        dist_to_v = cand['dist_to_v']

        row = []
        feasible = True
        for t in target_nodes:
            d_u = dist_to_targets[t].get(u, float('inf'))
            d_v = dist_to_targets[t].get(v, float('inf'))
            # # 假设边权重 = length（需确保 G 中边有 'time' 或 'length'）
            # time_u = G[u][v].get('time', G[u][v].get('length', L))
            d_total_u = dist_to_u + d_u
            d_total_v = dist_to_v + d_v
            # 插值：从 u 出发走 offset，或从 v 出发走 (L - offset)
            d_cand = min(d_total_u, d_total_v)
            if d_cand >= 1e9:#不可达
                feasible = False
            row.append(d_cand)
         
        if feasible:
            D.append(row)
            valid_candidates.append(cand)

    return np.array(D) if D else np.empty((0, len(target_nodes))), valid_candidates


if __name__ == "__main__":
    # =====构建矿井巷道网络=====
    mineB = MineNetworkBuilder()

    #确定各个网络节点
    mineB.load_data('real_map_data_one.json')
    print("Mine B target count:", len(mineB.get_target_nodes()))
    mineB.draw_2d()  # 可选
    
    # 1. 获取可部署边（假设你已有）
    deployable_edges = [(u, v) for u, v, _ in mineB.get_deployable_edges('main', 'branch')]

    # 2. 生成候选点
    candidates = discretize_deployable_edges(mineB.G, deployable_edges, interval=10.0)

    # 3. 预计算节点到目标的距离（用 'length' 或 'time'）
    target_nodes = [n for n, d in mineB.G.nodes(data=True) if d.get('critical', False)]
    dist_to_targets = compute_all_node_to_targets_distances(mineB.G, target_nodes, weight='length')

    # 4. 计算候选点到目标的距离矩阵
    D, valid_candidates = compute_distance_matrix(
        dist_to_targets, candidates, target_nodes, weight='length'
    )

    # 5. 找最优候选点（例如最小化总距离）
    if D.size > 0:
        total_distances = D.sum(axis=1)
        best_idx = np.argmin(total_distances)
        best_cand = valid_candidates[best_idx]
        print(f"最佳起点: 边 {best_cand['edge']}, 距 u 端 {best_cand['offset_from_u']:.1f} 米")
    else:
        print("无可行候选点！")



# # =====脆弱性评估相关函数=====（待完成）
# # 获取高危目标点     




# # =====构建矿井巷道网络=====
# mineB = MineNetworkBuilder()

# #确定各个网络节点
# mineB.load_data('real_map_data_one.json')
# # main = mineB.add_junction(xy=(0, -10))
# # mineB.add_edge(e2, main)

# # # 单链分支（无环）
# # for i in range(5):
# #     wf = mineB.add_working_face(xy=(i*10, -20))
# #     if i == 0: 
# #         mineB.add_edge(main, wf)
# #     else:
# #         prev_wf = mineB.get_target_nodes()[-2]
# #         mineB.add_edge(prev_wf, wf)

# print("Mine B target count:", len(mineB.get_target_nodes()))
# mineB.draw_2d()  # 可选

# # # =====起点选址模型=====
# # def solve_multi_start_location(
# #     D,                          # 距离矩阵 [K x M]
# #     weights,                    # 脆弱性权重 [M]
# #     n_starts=2,                 # 起点数量
# #     max_time=600,               # 最大响应时间（秒）
# #     lambda_load=0.1,            # 负载均衡权重
# #     epsilon=0.3                 # 负载允许偏差比例
# # ):
# #     K, M = D.shape
# #     avg_weight = np.sum(weights) / n_starts
# #     load_tol = epsilon * avg_weight

# #     # 创建问题
# #     prob = pulp.LpProblem("MultiStart_Location", pulp.LpMinimize)

# #     # 决策变量
# #     z = pulp.LpVariable.dicts("z", range(K), cat='Binary')          # 是否选候选点 k
# #     y = pulp.LpVariable.dicts("y", [(k, j) for k in range(K) for j in range(M)], cat='Binary')
# #     T_max = pulp.LpVariable("T_max", lowBound=0)                    # 最大加权响应时间

# #     # 目标函数
# #     prob += T_max + lambda_load * pulp.lpSum(
# #         (pulp.lpSum(y[(k, j)] * weights[j] for j in range(M)) - avg_weight)**2
# #         for k in range(K)
# #     )

# #     # 约束 1: 选 n_starts 个起点
# #     prob += pulp.lpSum(z[k] for k in range(K)) == n_starts

# #     # 约束 2: 每个目标至少被一个起点覆盖
# #     for j in range(M):
# #         prob += pulp.lpSum(y[(k, j)] for k in range(K)) >= 1

# #     # 约束 3: 电量/时间约束
# #     M_big = 1e6
# #     for k in range(K):
# #         for j in range(M):
# #             prob += D[k, j] <= max_time + M_big * (1 - y[(k, j)])
# #             prob += y[(k, j)] <= z[k]  # 只有选中的起点才能负责目标

# #     # 约束 4: 负载均衡（简化为线性约束）
# #     for k in range(K):
# #         load_k = pulp.lpSum(y[(k, j)] * weights[j] for j in range(M))
# #         prob += load_k <= avg_weight + load_tol
# #         prob += load_k >= avg_weight - load_tol

# #     # 约束 5: 定义 T_max >= w_j * d(k,j) * y(k,j)
# #     for j in range(M):
# #         prob += T_max >= weights[j] * pulp.lpSum(D[k, j] * y[(k, j)] for k in range(K))

# #     # 求解
# #     solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
# #     prob.solve(solver)

# #     # 提取结果
# #     selected_starts = [k for k in range(K) if pulp.value(z[k]) > 0.5]
# #     assignment = defaultdict(list)
# #     for k in range(K):
# #         for j in range(M):
# #             if pulp.value(y[(k, j)]) > 0.5:
# #                 assignment[k].append(j)

# #     return selected_starts, assignment, pulp.value(T_max)


# # # 1. 获取图
# # G = mineB.get_graph()

# # # 2. 脆弱性评估，使得边有权重
# # mineB.compute_edge_lengths()
# # for u, v, data in G.edges(data=True):
# #     data['time'] = data['length']  # 假设速度 1 m/s

# # # 3. 获取评估后的目标点
# # target_nodes = mineB.get_target_nodes()
# # if not target_nodes:
# #     raise ValueError("No target nodes found! Check 'is_target' attribute.")

# # # 4. 定义脆弱性权重
# # vulnerability_weights = np.array([G.nodes[t].get('vulnerability', 1.0) for t in target_nodes])

# # # 5. 获取可部署边
# # deployable_edges = mineB.get_deployable_edges('transport_roadway', 'return_airway', 'crosscut')

# # # 6. 离散化
# # candidates = discretize_deployable_edges(G, deployable_edges, interval=20.0)

# # # 7. 计算距离矩阵
# # D, valid_candidates = compute_distance_matrix(G, candidates, target_nodes, weight='time')

# # print(f"Generated {len(valid_candidates)} candidate points, {len(target_nodes)} targets.")

# # # 8. 求解
# # selected_indices, assignment, T_max = solve_multi_start_location(
# #     D, vulnerability_weights, n_starts=2, max_time=800
# # )

# # # 9. 输出结果
# # print(f"Optimal max weighted response time: {T_max:.1f}")
# # print("Selected start positions (xy):")
# # for idx in selected_indices:
# #     pos = valid_candidates[idx]['pos']
# #     print(f"  {pos}")