import numpy as np
import random
import copy
import networkx as nx
from sklearn.cluster import KMeans  # 需要安装 scikit-learn: pip install scikit-learn

# 尝试导入你的图构建模块
try:
    from tu1_25_12_22 import (
        MineNetworkBuilder,
        discretize_deployable_edges,
        compute_all_node_to_targets_distances,
        compute_distance_matrix
    )
except ImportError:
    # 如果文件名包含连字符，使用动态导入（备用方案）
    import importlib.util
    import sys
    import os
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_dir)
    sys.path.append(parent_dir)
    spec = importlib.util.spec_from_file_location("tu1_module", os.path.join(parent_dir, "tu1-25-12-22.py"))
    tu1_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tu1_module)
    MineNetworkBuilder = tu1_module.MineNetworkBuilder
    discretize_deployable_edges = tu1_module.discretize_deployable_edges
    compute_all_node_to_targets_distances = tu1_module.compute_all_node_to_targets_distances
    compute_distance_matrix = tu1_module.compute_distance_matrix

class G_HCES_Solver:
    """
    图结构引导的分层约束嵌入式进化优化框架 (G-HCES)
    """
    def __init__(self, dist_matrix, target_weights, candidates, target_nodes, 
                 num_centers=3, max_R=1000, lambda_balance=0.5, avg_load=0):
        """
        :param dist_matrix: (num_candidates, num_targets) numpy 矩阵
        :param target_weights: (num_targets,) 目标权重数组
        :param candidates: 候选点列表 (dict)
        :param target_nodes: 目标点 ID 列表
        :param num_centers: 需要选址的数量 (n)
        :param max_R: 这里的R指距离限制 (Electrical limit / Radius)
        :param lambda_balance: 负载均衡项的权重系数
        """
        self.D = dist_matrix
        self.W = target_weights
        self.candidates = candidates
        self.target_nodes = target_nodes
        self.n = num_centers
        self.R = max_R
        self.lam = lambda_balance
        self.avg_w = avg_load if avg_load > 0 else np.mean(target_weights) * (len(target_nodes) / num_centers)
        
        # 缓存一些索引供快速查找
        self.num_cands, self.num_targets = dist_matrix.shape
        self.all_target_indices = set(range(self.num_targets))

    # --- Stage 1: 图结构驱动的初始化 ---
    def generate_initial_population(self, pop_size=20):
        population = []
        
        # 使用 K-Means 基于距离特征对目标点进行聚类，形成初始“社区”
        # 这里用距离矩阵的列向量作为特征，意味着：如果你离相同的候选点近，你们就是一伙的
        kmeans = KMeans(n_clusters=self.n, n_init=10).fit(self.D.T)
        labels = kmeans.labels_

        # 构建基础分配方案
        base_assignment = [[] for _ in range(self.n)]
        for t_idx, cluster_id in enumerate(labels):
            base_assignment[cluster_id].append(t_idx)

        for _ in range(pop_size):
            individual = []
            # 对于每个社区，寻找一个最佳候选点作为局部中心
            current_assignment = copy.deepcopy(base_assignment)
            
            gene = []
            for cluster_targets in current_assignment:
                if not cluster_targets:
                    # 如果某个簇为空（KMeans偶尔发生），随机选一个候选点
                    rand_c = random.randint(0, self.num_cands - 1)
                    gene.append({'center_idx': rand_c, 'targets': []})
                    continue
                
                # 在该簇的“重心”附近找候选点：即到该簇所有点距离之和最小的候选点
                # 提取子矩阵：所有候选点 到 当前簇目标的距离
                sub_D = self.D[:, cluster_targets]
                sum_dist = np.sum(sub_D, axis=1)
                
                # 贪心选择：取前 5 个最好的里随机选一个（增加多样性）
                top_k_indices = np.argsort(sum_dist)[:5]
                chosen_c = random.choice(top_k_indices)
                
                gene.append({'center_idx': chosen_c, 'targets': cluster_targets})
            
            population.append(gene)
        
        return population

    # --- Stage 3: 适应度评估 ---
    def calculate_fitness(self, individual):
        """
        目标函数: min [ max_j (w_j * D_kj) + lambda * sum(load_diff^2) ]
        同时处理硬约束：如果超出 R，施加巨大惩罚
        """
        max_weighted_response = 0
        load_balance_term = 0
        penalty = 0
        
        covered_targets = set()
        
        for segment in individual:
            c_idx = segment['center_idx']
            t_indices = segment['targets']
            
            current_load = 0
            
            for t_idx in t_indices:
                dist = self.D[c_idx, t_idx]
                w = self.W[t_idx]
                
                # 1. 响应时间 (加权距离)
                response = w * dist
                if response > max_weighted_response:
                    max_weighted_response = response
                
                # 2. 负载累加
                current_load += w
                
                # a. 电量/距离硬约束 (Hard Constraint)
                if dist > self.R:
                    penalty += 1e5 * (dist - self.R) # 严厉惩罚
                
                covered_targets.add(t_idx)

            # 3. 负载均衡项计算
            load_balance_term += (current_load - self.avg_w) ** 2

        # b. 全覆盖硬约束
        uncovered_count = self.num_targets - len(covered_targets)
        if uncovered_count > 0:
            penalty += 1e6 * uncovered_count

        # 总目标值 (越小越好)
        fitness = max_weighted_response + self.lam * load_balance_term + penalty
        return fitness, max_weighted_response, load_balance_term

    # --- 进化操作 ---
    def mutate(self, individual, mutation_rate=0.2):
        new_ind = copy.deepcopy(individual)
        
        # 变异类型 1: 移动中心点 (Local Search)
        if random.random() < mutation_rate:
            seg_idx = random.randint(0, self.n - 1)
            # 在当前中心点附近随机游走，或完全随机跳跃
            # 这里简单实现为：随机重选一个
            new_ind[seg_idx]['center_idx'] = random.randint(0, self.num_cands - 1)
            
        # 变异类型 2: 边界目标点重新分配 (Load Rebalancing)
        if random.random() < mutation_rate:
            # 随机选两个区域
            idx_a, idx_b = random.sample(range(self.n), 2)
            targets_a = new_ind[idx_a]['targets']
            
            if targets_a:
                # 把 A 中的一个目标移给 B
                t_to_move = random.choice(targets_a)
                new_ind[idx_a]['targets'].remove(t_to_move)
                new_ind[idx_b]['targets'].append(t_to_move)
                
        return new_ind

    def run(self, generations=50):
        population = self.generate_initial_population()
        best_fitness = float('inf')
        best_sol = None
        
        print(f"开始 G-HCES 进化，初始种群 {len(population)}，迭代 {generations} 次...")
        
        for gen in range(generations):
            pop_fitness = []
            for ind in population:
                fit, _, _ = self.calculate_fitness(ind)
                pop_fitness.append((fit, ind))
            
            # 排序
            pop_fitness.sort(key=lambda x: x[0])
            
            # 记录最佳
            if pop_fitness[0][0] < best_fitness:
                best_fitness = pop_fitness[0][0]
                best_sol = pop_fitness[0][1]
                # 解析最佳得分成分
                _, best_mr, best_lb = self.calculate_fitness(best_sol)
                print(f"Gen {gen}: Best Fit={best_fitness:.2f} (MaxRes={best_mr:.1f}, LoadVar={best_lb:.1f})")
            
            # 选择 (保留前 50%)
            survivors = [x[1] for x in pop_fitness[:len(population)//2]]
            
            # 繁衍 (简单的变异填充)
            new_pop = survivors[:]
            while len(new_pop) < len(population):
                parent = random.choice(survivors)
                child = self.mutate(parent)
                new_pop.append(child)
            
            population = new_pop
            
        return best_sol

# ==========================================
# 主程序逻辑
# ==========================================
if __name__ == "__main__":
    # 1. 构建矿井巷道网络
    mineB = MineNetworkBuilder()
    try:
        mineB.load_data('real_map_data_one.json')
    except FileNotFoundError:
        print("警告: 找不到 json 文件，请确保路径正确。")
        # 这里为了防止代码直接报错退出，可以加一个退出或模拟数据
        # sys.exit()

    print(f"矿井图加载完毕: {len(mineB.G.nodes)} 节点, {len(mineB.G.edges)} 边")

    # 2. 获取可部署边 & 离散化候选点
    # 注意：这里假设 tu1 里的 get_deployable_edges 返回 (u, v, attr)
    deployable_edges = [(u, v) for u, v, _ in mineB.get_deployable_edges()]
    candidates = discretize_deployable_edges(mineB.G, deployable_edges, interval=20.0)
    print(f"生成候选点数量: {len(candidates)}")

    # 3. 准备目标点和权重
    # 假设 'critical' 属性标记了关键目标
    target_nodes = [n for n, d in mineB.G.nodes(data=True) if d.get('critical', False)]
    
    # 如果没有设定 critical，为了演示，随机选 20 个点作为目标
    if not target_nodes:
        print("未发现 critical 标记，随机选择 20 个目标点用于演示。")
        all_nodes = list(mineB.G.nodes())
        target_nodes = random.sample(all_nodes, min(20, len(all_nodes)))
    
    # 模拟权重 (重要程度 weights)
    # 实际项目中应从 node data 读取，例如 d.get('weight', 1.0)
    weights = np.array([random.uniform(1.0, 5.0) for _ in target_nodes])

    # 4. 计算距离矩阵 (Crucial for G-HCES)
    # 计算所有候选点到所有目标点的距离
    dist_to_targets = compute_all_node_to_targets_distances(mineB.G, target_nodes, weight='length')
    D_matrix, valid_candidates = compute_distance_matrix(
        dist_to_targets, candidates, target_nodes, weight='length'
    )
    
    if D_matrix.size == 0:
        print("错误: 无法计算距离矩阵，请检查图连通性。")
    else:
        # 5. 运行 G-HCES 求解器
        # 参数设定
        N_STATIONS = 3      # 设定的选址数量 (你模型里的 n)
        MAX_R = 3000        # 电量/距离限制
        LAMBDA = 0.01       # 负载均衡权重
        
        print("\n=== 初始化 G-HCES 求解器 ===")
        solver = G_HCES_Solver(
            dist_matrix=D_matrix,
            target_weights=weights,
            candidates=valid_candidates,
            target_nodes=target_nodes,
            num_centers=N_STATIONS,
            max_R=MAX_R,
            lambda_balance=LAMBDA
        )
        
        best_solution = solver.run(generations=30)
        
        # 6. 输出结果
        print("\n=== 最终选址方案 ===")
        for i, segment in enumerate(best_solution):
            c_idx = segment['center_idx']
            cand = valid_candidates[c_idx]
            targets_idx = segment['targets']
            
            # 计算该站点的负载
            load = sum(weights[t] for t in targets_idx)
            
            print(f"站点 {i+1}:")
            print(f"  - 部署位置: 边 {cand['edge']}, 偏移 {cand['offset']:.1f}m")
            print(f"  - 覆盖目标数: {len(targets_idx)}")
            print(f"  - 累计负载(权重和): {load:.2f}")
            print(f"  - 负责目标ID: {[target_nodes[t] for t in targets_idx]}")