import numpy as np
import random
from copy import deepcopy
from tu1_25_12_22 import (
    MineNetworkBuilder,
    discretize_deployable_edges,
    compute_all_node_to_targets_distances,
    compute_distance_matrix
    )

# import sys
# import os
# script_dir = os.path.dirname(os.path.abspath(__file__))#获取当前脚本所在目录
# #print("脚本所在目录:", script_dir) 
# project_root = os.path.dirname(script_dir)#先回到项目根目录
# data_file = os.path.join(project_root, 'data','json', 'real_map_data_one.json')#再进入项目的data/json目录,如此可以不用手动写 '/' 实现拼接可跨平台使用
# #print("数据文件路径:", data_file)
# sys.path.insert(0, project_root)
# #mine_builder.load_data(data_file)  # ← 传入绝对路径！










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