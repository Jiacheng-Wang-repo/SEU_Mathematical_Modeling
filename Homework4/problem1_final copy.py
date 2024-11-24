import pandas as pd
import numpy as np
from gurobipy import *
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data(data_path):
    """加载数据，并跳过第一行"""
    return pd.read_excel(data_path, skiprows=1)

def extract_coordinates(data, num_points=613):
    """提取坐标和属性值"""
    x = data.iloc[:num_points, 1].to_numpy()
    y = data.iloc[:num_points, 2].to_numpy()
    z = data.iloc[:num_points, 3].to_numpy()
    attributes = data.iloc[:num_points, 4]
    return x, y, z, attributes

def find_VH(attributes):
    """识别垂直和水平校正点"""
    V = np.where(attributes == 1)[0].tolist()  # 找到垂直校正点的索引
    H = np.where(attributes == 0)[0].tolist()  # 找到水平校正点的索引
    return V, H

def save_to_txt(V, H, txt_path):
    """将校正点保存到文本文件中"""
    with open(txt_path, 'w') as file:
        file.write('\n'.join(map(str, V + H)))  # 将 V 和 H 连接并写入文件

def calculate_distances(coords):
    """计算各点之间的欧几里得距离"""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return np.sqrt((x.reshape(-1, 1) - x) ** 2 + (y.reshape(-1, 1) - y) ** 2 + (z.reshape(-1, 1) - z) ** 2)

def pruning_distance(distances, V, H, alpha1, alpha2, beta1, beta2, theta, delta):
    """根据校正条件修剪距离矩阵"""
    dict_dist = {(i, j): distances[i, j] for i in range(distances.shape[0]) for j in range(distances.shape[1])}

    # 步骤 1: 修剪到垂直校正点的边
    for i in range(distances.shape[0]):
        for j in V:
            if distances[i, j] > min(alpha1, alpha2) / delta:
                dict_dist[i, j] = 0

    # 步骤 2: 修剪到水平校正点的边
    for i in range(distances.shape[0]):
        for j in H:
            if distances[i, j] > min(beta1, beta2) / delta:
                dict_dist[i, j] = 0

    # 步骤 3: 修剪距离终点 B 超过限制的边
    for i in range(distances.shape[0]):
        if distances[i, -1] > theta / delta:
            dict_dist[i, -1] = 0

    return dict_dist

def filter_edges(distances, V, H, max_distance=10):
    """删除距离大于10km的同类顶点之间的有向边"""
    num_points = distances.shape[0]
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if distances[i, j] > max_distance:  # 如果距离大于10km
                if (i in V and j in V) or (i in H and j in H):  # 如果两点属于同类
                    distances[i, j] = 0  # 删除边
                    distances[j, i] = 0  # 删除反向边
    return distances

def remove_reverse_edges(flight_path):
    """删除方向与前进方向相反的有向边"""
    filtered_path = [flight_path[0]]  # 保留起点
    for i in range(1, len(flight_path)):
        # 检查前一条边和当前边是否方向相反
        if flight_path[i][0] != filtered_path[-1][1]:
            filtered_path.append(flight_path[i])
    return filtered_path

def save_to_excel(dict_dist, num_points, excel_path):
    """将修剪后的邻接矩阵保存到 Excel 文件中"""
    dict_dist_save = pd.DataFrame(np.zeros((num_points, num_points)))
    for (i, j), d in dict_dist.items():
        dict_dist_save.iat[i, j] = d  # 将距离填入数据框
    dict_dist_save.to_excel(excel_path, index=False)  # 保存为 Excel 文件

def plot_3d_points(x, y, z, size, attributes):
    """绘制3D散点图，展示起点A，终点B，垂直与水平校正点"""
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(x[0], y[0], z[0], c='r', marker='o')  # 绘制起点A
    ax.scatter(x[size-1], y[size-1], z[size-1], c='y', marker='o')  # 绘制终点B

    x1, y1, z1 = [], [], []  # 垂直校正点
    x2, y2, z2 = [], [], []  # 水平校正点
    for i in range(1, size - 1):
        if attributes[i] == 1:
            x1.append(x[i])
            y1.append(y[i])
            z1.append(z[i])
        else:
            x2.append(x[i])
            y2.append(y[i])
            z2.append(z[i])

    # 绘制垂直与水平校正点
    ax.scatter(x1, y1, z1, s=6, c='g', marker='+', linewidth=1, label="垂直校正点")
    ax.scatter(x2, y2, z2, s=6, c='b', marker='^', linewidth=0, label="水平校正点")

    # 绘制A点到B点的直线
    ax.plot([x[0], x[size-1]], [y[0], y[size-1]], [z[0], z[size-1]], c='k', linestyle='--', linewidth=1, label="AB直线")

    plt.legend()
    plt.show()


def solve_shortest_path(distances, size, alpha1, alpha2, beta1, beta2, theta, delta):
    """通过Gurobi求解最短路径问题"""
    Adjacency_Matrix = {}
    for i in range(size):
        for j in range(size):
            Adjacency_Matrix[i, j] = distances[i, j]
    Adjacency_Matrix = tupledict(Adjacency_Matrix)

    # 创建Gurobi模型
    model = Model()
    x = model.addVars(size, size, vtype=GRB.BINARY, name='x')

    # 添加进出约束
    sum_xij = [0] * size
    sum_xji = [0] * size
    for i in range(size):
        for j in range(size):
            sum_xij[i] += x[i, j]
    for j in range(size):
        for i in range(size):
            sum_xji[i] += x[j, i]

    for i in range(size):
        if i == 0:
            model.addConstr(sum_xij[i] == 1)
            model.addConstr(sum_xji[i] == 0)
        elif 0 < i < size - 1:
            model.addConstr(sum_xij[i] == sum_xji[i])
        else:
            model.addConstr(sum_xij[i] == 0)
            model.addConstr(sum_xji[i] == 1)

    # 设置目标函数
    model.setObjectiveN(quicksum(distances[i][j] * x[i, j] for i in range(size) for j in range(size)), index=0, priority=1, name='Obj1')
    model.setObjectiveN(quicksum(x[i, j] for i in range(size) for j in range(size)), index=1, priority=2, name='Obj2')

    model.optimize()

    # 提取优化后的航迹路径
    flight_path = []
    if model.status == GRB.OPTIMAL:
        for var in model.getVars():
            if var.x == 1:
                i, j = map(int, re.findall(r"\d+", var.varName))
                flight_path.append((i, j))

    # 删除方向相反的有向边
    flight_path = remove_reverse_edges(flight_path)

    return flight_path

def plot_optimized_path(x, y, z, flight_path):
    """绘制优化后的航迹路径"""
    node = [flight_path[0][0]]
    for i in range(len(flight_path)):
        node.append(flight_path[i][1])

    flight_x, flight_y, flight_z = [], [], []
    for i in node:
        flight_x.append(x[i])
        flight_y.append(y[i])
        flight_z.append(z[i])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(flight_x, flight_y, flight_z, c='r', marker='o', label='航迹路径')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    num_points = 613
    alpha1, alpha2 = 25, 15
    beta1, beta2 = 20, 25
    theta = 30
    delta = 0.001

    data_path = 'Homework2/data/data1.xlsx'
    txt_path = 'Homework4/output/calibration_point.txt'
    excel_path = 'Homework4/output/dict_dist.xlsx'

    data = load_data(data_path)
    x, y, z, attributes = extract_coordinates(data, num_points)
    coords = np.column_stack((x, y, z))  # 合并坐标为一个数组

    # 1. 找到校正点并保存
    V, H = find_VH(attributes)
    save_to_txt(V, H, txt_path)

    # 2. 计算距离并修剪矩阵
    distances = calculate_distances(coords)
    distances = filter_edges(distances, V, H)
    dict_dist = pruning_distance(distances, V, H, alpha1, alpha2, beta1, beta2, theta, delta)
    save_to_excel(dict_dist, num_points, excel_path)

    # 3. 求解最短路径并绘制优化后的航迹
    flight_path = solve_shortest_path(distances, num_points, alpha1, alpha2, beta1, beta2, theta, delta)
    plot_optimized_path(x, y, z, flight_path)

