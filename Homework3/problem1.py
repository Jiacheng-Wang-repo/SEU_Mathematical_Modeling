import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from gurobipy import *

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
    return np.sqrt((x.reshape(-1, 1) - x) ** 2 + (y.reshape(-1, 1) - y) ** 2 + (z.reshape(-1, 1) - z) ** 2)

def pruning_distance(distances, V, H, alpha1, alpha2, beta1, beta2, theta, delta):
    """根据校正条件修剪距离矩阵"""
    dict_dist = {(i, j): distances[i, j] for i in range(distances.shape[0]) for j in range(distances.shape[1])}
    
    # 步骤 1: 修剪到垂直校正点的边
    for i in range(1, distances.shape[0] - 1):
        for j in V:
            if distances[i, j] > min(alpha1, alpha2) / delta:
                dict_dist[i, j] = 0

    # 步骤 2: 修剪到水平校正点的边
    for i in range(1, distances.shape[0] - 1):
        for j in H:
            if distances[i, j] > min(beta1, beta2) / delta:
                dict_dist[i, j] = 0

    # 步骤 3: 修剪距离终点 B 超过限制的边
    for i in range(distances.shape[0] - 1):
        if distances[i, -1] > theta / delta:
            dict_dist[i, -1] = 0

    # 收集剩余的边
    # edges = [(i, j) for (i, j), d in dict_dist.items() if d != 0]
    # print("修剪之后的边数:", len(edges))
    return dict_dist

def save_to_excel(dict_dist, num_points, excel_path):
    """将修剪后的邻接矩阵保存到 Excel 文件中"""
    dict_dist_save = pd.DataFrame(np.zeros((num_points, num_points)))
    for (i, j), d in dict_dist.items():
        dict_dist_save.iat[i, j] = d  # 将距离填入数据框
    dict_dist_save.to_excel(excel_path, index=False)  # 保存为 Excel 文件

if __name__ == "__main__":
    num_points = 613
    alpha1, alpha2 = 25, 15
    beta1, beta2 = 20, 25
    theta = 30
    delta = 0.001

    data_path = 'Homework2/data/data1.xlsx'
    txt_path = 'Homework3/output/calibration_point.txt'
    excel_path = 'Homework3/output/dict_dist.xlsx'

    data = load_data(data_path)
    x, y, z, attributes = extract_coordinates(data, num_points)
    coords = np.column_stack((x, y, z))  # 合并坐标为一个数组

    # 1.找到校正点并保存
    V, H = find_VH(attributes)
    save_to_txt(V, H, txt_path)

    # 2.计算距离并修剪矩阵
    distances = calculate_distances(coords)
    dict_dist = pruning_distance(distances, V, H, alpha1, alpha2, beta1, beta2, theta, delta)
    save_to_excel(dict_dist, num_points, excel_path)
