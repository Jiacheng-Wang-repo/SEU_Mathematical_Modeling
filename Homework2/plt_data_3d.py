import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def load_data(data_path):
    """读取数据并跳过第一行"""
    return pd.read_excel(data_path, skiprows=1)

def extract_coordinates(data, num_points=613):
    """提取坐标和属性值"""
    x = data.iloc[:num_points, 1].to_numpy()
    y = data.iloc[:num_points, 2].to_numpy()
    z = data.iloc[:num_points, 3].to_numpy()
    attributes = data.iloc[:num_points, 4]
    return x, y, z, attributes

def set_colors_and_markers(attributes):
    """根据属性值设置颜色和标记"""
    colors = []
    markers = []
    for attribute in attributes:
        if attribute == "A 点":
            colors.append('red')
            markers.append('o')
        elif attribute == "B点":
            colors.append('yellow')
            markers.append('o')
        elif attribute == 0:
            colors.append('green')
            markers.append('+')
        elif attribute == 1:
            colors.append('blue')
            markers.append('*')
        else:
            raise NotImplementedError    
    return colors, markers

def plot_3d_scatter(x, y, z, colors, markers):
    """绘制3D散点图"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], color=colors[i], marker=markers[i])

    # 绘制黑色连线
    ax.plot([x[0], x[612]], [y[0], y[612]], [z[0], z[612]], color='black', linewidth=1)

    # 绘制校正点之间的红色连线
    nodes = [0, 503, 294, 91, 607, 540, 250, 340, 277, 612]
    for i in range(len(nodes) - 1):
        ax.plot([x[nodes[i]], x[nodes[i + 1]]],
                [y[nodes[i]], y[nodes[i + 1]]],
                [z[nodes[i]], z[nodes[i + 1]]], color='red', linewidth=2)

    # 添加轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    return fig

def save_figure(fig, file_path):
    """保存图像"""
    fig.savefig(file_path)

def calculate_distances(x, y, z):
    """计算各点之间的欧氏距离"""
    return np.sqrt((x.reshape(-1, 1) - x) ** 2 + (y.reshape(-1, 1) - y) ** 2 + (z.reshape(-1, 1) - z) ** 2)

def export_to_excel(distances, output_path):
    """将距离矩阵导出到Excel文件"""
    distances_df = pd.DataFrame(distances)
    distances_df.insert(0, '', range(len(distances)))  # 添加编号列
    
    wb = Workbook()
    ws = wb.active
    for r in dataframe_to_rows(distances_df, index=False, header=True):
        ws.append(r)
    
    wb.save(output_path)
   

if __name__ == "__main__":
    data_path = 'Homework2/data/data1.xlsx'
    output_path = 'Homework2/data/data2.xlsx'
    figure_path = 'Homework2/data/FigData1.jpg'

    data = load_data(data_path)
    x, y, z, attributes = extract_coordinates(data)
    colors, markers = set_colors_and_markers(attributes)
    
    fig = plot_3d_scatter(x, y, z, colors, markers)
    save_figure(fig, figure_path)

    distances = calculate_distances(x, y, z)
    export_to_excel(distances, output_path)

