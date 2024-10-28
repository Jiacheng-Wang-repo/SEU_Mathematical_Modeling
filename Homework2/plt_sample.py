import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# 读取数据并跳过第一行
data_path = 'Homework2/data/data1.xlsx'
data = pd.read_excel(data_path, skiprows=1)  # 跳过第一行说明

# 提取B、C、D列数据（x、y、z坐标），并获取613个点
x = data.iloc[:613, 1].to_numpy()
y = data.iloc[:613, 2].to_numpy()
z = data.iloc[:613, 3].to_numpy()
attributes = data.iloc[:613, 4]
# print(attributes)

# 根据属性值设置颜色和标记
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

# 绘制不同颜色和标记的散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(x)):
    ax.scatter(x[i], y[i], z[i], color=colors[i], marker=markers[i])

# 在A点到B点绘制黑色连线
ax.plot([x[0], x[612]], [y[0], y[612]], [z[0], z[612]], color='black', linewidth=1)

# 定义校正点列表，并在这些点之间绘制红色连线
nodes = [0, 503, 294, 91, 607, 540, 250, 340, 277, 612]
for i in range(len(nodes) - 1):
    ax.plot([x[nodes[i]], x[nodes[i + 1]]],
            [y[nodes[i]], y[nodes[i + 1]]],
            [z[nodes[i]], z[nodes[i + 1]]], color='red', linewidth=2)

# 添加x、y、z轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 保存图像
fig.savefig('Homework2/data/FigData1.jpg')

# 计算各点之间的欧氏距离
distances = np.sqrt((x.reshape(-1, 1) - x) ** 2 + (y.reshape(-1, 1) - y) ** 2 + (z.reshape(-1, 1) - z) ** 2)

# 创建DataFrame并添加编号列
distances_df = pd.DataFrame(distances)
distances_df.insert(0, '', range(613))  # 添加第一列

# 将距离矩阵导出到Excel文件
output_path = 'Homework2/data/data2.xlsx'
wb = Workbook()
ws = wb.active

# 将DataFrame写入工作表
for r in dataframe_to_rows(distances_df, index=False, header=True):
    ws.append(r)

# 保存Excel文件
wb.save(output_path)



