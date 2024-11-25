# -*- coding: utf-8 -*-


"""
----------------------------------------------------------------
0.前期预处理工作
----------------------------------------------------------------
"""

# 分别导入相关模块
import numpy as np
import pandas as pd
from gurobipy import *
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取data1.xlsx文件B、C、D、E列的数据，跳过前两行，不设表头，赋给data
data = pd.read_excel("data1.xlsx", usecols="B:E", skiprows=2, header=None)
# 将data转换为numpy.array数组
data = np.array(data)
size = np.size(data, 0)

# 获取所有点的坐标及属性标记物
x0 = data[:, 0]
y0 = data[:, 1]
z0 = data[:, 2]
marker = data[:, 3]

# 初始化dist矩阵，置零
dist = np.zeros(shape=(size, size))

# 计算任意两点i，j之间的欧式距离，并赋值给距离矩阵dist[i][j]
for i in range(0, size):
    for j in range(0, size):
        # 欧式距离 = 两点间坐标差的平方和的平方根
        # 即 dist = sqrt((xi-xj)**2+(yi-yj)**2+(zi-zj)**2)
        dist[i][j] = np.sqrt(np.sum((data[i, 0:3] - data[j, 0:3]) ** 2))

# 初始化figure，绘制三维图，赋值给ax
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 设置绘图字体及大小
plt.rc('font', size=8)
plt.rc('font', family='SimHei')

# 绘制A点散点，颜色为红色，散点的标记为圆点
ax.scatter(x0[0], y0[0], z0[0], c='r', marker='o')

# 绘制B点散点，颜色为黄色，散点的标记为圆点
ax.scatter(x0[size - 1], y0[size - 1], z0[size - 1], c='y', marker='o')
x1 = [];
y1 = [];
z1 = []
x2 = [];
y2 = [];
z2 = []

# 根据属性标记物绘制校正点散点图
for i in range(1, 612):
    if marker[i] == 1:
        x1.append(x0[i])
        y1.append(y0[i])
        z1.append(z0[i])
    else:
        x2.append(x0[i])
        y2.append(y0[i])
        z2.append(z0[i])

# 绘制垂直校正点，颜色为绿色，散点的标记为+
ax.scatter(x1, y1, z1, s=6, c='g', marker='+', linewidth=1, label="v")

# 绘制水平校正点，颜色为蓝色，散点的标记为^
ax.scatter(x2, y2, z2, s=6, c='b', marker='^', linewidth=0, label="h")

# 绘制A点到B点的直线，颜色为黑色，直线宽度为1
ax.plot([x0[0], x0[size - 1]], [y0[0], y0[size - 1]], [z0[0], z0[size - 1]], c='k', linestyle='--', linewidth=1,
        label="AB")

"""
----------------------------------------------------------------
1.建立List格式的垂直校正点集合V和水平校正点集合H，保存在一个txt文件中
----------------------------------------------------------------
"""

# 建立垂直校正点集合V和水平校正点集合H
V = []
H = []

# 根据校正点属性将点分别赋予各校正点集合
for i in range(0, size):
    if marker[i] == 1:
        V.append(i)
    if marker[i] == 0:
        H.append(i)

# 将校正点集合先垂直再水平保存在一个txt文件中，文件名为calibration_point.txt
file = open('calibration_point.txt', 'w')
for i in range(len(V)):
    s = str(V[i]) + '\n'
    file.write(s)
for i in range(len(H)):
    s = str(H[i]) + '\n'
    file.write(s)
file.close()

"""
----------------------------------------------------------------
2.记距离矩阵元素为边长,利用校正条件减少最短路模型中有向边数量,并用python输出减少边之后
最短路模型的邻接矩阵，存放在excel文件中
----------------------------------------------------------------
"""

# 将距离矩阵dist转换为gurobi中的tupledict类型，用dict_dist.xlsx存放邻接矩阵
dict_dist = {}

for i in range(size):
    for j in range(size):
        dict_dist[i, j] = dist[i][j]

dict_dist = tupledict(dict_dist)

# 设置问题参数
a1 = 25
a2 = 15
b1 = 20
b2 = 25
theta = 30
delta = 0.001

# 根据校正条件分析，第一次减少最短路模型中有向边数量可以采用如下策略
# （1）以任意点为上一点i到某垂直校正点j不符合宽松垂直校正条件的边
#     即 dij*delta > min(a1,a2)
# （2）以任意点为上一点i到某水平校正点j不符合宽松水平校正条件的边
#     即 dij*delta > min(b1,b2)
# （3）以任意点为上一点i到终点B点不满足宽松距离条件小于𝜃/δ条件的边
#     即 dij*delta > theta

# （1）删除以任意点为上一点到某垂直校正点不符合宽松垂直校正条件的边
for i in range(0, size):
    for j in V:
        if dist[i][j] > min(a1, a2) / delta:
            dict_dist[i, j] = 0

# （2）删除以任意点为上一点到某水平校正点不符合宽松水平校正条件的边
for i in range(0, size):
    for j in H:
        if dist[i][j] > min(b1, b2) / delta:
            dict_dist[i, j] = 0

# （3）删除以任意点为上一点到终点B点不满足宽松距离小于𝜃/δ条件的边
for i in range(0, size - 1):
    if dist[i][size - 1] > theta / delta:
        dict_dist[i, size - 1] = 0

# 定义边集
edge = []
for i in range(size):
    for j in range(size):
        if dict_dist[i, j] != 0:
            edge.append((i, j))

print("第一次减边后的有向边数量：", len(edge))

# 输出减少边之后最短路模型的邻接矩阵，存放在excel文件'dict_dist.xlsx'中
dict_dist_Output = np.zeros(shape=(size, size))
for i in range(size):
    for j in range(size):
        dict_dist_Output[i][j] = dict_dist[i, j]

dict_dist_Output = pd.DataFrame(dict_dist_Output)
output = pd.ExcelWriter('dict_dist.xlsx')
dict_dist_Output.to_excel(output, "sheet1")
# output.to_excel(excel_path, index=False)

''' 
----------------------------------------------------------------
3.记hi表示飞行器到达i点时，校正前的水平偏差；vi表示i点校正前的垂直偏差；变量xij=1
表示有向边(i,j)在最短路上。采用“或”约束处理校正点类型，在校正点处如何构造校正前偏差
必须满足的约束；垂直（或水平）偏差校正后飞到下一个校正点j时偏差变量要满足的条件？
（提示：即通过对参数的分析和校正点类型，写出校正前、校正后偏差满足的约束）
----------------------------------------------------------------

根据上述条件，对各点类型进行分类后显然有如下约束：
for i in range(size):
    if i == 0:
        # 起点的垂直和水平误差为0
        v[i] == 0
        h[i] == 0
    elif 0 < i < size-1:
        # 校正点的误差约束条件
        if marker[i] == 1:
            # 垂直校正点前的误差约束条件
            v[i] <= a1
            h[i] <= a2
        else:
            # 水平校正点前的误差约束条件
            v[i] <= b1
            h[i] <= b2
    else:
        # 终点前的垂直和水平误差约束条件
        v[i] <= theta
        h[i] <= theta

'''

''' 
----------------------------------------------------------------
4.将起点A，终点B，垂直校正点和水平校正点看作顶点，构造边权为顶点间距离的有向网络。
为简化模型，可以根据给定参数和校正规则，删去距离大于10km的同类顶点之间的有向边；
删除方向与前进方向相反的有向边（提示：删除标准请自行设定）
----------------------------------------------------------------
'''

# 根据规划目标，航迹长度尽可能小，首先考虑删除方向与前进方向相反的有向边
# 前进方向为AB向量，若有向边向量与AB向量乘积为负或0，则该方向与AB方向相反或垂直，删除该边

# 定义AB向量
AB = [x0[size - 1] - x0[0], y0[size - 1] - y0[0], z0[size - 1] - z0[0]]

# 删除有向边与AB向量点乘积为非正值的边
for i in range(size):
    for j in range(size):
        if AB[0] * (x0[j] - x0[i]) + AB[1] * (y0[j] - y0[i]) + AB[2] * (z0[j] - z0[i]) <= 0:
            dict_dist[i, j] = 0

# 重定义边集并输出边数
edge = []
for i in range(size):
    for j in range(size):
        if dict_dist[i, j] != 0:
            edge.append((i, j))
print("第二次减边后的有向边数量：", len(edge))

# 根据规划目标，航迹长度尽可能小，其次考虑删除偏离核心路径范围的校正点
# 对核心路径范围定义为以AB为轴线，半径R=10km的圆柱体区间
# 需计算出各校正点到AB线的垂直距离，距离大于R的校正点删去

# 定义校正条件R，计算AB点直线距离dAB
R = 10000
dAB = np.sqrt(AB[0] ** 2 + AB[1] ** 2 + AB[2] ** 2)

# 建立偏离点集，并获取偏离点编号
Deviation_Point = []
for i in range(1, size - 1):
    # 利用三角形面积相等原则计算点到直线的距离，定义iA、iB向量，i到直线AB距离为r
    iA = np.array([x0[0] - x0[i], y0[0] - y0[i], z0[0] - z0[i]])
    iB = np.array([x0[size - 1] - x0[i], y0[size - 1] - y0[i], z0[size - 1] - z0[i]])
    r = np.sqrt(np.sum(np.cross(iA, iB) ** 2)) / dAB
    if r > R:
        Deviation_Point.append(i)

# 删除任意一点在偏离点集当中的有向边
for i in range(size):
    for j in range(size):
        if i in Deviation_Point or j in Deviation_Point:
            dict_dist[i, j] = 0

# 重定义边集并输出边数
edge = []
for i in range(size):
    for j in range(size):
        if dict_dist[i, j] != 0:
            edge.append((i, j))
print("第三次减边后的有向边数量：", len(edge))

dict_dist_Output = np.zeros(shape=(size, size))
for i in range(size):
    for j in range(size):
        dict_dist_Output[i][j] = dict_dist[i, j]

''' 
----------------------------------------------------------------
5.在步骤1中已经定义垂直校正点集合V，水平校正点集合H
  定义起点A到终点B的校正航迹点集合为P
  定义决策变量x[i,j]为0-1整数变量，若ij边位于校正航迹中，则x[i,j]=1，可以推出：
  （1）若i是A点，j是任意点，则 sum x[i,j] = 1 ， sum x[j,i] = 0
  （2）若i是B点，j是任意点，则 sum x[i,j] = 0 ， sum x[j,i] = 1
  （3）若i∈P且不为A、B点，j是任意点，则 sum x[i,j] = 1 ， sum x[j,i] = 1

  根据规划目标，可以得出模型为最短路0-1混合整数线性规划模型：
  a.航迹路径总长度尽可能小    
    即  min sum d[i,j]*x[i,j]
  b.航迹中经过校正点的次数尽可能少 => 航迹中的航线数尽可能少     
    即  min sum x[i,j]

  采用Gurobi求解器建立多目标混合优化模型并添加约束条件。
----------------------------------------------------------------
'''

# 建立模型
model = Model()

# ----------------------------------------------------------------
# 添加变量：x[i,j]
x = model.addVars(size, size, vtype=GRB.BINARY, name='x')
# 添加变量：在顶点i处校正前的垂直误差v[i],水平误差h[i]
v = model.addVars(size, vtype=GRB.CONTINUOUS, name='v')
h = model.addVars(size, vtype=GRB.CONTINUOUS, name='h')

# ----------------------------------------------------------------
# 添加约束1：邻接矩阵删除的零边不可能位于航迹上，对这部分决策变量赋零约束
for i in range(size):
    for j in range(size):
        if dict_dist[i, j] == 0:
            model.addConstr(x[i, j] == 0)

# ----------------------------------------------------------------
# 添加约束2：航迹点i的进出决策变量求和条件
# sum_xij[i]\sum_xji[i]对应于i为某航迹点，j为任意点
sum_xij = [0] * size
sum_xji = [0] * size

for i in range(size):
    for j in range(size):
        sum_xij[i] = sum_xij[i] + x[i, j]

for j in range(size):
    for i in range(size):
        sum_xji[i] = sum_xji[i] + x[j, i]

for i in range(size):
    if i == 0:
        # 若i是A点，j是任意点，则 sum x[i,j] = 1 ， sum x[j,i] = 0
        model.addConstr(sum_xij[i] == 1)
        model.addConstr(sum_xji[i] == 0)
    elif 0 < i < size - 1:
        # 若i∈P且不为A、B点，j是任意点，则 sum x[i,j] = 1 ， sum x[j,i] = 1
        model.addConstr(sum_xij[i] == sum_xji[i])
    else:
        # 若i是B点，j是任意点，则 sum x[i,j] = 0 ， sum x[j,i] = 1
        model.addConstr(sum_xij[i] == 0)
        model.addConstr(sum_xji[i] == 1)

# ----------------------------------------------------------------
# 添加约束3：航迹线上相邻校正点应满足的误差关系
for i in range(0, size - 1):
    for j in range(1, size):
        if i == 0:
            model.addConstr(dist[i][j] * delta - v[j] <= (1 - x[i, j]) * 10000)
            model.addConstr(dist[i][j] * delta - h[j] <= (1 - x[i, j]) * 10000)
        else:
            model.addConstr((1 - marker[i]) * v[i] + dist[i][j] * delta - v[j] <= (1 - x[i, j]) * 10000)
            model.addConstr(marker[i] * h[i] + dist[i][j] * delta - h[j] <= (1 - x[i, j]) * 10000)

# ----------------------------------------------------------------
# 添加约束4：航迹点i的垂直、水平偏差验证条件
for i in range(size):
    if i == 0:
        # 起点的垂直和水平误差为0
        model.addConstr(v[i] == 0)
        model.addConstr(h[i] == 0)
    elif 0 < i < size - 1:
        # 中间点的误差约束条件
        if marker[i] == 1:
            # 垂直校正点前的误差约束条件
            model.addConstr(v[i] <= a1)
            model.addConstr(h[i] <= a2)
        else:
            # 水平校正点前的误差约束条件
            model.addConstr(v[i] <= b1)
            model.addConstr(h[i] <= b2)
    else:
        # 终点的垂直和水平误差约束条件
        model.addConstr(v[i] <= theta)
        model.addConstr(h[i] <= theta)

# ----------------------------------------------------------------
# 添加目标函数
# 分别设置两个目标函数，多目标决策优先级(整数值)值越大优先级越高

#  1.目标函数1：航迹路径总长度之和，设置多目标决策优先级1
#    obj1 = Adjacency_Matrix.prod(x)
model.setObjectiveN(dict_dist.prod(x), index=0, priority=1, name='obj1')

#  2.目标函数2：航迹中的总航线数之和，设置多目标决策优先级2
#    obj2 = quicksum(x[i,j] for i,j in range(size))
model.setObjectiveN(quicksum(x[i, j] for i in range(size) for j in range(size)), index=1, priority=2, name='obj2')

# ----------------------------------------------------------------
# 更新变量空间
model.update()

''' 
----------------------------------------------------------------
6.采用Gurobi求解器，求解所建立的航迹规划模型
----------------------------------------------------------------
'''
# 显示求解过程
model.Params.LogToConsole = True

# 执行最优化
model.optimize()

# model.computeIIS()
# model.write("model.ilp")

# 判断并输出模型是否取得最优解
if model.status == gurobipy.GRB.Status.OPTIMAL:
    print("模型已取得最优解")
else:
    print("模型未取得最优解")

# 查看并输出多目标规划模型的目标函数值
model.setParam(gurobipy.GRB.Param.ObjNumber, 0)
print(f"航迹路径总长度之和：{model.ObjNVal}")
model.setParam(gurobipy.GRB.Param.ObjNumber, 1)
print(f"航迹中的总航线数之和：{model.ObjNVal}")

# 查看并输出航迹优化结果
for var in model.getVars():
    if var.x == 1:
        print(f"{var.varName}:{var.x}")

''' 
----------------------------------------------------------------
7.绘制优化后的航迹
----------------------------------------------------------------
'''

# 定义航迹集
Flight_Path = []

# 从变量名中获取航迹
for var in model.getVars():
    if var.x == 1:
        ij = re.findall(r"\d+", var.varName)
        i = int(ij[0])
        j = int(ij[1])
        Flight_Path.append((i, j))

    # 定义航迹点集
node = []

# 将航迹点按航线顺序依次存入航迹点集
for i in range(len(Flight_Path) + 1):
    if i == 0:
        node.append(Flight_Path[i][0])
        node.append(Flight_Path[i][1])
    else:
        for j in range(len(Flight_Path)):
            if Flight_Path[j][0] == node[i]:
                node.append(Flight_Path[j][1])

# 初始化node点坐标空序列
x3 = [];
y3 = [];
z3 = []

# 循环获取node点坐标并添加至序列中
for i in node:
    x3.append(x0[i])
    y3.append(y0[i])
    z3.append(z0[i])

# 绘制node中相邻两点之间的直线，颜色为红色，直线宽度为2.
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
ax.plot(x3, y3, z3, c="r", marker="o", label="flight_road")
ax.legend()
plt.show()
# ax.plot(x3, y3, z3, c='r', linewidth=1, label="优化后航迹")

# 绘制图例
# plt.legend(loc='upper left')

# 保存绘制的图像
# plt.savefig("Flight_Path_Fig2.jpg", dpi=500)

# 显示图像
# plt.show()