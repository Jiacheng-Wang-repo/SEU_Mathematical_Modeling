import numpy as np
import pandas as pd
from gurobipy import *
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_data(file_path):
    data = pd.read_excel(file_path, usecols="B:E", skiprows=2, header=None)
    return np.array(data)


def calculate_distances(data):
    size = data.shape[0]
    dist = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            dist[i, j] = np.linalg.norm(data[i, 0:3] - data[j, 0:3])
    return dist


def plot_points(ax, x0, y0, z0, marker, size):
    ax.scatter(x0[0], y0[0], z0[0], c='r', marker='o', label="A点")
    ax.scatter(x0[size - 1], y0[size - 1], z0[size - 1], c='y', marker='o', label="B点")

    x1, y1, z1 = [], [], []
    x2, y2, z2 = [], [], []

    for i in range(1, size):
        if marker[i] == 1:
            x1.append(x0[i])
            y1.append(y0[i])
            z1.append(z0[i])
        else:
            x2.append(x0[i])
            y2.append(y0[i])
            z2.append(z0[i])

    ax.scatter(x1, y1, z1, s=6, c='g', marker='+', linewidth=1, label="垂直校正点")
    ax.scatter(x2, y2, z2, s=6, c='b', marker='^', linewidth=0, label="水平校正点")
    ax.plot([x0[0], x0[size - 1]], [y0[0], y0[size - 1]], [z0[0], z0[size - 1]], c='k', linestyle='--', linewidth=1,
            label="AB线")
    ax.legend()


def save_calibration_points(file_name, V, H):
    with open(file_name, 'w') as file:
        for v in V:
            file.write(f"{v}\n")
        for h in H:
            file.write(f"{h}\n")


def filter_edges(size, dist, V, H, a1, a2, b1, b2, theta, delta):
    dict_dist = {(i, j): dist[i, j] for i in range(size) for j in range(size)}

    for i in range(size):
        for j in V:
            if dist[i][j] > min(a1, a2) / delta:
                dict_dist[i, j] = 0

    for i in range(size):
        for j in H:
            if dist[i][j] > min(b1, b2) / delta:
                dict_dist[i, j] = 0

    for i in range(size - 1):
        if dist[i][size - 1] > theta / delta:
            dict_dist[i, size - 1] = 0

    return dict_dist


def remove_invalid_edges(size, x0, y0, z0, dict_dist):
    AB = [x0[size - 1] - x0[0], y0[size - 1] - y0[0], z0[size - 1] - z0[0]]

    for i in range(size):
        for j in range(size):
            if AB[0] * (x0[j] - x0[i]) + AB[1] * (y0[j] - y0[i]) + AB[2] * (z0[j] - z0[i]) <= 0:
                dict_dist[i, j] = 0


def remove_deviation_edges(size, x0, y0, z0, dict_dist, R, dAB):
    Deviation_Point = []

    for i in range(1, size - 1):
        iA = np.array([x0[0] - x0[i], y0[0] - y0[i], z0[0] - z0[i]])
        iB = np.array([x0[size - 1] - x0[i], y0[size - 1] - y0[i], z0[size - 1] - z0[i]])
        r = np.sqrt(np.sum(np.cross(iA, iB) ** 2)) / dAB
        if r > R:
            Deviation_Point.append(i)

    for i in range(size):
        for j in range(size):
            if i in Deviation_Point or j in Deviation_Point:
                dict_dist[i, j] = 0


def create_model(size, dict_dist, delta, a1, a2, b1, b2, theta, marker, dist):
    model = Model()
    x = model.addVars(size, size, vtype=GRB.BINARY, name='x')
    v = model.addVars(size, vtype=GRB.CONTINUOUS, name='v')
    h = model.addVars(size, vtype=GRB.CONTINUOUS, name='h')

    for i in range(size):
        for j in range(size):
            if dict_dist[i, j] == 0:
                model.addConstr(x[i, j] == 0)

    sum_xij = {i: quicksum(x[i, j] for j in range(size)) for i in range(size)}
    sum_xji = {i: quicksum(x[j, i] for j in range(size)) for i in range(size)}

    model.addConstr(sum_xij[0] == 1)
    model.addConstr(sum_xji[0] == 0)
    for i in range(1, size - 1):
        model.addConstr(sum_xij[i] == sum_xji[i])
    model.addConstr(sum_xij[size - 1] == 0)
    model.addConstr(sum_xji[size - 1] == 1)

    for i in range(0, size - 1):
        for j in range(1, size):
            if i == 0:
                model.addConstr(dist[i][j] * delta - v[j] <= (1 - x[i, j]) * 10000)
                model.addConstr(dist[i][j] * delta - h[j] <= (1 - x[i, j]) * 10000)
            else:
                model.addConstr((1 - marker[i]) * v[i] + dist[i][j] * delta - v[j] <= (1 - x[i, j]) * 10000)
                model.addConstr(marker[i] * h[i] + dist[i][j] * delta - h[j] <= (1 - x[i, j]) * 10000)

    for i in range(size):
        if i == 0:
            model.addConstr(v[i] == 0)
            model.addConstr(h[i] == 0)
        elif 0 < i < size - 1:
            if marker[i] == 1:
                model.addConstr(v[i] <= a1)
                model.addConstr(h[i] <= a2)
            else:
                model.addConstr(v[i] <= b1)
                model.addConstr(h[i] <= b2)
        else:
            model.addConstr(v[i] <= theta)
            model.addConstr(h[i] <= theta)

    model.setObjectiveN(tupledict(dict_dist).prod(x), index=0, priority=1, name='obj1')
    model.setObjectiveN(quicksum(x[i, j] for i in range(size) for j in range(size)), index=1, priority=2, name='obj2')

    model.update()
    return model, x


def extract_flight_path(model, x):
    flight_path = []
    for var in model.getVars():
        if var.x == 1:
            i, j = map(int, re.findall(r"\d+", var.varName))
            flight_path.append((i, j))
    return flight_path


def plot_flight_path(ax, x0, y0, z0, flight_path):
    x3, y3, z3 = [], [], []
    node = []

    # 将航迹点按航线顺序依次存入航迹点集
    for i in range(len(flight_path) + 1):
        if i == 0:
            node.append(flight_path[i][0])
            node.append(flight_path[i][1])
        else:
            for j in range(len(flight_path)):
                if flight_path[j][0] == node[i]:
                    node.append(flight_path[j][1])
    # 循环获取node点坐标并添加至序列中
    for i in node:
        x3.append(x0[i])
        y3.append(y0[i])
        z3.append(z0[i])

    ax.plot(x3, y3, z3, c="r", marker="o", label="flight_road")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    file_path = "data1.xlsx"
    data = read_data(file_path)
    size = data.shape[0]
    x0, y0, z0, marker = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    dist = calculate_distances(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.rc('font', size=8)
    plt.rc('font', family='SimHei')
    plot_points(ax, x0, y0, z0, marker, size)

    V, H = np.where(marker == 1)[0], np.where(marker == 0)[0]
    save_calibration_points("calibration_point.txt", V, H)

    a1, a2, b1, b2, theta, delta = 25, 15, 20, 25, 30, 0.001
    dict_dist = filter_edges(size, dist, V, H, a1, a2, b1, b2, theta, delta)

    AB = [x0[size - 1] - x0[0], y0[size - 1] - y0[0], z0[size - 1] - z0[0]]
    remove_invalid_edges(size, x0, y0, z0, dict_dist)

    R = 10000
    dAB = np.linalg.norm(AB)
    remove_deviation_edges(size, x0, y0, z0, dict_dist, R, dAB)

    model, x = create_model(size, dict_dist, delta, a1, a2, b1, b2, theta, marker, dist)

    model.Params.LogToConsole = True
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        print("模型已取得最优解")
    else:
        print("模型未取得最优解")

    model.setParam(GRB.Param.ObjNumber, 0)
    print(f"航迹路径总长度之和：{model.ObjNVal}")
    model.setParam(GRB.Param.ObjNumber, 1)
    print(f"航迹中的总航线数之和：{model.ObjNVal}")

    flight_path = extract_flight_path(model, x)
    plot_flight_path(ax, x0, y0, z0, flight_path)

