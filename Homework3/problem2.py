from scipy.optimize import linprog

# 目标函数系数
c = [3, 5]  # 生产 A 和 B 果汁的成本分别是 3 元和 5 元

# 不等式约束的系数矩阵 A_ub 和右侧向量 b_ub
A_ub = [
    [-1, -1],  # 对应约束 -x1 - x2 <= -8 (等价于 x1 + x2 >= 8)
    [1, 0],    # 对应约束 x1 <= 6
    [0, -1]    # 对应约束 -x2 <= -3 (等价于 x2 >= 3)
]
b_ub = [-8, 6, -3]

# 变量边界
bounds = [(0, None), (0, None)]  # x1 >= 0, x2 >= 0

# 调用 linprog 求解
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# 输出结果
if result.success:
    print("最优解为：A 果汁产量 = {:.2f} 升, B 果汁产量 = {:.2f} 升".format(result.x[0], result.x[1]))
    print("最小化的生产成本为：{:.2f} 元".format(result.fun))
else:
    print("求解失败：", result.message)


