from scipy.optimize import linprog

# 目标函数系数
c = [1, 2]

# 不等式约束的系数矩阵 A_ub 和右侧向量 b_ub
A_ub = [
    [-1, 2],   # 对应约束 -x1 + 2x2 <= 4
    [-1, -1]   # 对应约束 -x1 - x2 <= -1 (等价于 x1 + x2 >= 1)
]
b_ub = [4, -1]

# 变量边界
bounds = [(0, None), (0, None)]  # x1 >= 0, x2 >= 0

# 调用 linprog 求解
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# 输出结果
if result.success:
    print("最优解为：", result.x)
    print("最小化的目标函数值为：", result.fun)
else:
    print("求解失败：", result.message)


