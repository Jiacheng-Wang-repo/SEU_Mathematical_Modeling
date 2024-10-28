def count_cute_fish(n, cute_levels):
    for i in range(n):
        num = 0
        for t in cute_levels[:i]:
            if t< cute_levels[i]:
                num += 1
        print(num,end=' ')

# 输入处理
n = int(input())  # 第一行表示鱼的数量
cute_levels = list(map(int, input().split()))  # 第二行输入鱼的可爱度列表

# 计算并输出
count_cute_fish(n, cute_levels)

