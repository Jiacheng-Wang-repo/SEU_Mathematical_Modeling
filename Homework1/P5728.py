def count_rivals(students):
    count = 0
    n = len(students)
    
    for i in range(n):
        for j in range(i + 1, n):
            # 分别计算语文、数学、英语的成绩差
            chinese_diff = abs(students[i][0] - students[j][0])
            math_diff = abs(students[i][1] - students[j][1])
            english_diff = abs(students[i][2] - students[j][2])
            total_diff = abs(sum(students[i]) - sum(students[j]))
            
            # 判断是否符合旗鼓相当的条件
            if chinese_diff <= 5 and math_diff <= 5 and english_diff <= 5 and total_diff <= 10:
                count += 1
                
    return count

# 读取输入
N = int(input())  # 学生人数
students = []

for _ in range(N):
    students.append(list(map(int, input().split())))

# 输出符合条件的对手对数
print(count_rivals(students))
