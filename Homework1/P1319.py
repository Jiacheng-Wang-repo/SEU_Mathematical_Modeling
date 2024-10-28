def decompress_matrix(compressed_code):
    current_col = 0  # 当前列数，表示输出一行时已经输出了多少列
    N = compressed_code[0]  # 压缩码的第一位表示矩阵的大小 N x N
    compressed_index = 0  # 当前处理的压缩码索引
    total_elements = 0  # 当前已经输出的矩阵元素总数
    
    while total_elements < N * N:
        compressed_index += 1  
        consecutive_count = compressed_code[compressed_index] 
        
        # 输出连续的 0 或 1，直到数量用完
        while consecutive_count >= 1:
            # 如果当前列数达到了 N，意味着一行已经输出完了，需要换行
            if current_col == N:
                print()  # 打印换行符
                current_col = 0  # 重置当前列数为 0
            
            # 偶数索引（如 1, 3, 5, ...）对应 0，奇数索引（如 2, 4, 6, ...）对应 1
            if compressed_index % 2 == 1:
                print("0", end='')
            else:
                print("1", end='') 
            
            current_col += 1  
            total_elements += 1  
            consecutive_count -= 1  
   
    # 输出结束时，确保最后一行打印换行符
    print()

# 输入处理
compressed_code = list(map(int, input().split()))  # 读取压缩码并转换为整数列表
decompress_matrix(compressed_code)  # 调用解压缩函数
