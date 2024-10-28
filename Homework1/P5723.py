def is_prime(n):
    """判断是否为质数"""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def fill_prime_bag(L):
    primes = []
    current_sum = 0
    num = 2

    while current_sum + num <= L:
        if is_prime(num):
            primes.append(num)
            current_sum += num
        num += 1

    # 按题目格式输出
    for prime in primes:
        print(prime)
    print(len(primes))

# 输入整数 L
L = int(input())
fill_prime_bag(L)
