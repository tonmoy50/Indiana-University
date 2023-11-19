def max_total_reward(M):
    n = len(M)
    opt = [[0] * n for _ in range(n)]

    # Base cases
    for i in range(n - 1):
        opt[i][i + 1] = M[i][i + 1]
    print(*opt)

    # Dynamic programming
    for l in range(2, n):
        for i in range(n - l):
            j = i + l
            max_reward = 0

            # Find the optimal k within the interval
            for k in range(i + 1, j):
                reward = opt[i][k - 1] + opt[k + 1][j - 1] + M[k][j]
                max_reward = max(max_reward, reward)

            opt[i][j] = max(max_reward, opt[i][j - 1])

    return opt[0][n - 1]

# Example usage:
M = [
    [0, 5, 2, 3],
    [5, 0, 6, 4],
    [2, 6, 0, 1],
    [3, 4, 1, 0]
]
result = max_total_reward(M)
print("Maximum total reward:", result)
