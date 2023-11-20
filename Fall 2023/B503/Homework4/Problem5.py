def calculate_maximum_reward(n, M):
    # Initialize the matrix to store results
    OPT = [[0] * (n + 1) for _ in range(n + 1)]

    # Set boundary conditions
    for i in range(1, n + 1):
        OPT[i][i] = 0
        if i < n:
            OPT[i][i + 1] = M[i][i + 1]

    print(OPT)
    print()

    # Fill the matrix using the recurrence relationship
    for l in range(2, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            max_reward = 0
            for k in range(i, j):
                max_reward = max(
                    max_reward, OPT[i][k - 1] + OPT[k + 1][j - 1] + M[k][j]
                )
            OPT[i][j] = max(max_reward, OPT[i][j - 1])
            print(OPT)

    # The result is stored in OPT[1][n]
    # print(OPT)
    return OPT[1][n]


def max_total_reward(n, M):
    # Initialize a 3D matrix to store results
    OPT = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(n + 1)]

    # Fill the matrix using the recurrence relationship
    for l in range(2, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            for k in range(i, j):
                OPT[i][j][k] = max(
                    OPT[i][j][k],
                    OPT[i][k - 1][k - 1] + OPT[k + 1][j][k + 1] + M[k][k + 1],
                )

            OPT[i][j][j] = max(OPT[i][j][j], OPT[i][j - 1][j - 1] + M[j - 1][j])

    # The result is stored in OPT[1][n][1]
    print(OPT)
    return OPT[1][n][1]


# Example usage:
n = 5
# Assuming M is a matrix of rewards, here's an example:
M = [
    [0, 2, 3, 4, 5, 6],
    [0, 0, 2, 3, 4, 5],
    [0, 0, 0, 1, 2, 3],
    [0, 0, 0, 0, 2, 4],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
]
# M = [[1, 5], [5, -1]]

result = calculate_maximum_reward(n, M)
print("Maximum total reward:", result)

# result = max_total_reward(n, M)
# print("Maximum total reward:", result)
