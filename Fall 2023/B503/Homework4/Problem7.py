def findMinimumDeletion(l, r, dp, s):
    if l > r:
        return 0
    if l == r:
        return 1
    if dp[l][r] != -1:
        return dp[l][r]

    steps = 1 + findMinimumDeletion(l + 1, r, dp, s)

    for i in range(l + 1, r + 1):
        if s[l] == s[i]:
            steps = min(
                steps,
                findMinimumDeletion(l + 1, i - 1, dp, s)
                + findMinimumDeletion(i, r, dp, s),
            )

    dp[l][r] = steps
    return steps


if __name__ == "__main__":
    s = "xyyzzqqqzzq"
    s = "abbaaaaaaaa"
    N = len(s)
    dp = [[-1 for i in range(N)] for j in range(N)]
    print(findMinimumDeletion(0, N - 1, dp, s))
