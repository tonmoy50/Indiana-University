def maxIncSubarr(arr, n):
    OPT = [0] * n
    OPT[0] = 1
    for i in range(1, n):
        if arr[i] > arr[i - 1]:
            OPT[i] = OPT[i - 1] + 1
        else:
            OPT[i] = 1

    OPT_reversed = [0] * n
    OPT_reversed[n - 1] = 1
    for i in range(n - 2, -1, -1):
        if arr[i] < arr[i + 1]:
            OPT_reversed[i] = OPT_reversed[i + 1] + 1
        else:
            OPT_reversed[i] = 1

    max_subsequence = max(OPT)
    for i in range(1, n - 1):
        if arr[i - 1] < arr[i + 1]:
            max_subsequence = max(max_subsequence, OPT[i - 1] + OPT_reversed[i + 1])

    return max_subsequence


if __name__ == "__main__":
    arr = [1, 2, 5, 3, 4]
    arr = [8, 5, 1, 1, 4, 10, 1, 9, 7, 2]
    arr = []
    n = len(arr)

    print(maxIncSubarr(arr, n))
