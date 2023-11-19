import math


def calculate_grundy_numbers(n):
    grundy_numbers = [0] * (n + 1)

    for i in range(2, n + 1):
        factors = set()
        for j in range(1, int(math.sqrt(i)) + 1):
            if i % j == 0:
                factors.add(grundy_numbers[j])
                factors.add(grundy_numbers[i // j])

        mex = 0
        while mex in factors:
            mex += 1

        grundy_numbers[i] = mex

    return grundy_numbers


def who_wins(n):
    grundy_numbers = calculate_grundy_numbers(n)
    winners = ["Joe" if grundy_numbers[i] != 0 else "Jane" for i in range(1, n + 1)]
    return winners


# Example usage:
n = 5
winners = who_wins(n)
print("Winners:", winners)
