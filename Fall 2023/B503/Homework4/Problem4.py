# Hoy nai


def find_moves(A, current_position):
    for i in range(len(A)):
        if (i - current_position) % A[current_position] == 0 and A[i] > A[
            current_position
        ]:
            yield i


def determine_winner(A):
    winners = [""] * len(A)
    for i in range(len(A)):
        if len(list(find_moves(A, i))) % 2 == 0:
            winners[i] = "Joe"
        else:
            winners[i] = "Jane"

    return winners


if __name__ == "__main__":
    A = [6, 5, 7, 1, 4, 3]
    # A = [1, 2, 3, 4, 5, 6, 7, 9]
    print(determine_winner(A))
