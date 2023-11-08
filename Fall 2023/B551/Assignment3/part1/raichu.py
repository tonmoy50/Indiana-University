#
# raichu.py : Play the game of Raichu
#
# Frangil Ramirez Koteich(fraramir) - Nilambar Halder Tonmoy(nhalder) - Paul Coen(pcoen)

import sys
import time
from queue import PriorityQueue as PQ
from dataclasses import dataclass, field
from typing import Any


# create class used to wrap data before pushing it into queue
# class adapted from: https://docs.python.org/3/library/queue.html?highlight=priorityqueue#queue.PriorityQueue
@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)

    def __getitem__(self, item):
        return self.item


# end of adapted code


def board_to_string(board, N):
    return "\n".join(board[i : i + N] for i in range(0, len(board), N))


def h(board, player):
    return 0


def valid_pos(x, y):
    if x >= 0 and x < N and y >= 0 and y < N:
        return True
    else:
        return False


def make_raichu_if_possible(board, x, y, player, raichu_made=False):
    opponents_end = N - 1 if player == "w" else 0
    if x == opponents_end:
        board[(x * N) + y] = "@" if player == "w" else "$"
        raichu_made = True

    return board, raichu_made


def find_pichus_move(board, N, pichus_curr_position, player):
    board = list(board)
    pichu = "w" if player == "w" else "b"
    opponent_pichu = "b" if player == "w" else "w"

    possible_boards = list()
    x, y = pichus_curr_position // N, pichus_curr_position % N

    all_possible_moves = [(1, -1), (1, 1)] if player == "w" else [(-1, -1), (-1, 1)]
    for i, j in all_possible_moves:
        if valid_pos(x + i, y + j):
            if board[((x + i) * N) + (y + j)] == ".":
                temp_board = board.copy()
                temp_board[((x + i) * N) + (y + j)] = pichu
                temp_board, raichu_made = make_raichu_if_possible(
                    temp_board, (x + i) * N, y + j, player
                )
                temp_board[pichus_curr_position] = "."
                possible_boards.append(
                    ["".join([i for i in temp_board]), -4 if raichu_made else 0]
                )
                del temp_board
            elif (
                board[((x + i) * N) + (y + j)] == opponent_pichu
                and valid_pos(x + (i * 2), y + (j * 2))
                and board[((x + (i * 2)) * N) + (y + (j * 2))] == "."
            ):
                temp_board = board.copy()
                temp_board[pichus_curr_position] = "."
                temp_board[((x + i) * N) + (y + j)] = "."
                temp_board[((x + (i * 2)) * N) + (y + (j * 2))] = pichu
                temp_board, raichu_made = make_raichu_if_possible(
                    temp_board, (x + i) * N, y + j, player
                )
                possible_boards.append(
                    ["".join([i for i in temp_board]), -5 if raichu_made else -1]
                )
                del temp_board

    return possible_boards


def move_raichu(board, player, j, N):
    yield (board, 20)
    # if player = white raichu:
    if player == "w":
        backward = 0
        forward = 0
        backward_eaten = 0
        forward_eaten = 0
        i = 1
        coord = 0
        coord1 = 0

        H_backward = 0
        H_forward = 0
        H_backward_eaten = 0
        H_forward_eaten = 0
        H_coord = 0
        H_coord1 = 0
        even_remainder = 20
        odd_remainder = 20

        D_right = 0
        D_left = 0
        D_right_eaten = 0
        D_left_eaten = 0
        D_coord = 0
        D_coord1 = 0
        D_even_remainder = 20
        D_odd_remainder = 20

        DD_right = 0
        DD_left = 0
        DD_right_eaten = 0
        DD_left_eaten = 0
        DD_coord = 0
        DD_coord1 = 0
        DD_even_remainder = 20
        DD_odd_remainder = 20

        while i < N:
            # up vertical move
            try:
                if (j - (i * N) >= 0) and (board[j - (i * N)] in "@wW"):
                    backward = 1
                if (j - (i * N) >= 0) and (board[j - (i * N)] in "$bB"):
                    backward_eaten += 1
                    coord = j - (i * N)
                    if board[j - (i * N)] == "b":
                        rewardU = -1
                    if board[j - (i * N)] == "B":
                        rewardU = -2
                    if board[j - (i * N)] == "$":
                        rewardU = -3
                if (
                    (j - (i * N) >= 0)
                    and (board[j - (i * N)] == ".")
                    and (backward == 0)
                    and (backward_eaten < 2)
                ):
                    # print("Move corresponding to i: %3d j: %3d back_eaten: %3d" % (i, j, backward_eaten))
                    if backward_eaten == 0:
                        yield (
                            board[0 : j - (i * N)]
                            + "@"
                            + board[j - (i * N) + 1 : j]
                            + "."
                            + board[j + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0 : j - (i * N)]
                            + "@"
                            + board[j - (i * N) + 1 : coord]
                            + "."
                            + board[coord + 1 : j]
                            + "."
                            + board[j + 1 :],
                            rewardU,
                        )
            except:
                None

            # down vertical move
            try:
                if board[j + (i * N)] in "@wW":
                    forward = 1
                if board[j + (i * N)] in "$bB":
                    forward_eaten += 1
                    coord1 = j + (i * N)
                    if board[j + (i * N)] == "b":
                        rewardD = -1
                    if board[j + (i * N)] == "B":
                        rewardD = -2
                    if board[j + (i * N)] == "$":
                        rewardD = -3
                if (
                    (board[j + (i * N)] == ".")
                    and (forward == 0)
                    and (forward_eaten < 2)
                ):
                    # print("Move corresponding to i: %3d j: %3d for_eaten: %3d" % (i, j, forward_eaten))
                    if forward_eaten == 0:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : j + (i * N)]
                            + "@"
                            + board[j + (i * N) + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : coord1]
                            + "."
                            + board[coord1 + 1 : j + (i * N)]
                            + "@"
                            + board[j + (i * N) + 1 :],
                            rewardD,
                        )
            except:
                None

            # left horizontal move
            try:
                if (j % N) == 0:
                    even_remainder = 0
                if (j - i >= 0) and (board[j - i] in "@wW"):
                    H_backward = 1
                if (j - i >= 0) and (board[j - i] in "$bB"):
                    H_backward_eaten += 1
                    H_coord = j - i
                    if board[j - i] == "b":
                        rewardL = -1
                    if board[j - i] == "B":
                        rewardL = -2
                    if board[j - i] == "$":
                        rewardL = -3
                if (
                    (even_remainder != 0)
                    and (j - i >= 0)
                    and (H_backward == 0)
                    and (H_backward_eaten < 2)
                    and (board[j - i] == ".")
                ):
                    if H_backward_eaten == 0:
                        yield (
                            board[0 : j - i]
                            + "@"
                            + board[j - i + 1 : j]
                            + "."
                            + board[j + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0 : j - i]
                            + "@"
                            + board[j - i + 1 : H_coord]
                            + "."
                            + board[H_coord + 1 : j]
                            + "."
                            + board[j + 1 :],
                            rewardL,
                        )
                if ((j - i) % N) == 0:
                    even_remainder = 0
            except:
                None

            # right horizontal move
            try:
                if (j % N) == (N - 1):
                    odd_remainder = 0
                if board[j + i] in "@wW":
                    H_forward = 1
                if board[j + i] in "$bB":
                    H_forward_eaten += 1
                    H_coord1 = j + i
                    if board[j + i] == "b":
                        rewardR = -1
                    if board[j + i] == "B":
                        rewardR = -2
                    if board[j + i] == "$":
                        rewardR = -3
                if (
                    (odd_remainder != 0)
                    and (H_forward == 0)
                    and (H_forward_eaten < 2)
                    and (board[j + i] == ".")
                ):
                    if H_forward_eaten == 0:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : j + i]
                            + "@"
                            + board[j + i + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : H_coord1]
                            + "."
                            + board[H_coord1 + 1 : j + i]
                            + "@"
                            + board[j + i + 1 :],
                            rewardR,
                        )
                if ((j + i) % N) == (N - 1):
                    odd_remainder = 0
            except:
                None

            # left upper diagonal move
            try:
                if (j % N) == 0:
                    D_even_remainder = 0
                if ((j - (i * (N + 1))) >= 0) and (board[j - (i * (N + 1))] in "@wW"):
                    D_left = 1
                if (j - (i * (N + 1)) >= 0) and (board[j - (i * (N + 1))] in "$bB"):
                    D_left_eaten += 1
                    D_coord = j - (i * (N + 1))
                    if board[j - (i * (N + 1))] == "b":
                        rewardLU = -1
                    if board[j - (i * (N + 1))] == "B":
                        rewardLU = -2
                    if board[j - (i * (N + 1))] == "$":
                        rewardLU = -3

                if (
                    (D_even_remainder != 0)
                    and (j - (i * (N + 1)) >= 0)
                    and (D_left == 0)
                    and (D_left_eaten < 2)
                    and (board[j - (i * (N + 1))] == ".")
                ):
                    # print("LU diagonal move")
                    if D_left_eaten == 0:
                        yield (
                            board[0 : j - (i * (N + 1))]
                            + "@"
                            + board[j - (i * (N + 1)) + 1 : j]
                            + "."
                            + board[j + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0 : j - (i * (N + 1))]
                            + "@"
                            + board[j - (i * (N + 1)) + 1 : D_coord]
                            + "."
                            + board[D_coord + 1 : j]
                            + "."
                            + board[j + 1 :],
                            rewardLU,
                        )
                if ((j - (i * (N + 1))) % N) == 0:
                    D_even_remainder = 0

            except:
                None

            # right upper diagonal move
            try:
                if (j % N) == (N - 1):
                    D_odd_remainder = 0
                if ((j - (i * (N - 1))) >= 0) and (board[j - (i * (N - 1))] in "@wW"):
                    D_right = 1
                if (j - (i * (N - 1)) >= 0) and (board[j - (i * (N - 1))] in "$bB"):
                    D_right_eaten += 1
                    D_coord1 = j - (i * (N - 1))
                    if board[j - (i * (N - 1))] == "b":
                        rewardRU = -1
                    if board[j - (i * (N - 1))] == "B":
                        rewardRU = -2
                    if board[j - (i * (N - 1))] == "$":
                        rewardRU = -3

                if (
                    (D_odd_remainder != 0)
                    and (j - (i * (N - 1)) >= 0)
                    and (D_right == 0)
                    and (D_right_eaten < 2)
                    and (board[j - (i * (N - 1))] == ".")
                ):
                    # print("RU diagonal move")
                    if D_right_eaten == 0:
                        yield (
                            board[0 : j - (i * (N - 1))]
                            + "@"
                            + board[j - (i * (N - 1)) + 1 : j]
                            + "."
                            + board[j + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0 : j - (i * (N - 1))]
                            + "@"
                            + board[j - (i * (N - 1)) + 1 : D_coord1]
                            + "."
                            + board[D_coord1 + 1 : j]
                            + "."
                            + board[j + 1 :],
                            rewardRU,
                        )
                if ((j - (i * (N - 1))) % N) == (N - 1):
                    D_odd_remainder = 0

            except:
                None

            # left lower diagonal move
            try:
                if (j % N) == 0:
                    DD_even_remainder = 0
                if board[j + (i * (N - 1))] in "@wW":
                    DD_left = 1
                if board[j + (i * (N - 1))] in "$bB":
                    DD_left_eaten += 1
                    DD_coord = j + (i * (N - 1))
                    if board[j + (i * (N - 1))] == "b":
                        rewardLL = -1
                    if board[j + (i * (N - 1))] == "B":
                        rewardLL = -2
                    if board[j + (i * (N - 1))] == "$":
                        rewardLL = -3

                if (
                    (DD_even_remainder != 0)
                    and (DD_left == 0)
                    and (DD_left_eaten < 2)
                    and (board[j + (i * (N - 1))] == ".")
                ):
                    # print("LL diagonal move")
                    if DD_left_eaten == 0:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : j + (i * (N - 1))]
                            + "@"
                            + board[j + (i * (N - 1)) + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : DD_coord]
                            + "."
                            + board[DD_coord + 1 : j + (i * (N - 1))]
                            + "@"
                            + board[j + (i * (N - 1)) + 1 :],
                            rewardLL,
                        )
                if ((j + (i * (N - 1))) % N) == 0:
                    DD_even_remainder = 0

            except:
                None

            # right lower diagonal move
            try:
                if (j % N) == N - 1:
                    DD_odd_remainder = 0
                if board[j + (i * (N + 1))] in "@wW":
                    DD_right = 1
                if board[j + (i * (N + 1))] in "$bB":
                    DD_right_eaten += 1
                    DD_coord1 = j + (i * (N + 1))
                    if board[j + (i * (N + 1))] == "b":
                        rewardRL = -1
                    if board[j + (i * (N + 1))] == "B":
                        rewardRL = -2
                    if board[j + (i * (N + 1))] == "$":
                        rewardRL = -3

                if (
                    (DD_odd_remainder != 0)
                    and (DD_right == 0)
                    and (DD_right_eaten < 2)
                    and (board[j + (i * (N + 1))] == ".")
                ):
                    # print("RL diagonal move")
                    if DD_right_eaten == 0:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : j + (i * (N + 1))]
                            + "@"
                            + board[j + (i * (N + 1)) + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : DD_coord1]
                            + "."
                            + board[DD_coord1 + 1 : j + (i * (N + 1))]
                            + "@"
                            + board[j + (i * (N + 1)) + 1 :],
                            rewardRL,
                        )
                if ((j + (i * (N + 1))) % N) == (N - 1):
                    DD_odd_remainder = 0

            except:
                None

            i += 1
    # if player = black raichu:
    else:
        backward = 0
        forward = 0
        backward_eaten = 0
        forward_eaten = 0
        i = 1
        coord = 0
        coord1 = 0

        H_backward = 0
        H_forward = 0
        H_backward_eaten = 0
        H_forward_eaten = 0
        H_coord = 0
        H_coord1 = 0
        even_remainder = 20
        odd_remainder = 20

        D_right = 0
        D_left = 0
        D_right_eaten = 0
        D_left_eaten = 0
        D_coord = 0
        D_coord1 = 0
        D_even_remainder = 20
        D_odd_remainder = 20

        DD_right = 0
        DD_left = 0
        DD_right_eaten = 0
        DD_left_eaten = 0
        DD_coord = 0
        DD_coord1 = 0
        DD_even_remainder = 20
        DD_odd_remainder = 20

        while i < N:
            # up vertical move
            try:
                if (j - (i * N) >= 0) and (board[j - (i * N)] in "$bB"):
                    backward = 1
                if (j - (i * N) >= 0) and (board[j - (i * N)] in "wW@"):
                    backward_eaten += 1
                    coord = j - (i * N)
                    if board[j - (i * N)] == "w":
                        rewardU = -1
                    if board[j - (i * N)] == "W":
                        rewardU = -2
                    if board[j - (i * N)] == "@":
                        rewardU = -3
                if (
                    (j - (i * N) >= 0)
                    and (board[j - (i * N)] == ".")
                    and (backward == 0)
                    and (backward_eaten < 2)
                ):
                    # print("Move corresponding to i: %3d j: %3d back_eaten: %3d" % (i, j, backward_eaten))
                    if backward_eaten == 0:
                        yield (
                            board[0 : j - (i * N)]
                            + "$"
                            + board[j - (i * N) + 1 : j]
                            + "."
                            + board[j + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0 : j - (i * N)]
                            + "$"
                            + board[j - (i * N) + 1 : coord]
                            + "."
                            + board[coord + 1 : j]
                            + "."
                            + board[j + 1 :],
                            rewardU,
                        )
            except:
                None

            # down vertical move
            try:
                if board[j + (i * N)] in "$bB":
                    forward = 1
                if board[j + (i * N)] in "wW@":
                    forward_eaten += 1
                    coord1 = j + (i * N)
                    if board[j + (i * N)] == "w":
                        rewardD = -1
                    if board[j + (i * N)] == "W":
                        rewardD = -2
                    if board[j + (i * N)] == "@":
                        rewardD = -3
                if (
                    (board[j + (i * N)] == ".")
                    and (forward == 0)
                    and (forward_eaten < 2)
                ):
                    # print("Move corresponding to i: %3d j: %3d for_eaten: %3d" % (i, j, forward_eaten))
                    if forward_eaten == 0:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : j + (i * N)]
                            + "$"
                            + board[j + (i * N) + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : coord1]
                            + "."
                            + board[coord1 + 1 : j + (i * N)]
                            + "$"
                            + board[j + (i * N) + 1 :],
                            rewardD,
                        )
            except:
                None

            # left horizontal move
            try:
                if (j % N) == 0:
                    even_remainder = 0
                if (j - i >= 0) and (board[j - i] in "$bB"):
                    H_backward = 1
                if (j - i >= 0) and (board[j - i] in "wW@"):
                    H_backward_eaten += 1
                    H_coord = j - i
                    if board[j - i] == "w":
                        rewardL = -1
                    if board[j - i] == "W":
                        rewardL = -2
                    if board[j - i] == "@":
                        rewardL = -3
                if (
                    (even_remainder != 0)
                    and (j - i >= 0)
                    and (H_backward == 0)
                    and (H_backward_eaten < 2)
                    and (board[j - i] == ".")
                ):
                    if H_backward_eaten == 0:
                        yield (
                            board[0 : j - i]
                            + "$"
                            + board[j - i + 1 : j]
                            + "."
                            + board[j + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0 : j - i]
                            + "$"
                            + board[j - i + 1 : H_coord]
                            + "."
                            + board[H_coord + 1 : j]
                            + "."
                            + board[j + 1 :],
                            rewardL,
                        )
                if ((j - i) % N) == 0:
                    even_remainder = 0
            except:
                None

            # right horizontal move
            try:
                if (j % N) == (N - 1):
                    odd_remainder = 0
                if board[j + i] in "$bB":
                    H_forward = 1
                if board[j + i] in "wW@":
                    H_forward_eaten += 1
                    H_coord1 = j + i
                    if board[j + i] == "w":
                        rewardR = -1
                    if board[j + i] == "W":
                        rewardR = -2
                    if board[j + i] == "@":
                        rewardR = -3
                if (
                    (odd_remainder != 0)
                    and (H_forward == 0)
                    and (H_forward_eaten < 2)
                    and (board[j + i] == ".")
                ):
                    if H_forward_eaten == 0:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : j + i]
                            + "$"
                            + board[j + i + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : H_coord1]
                            + "."
                            + board[H_coord1 + 1 : j + i]
                            + "$"
                            + board[j + i + 1 :],
                            rewardR,
                        )
                if ((j + i) % N) == (N - 1):
                    odd_remainder = 0
            except:
                None

            # left upper diagonal move
            try:
                if (j % N) == 0:
                    D_even_remainder = 0
                if ((j - (i * (N + 1))) >= 0) and (board[j - (i * (N + 1))] in "$bB"):
                    D_left = 1
                if (j - (i * (N + 1)) >= 0) and (board[j - (i * (N + 1))] in "wW@"):
                    D_left_eaten += 1
                    D_coord = j - (i * (N + 1))
                    if board[j - (i * (N + 1))] == "w":
                        rewardLU = -1
                    if board[j - (i * (N + 1))] == "W":
                        rewardLU = -2
                    if board[j - (i * (N + 1))] == "@":
                        rewardLU = -3

                if (
                    (D_even_remainder != 0)
                    and (j - (i * (N + 1)) >= 0)
                    and (D_left == 0)
                    and (D_left_eaten < 2)
                    and (board[j - (i * (N + 1))] == ".")
                ):
                    # print("LU diagonal move")
                    if D_left_eaten == 0:
                        yield (
                            board[0 : j - (i * (N + 1))]
                            + "$"
                            + board[j - (i * (N + 1)) + 1 : j]
                            + "."
                            + board[j + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0 : j - (i * (N + 1))]
                            + "$"
                            + board[j - (i * (N + 1)) + 1 : D_coord]
                            + "."
                            + board[D_coord + 1 : j]
                            + "."
                            + board[j + 1 :],
                            rewardLU,
                        )
                if ((j - (i * (N + 1))) % N) == 0:
                    D_even_remainder = 0

            except:
                None

            # right upper diagonal move
            try:
                if (j % N) == (N - 1):
                    D_odd_remainder = 0
                if ((j - (i * (N - 1))) >= 0) and (board[j - (i * (N - 1))] in "$bB"):
                    D_right = 1
                if (j - (i * (N - 1)) >= 0) and (board[j - (i * (N - 1))] in "wW@"):
                    D_right_eaten += 1
                    D_coord1 = j - (i * (N - 1))
                    if board[j - (i * (N - 1))] == "w":
                        rewardRU = -1
                    if board[j - (i * (N - 1))] == "W":
                        rewardRU = -2
                    if board[j - (i * (N - 1))] == "@":
                        rewardRU = -3

                if (
                    (D_odd_remainder != 0)
                    and (j - (i * (N - 1)) >= 0)
                    and (D_right == 0)
                    and (D_right_eaten < 2)
                    and (board[j - (i * (N - 1))] == ".")
                ):
                    # print("RU diagonal move")
                    if D_right_eaten == 0:
                        yield (
                            board[0 : j - (i * (N - 1))]
                            + "$"
                            + board[j - (i * (N - 1)) + 1 : j]
                            + "."
                            + board[j + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0 : j - (i * (N - 1))]
                            + "$"
                            + board[j - (i * (N - 1)) + 1 : D_coord1]
                            + "."
                            + board[D_coord1 + 1 : j]
                            + "."
                            + board[j + 1 :],
                            rewardRU,
                        )
                if ((j - (i * (N - 1))) % N) == (N - 1):
                    D_odd_remainder = 0

            except:
                None

            # left lower diagonal move
            try:
                if (j % N) == 0:
                    DD_even_remainder = 0
                if board[j + (i * (N - 1))] in "$bB":
                    DD_left = 1
                if board[j + (i * (N - 1))] in "wW@":
                    DD_left_eaten += 1
                    DD_coord = j + (i * (N - 1))
                    if board[j + (i * (N - 1))] == "w":
                        rewardLL = -1
                    if board[j + (i * (N - 1))] == "W":
                        rewardLL = -2
                    if board[j + (i * (N - 1))] == "@":
                        rewardLL = -3

                if (
                    (DD_even_remainder != 0)
                    and (DD_left == 0)
                    and (DD_left_eaten < 2)
                    and (board[j + (i * (N - 1))] == ".")
                ):
                    # print("LL diagonal move")
                    if DD_left_eaten == 0:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : j + (i * (N - 1))]
                            + "$"
                            + board[j + (i * (N - 1)) + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : DD_coord]
                            + "."
                            + board[DD_coord + 1 : j + (i * (N - 1))]
                            + "$"
                            + board[j + (i * (N - 1)) + 1 :],
                            rewardLL,
                        )
                if ((j + (i * (N - 1))) % N) == 0:
                    DD_even_remainder = 0

            except:
                None

            # right lower diagonal move
            try:
                if (j % N) == (N - 1):
                    DD_odd_remainder = 0
                if board[j + (i * (N + 1))] in "$bB":
                    DD_right = 1
                if board[j + (i * (N + 1))] in "wW@":
                    DD_right_eaten += 1
                    DD_coord1 = j + (i * (N + 1))
                    if board[j + (i * (N + 1))] == "w":
                        rewardRL = -1
                    if board[j + (i * (N + 1))] == "W":
                        rewardRL = -2
                    if board[j + (i * (N + 1))] == "@":
                        rewardRL = -3

                if (
                    (DD_odd_remainder != 0)
                    and (DD_right == 0)
                    and (DD_right_eaten < 2)
                    and (board[j + (i * (N + 1))] == ".")
                ):
                    # print("RL diagonal move")
                    if DD_right_eaten == 0:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : j + (i * (N + 1))]
                            + "$"
                            + board[j + (i * (N + 1)) + 1 :],
                            0,
                        )
                    else:
                        yield (
                            board[0:j]
                            + "."
                            + board[j + 1 : DD_coord1]
                            + "."
                            + board[DD_coord1 + 1 : j + (i * (N + 1))]
                            + "$"
                            + board[j + (i * (N + 1)) + 1 :],
                            rewardRL,
                        )
                if ((j + (i * (N + 1))) % N) == (N - 1):
                    DD_odd_remainder = 0

            except:
                None

            i += 1


def find_pikachus_move(board, N, pikachus_curr_position, player):
    board = list(board)
    pikachu = "W" if player == "w" else "B"
    opponent_pikachu = "B" if player == "w" else "W"
    opponent_pichu = "b" if player == "w" else "w"

    possible_boards = list()
    x, y = pikachus_curr_position // N, pikachus_curr_position % N

    def add_space_to_jumped_board(board, x, y):
        temp_board = list(board[0])
        temp_board[(x * N) + y] = "."
        return ["".join([i for i in temp_board]), board[1]]

    def move_pikachu(
        board,
        x,
        y,
        score=0,
    ):
        temp_board = board.copy()
        temp_board[(x * N) + y] = pikachu
        temp_board[pikachus_curr_position] = "."
        temp_board, raichu_made = make_raichu_if_possible(temp_board, x, y, player)
        possible_boards.append(
            ["".join([i for i in temp_board]), -4 + score if raichu_made else score]
        )
        del temp_board

    all_possible_moves = (
        [(0, -1), (0, 1), (1, 0)] if player == "w" else [(0, -1), (0, 1), (-1, 0)]
    )
    for i, j in all_possible_moves:
        if valid_pos(x + (i * 1), y + (j * 1)):
            if board[((x + (i * 1)) * N) + (y + (j * 1))] == ".":
                move_pikachu(
                    board, x + (i * 1), y + (j * 1), score=-0.5 if i != 0 else 0
                )
                if valid_pos(x + (i * 2), y + (j * 2)):
                    if board[((x + (i * 2)) * N) + (y + (j * 2))] == ".":
                        move_pikachu(
                            board, x + (i * 2), y + (j * 2), score=-0.5 if i != 0 else 0
                        )
                    elif valid_pos(x + (i * 2), y + (j * 2)) and valid_pos(
                        x + (i * 3), y + (j * 3)
                    ):
                        if (
                            board[((x + (i * 2)) * N) + (y + (j * 2))]
                            in [opponent_pikachu, opponent_pichu]
                            and board[((x + (i * 3)) * N) + (y + (j * 3))] == "."
                        ):
                            move_pikachu(
                                board,
                                x + (i * 3),
                                y + (j * 3),
                                score=-1
                                if board[((x + (i * 2)) * N) + (y + (j * 2))]
                                == opponent_pichu
                                else -2,
                            )
                            possible_boards[-1] = add_space_to_jumped_board(
                                possible_boards[-1], x + (i * 2), y + (j * 2)
                            )
            elif (
                board[((x + (i * 1)) * N) + (y + (j * 1))]
                in [
                    opponent_pichu,
                    opponent_pikachu,
                ]
                and valid_pos(x + (i * 2), y + (j * 2))
                and board[((x + (i * 2)) * N) + (y + (j * 2))] == "."
            ):
                move_pikachu(
                    board,
                    x + (i * 2),
                    y + (j * 2),
                    score=-1
                    if board[((x + (i * 1)) * N) + (y + (j * 1))] == opponent_pichu
                    else -2,
                )
                possible_boards[-1] = add_space_to_jumped_board(
                    possible_boards[-1], x + (i * 1), y + (j * 1)
                )

    return possible_boards


def get_opponents_move(board, N, player, timelimit):
    opponent_player = "b" if player == "w" else "w"
    opponent_pichu = "b" if player == "w" else "w"
    opponent_pikachu = "B" if player == "w" else "W"
    opponent_raichu = "$" if player == "w" else "@"
    player_pieces = "wW$" if player == "w" else "bB@"
    opponents_total_pieces = sum(
        [1 for i in range(len(board)) if board[i] in player_pieces]
    )
    all_opponents_move = list()

    for i in range(len(board)):
        if board[i] == opponent_pichu:
            for move in find_pichus_move(board, N, i, opponent_player):
                move[1] = 0
                for j in range(len(move[0])):
                    if move[0][j] in player_pieces:
                        move[1] += 1
                move[1] -= opponents_total_pieces
                yield move
        elif board[i] == opponent_pikachu:
            for move in find_pikachus_move(board, N, i, opponent_player):
                move[1] = 0
                for j in range(len(move[0])):
                    if move[0][j] in player_pieces:
                        move[1] += 1
                move[1] -= opponents_total_pieces
                yield move
        elif board[i] == opponent_raichu:
            for move in move_raichu(board, opponent_player, i, N):
                new_move = [move[0], move[1]]
                new_move[1] = 0
                for j in range(len(new_move[0])):
                    if new_move[0][j] in player_pieces:
                        new_move[1] += 1
                new_move[1] -= opponents_total_pieces
                yield new_move


def find_best_move(board, N, player, timelimit):
    player_pichu = "w" if player == "w" else "b"
    player_pikachu = "W" if player == "w" else "B"
    player_raichu = "@" if player == "w" else "$"

    for i in range(len(board)):
        if board[i] == player_pichu:
            for move in find_pichus_move(board, N, i, player):
                yield move
                try:
                    move[1] -= max(
                        [
                            opponents_move[1]
                            for opponents_move in get_opponents_move(
                                move[0], N, player, timelimit
                            )
                        ]
                    )
                except ValueError:
                    move[1] -= 0
                yield move
        elif board[i] == player_pikachu:
            for move in find_pikachus_move(board, N, i, player):
                yield move
                try:
                    move[1] -= max(
                        [
                            opponents_move[1]
                            for opponents_move in get_opponents_move(
                                move[0], N, player, timelimit
                            )
                        ]
                    )
                except ValueError:
                    move[1] -= 0
                yield move
        elif board[i] == player_raichu:
            try:
                for move in move_raichu(board, player, i, N):
                    yield move
                    new_move = [move[0], move[1]]
                    new_move[1] -= max(
                        [
                            opponents_move[1]
                            for opponents_move in get_opponents_move(
                                move[0], N, player, timelimit
                            )
                        ]
                    )
            except ValueError:
                yield move


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Usage: Raichu.py N player board timelimit")

    (_, N, player, board, timelimit) = sys.argv
    N = int(N)
    timelimit = int(timelimit)
    if player not in "wb":
        raise Exception("Invalid player.")

    if len(board) != N * N or 0 in [c in "wb.WB@$" for c in board]:
        raise Exception("Bad board string.")
    myPQ = PQ()
    print(
        "Searching for best move for "
        + player
        + " from board state: \n"
        + board_to_string(board, N)
    )
    print("Here's what I decided:")
    for new_board in find_best_move(board, N, player, timelimit):
        print(new_board[0])

        myPQ.put(PrioritizedItem(new_board[1], new_board[0]))
    best_move = myPQ.get()
    print(best_move[1])
