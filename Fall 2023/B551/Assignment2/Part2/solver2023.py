#!/usr/local/bin/python3
# solver2023.py : 2023 Sliding tile puzzle solver
#
# Code by: name IU ID
#
# Based on skeleton code by B551 Staff, Fall 2023
#

import sys
import numpy as np
import copy

ROWS = 5
COLS = 5


def printable_board(board):
    return [
        ("%3d ") * COLS % board[j : (j + COLS)] for j in range(0, ROWS * COLS, COLS)
    ]


def move_right(board, row):
    """Move the given row to one position right"""
    board[row] = board[row][-1:] + board[row][:-1]
    return board


def move_left(board, row):
    """Move the given row to one position left"""
    board[row] = board[row][1:] + board[row][:1]
    return board


def rotate_right(board, row, residual):
    board[row] = [board[row][0]] + [residual] + board[row][1:]
    residual = board[row].pop()
    return residual


def rotate_left(board, row, residual):
    board[row] = board[row][:-1] + [residual] + [board[row][-1]]
    residual = board[row].pop(0)
    return residual


def move_clockwise(board):
    """Move the outer ring clockwise"""
    board[0] = [board[1][0]] + board[0]
    residual = board[0].pop()
    board = transpose_board(board)
    residual = rotate_right(board, -1, residual)
    board = transpose_board(board)
    residual = rotate_left(board, -1, residual)
    board = transpose_board(board)
    residual = rotate_left(board, 0, residual)
    board = transpose_board(board)
    return board


def move_cclockwise(board):
    """Move the outer ring counter-clockwise"""
    board[0] = board[0] + [board[1][-1]]
    residual = board[0].pop(0)
    board = transpose_board(board)
    residual = rotate_right(board, 0, residual)
    board = transpose_board(board)
    residual = rotate_right(board, -1, residual)
    board = transpose_board(board)
    residual = rotate_left(board, -1, residual)
    board = transpose_board(board)
    return board


def transpose_board(board):
    """Transpose the board --> change row to column"""
    return [list(col) for col in zip(*board)]


# return a list of possible successor states
def successors(state):
    moves = {
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "U1",
        "U2",
        "U3",
        "U4",
        "U5",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "Oc",
        "Ic",
        "Occ",
        "Icc",
    }

    successive_boards = list()
    for move in moves:
        turn, row = move[0], move[1 : len(move)]
        # print(move)
        # print(state)
        if turn == "R":
            copy_state = copy.deepcopy(state)
            successive_boards.append(
                (move_right(copy_state, int(row) - 1), "R" + str(int(row) - 1))
            )
        elif turn == "L":
            copy_state = copy.deepcopy(state)
            successive_boards.append(
                (move_left(copy_state, int(row) - 1), "L" + str(int(row) - 1))
            )
        elif turn == "U":
            copy_state = copy.deepcopy(state)
            successive_boards.append(
                (
                    transpose_board(
                        move_left(transpose_board(copy_state), int(row) - 1)
                    ),
                    "U" + str(int(row) - 1),
                )
            )
        elif turn == "D":
            copy_state = copy.deepcopy(state)
            successive_boards.append(
                (
                    transpose_board(
                        move_right(transpose_board(copy_state), int(row) - 1)
                    ),
                    "D" + str(int(row) - 1),
                )
            )
        elif turn == "O" and row == "c":
            copy_state = copy.deepcopy(state)
            successive_boards.append((move_clockwise(copy_state), "Oc"))
        elif turn == "O" and row == "cc":
            copy_state = copy.deepcopy(state)
            successive_boards.append((move_cclockwise(copy_state), "Occ"))
        elif turn == "I" and row == "c":
            copy_state = copy.deepcopy(state)
            copy_state = np.array(copy_state)
            inner_state = copy_state[1:-1, 1:-1].tolist()
            inner_state = move_clockwise(inner_state)
            copy_state[1:-1, 1:-1] = np.array(inner_state)
            copy_state = copy_state.tolist()
            successive_boards.append((copy_state, "Ic"))
        elif turn == "I" and row == "cc":
            copy_state = copy.deepcopy(state)
            copy_state = np.array(copy_state)
            inner_state = copy_state[1:-1, 1:-1].tolist()
            inner_state = move_cclockwise(inner_state)
            copy_state[1:-1, 1:-1] = np.array(inner_state)
            copy_state = copy_state.tolist()
            successive_boards.append((copy_state, "Icc"))

    return successive_boards


# check if we've reached the goal
def is_goal(state):
    # if state == tuple(range(1, 26)):
    #     return True
    # else:
    #     return False

    if heuristic_cost(state) == 0:
        return True
    else:
        return False


# Calculate the misplaced numbers for heuristic
def heuristic_cost(state):
    count = 0
    for i in range(len(state)):
        if i != state[i]:
            count += 1

    return count


def find_distance_to_correct_position(curr_pos, correct_pos):
    return abs(curr_pos[0] - correct_pos[0]) + abs(curr_pos[1] - correct_pos[1])


def find_correct_position_of_block(block):
    for i in range(5):
        for j in range(5):
            formulator = 5 * i  # if i != 0 else i
            if block - (formulator + j + 1) == 0:
                return (i, j)


# Calculate manhattan distance from goal to current position of the whole board
# TO DO:
# 1. Need to figure out number rotation based heuristics
# 2. Check for total heuristic
def heuristic_cost(state):
    total = 0
    for i in range(len(state)):
        for j in range(len(state[0])):
            block = state[i][j]
            if block != ((5 * i) + j + 1):
                total += find_distance_to_correct_position(
                    (i, j), find_correct_position_of_block(block)
                )
                # print(
                #     block,
                #     find_distance_to_correct_position(
                #         (i, j), find_correct_position_of_block(block)
                #     ),
                # )

    return total


# TO DO:
# 1. Prepare rotation functions
# 2. Make use of the success functions
# 3. Formulate Plan
def solve(initial_board):
    """
    1. This function should return the solution as instructed in assignment, consisting of a list of moves like ["R2","D2","U1"].
    2. Do not add any extra parameters to the solve() function, or it will break our grading and testing code.
       For testing we will call this function with single argument(initial_board) and it should return
       the solution.
    3. Please do not use any global variables, as it may cause the testing code to fail.
    4. You can assume that all test cases will be solvable.
    5. The current code just returns a dummy solution.
    """

    goal = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]
    curr_board = [
        list(initial_board[i : i + 5]) for i in range(0, len(initial_board), 5)
    ]
    curr_board = [(curr_board), ""]
    # print(heuristic_cost(curr_board))
    # print(is_goal(curr_board))

    # print(successors(curr_board))

    # print(heuristic_cost(state=initial_board))

    # goal_board = is_goal(state=curr_board)
    move_made = list()
    min_cost_successor = sys.maxsize
    while not is_goal(state=curr_board[0]):
        print(min_cost_successor)
        print(curr_board)
        # all_successors = successors(state=curr_board)

        for successor in successors(state=curr_board[0]):
            # print(successor)
            # break
            new_succesor_cost = heuristic_cost(state=successor[0])
            if new_succesor_cost < min_cost_successor:
                min_cost_successor = new_succesor_cost
                curr_board = copy.deepcopy(successor)

            # if new_succesor_cost < min_cost_successor:
            #     min_cost_successor = new_succesor_cost
            #     curr_board = copy.deepcopy(successor)
            #     print()
            #     print("Found New Successor with cost ", min_cost_successor)
        # break
        # goal_board = is_goal(state=curr_board)
        move_made.append(curr_board[1])
        print(move_made)
    # print()
    # print(curr_board)
    # print(min_cost_successor)
    print()
    return move_made


# Please don't modify anything below this line
#
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise (Exception("Error: expected a board filename"))

    start_state = []
    with open(sys.argv[1], "r") as file:
        for line in file:
            start_state += [int(i) for i in line.split()]

    if len(start_state) != ROWS * COLS:
        raise (Exception("Error: couldn't parse start state file"))

    print("Start state: \n" + "\n".join(printable_board(tuple(start_state))))

    print("Solving...")
    route = solve(tuple(start_state))

    print("Solution found in " + str(len(route)) + " moves:" + "\n" + " ".join(route))
