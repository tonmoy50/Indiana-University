import heapq
import copy

# Define the goal state (canonical configuration)
goal_state = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]

# Define a heuristic function (you may need to customize this)
def heuristic(state):
    # Calculate the Manhattan distance for each tile
    distance = 0
    for i in range(5):
        for j in range(5):
            if state[i][j] != goal_state[i][j]:
                distance += 1
    return distance

# Define the possible moves (slide rows, slide columns, rotate rings)
moves = ["R1", "R2", "R3", "R4", "R5", "L1", "L2", "L3", "L4", "L5",
         "U1", "U2", "U3", "U4", "U5", "D1", "D2", "D3", "D4", "D5",
         "Oc", "Occ", "Ic", "Icc"]

# Implement the A* search algorithm
def solve(initial_state):
    open_set = [(heuristic(initial_state), initial_state)]
    heapq.heapify(open_set)
    closed_set = set()

    while open_set:
        _, current_state = heapq.heappop(open_set)
        print(current_state)
        if current_state == goal_state:
            return current_state

        closed_set.add(tuple(map(tuple, current_state)))

        for move in moves:
            new_state = apply_move(current_state, move)
            if tuple(map(tuple, new_state)) not in closed_set:
                priority = len(closed_set) + heuristic(new_state)
                heapq.heappush(open_set, (priority, new_state))

    return None

# Apply a move to the current state
def apply_move(state, move):
    new_state = copy.deepcopy(state)
    if move[0] == "R":
        row = int(move[1]) - 1
        new_state[row].insert(0, new_state[row].pop())
    elif move[0] == "L":
        row = int(move[1]) - 1
        new_state[row].append(new_state[row].pop(0))
    elif move[0] == "U":
        col = int(move[1]) - 1
        new_state[0][col], new_state[4][col] = new_state[4][col], new_state[0][col]
        new_state[1][col], new_state[3][col] = new_state[3][col], new_state[1][col]
    elif move[0] == "D":
        col = int(move[1]) - 1
        new_state[0][col], new_state[4][col] = new_state[4][col], new_state[0][col]
        new_state[1][col], new_state[3][col] = new_state[3][col], new_state[1][col]
    elif move == "Oc":
        ring_outer_clockwise(new_state)
    elif move == "Occ":
        ring_outer_counterclockwise(new_state)
    elif move == "Ic":
        ring_inner_clockwise(new_state)
    elif move == "Icc":
        ring_inner_counterclockwise(new_state)
    return new_state

# Implement ring rotations (outer and inner)
def ring_outer_clockwise(state):
    temp = state[0][0]
    state[0][0] = state[0][4]
    state[0][4] = state[4][4]
    state[4][4] = state[4][0]
    state[4][0] = temp

    temp = state[0][1]
    state[0][1] = state[1][4]
    state[1][4] = state[4][3]
    state[4][3] = state[3][0]
    state[3][0] = temp

    temp = state[0][2]
    state[0][2] = state[2][4]
    state[2][4] = state[4][2]
    state[4][2] = state[2][0]
    state[2][0] = temp

    temp = state[1][0]
    state[1][0] = state[0][3]
    state[0][3] = state[3][4]
    state[3][4] = state[4][1]
    state[4][1] = temp

    temp = state[1][2]
    state[1][2] = state[2][3]
    state[2][3] = state[3][2]
    state[3][2] = state[2][1]
    state[2][1] = temp

def ring_outer_counterclockwise(state):
    for _ in range(3):
        ring_outer_clockwise(state)

def ring_inner_clockwise(state):
    temp = state[1][1]
    state[1][1] = state[1][3]
    state[1][3] = state[3][3]
    state[3][3] = state[3][1]
    state[3][1] = temp

    temp = state[1][2]
    state[1][2] = state[2][3]
    state[2][3] = state[3][2]
    state[3][2] = state[2][1]
    state[2][1] = temp

def ring_inner_counterclockwise(state):
    for _ in range(3):
        ring_inner_clockwise(state)

# Example initial state
initial_state = [
    [2, 3, 4, 5, 1],
    [8, 9, 10, 6, 7],
    [13, 14, 15, 11, 12],
    [18, 19, 20, 16, 17],
    [24, 25, 21, 22, 23]
]

# Solve the puzzle
solution = solve(initial_state)
if solution:
    print("Solution found:")
    for row in solution:
        print(row)
else:
    print("No solution found.")
