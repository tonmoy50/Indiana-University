#!/usr/local/bin/python3
#
# place_turrets.py : arrange turrets on a grid, avoiding conflicts
#
# Submitted by : Nilambar Halder Tonmoy - nhaldert
#
# Based on skeleton code in CSCI B551, Fall 2022.

import sys


# Parse the map from a given filename
def parse_map(filename):
    with open(filename, "r") as f:
        return [[char for char in line] for line in f.read().rstrip("\n").split("\n")][
            3:
        ]


# Count total # of turrets on castle_map
def count_turrets(castle_map):
    return sum([row.count("p") for row in castle_map])


# Return a string with the castle_map rendered in a human-turretly format
def printable_castle_map(castle_map):
    return "\n".join(["".join(row) for row in castle_map])


# Add a turret to the castle_map at the given position, and return a new castle_map (doesn't change original)
def add_turret(castle_map, row, col):
    return (
        castle_map[0:row]
        + [
            castle_map[row][0:col]
            + [
                "p",
            ]
            + castle_map[row][col + 1 :]
        ]
        + castle_map[row + 1 :]
    )


# Get list of successors of given castle_map state
def successors(castle_map):
    return [
        add_turret(castle_map, r, c)
        for r in range(0, len(castle_map))
        for c in range(0, len(castle_map[0]))
        if castle_map[r][c] == "."
    ]


# check if castle_map is a goal state
def is_goal(castle_map, k):
    return count_turrets(castle_map) == k

# Finds and returns the position of placed turret by the successor function
def get_turrets_position(castle_map):
    turret_positions = []
    for i in range(len(castle_map)):
        for j in range(len(castle_map[0])):
            if castle_map[i][j] == "p":
                turret_positions.append((i, j))

    return turret_positions

# Checks whether there is a conflict in the left side of the given turret
def is_conflict_left(castle_map, turret_position):
    i = turret_position[1] - 1
    while i >= 0:
        if castle_map[turret_position[0]][i] == "p":
            return True
        elif castle_map[turret_position[0]][i] in "X@":
            return False
        i -= 1
    return False

# Checks whether there is a conflict in the right side of the given turret
def is_conflict_right(castle_map, turret_position):
    i = turret_position[1] + 1
    while i < len(castle_map[0]):
        if castle_map[turret_position[0]][i] == "p":
            return True
        elif castle_map[turret_position[0]][i] in "X@":
            return False
        i += 1
    return False

# Checks whether there is a conflict in the upper side of the given turret
def is_conflict_up(castle_map, turret_position):
    i = turret_position[0] - 1
    while i >= 0:
        if castle_map[i][turret_position[1]] == "p":
            return True
        elif castle_map[i][turret_position[1]] in "X@":
            return False
        i -= 1
    return False

# Checks whether there is a conflict in the lower part of the given turret
def is_conflict_down(castle_map, turret_position):
    i = turret_position[0] + 1
    while i < len(castle_map):
        if castle_map[i][turret_position[1]] == "p":
            return True
        elif castle_map[i][turret_position[1]] in "X@":
            return False
        i += 1
    return False

# Checks whether there is a conflict in the diagonal upper left side of the given turret
def is_diagonal_up_left_conflicted(castle_map, turret_position):
    i = turret_position[0] - 1
    j = turret_position[1] - 1
    while i >= 0 and j >= 0:
        if castle_map[i][j] == "p":
            return True
        elif castle_map[i][j] in "X@":
            return False
        i -= 1
        j -= 1

    return False

# Checks whether there is a conflict in the diagonal lower right side of the given turret
def is_diagonal_down_right_conflicted(castle_map, turret_position):
    i = turret_position[0] + 1
    j = turret_position[1] + 1
    while i < len(castle_map) and j < len(castle_map[0]):
        if castle_map[i][j] == "p":
            return True
        elif castle_map[i][j] in "X@":
            return False
        i += 1
        j += 1

    return False

# Checks whether there is a conflict in the diagonal upper right side of the given turret
def is_diagonally_up_right_conflicted(castle_map, turret_position):
    i = turret_position[0] - 1
    j = turret_position[1] + 1
    while i > 0 and i < len(castle_map) and j > 0 and j < len(castle_map[0]):
        if castle_map[i][j] == "p":
            return True
        elif castle_map[i][j] in "X@":
            return False
        i -= 1
        j += 1

    return False

# Checks whether there is a conflict in the diagonal lower left side of the given turret
def is_diagonally_down_left_conflicted(castle_map, turret_position):
    i = turret_position[0] + 1
    j = turret_position[1] - 1
    while i < len(castle_map) and j < len(castle_map[0]) and j > 0:
        if castle_map[i][j] == "p":
            return True
        elif castle_map[i][j] in "X@":
            return False
        i += 1
        j -= 1

    return False


# Arrange turrets on the map
#
# This function MUST take two parameters as input -- the castle map and the value k --
# and return a tuple of the form (new_castle_map, success), where:
# - new_castle_map is a new version of the map with k turrets,
# - success is True if a solution was found, and False otherwise.
#
def solve(initial_castle_map, k):
    fringe = [initial_castle_map]
    while len(fringe) > 0:
        for new_castle_map in successors(fringe.pop()):
            turret_positions = get_turrets_position(castle_map=new_castle_map)
            valid_turret_positions = [] # For storing valid turret positions
            for turret_position in turret_positions:
                # Checking whether the placed turret is a valid postion and don't conflict with other turrets
                if (
                    is_conflict_left(new_castle_map, turret_position) == False
                    and is_conflict_right(new_castle_map, turret_position) == False
                    and is_conflict_up(new_castle_map, turret_position) == False
                    and is_conflict_down(new_castle_map, turret_position) == False
                    and is_diagonal_down_right_conflicted(
                        new_castle_map, turret_position
                    )
                    == False
                    and is_diagonally_down_left_conflicted(
                        new_castle_map, turret_position
                    )
                    == False
                    and is_diagonal_up_left_conflicted(new_castle_map, turret_position)
                    == False
                    and is_diagonally_up_right_conflicted(
                        new_castle_map, turret_position
                    )
                    == False
                ):
                    valid_turret_positions.append(turret_position)
            # Adding to the fringe if and only if the new_castle_map is a valid map that doesn't have conflicting turret position
            # Also checking if our goal has been reached inside the check as well as otherwise will give us wrong castle map
            if len(turret_positions) == len(valid_turret_positions):
                fringe.append(new_castle_map)
                if is_goal(new_castle_map, k):
                    return (new_castle_map, True)

    return (castle_map, "")


# Main Function
if __name__ == "__main__":
    castle_map = parse_map(sys.argv[1])
    # This is k, the number of turrets
    k = int(sys.argv[2])
    print(
        "Starting from initial castle map:\n"
        + printable_castle_map(castle_map)
        + "\n\nLooking for solution...\n"
    )
    solution = solve(castle_map, k)

    print("Here's what we found:")
    print(printable_castle_map(solution[0]) if solution[1] else "False")
