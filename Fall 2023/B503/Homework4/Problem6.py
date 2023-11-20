from collections import defaultdict


# Perform a topological sort of the graph
def topological_sort(graph):
    visited = set()
    sorted_nodes = []

    def visit(node):
        if node not in visited:
            visited.add(node)
            for successor in graph[node]:
                visit(successor)
            sorted_nodes.append(node)

    for node in graph:
        visit(node)

    return sorted_nodes[::-1]


# Find the longest path in a directed acyclic graph
def longest_path(graph, weights):
    sorted_nodes = topological_sort(graph)
    dist = defaultdict(lambda: float("-inf"))
    dist[sorted_nodes[0]] = 0

    for node in sorted_nodes:
        # for i in range(1, len(graph[node])):
        #     if weights[node, graph[node][i-1]] < weights[node, graph[node][i-1]]:
        #         dist[graph[node][i]] = max(dist[graph[node][i]], dist[node] + weights[(node, graph[node][i])])
        successors = list()
        for successor in graph[node]:
            dist[successor] = max(
                dist[successor], dist[node] + weights[(node, successor)]
            )
            successors.append(successor)

    # for node in graph:
    #     for i in range(1, len(graph[node])):
    #         if weights[node, graph[node][i]] < weights[node, graph[node][i - 1]]:
    #             dist[node] = -1
    #             break
    print(dist)
    return max(dist.values())


# Example graph
graph = {"A": ["B", "C"], "B": ["D", "E"], "C": ["E"], "D": [], "E": []}

weights = {("A", "B"): 2, ("A", "C"): 4, ("B", "D"): 3, ("B", "E"): 1, ("C", "E"): 4}

print(longest_path(graph, weights))  # Output: 8
