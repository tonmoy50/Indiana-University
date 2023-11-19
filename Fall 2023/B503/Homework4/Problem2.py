def count_paths(graph, s, t):
    opt = {v: 0 for v in graph}
    opt[t] = 1

    top_order = topological_sort(graph)
    print(top_order)
    for v in reversed(top_order):
        for u in graph[v]:
            opt[v] += opt[u]

    return opt[s]


def topological_sort(graph):
    in_degree = {v: 0 for v in graph}
    for v in graph:
        for u in graph[v]:
            in_degree[u] += 1

    queue = [v for v in graph if in_degree[v] == 0]
    result = []

    while queue:
        current = queue.pop(0)
        result.append(current)
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(graph):
        raise ValueError("Graph contains a cycle")

    return result


# Example usage:
graph = {
    "s": ["a", "b"],
    "a": ["c", "d"],
    "b": ["c"],
    "c": ["t"],
    # "d": ["t"],
    "d": [],
    "t": [],
    "b": ["t"],
}

result = count_paths(graph, "s", "t")
print("Number of paths from s to t is odd:", result)
