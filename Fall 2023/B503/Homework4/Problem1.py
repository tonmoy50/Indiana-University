class TreeNode:
    def __init__(self, value=None, operator=None, left=None, right=None):
        self.is_leaf = value is not None
        self.value = value
        self.operator = operator
        self.left = left
        self.right = right


def min_cost(root):
    memo = {}

    def dp(node):
        if node in memo:
            return memo[node]

        if node.is_leaf:
            cost = 0 if node.value else 1
        else:
            cost_left = dp(node.left)
            cost_right = dp(node.right)

            if node.operator == "and":
                cost = cost_left + cost_right
            elif node.operator == "or":
                cost = min(cost_left, cost_right)

        memo[node] = cost
        return cost

    return dp(root)

if __name__ == "__main__":
    root = TreeNode(
        operator="and",
        left=TreeNode(
            operator="or", left=TreeNode(value=True), right=TreeNode(value=False)
        ),
        right=TreeNode(
            operator="and",
            left=TreeNode(value=False),
            right=TreeNode(
                operator="and", left=TreeNode(value=False), right=TreeNode(value=False)
            ),
        ),
    )

    result = min_cost(root)
    print("Minimum cost for the cake:", result)

    root = TreeNode(
        operator="and",
        left=TreeNode(
            operator="or", left=TreeNode(value=True), right=TreeNode(value=False)
        ),
        right=TreeNode(
            operator="and",
            left=TreeNode(value=True),
            right=TreeNode(value=True),
        ),
    )


    result = min_cost(root)
    print("Minimum cost for the cake:", result)
