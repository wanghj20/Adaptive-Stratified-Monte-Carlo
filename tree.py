# tree.py
import numpy as np

class TreeNode:
    def __init__(self, points):
        self.points = points
        self.left = None
        self.right = None
        self.split_dim = None
        self.split_val = None

def split_node(node, depth, max_depth, dim):
    if depth >= max_depth:
        return

    n_pts = len(node.points)
    if n_pts <= 1:
        split_dim = np.random.randint(0, dim)
        split_val = 0.5
    else:
        X_local = node.points[:, :dim]                    
        variances = np.var(X_local, axis=0)
        max_var = np.max(variances)
        candidates = np.flatnonzero(np.isclose(variances, max_var))
        split_dim = np.random.choice(candidates)
        split_val = np.median(X_local[:, split_dim])

    node.split_dim = split_dim
    node.split_val = split_val

    left_points  = node.points[node.points[:, split_dim] <= split_val]
    right_points = node.points[node.points[:, split_dim] >  split_val]

    node.left = TreeNode(left_points)
    node.right = TreeNode(right_points)

    split_node(node.left,  depth + 1, max_depth, dim)
    split_node(node.right, depth + 1, max_depth, dim)

def get_leaf_bounds(node, lower_bounds, upper_bounds):
   
    if node.left is None and node.right is None:
        return [(lower_bounds, upper_bounds)]

    bounds = []
    d = node.split_dim
    new_upper = upper_bounds.copy()
    new_upper[d] = node.split_val
    
    bounds.extend(get_leaf_bounds(node.left, lower_bounds, new_upper))
    new_lower = lower_bounds.copy()
    new_lower[d] = node.split_val
    bounds.extend(get_leaf_bounds(node.right, new_lower, upper_bounds))

    return bounds
















