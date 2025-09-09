from joblib import Parallel, delayed 
import numpy as np

class TreeNode:
    def __init__(self, points):
        self.points = points
        self.left = None
        self.right = None
        self.split_dim = None
        self.split_val = None

def split_node(node, depth, max_depth):
    
    if depth >= max_depth:
        return
      
    if len(node.points) <= 1:
        
        split_dim = np.random.randint(0, dim)
        
        node.split_val = 0.5
    else:
        variances = np.var(node.points[:, :-1], axis=0)
        split_dim = np.argmax(variances)
        node.split_val = np.median(node.points[:, split_dim])
    
    node.split_dim = split_dim
    left_points = node.points[node.points[:, split_dim] <= node.split_val]
    right_points = node.points[node.points[:, split_dim] > node.split_val]
    node.left = TreeNode(left_points)
    node.right = TreeNode(right_points)

    split_node(node.left, depth + 1, max_depth)
    split_node(node.right, depth + 1, max_depth)

def get_leaf_bounds(node, lower_bounds, upper_bounds):
    if node.left is None and node.right is None:
        return [(lower_bounds, upper_bounds)]
    
    bounds = []
    split_dim = node.split_dim
    new_lower = lower_bounds.copy()
    new_upper = upper_bounds.copy()
    
    new_upper[split_dim] = node.split_val
    bounds.extend(get_leaf_bounds(node.left, lower_bounds, new_upper))
    
    new_lower[split_dim] = node.split_val
    bounds.extend(get_leaf_bounds(node.right, new_lower, upper_bounds))
    
    return bounds

def monte_carlo_single_run(size, i, func):
    
    X = np.random.rand(size, dim)
    y = np.array([[func(x)] for x in X])
    points = np.hstack((X, y))
    
    root = TreeNode(points)
    split_node(root, 0, i)
    leaf_bounds = get_leaf_bounds(root, np.zeros(dim), np.ones(dim))
    
    results = []
    for bound in leaf_bounds:
        lower, upper = bound
        point = np.random.uniform(lower, upper)
        results.append(func(point))
    
    return np.mean(results)


dim = 5
lambdas = np.array([1/(10**i) for i in range(1, dim+1)])

def f(X):
    return np.exp(np.sum(lambdas*X))


np.random.seed(0)
sample_sizes = [2**i for i in range(4, 12)]
num_repetitions = 256
results = []

for size in sample_sizes:
    i = np.log2(size).astype(int)  
    
  
    results_for_size = Parallel(n_jobs=-1)(
        delayed(monte_carlo_single_run)(size, i, f) 
        for _ in range(num_repetitions)
    )
    
    mean = np.mean(results_for_size)
    std_dev = np.std(results_for_size)
    results.append((size, mean, std_dev))


for result in results:
    print(f"Sample Size: {result[0]}, Mean: {result[1]}, Std Dev: {result[2]}")














