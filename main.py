# main.py
from joblib import Parallel, delayed
import numpy as np

from tree import TreeNode, split_node, get_leaf_bounds
from func import make_exp_linear
from mc import ordinary_mc_single_run
from stratified import uniform_stratified_single_run


def stratified_tree_single_run(size, depth_for_tree, func, dim):
    X = np.random.rand(size, dim)
    y = np.array([[func(x)] for x in X], dtype=float)
    points = np.hstack((X, y))

    root = TreeNode(points)
    split_node(root, depth=0, max_depth=depth_for_tree, dim=dim)
    leaf_bounds = get_leaf_bounds(root, np.zeros(dim), np.ones(dim))

    if len(leaf_bounds) == 0:
        return float("nan")

    vals = []
    for lower, upper in leaf_bounds:
        s = np.random.uniform(lower, upper)  
        vals.append(func(s))
    return float(np.mean(vals))

def main():
    dim = 5
    f, lambdas = make_exp_linear(dim)

    sample_sizes = [2 ** i for i in range(10, 20)]   
    num_repetitions = 256

    print("=== Monte Carlo Estimation Comparison ===")
    print(f"dim={dim}, repetitions={num_repetitions}")
    print(f"methods: Ordinary MC | Tree-Stratified")
    print("-" * 78)

    for size in sample_sizes:
        depth_for_tree = int(np.log2(size))

        # Ordinary MC
        mc_runs = Parallel(n_jobs=-1)(
            delayed(ordinary_mc_single_run)(size, f, dim)
            for _ in range(num_repetitions)
        )
        mc_mean = np.mean(mc_runs)
        mc_std  = np.std(mc_runs)

        # Tree-Stratified
        tree_runs = Parallel(n_jobs=-1)(
            delayed(stratified_tree_single_run)(size, depth_for_tree, f, dim)
            for _ in range(num_repetitions)
        )
        tree_mean = np.mean(tree_runs)
        tree_std  = np.std(tree_runs)

        print(f"N={size:4d} | "
              f"MC: mean={mc_mean:.6f}, sd={mc_std:.6f} | "
              f"UniformStrat: mean={uni_mean:.6f}, sd={uni_std:.6f} | "
              f"Tree: mean={tree_mean:.6f}, sd={tree_std:.6f}")

if __name__ == "__main__":
    main()

