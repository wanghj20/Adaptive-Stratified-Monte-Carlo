
# func.py
import numpy as np

def make_exp_linear(dim):
   
    lambdas = np.array([1.0 / (i**2) for i in range(1, dim + 1)], dtype=float)

    def f(x):
        
        return float(np.exp(np.sum(lambdas * x)))

    return f, lambdas
