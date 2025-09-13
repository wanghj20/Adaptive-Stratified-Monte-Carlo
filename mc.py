# mc.py
import numpy as np

def ordinary_mc_single_run(size, func, dim):
    
    X = np.random.rand(size, dim)  
    vals = np.array([func(x) for x in X], dtype=float)
    return float(np.mean(vals))
