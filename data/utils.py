import numpy as np


def split_indices(n, val_pct=0.1, seed=99):
    # Determine size of validation set
    n_val = int(val_pct * n)
    # Set the random seed
    np.random.seed(seed)
    # Create permutation of 0 to n-1
    idxs = np.random.permutation(n)
    # Pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]
