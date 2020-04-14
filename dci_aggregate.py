"""
Aggregate and plot dci scores of models.

Simon Bing,
ETHZ 2020
"""

import numpy as np
import os
from glob import glob
from absl import flags, app

def aggregate(N, latent_dims, base_dir='dsprites_dim_'):
    """
    Collects all dci scores and aggregates into single array.
    Args:
        N, int:             Number of random seeds
        latent_dims, list:  Latent dims tested
        base_dir, string:   Base directory naming scheme

    Returns:
        dci_scores, [3xNxM] np array
    """
    dci_scores = np.zeros((3,N,len(latent_dims)), dtype=np.float32)

    for m, dim in enumerate(latent_dims):
        dim_dir = base_dir+"{}".format(dim)
        n = 0
        for _, dirs, _ in os.walk(os.path.join("models", dim_dir)):
            for dir in dirs
                dci_file = glob(os.path.join(dir,'dci*')) # This should find the file that begin with dci so full name doesnt have to be specified MUST STILL BE TESTED
                print(dci_file)
                dci = np.load(dci_file)
                # dci_scores[0,n,m]

def main(argv):
    del argv # Unused

    n_experiments = 10
    latent_dims = [8, 16, 32, 64, 128]

    dci_scores = aggregate(n_experiments, latent_dims)

if __name__ == '__main__':
    app.run(main)
