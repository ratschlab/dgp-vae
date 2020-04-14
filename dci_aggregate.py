"""
Aggregate and plot dci scores of models.

Simon Bing,
ETHZ 2020
"""

import numpy as np
import os
from glob import glob
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'gpvae', 'Model for dci scores should be evaluated')

def aggregate_gpvae(N, latent_dims, base_dir='dsprites_dim_'):
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
        # n = 0
        models_path = os.path.join('models', dim_dir)
        for _, dirs, _ in os.walk(models_path):
            for n, dir in enumerate(dirs):
                for _, _, files in os.walk(os.path.join(models_path,dir)):
                    for filename in files:
                        if filename.startswith('dci'):
                            # print(n, filename)
                            dci = np.load(filename) # Might need abspath here
                            dci_scores[0,n,m] = dci['disentanglement']
                            dci_scores[1,n,m] = dci['completeness']
                            dci_scores[2,n,m] = dci['informativeness']

    return dci_scores

def main(argv):
    del argv # Unused

    n_experiments = 10

    if FLAGS.model == 'gpvae':
        latent_dims = [8, 16, 32, 64, 128]
        dci_scores = aggregate_gpvae(n_experiments, latent_dims)
    elif FLAGS.model in ['betatcvae', 'factorvae', 'dipvae_i']:
        blabla
    else:
        raise ValueError("Model must be one of: ['gpvae', 'betatcvae', 'factorvae', 'dipvae_i']")

if __name__ == '__main__':
    app.run(main)
