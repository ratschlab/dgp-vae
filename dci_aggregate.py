"""
Aggregate and plot dci scores of models.

Simon Bing,
ETHZ 2020
"""

import numpy as np
import os
from glob import glob
import json
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'gpvae', 'Model for dci scores should be evaluated')
flags.DEFINE_boolean('save', False, 'Save aggregated scores')

def walklevel(some_dir, level=0):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

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
        dim_dir = base_dir+"{}_tensor2".format(dim)
        # n = 0
        models_path = os.path.join('models', dim_dir)
        for _, dirs, _ in os.walk(models_path):
            for n, dir in enumerate(dirs):
                for _, _, files in os.walk(os.path.join(models_path, dir)):
                    for filename in files:
                        if filename.startswith('dci'):
                            # print(n, filename)
                            dci = np.load(os.path.join(models_path, dir, filename)) # Might need abspath here
                            dci_scores[0,n,m] = dci['disentanglement']
                            dci_scores[1,n,m] = dci['completeness']
                            dci_scores[2,n,m] = dci['informativeness_test']

    return dci_scores

def aggregate_baseline(N, base_dir='dim_64'):
    """
    Collects all dci scores and aggregates into single array.
    Args:
        N, int:             Number of random seeds
        base_dir, string:   Base directory naming scheme

    Returns:
        dci_scores, [3xN] np array
    """
    dci_scores = np.zeros((3,N), dtype=np.float32)

    base_path = os.path.join('baselines', FLAGS.model, base_dir)

    for _, dirs, _ in walklevel(base_path):
        for n, dir in enumerate(dirs):
            print(n, dir)
            json_path = os.path.join(base_path, dir, 'metrics', 'dci', 'results', 'aggregate', 'evaluation.json')
            with open(json_path) as json_file:
                dci = json.load(json_file) # PROPERLY PARSE JSON FILE
                dci_scores[0,n] = dci['evaluation_results.disentanglement']
                dci_scores[1,n] = dci['evaluation_results.completeness']
                dci_scores[2,n] = dci['evaluation_results.informativeness_test']

    return dci_scores

def main(argv):
    del argv # Unused

    n_experiments = 9

    if FLAGS.model == 'gpvae':
        latent_dims = [8, 16, 32, 64, 128]
        dci_scores = aggregate_gpvae(n_experiments, latent_dims)
    elif FLAGS.model in ['betatcvae', 'factorvae', 'dipvae_i']:
        dci_scores = aggregate_baseline(n_experiments, base_dir='dim_64')
    else:
        raise ValueError("Model must be one of: ['gpvae', 'betatcvae', 'factorvae', 'dipvae_i']")

    if FLAGS.save:
        np.save('{}_64_dci_aggr.npy'.format(FLAGS.model), dci_scores)

if __name__ == '__main__':
    app.run(main)
