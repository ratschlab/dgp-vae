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

flags.DEFINE_string('model', 'gpvae', 'Model for which dci score should be calculated')
flags.DEFINE_string('base_dir', '', 'base directory of models')
flags.DEFINE_list('params', [64], 'Parameters tested in experiment')
flags.DEFINE_string('exp_name', '', 'Experiment name')
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

def aggregate_gpvae(N, params, base_dir='dsprites_dim_'):
    """
    Collects all dci scores and aggregates into single array.
    Args:
        N, int:             Number of random seeds
        latent_dims, list:  Latent dims tested
        base_dir, string:   Base directory naming scheme

    Returns:
        dci_scores, [3xNxM] np array
    """
    dci_scores = np.zeros((3,N,len(params)), dtype=np.float32)

    for m, param in enumerate(params):
        param_dir = base_dir+"_{}_{}".format(param, FLAGS.exp_name)

        models_path = os.path.join('models', param_dir)
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

def aggregate_baseline(N, base_dir='dim', exp_name='test_out'):
    """
    Collects all dci scores and aggregates into single array.
    Args:
        N, int:             Number of random seeds
        base_dir, string:   Base directory naming scheme

    Returns:
        dci_scores, [3xN] np array
    """
    dci_scores = np.zeros((3,N), dtype=np.float32)

    base_path = os.path.join('baselines', FLAGS.model, base_dir, exp_name)

    for _, dirs, _ in walklevel(base_path):
        for n, dir in enumerate(dirs):
            # print(n, dir)
            json_path = os.path.join(base_path, dir, 'metrics', 'dci', 'results', 'aggregate', 'evaluation.json')
            with open(json_path) as json_file:
                dci = json.load(json_file) # PROPERLY PARSE JSON FILE
                dci_scores[0,n] = dci['evaluation_results.disentanglement']
                dci_scores[1,n] = dci['evaluation_results.completeness']
                dci_scores[2,n] = dci['evaluation_results.informativeness_test']

    return dci_scores

def main(argv):
    del argv # Unused

    n_experiments = 10

    if FLAGS.model == 'gpvae':
        # params = [64]
        dci_scores = aggregate_gpvae(n_experiments, FLAGS.params, FLAGS.base_dir)
    elif FLAGS.model in ['betatcvae', 'factorvae', 'dipvae_i']:
        dci_scores = aggregate_baseline(n_experiments, FLAGS.base_dir)
    else:
        raise ValueError("Model must be one of: ['gpvae', 'betatcvae', 'factorvae', 'dipvae_i']")

    if FLAGS.save:
        if FLAGS.model == 'gpvae':
            np.save('dci_scores/{}_64_sin_rand_dci_aggr.npy'.format(FLAGS.model), dci_scores)
        elif FLAGS.model in ['betatcvae', 'factorvae', 'dipvae_i']:
            np.save(os.path.join('baselines', FLAGS.model, FLAGS.base_dir,
                                 FLAGS.exp_name, '{}_dci_aggr.npy'.format(FLAGS.model))
                    , dci_scores)
        else:
            raise ValueError(
                "Model must be one of: ['gpvae', 'betatcvae', 'factorvae', 'dipvae_i']")

if __name__ == '__main__':
    app.run(main)
