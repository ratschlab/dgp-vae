"""
Simon Bing ETHZ, 2020

Script to compute dci score of learned representation.
"""
import warnings
from typing import Union, Iterable

from numpy.core._multiarray_umath import ndarray

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from absl import flags, app
from sklearn.model_selection import train_test_split
from disentanglement_lib.evaluation.metrics import dci
from disentanglement_lib.visualize import visualize_scores
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('c_path', '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_5000.npy', 'File path for underlying factors z')
flags.DEFINE_string('model_name', '', 'Name of model directory to get learned latent code')
flags.DEFINE_bool('visualize_score', False, 'Whether or not to visualize score')
flags.DEFINE_bool('save_score', False, 'Whether or not to save calculated score')

def load_z_c(c_path, z_path):
    try:
        c_full = np.load(c_path)['factors_test']
    except IndexError:
        c_full = np.load(c_path)
    z = np.load(z_path)

    # Check length of c and only take same amount of z values. Corresponds to z_test.
    c = c_full[:z.shape[0],:,:]
    assert z.shape[0] == c.shape[0]

    return c, z

def main(argv, model_dir=None):
    del argv # Unused

    if model_dir is None:
        out_dir = FLAGS.model_name
    else:
        out_dir = model_dir

    z_path = '{}/z_mean.npy'.format(out_dir)
    # project_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_5000.npy'
    # z_path = os.path.join(project_path, FLAGS.z_name)

    c, z = load_z_c(FLAGS.c_path, z_path)

    z_shape = z.shape
    c_shape = c.shape

    z_reshape = np.reshape(np.transpose(z, (0,2,1)),(z_shape[0]*z_shape[2],z_shape[1]))
    c_reshape = np.reshape(np.transpose(c, (0,2,1)),(c_shape[0]*c_shape[2],c_shape[1]))

    # Check if latent factor doesn't change and remove if is the case
    mask = np.ones(c_reshape.shape[1], dtype=bool)
    for i in range(c_reshape.shape[1]):
        c_change = np.sum(np.diff(c_reshape[:,i]))
        if not c_change:
            mask[i] = False
    c_reshape = c_reshape[:,mask]

    c_train, c_test, z_train, z_test = train_test_split(c_reshape, z_reshape, test_size=0.2, shuffle=False)
    scores = dci._compute_dci(z_train[:800,:].transpose(), c_train[:800,:].transpose(), z_test[:200,:].transpose(), c_test[:200,:].transpose())

    # Visualization
    if FLAGS.visualize_score:
        importance_matrix, _, _ = dci.compute_importance_gbt(
            z_train[:800,:].transpose(), c_train[:800,:].transpose(),
            z_test[:200,:].transpose(), c_test[:200,:].transpose())

        visualize_scores.heat_square(importance_matrix, out_dir, "dci_matrix",
                                     "x_axis", "y_axis")

    print('D: {}'.format(scores['disentanglement']))
    print('C: {}'.format(scores['completeness']))
    print('I: {}'.format(scores['informativeness_test']))
    print("Evaluation finished")

    if FLAGS.save_score:
        np.savez('{}/dci_{}_{}_{}'.format(out_dir, z_shape[1], c_shape[1], z_shape[0]),
                 informativeness_train=scores['informativeness_train'],
                 informativeness_test=scores['informativeness_test'],
                 disentanglement=scores['disentanglement'],
                 completeness=scores['completeness'])
        print("Score saved")


if __name__ == '__main__':
    app.run(main)
