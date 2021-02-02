"""
Simon Bing ETHZ, 2020

Script to compute dci score of learned representation.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from absl import flags, app
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from disentanglement_lib.evaluation.metrics import dci
from disentanglement_lib.visualize import visualize_scores
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('c_path', '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_5000.npz', 'File path for underlying factors c')
flags.DEFINE_string('assign_mat_path', '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy', 'Path for assignment matrix')
flags.DEFINE_string('model_name', '', 'Name of model directory to get learned latent code')
flags.DEFINE_enum('data_type_dci', 'dsprites', ['hmnist', 'physionet', 'hirid', 'sprites', 'dsprites', 'smallnorb', 'cars3d'], 'Type of data and how to evaluate')
flags.DEFINE_list('score_factors', [], 'Underlying factors to consider in DCI score calculation')
flags.DEFINE_enum('rescaling', 'linear', ['linear', 'standard'], 'Rescaling of ground truth factors')
flags.DEFINE_bool('shuffle', False, 'Whether or not to shuffle evaluation data.')
flags.DEFINE_integer('dci_seed', 42, 'Random seed.')
flags.DEFINE_bool('visualize_score', False, 'Whether or not to visualize score')
flags.DEFINE_bool('save_score', False, 'Whether or not to save calculated score')

def load_z_c(c_path, z_path):
    try:
        c_full = np.load(c_path)['factors_test']
    except IndexError:
        c_full = np.load(c_path)
    z = np.load(z_path)
    c = c_full

    return c, z

def main(argv, model_dir=None):
    del argv # Unused

    if model_dir is None:
        out_dir = FLAGS.model_name
    else:
        out_dir = model_dir

    z_path = '{}/z_mean.npy'.format(out_dir)
    if FLAGS.data_type_dci == "physionet":
        # Use imputed values as ground truth for physionet data
        c, z = load_z_c('{}/imputed.npy'.format(out_dir), z_path)
        c = np.transpose(c, (0,2,1))
    elif FLAGS.data_type_dci == "hirid":
        c = np.load(FLAGS.c_path)['x_test_miss']
        c = np.transpose(c, (0, 2, 1))
        c = c.astype(int)
        z = np.load(z_path)
    else:
        c, z = load_z_c(FLAGS.c_path, z_path)

    z_shape = z.shape
    c_shape = c.shape

    z_reshape = np.reshape(np.transpose(z, (0,2,1)),(z_shape[0]*z_shape[2],z_shape[1]))
    c_reshape = np.reshape(np.transpose(c, (0,2,1)),(c_shape[0]*c_shape[2],c_shape[1]))
    c_reshape = c_reshape[:z_reshape.shape[0], ...]

    # Experimental physionet rescaling
    if FLAGS.data_type_dci == 'physionet':
        if FLAGS.rescaling == 'linear':
            # linear rescaling
            c_rescale = 10 * c_reshape
            c_reshape = c_rescale.astype(int)
        elif FLAGS.rescaling == 'standard':
            # standardizing
            scaler = StandardScaler()
            c_rescale = scaler.fit_transform(c_reshape)
            c_reshape = (10*c_rescale).astype(int)
        else:
            raise ValueError("Rescaling must be 'linear' or 'standard'")


    # Include all factors in score calculation, if not specified otherwise
    if not FLAGS.score_factors:
        FLAGS.score_factors = np.arange(c_shape[1]).astype(str)

    # Check if ground truth factor doesn't change and remove if is the case
    mask = np.ones(c_reshape.shape[1], dtype=bool)
    for i in range(c_reshape.shape[1]):
        c_change = np.sum(abs(np.diff(c_reshape[:8000,i])))
        if (not c_change) or (F"{i}" not in FLAGS.score_factors):
            mask[i] = False
    c_reshape = c_reshape[:,mask]
    print(F'C shape: {c_reshape.shape}')
    print(F'Z shape: {z_reshape.shape}')
    print(F'Shuffle: {FLAGS.shuffle}')

    c_train, c_test, z_train, z_test = train_test_split(c_reshape, z_reshape, test_size=0.2, shuffle=FLAGS.shuffle, random_state=FLAGS.dci_seed)

    if FLAGS.data_type_dci == "hirid":
        n_train = 20000
        n_test = 5000
    else:
        n_train = 8000
        n_test = 2000

    importance_matrix, i_train, i_test = dci.compute_importance_gbt(
        z_train[:n_train, :].transpose(),
        c_train[:n_train, :].transpose().astype(int),
        z_test[:n_test, :].transpose(), c_test[:n_test, :].transpose().astype(int))
    # Calculate scores
    d = dci.disentanglement(importance_matrix)
    c = dci.completeness(importance_matrix)
    print(F'D: {d}')
    print(F'C: {c}')
    print(F'I: {i_test}')

    if FLAGS.data_type_dci in ['hirid', 'physionet']:
        miss_idxs = np.nonzero(np.invert(mask))[0]
        for idx in miss_idxs:
            importance_matrix = np.insert(importance_matrix,
                                          idx,
                                          0, axis=1)
        assign_mat = np.load(FLAGS.assign_mat_path)
        impt_mat_assign = np.matmul(importance_matrix, assign_mat)
        impt_mat_assign_norm = np.nan_to_num(
            impt_mat_assign / np.sum(impt_mat_assign, axis=0))
        d_assign = dci.disentanglement(impt_mat_assign_norm)
        c_assign = dci.completeness(impt_mat_assign_norm)
        print(F'D assign: {d_assign}')
        print(F'C assign: {c_assign}')

    if FLAGS.save_score:
        if FLAGS.data_type_dci in ['hirid', 'physionet']:
            np.savez(F'{out_dir}/dci_assign_2_{FLAGS.dci_seed}', informativeness_train=i_train, informativeness_test=i_test,
                     disentanglement=d, completeness=c,
                     disentanglement_assign=d_assign, completeness_assign=c_assign)
        else:
            np.savez(F'{out_dir}/dci_{FLAGS.dci_seed}', informativeness_train=i_train, informativeness_test=i_test,
                     disentanglement=d, completeness=c)

    # Visualization
    if FLAGS.visualize_score:
        if FLAGS.data_type_dci == 'hirid':
            # Visualize
            visualize_scores.heat_square(np.transpose(importance_matrix), out_dir,
                                         F"dci_matrix_{FLAGS.dci_seed}",
                                         "feature", "latent dim")
            visualize_scores.heat_square(np.transpose(impt_mat_assign_norm), out_dir,
                                         F"dci_matrix_assign_{FLAGS.dci_seed}",
                                         "feature", "latent_dim")

            # Save importance matrices
            if FLAGS.save_score:
                np.save(F"{out_dir}/impt_matrix_{FLAGS.dci_seed}", importance_matrix)
                np.save(F"{out_dir}/impt_matrix_assign_{FLAGS.dci_seed}", impt_mat_assign_norm)

        else:
            # Visualize
            visualize_scores.heat_square(importance_matrix, out_dir,
                                         F"dci_matrix_{FLAGS.dci_seed}",
                                         "x_axis", "y_axis")
            # Save importance matrices
            np.save(F"{out_dir}/impt_matrix_{FLAGS.dci_seed}", importance_matrix)

    print("Evaluation finished")


if __name__ == '__main__':
    app.run(main)
