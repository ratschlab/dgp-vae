"""
Simon Bing ETHZ, 2020
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from absl import flags, app
from sklearn.model_selection import train_test_split
from disentanglement_lib.evaluation.metrics import dci

FLAGS = flags.FLAGS

flags.DEFINE_string('z_name', '', 'Filename for underlying factors z')
flags.DEFINE_string('model_name', '', 'Name of model directory to get learned latent code')

def load_z_c(z_path, c_path):
    z_full = np.load(z_path)
    c = np.load(c_path)

    # Check length of c and only take same amount of z values. Corresponds to z_test.
    z = z_full[:c.shape[0],:,:]
    assert z.shape[0] == c.shape[0]

    return z, c

def main(argv):
    del argv # Unused

    # if FLAGS.z_name == '':
    #     z_path = 'data/dsprites/factors_5000.npy'
    # else:
    #     z_path = 'data/dsprites/{}'.format(FLAGS.z_name)
    z_path = 'data/dsprites/{}'.format(FLAGS.z_name)
    c_path = '{}/z_mean.npy'.format(FLAGS.model_name)

    z, c = load_z_c(z_path, c_path)

    # flags.DEFINE_string('data_dir', "", 'Directory from where the data should be read in')
    # flags.DEFINE_boolean('save_score', False, 'Save scores')
    #
    # if FLAGS.data_dir == "":
    #     FLAGS.data_dir = "data/z_c_5000.npz"
    # data = np.load(FLAGS.data_dir)

    # z = data['z']
    # c = data['c']

    z_shape = z.shape
    c_shape = c.shape

    z_reshape = np.reshape(np.transpose(z, (0,2,1)),(z_shape[0]*z_shape[2],z_shape[1]))
    c_reshape = np.reshape(np.transpose(c, (0,2,1)),(c_shape[0]*c_shape[2],c_shape[1]))

    # Check if latent factor doesn't change and remove if is the case
    mask = np.ones(z_reshape.shape[1], dtype=bool)
    for i in range(z_reshape.shape[1]):
        z_change = np.sum(np.diff(z_reshape[:,i]))
        if not z_change:
            mask[i] = False
    z_reshape = z_reshape[:,mask]

    c_train, c_test, z_train, z_test = train_test_split(c_reshape, z_reshape, test_size=0.2, shuffle=False)
    scores = dci._compute_dci(c_train.transpose(), z_train.transpose(), c_test.transpose(), z_test.transpose())

    save_score = True
    if save_score:
        np.savez('{}/dci_{}_{}_{}'.format(FLAGS.model_name, z_shape[1], c_shape[1], z_shape[0]),
                 informativeness_train=scores['informativeness_train'],
                 informativeness_test=scores['informativeness_test'],
                 disentanglement=scores['disentanglement'],
                 completeness=scores['completeness'])

if __name__ == '__main__':
    app.run(main)
