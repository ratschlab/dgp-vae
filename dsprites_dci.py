"""
Simon Bing ETHZ, 2020
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from absl import flags, app
from sklearn.model_selection import train_test_split
from disentanglement_lib.evaluation.metrics import dci

def main(argv):
    del argv # Unused

    FLAGS = flags.FLAGS

    flags.DEFINE_string('data_dir', "", 'Directory from where the data should be read in')
    flags.DEFINE_boolean('save_score', False, 'Save scores')

    if FLAGS.data_dir == "":
        FLAGS.data_dir = "data/z_c_5000.npz"
    data = np.load(FLAGS.data_dir)

    z = data['z']
    c = data['c']

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
        np.savez('data/dci_scores/dci_{}_{}_{}.npz'.format(z_shape[1], c_shape[1], z_shape[0]),
                 informativeness_train=scores['informativeness_train'],
                 informativeness_test=scores['informativeness_test'],
                 disentanglement=scores['disentanglement'],
                 completeness=scores['completeness'])

if __name__ == '__main__':
    app.run(main)
