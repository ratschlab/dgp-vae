"""
Classifier for downstream proxy task of hirid representations.
"""

import numpy as np
import os
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_string('labels_path', '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/mort_labels_test.npy', 'Hirid classification labels')
flags.DEFINE_string('representation_path', '/cluster/home/bings/dgpvae/models/hirid/comp/base/dim_8/len_25/scaled/n_scales_4/210126_n_1', 'Path to latent representation that are used as input features.')

def main(argv):
    del argv # Unused

    # Load labels and representations
    labels_full = np.load(FLAGS.labels_path)
    reps_path = os.path.join(FLAGS.representation_path, 'z_mean.npy')
    reps_full = np.load(reps_path)
    # Reshape representations
    reps_full_re = np.reshape(reps_full, (labels_full.shape[0], reps_full.shape[1], -1))

    print(labels_full.shape)
    print(reps_full.shape)
    print(reps_full_re.shape)

if __name__ == '__main__':
    app.run(main)
