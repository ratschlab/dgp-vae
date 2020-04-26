"""
BetaTCVAE training script for dsprites dataset, using disentanglement_lib.
Also evaluates DCI metric and saves outputs.

Simon Bing
ETHZ 2020
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import tensorflow as tf
import gin.tf
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', 'test_output', 'Directory to save results in')
flags.DEFINE_integer('dim', 32, 'Latent dimension of encoder')
flags.DEFINE_string('subset', "", 'Subset of factors of tested dataset')
flags.DEFINE_integer('seed', 42, 'Seed for the random number generator')

def main(argv):
    del argv # Unused

    # Save all results in subdirectories of following path
    base_path = 'baselines/betatcvae/dim_{}_subset_{}'.format(FLAGS.dim, FLAGS.subset)

    # Overwrite output or not (for rerunning script)
    overwrite = True

    # Results directory of BetaTCVAE
    path_btcvae = os.path.join(base_path,FLAGS.output_dir)

    gin_bindings = [
        "model.random_seed = {}".format(FLAGS.seed),
        "subset.name = '{}'".format(FLAGS.subset),
        "encoder.num_latent = {}".format(FLAGS.dim)
    ]
    # Train model. Training is configured with a gin config
    train.train_with_gin(os.path.join(path_btcvae, 'model'), overwrite,
                         [os.path.realpath('baselines/betatcvae/btcvae_train.gin')], gin_bindings)

    # Extract mean representation of latent space
    representation_path = os.path.join(path_btcvae, "representation")
    model_path = os.path.join(path_btcvae, "model")
    postprocess_gin = [os.path.realpath('baselines/betatcvae/btcvae_postprocess.gin')]  # This contains the settings.
    postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                     postprocess_gin)

    # Compute DCI metric
    result_path = os.path.join(path_btcvae, "metrics", "dci")
    representation_path = os.path.join(path_btcvae, "representation")
    evaluate.evaluate_with_gin(representation_path, result_path, overwrite, [os.path.realpath('baselines/betatcvae/btcvae_dci.gin')])

if __name__ == '__main__':
    app.run(main)
