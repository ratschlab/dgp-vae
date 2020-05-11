"""
Unified baseline training script

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
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.postprocessing import postprocess
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'betatcvae', 'Baseline to train')
flags.DEFINE_string('base_dir', 'base', 'Base directory')
flags.DEFINE_string('output_dir', 'test_output', 'Directory to save results in')
flags.DEFINE_integer('dim', 32, 'Latent dimension of encoder')
flags.DEFINE_string('subset', "", 'Subset of factors of tested dataset')
flags.DEFINE_integer('seed', 42, 'Seed for the random number generator')

def main(argv):
    del argv # Unused

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             FLAGS.model)

    # Save all results in subdirectories of following path
    base_path = os.path.join(file_path, FLAGS.base_dir)

    # Overwrite output or not (for rerunning script)
    overwrite = True

    # Results directory of BetaTCVAE
    path_baseline = os.path.join(base_path,FLAGS.output_dir)

    gin_bindings = [
        "model.random_seed = {}".format(FLAGS.seed),
        "subset.name = '{}'".format(FLAGS.subset),
        "encoder.num_latent = {}".format(FLAGS.dim)
    ]
    # Train model. Training is configured with a gin config
    train.train_with_gin(os.path.join(path_baseline, 'model'), overwrite,
                         ['baselines/adagvae/adagvae_train.gin'], gin_bindings)

    # Extract mean representation of latent space
    representation_path = os.path.join(path_baseline, "representation")
    model_path = os.path.join(path_baseline, "model")
    postprocess_gin = ['baselines/adagvae/adagvae_postprocess.gin']  # This contains the settings.
    postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                     postprocess_gin)

    # Compute DCI metric
    result_path = os.path.join(path_baseline, "metrics", "dci")
    representation_path = os.path.join(path_baseline, "representation")
    evaluate.evaluate_with_gin(representation_path, result_path, overwrite, ['baselines/adagvae/adagvae_dci.gin'])

if __name__ == '__main__':
    app.run(main)