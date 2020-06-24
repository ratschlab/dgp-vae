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

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.postprocessing import postprocess
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'betatcvae', 'Baseline to train')
flags.DEFINE_string('base_dir', 'base', 'Base directory')
flags.DEFINE_string('output_dir', 'test_output', 'Directory to save results in')
flags.DEFINE_integer('dim', 64, 'Latent dimension of encoder')
flags.DEFINE_string('subset', "", 'Subset of factors of tested dataset')
flags.DEFINE_integer('seed', 42, 'Seed for the random number generator')
flags.DEFINE_integer('steps', 15620, 'Training steps')

def main(argv):
    del argv # Unused

    baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             FLAGS.model)

    # Save all results in subdirectories of following path
    base_path = os.path.join(baseline_path, FLAGS.base_dir)

    # Overwrite output or not (for rerunning script)
    overwrite = True

    # Results directory of BetaTCVAE
    output_path = os.path.join(base_path, FLAGS.output_dir)

    gin_bindings = [
        "model.random_seed = {}".format(FLAGS.seed),
        "subset.name = '{}'".format(FLAGS.subset),
        "encoder.num_latent = {}".format(FLAGS.dim),
        "model.training_steps = {}".format(FLAGS.steps)
    ]
    # Train model. Training is configured with a gin config
    train.train_with_gin(os.path.join(output_path, 'model'), overwrite,
                         [os.path.join(baseline_path, '{}_train.gin'.format(FLAGS.model))],
                         gin_bindings)

    # Extract mean representation of latent space
    representation_path = os.path.join(output_path, "representation")
    model_path = os.path.join(output_path, "model")
    postprocess_gin = [os.path.join(baseline_path, '{}_postprocess.gin'.format(FLAGS.model))]
    postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                     postprocess_gin)

    # Compute DCI metric
    result_path = os.path.join(output_path, "metrics", "dci")
    representation_path = os.path.join(output_path, "representation")
    evaluate.evaluate_with_gin(representation_path, result_path, overwrite,
                               [os.path.join(baseline_path, '{}_dci.gin'.format(FLAGS.model))])

if __name__ == '__main__':
    app.run(main)