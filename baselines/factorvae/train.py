"""
Factor VAE training script for dsprites dataset, using disentanglement_lib.
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

def main(argv):
    del argv # Unused

    # Save all results in subdirectories of following path
    base_path = FLAGS.output_dir

    # Overwrite output or not (for rerunning script)
    overwrite = True

    # Results directory of Factor VAE
    path_btcvae = os.path.join(base_path,'factorvae')

    # Train model. Training is configured with a gin config
    train.train_with_gin(os.path.join(path_btcvae, 'model'), overwrite, ['factorvae_train.gin'])

    # Extract mean representation of latent space
    representation_path = os.path.join(path_btcvae, "representation")
    model_path = os.path.join(path_btcvae, "model")
    postprocess_gin = ["factorvae_postprocess.gin"]  # This contains the settings.
    postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                     postprocess_gin)

    # Compute DCI metric
    result_path = os.path.join(path_btcvae, "metrics", "dci")
    representation_path = os.path.join(path_btcvae, "representation")
    evaluate.evaluate_with_gin(representation_path, result_path, overwrite, ['factorvae_dci.gin'])

if __name__ == '__main__':
    app.run(main)