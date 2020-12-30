"""
Simon Bing ETHZ, 2020

Script to compute factor VAE score (or something close) of learned representation.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_smpls_var', 10000, 'Number of samples for empirical variance estimation.')
flags.DEFINE_string('z_path', '', 'Path for latent representations.')

def compute_variance(representations, num_samples):
    """
    Args:
        representations: Latent representations of input features.
        num_samples: Number of samples to consider.

    Returns:
        Vector with the variance of each latent dimension.
    """
    representations = representations[:num_samples, ...]
    return np.var(representations, axis=0, ddof=1)

def prune_dims(variances, threshold=0.05):
  """Mask for dimensions collapsed to the prior."""
  scale_z = np.sqrt(variances)
  return scale_z >= threshold

def generate_training_sample(feature_batch, variances, active_dims):
    # Encode features to get latent representation
    representations = 0
    # Compute variances of representations
    rep_variances = np.var(representations, axis=0, ddof=1)
    # Get argmin of variances masked with active_dims
    argmin = np.argmin(rep_variances[active_dims] / variances[active_dims])

    return argmin

def generate_training_batch(num_features, num_samples, feature_idxs,
                            features_batches, variances, active_dims):
    votes = np.zeros((num_features, variances.shape[0]), dtype=np.int64)

    for i in range(num_samples):
        argmin = generate_training_sample(features_batches[i], variances, active_dims)
        votes[feature_idxs[i], argmin] += 1
    return votes

def main(argv):
    del argv #Unused

    z = np.load(FLAGS.z_path)
    # Estimate variance empirically
    variances = compute_variance(z, FLAGS.num_smpls_var)
    # Prune collapsed dimensions
    active_dims = prune_dims(variances)

    # Generate training votes
    training_votes = generate_training_batch(#TODO)


if __name__ == '__main__':
    app.run(main)

