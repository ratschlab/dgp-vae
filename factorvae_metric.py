"""
Simon Bing ETHZ, 2020

Script to compute factor VAE score (or something close) of learned representation.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from absl import flags, app
import os
from disentanglement_lib.visualize import visualize_scores

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_smpls_var', 10000, 'Number of samples for empirical variance estimation.')
flags.DEFINE_string('model_dir', '', 'Path to model and output files.')
flags.DEFINE_string('eval_dir', '', 'Path to metric evaluation data.')
flags.DEFINE_bool('visualize', False, 'Plot matrices of sensitivity analysis.')
flags.DEFINE_bool('save', False, 'Save visalization plots and scores.')
flags.DEFINE_string('assign_mat', '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/physionet/assignment_matrix.npy', 'Path for assignment matrix')


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

def generate_training_sample(representation_batch, variances, active_dims):
    # Compute variances of representations
    rep_variances = np.var(representation_batch, axis=1, ddof=1)
    # Get argmin of variances masked with active_dims
    argmin = np.argmin(rep_variances[active_dims] / variances[active_dims])

    return argmin

def generate_training_batch(num_features, num_samples, feature_idxs,
                            representation_batches, variances, active_dims):
    votes = np.zeros((num_features, variances.shape[0]), dtype=np.int64)

    for i in range(num_samples):
        argmin = generate_training_sample(representation_batches[i], variances, active_dims)
        votes[feature_idxs[i], argmin] += 1
    return votes

def main(argv, model_dir=None):
    del argv # Unused

    if model_dir is None:
        out_dir = FLAGS.model_name
    else:
        out_dir = model_dir

    z_var_path = os.path.join(out_dir, "z_mean.npy")
    z_var = np.load(z_var_path)
    z_eval_path = os.path.join(out_dir, "z_eval_mean.npy")
    z_eval = np.load(z_eval_path)

    data_eval = np.load(FLAGS.eval_dir)
    idxs_eval = data_eval['feature_idxs']

    # Reshape representations
    z_var = np.reshape(z_var, (z_var.shape[0] * z_var.shape[2], z_var.shape[1]))
    # Estimate variance empirically
    variances = compute_variance(z_var, FLAGS.num_smpls_var)
    # Prune collapsed dimensions
    active_dims = prune_dims(variances)
    scores = {}
    if not active_dims.any():
        scores["train_accuracy"] = 0.
        scores["eval_accuracy"] = 0.
        scores["num_active_dims"] = 0
        np.save('sensitivity_score.npy', scores)
        return

    # Generate training votes
    # n_samples = min(z_eval.shape[0], 10000)
    n_samples = z_eval.shape[0]
    training_votes = generate_training_batch(num_features=18, num_samples=n_samples,
                                             feature_idxs= idxs_eval, representation_batches=z_eval,
                                             variances=variances, active_dims=active_dims)
    # Aggregate votes according to underlying organs systems
    # assign_mat = np.load(FLAGS.assign_mat)
    # training_votes_aggregate = np.matmul(np.transpose(assign_mat), training_votes)

    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])
    train_accuracy = np.sum(
        training_votes[classifier, other_index]) * 1. / np.sum(training_votes)

    # classifier_aggr = np.argmax(training_votes_aggregate, axis=0)
    # other_index_aggr = np.arange(training_votes_aggregate.shape[1])
    # train_accuracy_aggr = np.sum(
    #     training_votes_aggregate[classifier_aggr, other_index_aggr]) * 1. / np.sum(training_votes_aggregate)

    if FLAGS.visualize:
        visualize_scores.heat_square(training_votes, out_dir, 'sensitivity',
                                     'feature', 'latent dim')
        # visualize_scores.heat_square(training_votes_aggregate, out_dir, 'sensitivity_aggr',
        #                              'organ system', 'latent dim')
        if FLAGS.save:
            np.save(F"{out_dir}/sens_matrix", training_votes)
            # np.save(F"{out_dir}/sens_matrix_aggr", training_votes_aggregate)

    # TODO: add eval accuracy, save score, do summation with assignment matrix

    print(F"Train accuracy: {train_accuracy}")
    # print(F"Train accuracy aggregate: {train_accuracy_aggr}")


if __name__ == '__main__':
    app.run(main)

