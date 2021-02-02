"""
Script to synthesize observational data from ground truth factors.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
from disentanglement_lib.data.ground_truth import dsprites, norb, cars3d, shapes3d

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_enum('data_type', 'dsprites', ['dsprites', 'smallnorb', 'cars3d', 'shapes3d'], 'Data set type.')
flags.DEFINE_string('factors_path', '', 'Path to where factors are saved.')
flags.DEFINE_string('out_dir', '', 'Directory to where to save data to.')
flags.DEFINE_integer('seed', 42, 'Random seed.')


def create_data(factors):
    """
    :param factors: underlying factors of variation
    :return: data: obervational data from underlying factors
    """
    if FLAGS.data_type == "dsprites":
        dsp = dsprites.DSprites()
    elif FLAGS.data_type == "smallnorb":
        snb = norb.SmallNORB()
    elif FLAGS.data_type == "cars3d":
        cars = cars3d.Cars3D()
    elif FLAGS.data_type == "shapes3d":
        shp = shapes3d.Shapes3D()

    random_state = np.random.RandomState(FLAGS.seed)

    factors_train = np.transpose(factors['factors_train'], (0,2,1))
    factors_test = np.transpose(factors['factors_test'], (0,2,1))

    N_train = factors_train.shape[0]
    N_test = factors_test.shape[0]
    time_len = factors_train.shape[1]

    if FLAGS.data_type in ["dsprites", "smallnorb"]:
        data_train = np.zeros([N_train, time_len, 64 * 64])
        data_test = np.zeros([N_test, time_len, 64 * 64])
    elif FLAGS.data_type in ["cars3d", "shapes3d"]:
        data_train = np.zeros([N_train, time_len, 64 * 64 * 3])
        data_test = np.zeros([N_test, time_len, 64 * 64 * 3])

    # Training data
    for i in range(N_train):
        if FLAGS.data_type == "dsprites":
            data_point_train = np.squeeze(dsp.sample_observations_from_factors_no_color(
                factors=factors_train[i, :, :], random_state=random_state))
            data_train_reshape = data_point_train.reshape(data_point_train.shape[0], 64 * 64)
        elif FLAGS.data_type == "smallnorb":
            data_point_train = np.squeeze(snb.sample_observations_from_factors(
                factors=factors_train[i, :, :], random_state=random_state))
            data_train_reshape = data_point_train.reshape(data_point_train.shape[0], 64 * 64)
        elif FLAGS.data_type == "cars3d":
            data_point_train = cars.sample_observations_from_factors(
                factors=factors_train[i, :, :],
                random_state=random_state)
            data_train_reshape = data_point_train.reshape(data_point_train.shape[0], 64 * 64 * 3)
        elif FLAGS.data_type == "shapes3d":
            data_point_train = shp.sample_observations_from_factors(
                factors=factors_train[i, :, :],
                random_state=random_state)
            data_train_reshape = data_point_train.reshape(data_point_train.shape[0], 64 * 64 * 3)
        data_train[i, :, :] = data_train_reshape

    # Test data
    for i in range(N_test):
        if FLAGS.data_type == "dsprites":
            data_point_test = np.squeeze(dsp.sample_observations_from_factors_no_color(
                factors=factors_test[i, :, :], random_state=random_state))
            data_test_reshape = data_point_test.reshape(data_point_test.shape[0], 64 * 64)
        elif FLAGS.data_type == "smallnorb":
            data_point_test = np.squeeze(snb.sample_observations_from_factors(
                factors=factors_test[i, :, :], random_state=random_state))
            data_test_reshape = data_point_test.reshape(data_point_test.shape[0], 64 * 64)
        elif FLAGS.data_type == "cars3d":
            data_point_test = cars.sample_observations_from_factors(
                factors=factors_test[i, :, :],
                random_state=random_state)
            data_test_reshape = data_point_test.reshape(data_point_test.shape[0], 64 * 64 * 3)
        elif FLAGS.data_type == "shapes3d":
            data_point_test = shp.sample_observations_from_factors(
                factors=factors_test[i, :, :],
                random_state=random_state)
            data_test_reshape = data_point_test.reshape(data_point_test.shape[0], 64 * 64 * 3)
        data_test[i, :, :] = data_test_reshape

    return data_train.astype('float32'), data_test.astype('float32')

def main(argv):
    del argv

    if FLAGS.out_dir == '':
        out_dir = FLAGS.data_type
    else:
        out_dir = FLAGS.out_dir

    if FLAGS.factors_path == '':
        factors_path = os.path.join(FLAGS.data_type, F'factors_{FLAGS.data_type}.npz')
    else:
        factors_path = FLAGS.factors_path

    # Load factors
    factors_full = np.load(factors_path)
    # Synthesize observational data from factors
    data_train, data_test = create_data(factors_full)
    # Save data
    save_path = os.path.join(out_dir, F'{FLAGS.data_type}.npz')
    np.savez(save_path, x_train_full=data_train, x_train_miss=data_train,
             m_train_miss=np.zeros_like(data_train), x_test_full=data_test,
             x_test_miss=data_test, m_test_miss=np.zeros_like(data_test))

    print(F'Data set successfully created and saved at: {save_path}')


if __name__ == '__main__':
    app.run(main)
