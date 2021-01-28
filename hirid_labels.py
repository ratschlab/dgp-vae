import numpy as np
from absl import flags, app
import os

FLAGS = flags.FLAGS

flags.DEFINE_integer('n', 0, 'Slice of test set.')

def main(argv):
    del argv

    hirid_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid'

    hirid_full = np.load(os.path.join(hirid_path, 'hirid_filter_reshape.npy'))
    hirid_shuffle = np.load(os.path.join(hirid_path, 'hirid_no_std.npz'))

    hirid_shuffle_test = hirid_shuffle['x_test_full']

    perm_slice = np.zeros(100)
    start_idx = FLAGS.n * 100
    end_idx = start_idx + 100
    idxs = np.arange(start_idx, end_idx)

    for i in range(100):
        test_arr = hirid_shuffle_test[idxs[i], :, :]
        try:
            idx = np.where(np.all(test_arr==hirid_full[:,:,2:], axis=(1,2)))[0]
            perm_slice[i] = idx
        except:
            print(F'Failed at step {i}!')
            print(F'Result of where: {np.where(np.all(test_arr==hirid_full[:,:,2:], axis=(1,2)))}')


    # perm = np.zeros(len(hirid_shuffle_test))
    # print(F'Permutation list len: {len(perm)}')
    #
    # for i in range(len(perm)):
    #     test_arr = hirid_shuffle_test[i,:,:]
    #     try:
    #         idx = np.where(np.all(test_arr==hirid_full[:,:,2:], axis=(1,2)))[0]
    #         perm[i] = idx
    #     except:
    #         print(F'Failed at step {i}!')
    #         print(F'Rsult of where: {np.where(np.all(test_arr==hirid_full[:,:,2:], axis=(1,2)))}')

    np.save(os.path.join(hirid_path, 'perm_slices', F'perm_slice_{FLAGS.n:03d}.npy'), perm_slice)


if __name__ == '__main__':
    app.run(main)