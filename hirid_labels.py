import numpy as np
import pandas as pd
import os

hirid_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid'

hirid_full = np.load(os.path.join(hirid_path, 'hirid_filter_reshape.npy'))
hirid_shuffle = np.load(os.path.join(hirid_path, 'hirid_no_std.npz'))

hirid_shuffle_test = hirid_shuffle['x_test_full']

perm = np.zeros(len(hirid_shuffle_test))
print(F'Permutation list len: {len(perm)}')

for i in range(len(perm)):
    test_arr = hirid_shuffle_test[i,:,:]
    try:
        idx = np.where(np.all(test_arr==hirid_full[:,:,2:], axis=(1,2)))[0]
        perm[i] = idx
    except:
        print(F'Failed at step {i}!')
        print(F'Rsult of where: {np.where(np.all(test_arr==hirid_full[:,:,2:], axis=(1,2)))}')

np.save(os.path.join(hirid_path, 'hirid_test_permutation.npy'), perm)
