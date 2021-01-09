"""
Simon Bing ETHZ, 2021

Script to aggregate and maniplate HiRID data.
"""
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', '/cluster/work/grlab/clinical/hirid_public/v1/imputed_stage',
                    'Base directory of pq partitions.')
flags.DEFINE_bool('save', False, 'Save data.')
flags.DEFINE_string('out_dir', '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid', 'Where to save data.')

def aggregate_pq(base_dir):
    base_dir = Path(base_dir)
    full_df = pd.concat(pd.read_parquet(pq_file) for pq_file in base_dir.glob('*.parquet'))

    return full_df

def filter_and_reshape(data, idxs, counts, time_len):
    """
    Throw out time series below len threshold and reshape for later training.
    """
    min_len_idxs = np.where(counts >= time_len)
    idxs_filter = idxs[min_len_idxs]
    counts_filter = counts[min_len_idxs]

    # Calculate total number of samples
    N = np.sum(counts_filter//time_len)
    data_filter_reshape = np.zeros((N, time_len, data.shape[1]), dtype=np.float32)

    k=0
    for i in range(len(idxs_filter)):
        start_idx = idxs_filter[i]
        num_chunks = counts_filter[i] // time_len
        for j in range(num_chunks):
            chunk = data[start_idx:start_idx+time_len,:]
            assert np.sum(np.diff(chunk[:,0])) == 0 # Check that patient id really is shared i.e. one coherent time series
            data_filter_reshape[k,:,:] = chunk
            start_idx = start_idx + time_len
            k = k + 1
    print(F'Filtered and reshaped data shape: {data_filter_reshape.shape}')

    return data_filter_reshape

def pre_process(data):
    # Remove patient_id and timestamp
    data_no_id = data[:,:,2:]
    # Standardize data
    means = np.mean(data_no_id, axis=(0,1))
    std_devs = np.std(data_no_id, axis=(0,1))
    data_standardized = (data_no_id - means) / std_devs
    data_standardized[:,:,-1] = data_no_id[:,:,-1] # Last feature is binary

    return data_no_id, data_standardized

def main(argv):
    del argv # Unused

    full_df = aggregate_pq(FLAGS.base_dir)
    # Convert to np array
    full_np = full_df.to_numpy(dtype=np.float32)
    unique_patients, idxs, counts = np.unique(full_np[:,0], return_index=True, return_counts=True)

    print(F'Unique idxs: {len(unique_patients)}')
    print(F'Min time series len: {np.min(counts)}')
    print(F'Max time series len: {np.max(counts)}')

    min_len_idxs = np.where(counts >= 100)
    min_len_patients = unique_patients[min_len_idxs]
    print(F'Number of min len patients: {len(min_len_patients)}')

    filtered_np = filter_and_reshape(full_np, idxs, counts, time_len=100)

    data_raw, data_std = pre_process(filtered_np)
    print(data_raw.shape)
    print(data_std.shape)

    # Eval data for FactorVAE metric
    N = data_raw.shape[0]
    n_feats = data_raw.shape[2]
    feature_idxs_raw = np.zeros(N, dtype=np.int64)
    feature_batches_raw = np.copy(data_raw)
    feature_idxs_std = np.zeros(N, dtype=np.int64)
    feature_batches_std = np.copy(data_std)
    for i in range(N):
        idx = np.random.choice(n_feats)
        feature_idxs_raw[i] = idx
        feature_idxs_std[i] = idx
        feature_batches_raw[i, :, idx] = feature_batches_raw[i, 0, idx]
        feature_batches_std[i, :, idx] = feature_batches_std[i, 0, idx]

    if FLAGS.save:
        # full_save_path = os.path.join(FLAGS.out_dir, 'hirid_full.npy')
        # np.save(full_save_path, full_np)
        # filter_reshape_save_path = os.path.join(FLAGS.out_dir, 'hirid_filter_reshape.npy')
        # np.save(filter_reshape_save_path, filtered_np)

        # data_raw_train, data_raw_test, data_std_train, data_std_test = train_test_split(data_raw, data_std, train_size=(100/105))
        # print(data_raw_train.shape)
        # print(data_raw_test.shape)
        # print(data_std_train.shape)
        # print(data_std_test.shape)
        # raw_path = os.path.join(FLAGS.out_dir, 'hirid_no_std.npz')
        # np.savez(raw_path, x_train_full=data_raw_train,
        #          x_train_miss=data_raw_train,
        #          m_train_miss=np.zeros_like(data_raw_train),
        #          x_test_full=data_raw_test,
        #          x_test_miss=data_raw_test,
        #          m_test_miss=np.zeros_like(data_raw_test))
        # std_path = os.path.join(FLAGS.out_dir, 'hirid_std.npz')
        # np.savez(std_path, x_train_full=data_std_train,
        #          x_train_miss=data_std_train,
        #          m_train_miss=np.zeros_like(data_std_train),
        #          x_test_full=data_std_test,
        #          x_test_miss=data_std_test,
        #          m_test_miss=np.zeros_like(data_std_test))

        eval_path_no_std = os.path.join(FLAGS.out_dir, 'hirid_sensitivity_eval_no_std.npz')
        np.savez(eval_path_no_std,
                 feature_batches=feature_batches_raw,
                 feature_idxs=feature_idxs_raw)
        eval_path_std = os.path.join(FLAGS.out_dir,
                                        'hirid_sensitivity_eval_std.npz')
        np.savez(eval_path_std,
                 feature_batches=feature_batches_std,
                 feature_idxs=feature_idxs_std)


if __name__ == '__main__':
    app.run(main)