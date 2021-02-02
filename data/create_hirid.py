"""
Script to synthesize chunks of time series from the HiRID data in the merged stage.
"""

import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('hirid_merged_dir', '/hirid/partitions', 'Base directory of hirid (merged) pq partitions.')
flags.DEFINE_string('out_dir', 'hirid', 'Where to save data.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

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

    return data_standardized

def main(argv):
    del argv

    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    # Aggregate partitions into one dataframe
    base_dir = Path(FLAGS.hirid_merged_dir)
    full_df = pd.concat(pd.read_parquet(pq_file) for pq_file in base_dir.glob('*.parquet'))

    # Convert to np array and find unique patients
    full_np = full_df.to_numpy(dtype=np.float32)
    unique_patients, idxs, counts = np.unique(full_np[:, 0], return_index=True, return_counts=True)

    # Filter out patients below certain threshold and reshape for later training
    filtered_np = filter_and_reshape(full_np, idxs, counts, time_len=100)

    # Standardize data and remove ids and timestamps
    data_std = pre_process(filtered_np)

    data_std_train, data_std_test = train_test_split(data_std, train_size=(100 / 105), random_state=FLAGS.seed)
    # Save data
    out_path = os.path.join(FLAGS.out_dir, 'hirid.npz')
    np.savez(out_path, x_train_full=data_std_train,
             x_train_miss=data_std_train,
             m_train_miss=np.zeros_like(data_std_train),
             x_test_full=data_std_test,
             x_test_miss=data_std_test,
             m_test_miss=np.zeros_like(data_std_test))

    print(F'Data successfully created and saved at: {out_path}')

if __name__ == '__main__':
    app.run(main)