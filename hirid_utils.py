"""
Simon Bing ETHZ, 2021

Script to aggregate and maniplate HiRID data.
"""
import numpy as np
import os
import pandas as pd
from pathlib import Path
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', '/cluster/work/grlab/clinical/hirid_public/v1/imputed_stage',
                    'Base directory of pq partitions.')

def aggregate_pq(base_dir):
    base_dir = Path(base_dir)
    full_df = pd.concat(pd.read_parquet(pq_file) for pq_file in base_dir.glob('*.parquet'))

    return full_df

def filter_and_reshape(data, idxs, counts, patients, time_len):
    """
    Throw out time series below len threshold and reshape for later training.
    """
    min_len_idxs = np.where(counts >= time_len)
    idxs_filter = idxs[min_len_idxs]
    counts_filter = counts[min_len_idxs]
    patients_filter = patients[min_len_idxs]

    # Calculate total number of samples
    N = np.sum(counts_filter//time_len)
    data_filter_reshape = np.zeros((N, time_len, data.shape[1]), dtype=np.float32)

    k=0
    for i in range(len(idxs_filter)):
        start_idx = idxs_filter[i]
        num_chunks = counts_filter[i] // time_len
        for j in range(num_chunks):
            chunk = data[start_idx:start_idx+time_len,:]
            assert np.sum(np.diff(chunk[:,0])) == 0
            data_filter_reshape[k,:,:] = chunk
            start_idx = start_idx + time_len
            k = k + 1
    print(F'Huge loop finished. N={N}, k={k}')

    return data_filter_reshape

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

    filtered_np = filter_and_reshape(full_np, idxs, counts, unique_patients, time_len=100)

if __name__ == '__main__':
    app.run(main)