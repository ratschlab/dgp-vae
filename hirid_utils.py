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
    full_df = pd.concat(pd.read_parquet(pq_file for pq_file in base_dir.glob('*.parquet')))

    return full_df

def main(argv):
    del argv # Unused

    full_df = aggregate_pq(FLAGS.base_dir)
    # Convert to np array
    full_np = full_df.to_numpy(dtype=np.float32)
    unique_idxs, counts = np.unique(full_np[:,0], return_counts=True)

    print(F'Unique idxs: {len(unique_idxs)}')
    print(F'Min time series len: {np.min(counts)}')
    print(F'Max time series len: {np.max(counts)}')

if __name__ == '__main__':
    app.run(main)