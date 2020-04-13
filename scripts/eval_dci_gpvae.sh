#!/bin/bash

# Evaluate DCI metric for gpvae models

for dim in 8; do
  for base_dir in models/dsprites_dim_$dim/; do
    bsub -R "rusage[mem=16000]" python dsprites_dci.py --z_name factors_5000.npy \
    --model_name /dsprites_dim_$dim/$base_dir
  done
done
