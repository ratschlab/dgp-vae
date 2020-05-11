#!/bin/bash

# Evaluate DCI metric for gpvae models

for dim in 6 16 32 64 128; do
  for base_dir in models/dim/dsprites_dimsweep_"$dim"_0/*; do
    bsub -g /gpvae_norm -R "rusage[mem=16000]" python dsprites_dci.py --z_name factors_100k_5k.npz \
    --model_name "$base_dir"
  done
done
