#!/bin/bash

# Evaluate DCI metric for gpvae models

for epochs in 1 4; do
  for base_dir in models/epoch/dsprites_epochs_"$epochs"_sweep0/*; do
    bsub -g /gpvae_norm -R "rusage[mem=16000]" python dsprites_dci.py --z_name factors_100k_5k.npz \
    --model_name "$base_dir"
  done
done
