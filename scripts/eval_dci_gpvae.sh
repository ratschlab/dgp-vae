#!/bin/bash

# Evaluate DCI metric for gpvae models

for len in 1e-1 5e-1 1 2 4 8; do
  for base_dir in models/len/dsprites_len_"$len"_1/*; do
#    bsub -g /gpvae_norm -R "rusage[mem=16000]" python dsprites_dci.py --z_name factors_100k_5k.npz \
#    --model_name "$base_dir"
     echo AHHHHHHHHHHHH
     echo $"base_dir"
  done
done
