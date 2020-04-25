#!/bin/bash

# Create directory for each tested latent dimension
for len in 1e-1 5e-1 1 2 4 8; do
  mkdir -p models/len/dsprites_len_"$len"_sin_no_ss1
done

for n in {1..10}; do
  SEED=$RANDOM
  for len in 1e-1 5e-1 1 2 4 8; do

  bsub -o models/len/dsprites_len_"$len"_sin_no_ss1/lfs_log_"$len"_$n -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python train.py --model_type gp-vae --data_type dsprites --exp_name dsprites_"$len"_n$n \
  --basedir models/len/dsprites_len_"$len"_sin_no_ss1 --seed $SEED --banded_covar --latent_dim 64 \
  --encoder_sizes=32,256,256 --decoder_sizes=256,256,256 --window_size 3 --sigma 1 \
  --length_scale $len --beta 1.0 --num_epochs 20

  done
done
