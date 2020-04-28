#!/bin/bash

mkdir -p models/dsprites_dim_64_sin_10k

for n in {1..10}; do
  SEED=$RANDOM
  for dim in 64; do

  bsub -o models/dsprites_dim_"$dim"_sin_10k/lfs_log_"$dim"_$n -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python train.py --model_type gp-vae --data_type dsprites --exp_name dsprites_"$dim"_n$n \
  --basedir models/dsprites_dim_"$dim"_sin_10k --data_dir data/dsrpites/dsprites_10000.npz --seed $SEED --banded_covar --latent_dim $dim \
  --encoder_sizes=32,256,256 --decoder_sizes=256,256,256 --window_size 3 --sigma 1 \
  --length_scale 2 --beta 1.0 --num_epochs 20

  done
done
