#!/bin/bash

# Create directory for each tested latent dimension
for dim in 8 16 32 64 128; do
  mkdir -p models/dsprites_dim_"$dim"_debug3
done

for n in 1; do
  SEED=0
  for dim in 8 16 32 64 128; do

  bsub -o models/dsprites_dim_"$dim"_debug3/lfs_log_"$dim"_$n -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python train.py --model_type gp-vae --data_type dsprites --exp_name dsprites_"$dim"_debug_n$n \
  --basedir models/dsprites_dim_"$dim"_debug3 --seed $SEED --banded_covar --latent_dim $dim \
  --encoder_sizes=32,256,256 --decoder_sizes=256,256,256 --window_size 3 --sigma 1 \
  --length_scale 2 --beta 1.0 --num_epochs 20

  done
done
