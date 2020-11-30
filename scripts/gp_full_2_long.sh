#!/bin/bash

mkdir -p models/gp_full2/base/len_10/scaled/epoch_5

for n in {1..5}; do
  seed=$RANDOM

  # Base model time window len 10, epochs
  bsub -W 10:00 -o models/gp_full2/base/len_10/scaled/epoch_5/log_%J -g /gpvae_disent \
  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 10 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range2.npz \
  --exp_name n_"$n" --basedir models/gp_full2/base/len_10/scaled/epoch_5 --len_init scaled --kernel_scales 16 \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
  --num_epochs 5 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range2.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score
done