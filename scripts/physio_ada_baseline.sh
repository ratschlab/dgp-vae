#!/bin/bash

mkdir -p models/physionet/ada_baseline/cauchy
mkdir -p models/physionet/ada_baseline/const

for n in 1 2 3; do
  seed=$RANDOM
  for kernel in const cauchy; do
      bsub -o models/physionet/ada_baseline/"$kernel"/log_%J \
      -g /gpvae_norm -W 12:00 -R "rusage[mem=10000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type ada-gp-vae --data_type physionet --time_len 1 \
      --testing --batch_size 64 --exp_name n_"$n" --basedir models/physionet/ada_baseline/"$kernel" \
      --len_init scaled --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/physionet/physionet_norm.npz \
      --seed "$seed" --banded_covar --latent_dim 16 --encoder_sizes=128,128 \
      --decoder_sizes=256,256 --window_size 24 --sigma 1.005 --length_scale 1.0 --beta 1.0 \
      --num_epochs 10 --kernel "$kernel" --save_score --visualize_score \
      --data_type_dci physionet --rescaling standard
  done
done