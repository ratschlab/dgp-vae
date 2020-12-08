#!/bin/bash

mkdir -p models/physionet/explore_1/standard/beta_0.2/lat_8
mkdir -p models/physionet/explore_1/standard/beta_0.2/lat_16
mkdir -p models/physionet/explore_1/standard/beta_0.2/lat_32
mkdir -p models/physionet/explore_1/standard/beta_1.0/lat_8
mkdir -p models/physionet/explore_1/standard/beta_1.0/lat_16
mkdir -p models/physionet/explore_1/standard/beta_1.0/lat_32

mkdir -p models/physionet/explore_1/linear/beta_0.2/lat_8
mkdir -p models/physionet/explore_1/linear/beta_0.2/lat_16
mkdir -p models/physionet/explore_1/linear/beta_0.2/lat_32
mkdir -p models/physionet/explore_1/linear/beta_1.0/lat_8
mkdir -p models/physionet/explore_1/linear/beta_1.0/lat_16
mkdir -p models/physionet/explore_1/linear/beta_1.0/lat_32

for n in 1 2 3; do
  seed=$RANDOM
  for rescale in standard linear; do
    for beta in 0.2 1.0; do
      for lat_dim in 8 16 32; do
        bsub -o models/physionet/explore_1/"$rescale"/beta_"$beta"/lat_"$lat_dim"/log_%J \
        -g /gpvae_norm -W 12:00 -R "rusage[mem=10000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
        python run_experiment.py --model_type gp-vae --data_type physionet --time_len 24 \
        --testing --batch_size 64 --exp_name n_"$n" --basedir models/physionet/explore_1/"$rescale"/beta_"$beta"/lat_"$lat_dim" \
        --len_init scaled --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/physionet/physionet_norm.npz \
        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 \
        --decoder_sizes=256,256 --window_size 24 --sigma 1.005 --length_scale 10.0 --beta "$beta" \
        --kernel_scales 8 --num_epochs 10 --kernel cauchy --save_score --visualize_score \
        --data_type_dci physionet --rescaling "$rescale"
      done
    done
  done
done