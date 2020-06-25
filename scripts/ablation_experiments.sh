#!/bin/bash

mkdir -p models/extensions/all
mkdir -p models/extensions/ws_ls
mkdir -p models/extensions/ws_at
mkdir -p models/extensions/ls_at

# Depending on what previous experiments show, change params for learnable len.
# Fix to something for now, most likely all_same as init.

for n in {1..10}; do
  seed=$RANDOM
  # all
  bsub -o models/extensions/all/log_"$n" -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python run_experiment.py --model_type ada-gp-vae --data_type dsprites --testing\
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
  --exp_name ext_all_n"$n" --basedir models/extensions/all \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 \
  --beta 1.0 --num_epochs 1 --kernel cauchy --learn_len --kernel_scales 64 \
  --len_init same --aggressive_train\
  --z_name factors_100k_5k.npz --save_score
  # ws + ls
  bsub -o models/extensions/ws_ls/log_"$n" -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python run_experiment.py --model_type ada-gp-vae --data_type dsprites --testing\
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
  --exp_name ws_ls_n"$n" --basedir models/extensions/ws_ls \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 \
  --beta 1.0 --num_epochs 1 --kernel cauchy --learn_len --kernel_scales 64 \
  --len_init same --z_name factors_100k_5k.npz --save_score
  # ws + at
  bsub -o models/extensions/ws_at/log_"$n" -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python run_experiment.py --model_type ada-gp-vae --data_type dsprites --testing\
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
  --exp_name ws_at_n"$n" --basedir models/extensions/ws_at \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 \
  --beta 1.0 --num_epochs 1 --kernel cauchy --kernel_scales 1 --aggressive_train \
  --z_name factors_100k_5k.npz --save_score
  # ls + at
  bsub -o models/extensions/ls_at/log_"$n" -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python run_experiment.py --model_type gp-vae --data_type dsprites --testing\
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
  --exp_name ls_at_n"$n" --basedir models/extensions/ls_at \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 \
  --beta 1.0 --num_epochs 1 --kernel cauchy --kernel_scales 64 --learn_len \
  --len_init same --aggressive_train \
  --z_name factors_100k_5k.npz --save_score
done