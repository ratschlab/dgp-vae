#!/bin/bash

mkdir -p models/extensions/weak_supiv
mkdir -p models/extensions/len_scale/single
mkdir -p models/extensions/len_scale/all_scaled
mkdir -p models/extensions/len_scale/all_same

# weak supervision

# length scales: all_scaled, all_same, single
for n in {1..10}; do
  seed=$RANDOM
  # weak supervision
  bsub -o models/extensions/weak_supiv/log_"$n" -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python run_experiment.py --model_type ada-gp-vae --data_type dsprites --testing\
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
  --exp_name weak_supiv_n"$n" --basedir models/extensions/weak_supiv \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 \
  --beta 1.0 --num_epochs 1 --kernel cauchy \
  --z_name factors_100k_5k.npz --save_score
  # length scales: single
  bsub -o models/extensions/len_scale/single/log_"$n" -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python run_experiment.py --model_type gp-vae --data_type dsprites --testing\
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
  --exp_name len_scale_single_n"$n" --basedir models/extensions/len_scale/single \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 \
  --beta 1.0 --num_epochs 1 --kernel cauchy --learn_len --kernel_scales 1 \
  --z_name factors_100k_5k.npz --save_score
  # length scales: all_scaled
  bsub -o models/extensions/len_scale/all_scaled/log_"$n" -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python run_experiment.py --model_type gp-vae --data_type dsprites --testing\
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
  --exp_name len_scale_all_scaled_n"$n" --basedir models/extensions/len_scale/all_scaled \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 \
  --beta 1.0 --num_epochs 1 --kernel cauchy --learn_len --kernel_scales 64 --len_init scaled \
  --z_name factors_100k_5k.npz --save_score
  # length scales: all_same
  bsub -o models/extensions/len_scale/all_same/log_"$n" -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python run_experiment.py --model_type gp-vae --data_type dsprites --testing\
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
  --exp_name len_scale_all_same_n"$n" --basedir models/extensions/len_scale/all_same \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 \
  --beta 1.0 --num_epochs 1 --kernel cauchy --learn_len --kernel_scales 64 --len_init same \
  --z_name factors_100k_5k.npz --save_score
done
