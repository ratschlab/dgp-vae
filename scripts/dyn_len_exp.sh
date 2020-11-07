#!/bin/bash

mkdir -p models/dyn_len_exp/base/len_1
mkdir -p models/dyn_len_exp/base/len_10
mkdir -p models/dyn_len_exp/base/len_50
mkdir -p models/dyn_len_exp/ada/len_1
mkdir -p models/dyn_len_exp/ada/len_10
mkdir -p models/dyn_len_exp/ada/len_50

for len in 1 10 50; do
  for n in {1..5}; do
    seed=$RANDOM
    # Base
    bsub -o models/dci_factors_comp/log_%J -g /gpvae_disent \
    -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
    python run_experiment.py --model_type gp-vae --data_type dsprites --time_len "$len" --testing \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k_100.npz \
    --exp_name n_"$n" --basedir models/dyn_len_exp/base/len_"$len" \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
    --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_100k_5k_100.npz \
    --score_factors=1,2,3,4,5 --save_score --visualize_score

    # Weak supervision
    bsub -o models/dci_factors_comp/log_%J -g /gpvae_disent \
    -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
    python run_experiment.py --model_type ada-gp-vae --data_type dsprites --time_len "$len" --testing \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k_100.npz \
    --exp_name n_"$n" --basedir models/dyn_len_exp/ada/len_"$len" \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
    --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_100k_5k_100.npz \
    --score_factors=1,2,3,4,5 --save_score --visualize_score

  done
done