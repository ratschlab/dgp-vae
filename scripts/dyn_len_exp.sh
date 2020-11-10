#!/bin/bash

mkdir -p models/dyn_len_exp/base/len_1_old
mkdir -p models/dyn_len_exp/base/len_10_old
mkdir -p models/dyn_len_exp/base/len_5_old
mkdir -p models/dyn_len_exp/ada/len_1_old
mkdir -p models/dyn_len_exp/ada/len_10_old
mkdir -p models/dyn_len_exp/ada/len_5_old

for len in 1 5 10; do
  for n in {1..5}; do
    seed=$RANDOM
    # Base
    bsub -o models/dci_factors_comp/log_%J -g /gpvae_disent \
    -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
    python run_experiment.py --model_type gp-vae --data_type dsprites --time_len "$len" --testing --batch_size 32 \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
    --exp_name n_"$n" --basedir models/dyn_len_exp/base/len_"$len"_old \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
    --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_100k_5k.npz \
    --score_factors=1,2,3,4,5 --save_score --visualize_score

    # Weak supervision
    bsub -o models/dci_factors_comp/log_%J -g /gpvae_disent \
    -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
    python run_experiment.py --model_type ada-gp-vae --data_type dsprites --time_len "$len" --testing --batch_size 32 \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
    --exp_name n_"$n" --basedir models/dyn_len_exp/ada/len_"$len"_old \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
    --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_100k_5k.npz \
    --score_factors=1,2,3,4,5 --save_score --visualize_score

  done
done
