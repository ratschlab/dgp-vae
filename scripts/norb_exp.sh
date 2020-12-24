#!/bin/bash

mkdir -p models/norb_full1/base/len_5/same
mkdir -p models/norb_full1/base/len_5/scaled
mkdir -p models/norb_full1/base/len_10/same
mkdir -p models/norb_full1/base/len_10/scaled
mkdir -p models/norb_full1/ada/len_5/same
mkdir -p models/norb_full1/ada/len_5/scaled
mkdir -p models/norb_full1/ada/len_10/same
mkdir -p models/norb_full1/ada/len_10/scaled

for n in {1..10};do
  seed=$RANDOM
  for len in 5 10;do
    bsub -o models/norb_full1/base/len_"$len"/same/log_%J -g /gpvae_disent \
    -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
    python run_experiment.py --model_type gp-vae --data_type smallnorb --time_len "$len" --testing --batch_size 32 \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/norb_full1.npz \
    --exp_name n_"$n" --basedir models/norb_full1/base/len_"$len"/same \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init same \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
    --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
    --save_score

    bsub -o models/norb_full1/base/len_"$len"/scaled/log_%J -g /gpvae_disent \
    -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
    python run_experiment.py --model_type gp-vae --data_type smallnorb --time_len "$len" --testing --batch_size 32 \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/norb_full1.npz \
    --exp_name n_"$n" --basedir models/norb_full1/base/len_"$len"/scaled \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init scaled --kernel_scales 16 \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
    --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
    --save_score

    bsub -o models/norb_full1/ada/len_"$len"/same/log_%J -g /gpvae_disent \
    -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
    python run_experiment.py --model_type ada-gp-vae --data_type smallnorb --time_len "$len" --testing --batch_size 32 \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/norb_full1.npz \
    --exp_name n_"$n" --basedir models/norb_full1/ada/len_"$len"/same \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init same \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
    --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
    --save_score

    bsub -o models/norb_full1/ada/len_"$len"/scaled/log_%J -g /gpvae_disent \
    -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
    python run_experiment.py --model_type ada-gp-vae --data_type smallnorb --time_len "$len" --testing --batch_size 32 \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/norb_full1.npz \
    --exp_name n_"$n" --basedir models/norb_full1/ada/len_"$len"/scaled \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init scaled --kernel_scales 16 \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
    --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
    --save_score
  done
done