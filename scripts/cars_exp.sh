#!/bin/bash

mkdir -p models/cars_full1/base/len_5/same
mkdir -p models/cars_full1/base/len_5/scaled
mkdir -p models/cars_full1/base/len_10/same
mkdir -p models/cars_full1/base/len_10/scaled
mkdir -p models/cars_full1/ada/len_5/same
mkdir -p models/cars_full1/ada/len_5/scaled
mkdir -p models/cars_full1/ada/len_10/same
mkdir -p models/cars_full1/ada/len_10/scaled

for n in {1..10};do
  seed=$RANDOM
  for dataset in cars_part1 cars_full2 cars_full3 cars_full4 cars_full5;do
    for len in 5 10;do
      bsub -o models/"$dataset"/base/len_"$len"/same/log_%J -g /gpvae_disent \
      -R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type gp-vae --data_type cars3d --time_len "$len" --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/"$dataset".npz \
      --exp_name n_"$n" --basedir models/"$dataset"/base/len_"$len"/same \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init same \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_"$dataset".npz \
      --save_score

      bsub -o models/"$dataset"/base/len_"$len"/scaled/log_%J -g /gpvae_disent \
      -R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type gp-vae --data_type cars3d --time_len "$len" --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/"$dataset".npz \
      --exp_name n_"$n" --basedir models/"$dataset"/base/len_"$len"/scaled \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init scaled --kernel_scales 16 \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_"$dataset".npz \
      --save_score

  #    bsub -o models/"$dataset"/ada/len_"$len"/same/log_%J -g /gpvae_disent \
  #    -R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  #    python run_experiment.py --model_type ada-gp-vae --data_type cars3d --time_len "$len" --testing --batch_size 32 \
  #    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/"$dataset".npz \
  #    --exp_name n_"$n" --basedir models/"$dataset"/ada/len_"$len"/same \
  #    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init same \
  #    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  #    --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_"$dataset".npz \
  #    --save_score
  #
  #    bsub -o models/"$dataset"/ada/len_"$len"/scaled/log_%J -g /gpvae_disent \
  #    -R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  #    python run_experiment.py --model_type ada-gp-vae --data_type cars3d --time_len "$len" --testing --batch_size 32 \
  #    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/"$dataset".npz \
  #    --exp_name n_"$n" --basedir models/"$dataset"/ada/len_"$len"/scaled \
  #    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init scaled --kernel_scales 16 \
  #    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
  #    --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_"$dataset".npz \
  #    --save_score
    done
  done
done