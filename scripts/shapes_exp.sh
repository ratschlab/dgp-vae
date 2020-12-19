#!/bin/bash

mkdir -p models/shapes_full1/base/len_5/same
mkdir -p models/shapes_full1/base/len_5/scaled
mkdir -p models/shapes_full1/base/len_10/same
mkdir -p models/shapes_full1/base/len_10/scaled
mkdir -p models/shapes_full1/ada/len_5/same
mkdir -p models/shapes_full1/ada/len_5/scaled
mkdir -p models/shapes_full1/ada/len_10/same
mkdir -p models/shapes_full1/ada/len_10/scaled

mkdir -p models/shapes_full2/base/len_5/same
mkdir -p models/shapes_full2/base/len_5/scaled
mkdir -p models/shapes_full2/base/len_10/same
mkdir -p models/shapes_full2/base/len_10/scaled
mkdir -p models/shapes_full2/ada/len_5/same
mkdir -p models/shapes_full2/ada/len_5/scaled
mkdir -p models/shapes_full2/ada/len_10/same
mkdir -p models/shapes_full2/ada/len_10/scaled

for n in {1..5};do
  seed=$RANDOM
  for num in 1 2;do
    for len in 5 10;do
      bsub -o models/shapes_full"$num"/base/len_"$len"/same/log_%J -g /gpvae_disent \
      -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type gp-vae --data_type shapes3d --time_len 5 --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/shapes_full"$num".npz \
      --exp_name n_"$n" --basedir models/shapes_full"$num"/base/len_"$len"/same \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init same \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/factors_shapes_full"$num".npz \
      --save_score

      bsub -o models/shapes_full"$num"/base/len_"$len"/scaled/log_%J -g /gpvae_disent \
      -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type gp-vae --data_type shapes3d --time_len 5 --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/shapes_full"$num".npz \
      --exp_name n_"$n" --basedir models/shapes_full"$num"/base/len_"$len"/same \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init scaled --kernel_scales 16 \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/factors_shapes_full"$num".npz \
      --save_score

      bsub -o models/shapes_full"$num"/ada/len_"$len"/same/log_%J -g /gpvae_disent \
      -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type ada-gp-vae --data_type shapes3d --time_len 5 --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/shapes_full"$num".npz \
      --exp_name n_"$n" --basedir models/shapes_full"$num"/ada/len_"$len"/same \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init same \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/factors_shapes_full"$num".npz \
      --save_score

      bsub -o models/shapes_full"$num"/ada/len_"$len"/scaled/log_%J -g /gpvae_disent \
      -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type ada-gp-vae --data_type shapes3d --time_len 5 --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/shapes_full"$num".npz \
      --exp_name n_"$n" --basedir models/shapes_full"$num"/ada/len_"$len"/same \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init scaled --kernel_scales 16 \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/factors_shapes_full"$num".npz \
      --save_score
    done
  done
done