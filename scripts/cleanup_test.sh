#!/bin/bash

# Test script to see if refactoring didnt break anything

# dSprites
  bsub -o /cluster/home/bings/dgpvae/models/cleanup_test/dsprites2_log_%J -g /gpvae_disent \
  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type dgp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites.npz \
  --exp_name dsprites --basedir /cluster/home/bings/dgpvae/models/cleanup_test \
  --seed 0 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --print_interval 1 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_steps 100 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range4.npz \
  --data_type_dci dsprites --shuffle --save_score --visualize_score

# NORB
  bsub -o /cluster/home/bings/dgpvae/models/cleanup_test/norb2_log_%J -g /gpvae_disent \
  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type dgp-vae --data_type smallnorb --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/smallnorb.npz \
  --exp_name norb --basedir /cluster/home/bings/dgpvae/models/cleanup_test \
  --seed 0 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --print_interval 1 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_steps 100 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
  --data_type_dci smallnorb --shuffle --save_score --visualize_score

# Cars3D
  bsub -o /cluster/home/bings/dgpvae/models/cleanup_test/cars2_log_%J -g /gpvae_disent \
  -R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type dgp-vae --data_type cars3d --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/cars3d.npz \
  --exp_name cars --basedir /cluster/home/bings/dgpvae/models/cleanup_test \
  --seed 0 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --print_interval 1 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_steps 100 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_cars_part1.npz \
  --data_type_dci cars3d --shuffle --save_score --visualize_score

# Shapes3D
  bsub -o /cluster/home/bings/dgpvae/models/cleanup_test/shapes2_log_%J -g /gpvae_disent \
  -R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type dgp-vae --data_type shapes3d --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/shapes3d.npz \
  --exp_name shapes --basedir /cluster/home/bings/dgpvae/models/cleanup_test \
  --seed 0 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --print_interval 1 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_steps 100 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/factors_shapes_part2.npz \
  --data_type_dci shapes3d --shuffle --save_score --visualize_score

# HiRID
  bsub -o /cluster/home/bings/dgpvae/models/cleanup_test/hirid2_log_%J -g /gpvae_disent \
  -R "rusage[mem=65000,ngpus_excl_p=1]" \
  python run_experiment.py --model_type dgp-vae --data_type hirid --time_len 25 --testing --batch_size 64 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid.npz \
  --exp_name hirid --basedir /cluster/home/bings/dgpvae/models/cleanup_test --len_init scaled\
  --seed 0 --banded_covar --latent_dim 8 --encoder_sizes=128,128 --kernel_scales 4 --print_interval 1 \
  --decoder_sizes=256,256 --window_size 12 --sigma 1.005 --length_scale 20.0 --beta 1.0 \
  --num_steps 100 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid.npz \
  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_2.npy \
  --shuffle --data_type_dci hirid --save_score --visualize_score