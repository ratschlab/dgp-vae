#!/bin/bash

mkdir -p models/new_data_test

bsub -o models/new_data_test/log_%J -g /gpvae_disent \
-R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
python run_experiment.py --model_type gp-vae --data_type smallnorb --time_len 5 --testing --batch_size 32 \
--data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/norb_full1.npz \
--exp_name norb --basedir models/new_data_test \
--seed 0 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
--decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
--num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
--save_score

bsub -o models/new_data_test/log_%J -g /gpvae_disent \
-R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
python run_experiment.py --model_type gp-vae --data_type cars3d --time_len 5 --testing --batch_size 32 \
--data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/cars_full1.npz \
--exp_name cars --basedir models/new_data_test \
--seed 0 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
--decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
--num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_cars_full1.npz \
--save_score

bsub -o models/new_data_test/log_%J -g /gpvae_disent \
-R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
python run_experiment.py --model_type gp-vae --data_type shapes3d --time_len 5 --testing --batch_size 32 \
--data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/shapes_full1.npz \
--exp_name shapes --basedir models/new_data_test \
--seed 0 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
--decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
--num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/factors_shapes_full1.npz \
--save_score
