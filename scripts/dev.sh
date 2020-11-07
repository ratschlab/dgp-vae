#!/bin/bash

# Script for development training

bsub -o models/dev/log_%J -g /gpvae_disent \
-R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
python run_experiment.py --model_type ada-gp-vae --data_type dsprites --testing\
--data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
--exp_name dev --basedir models/dev \
--seed 42 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
--decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 \
--beta 1.0 --num_epochs 1 --kernel cauchy \
--z_name factors_100k_5k.npz --save_score --visualize_score

bsub -o models/dev/log_%J -g /gpvae_disent -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python run_experiment.py --model_type ada-gp-vae --data_type dsprites --testing --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz --exp_name ada_init_test_0 --basedir models/dev --seed 42 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_100k_5k.npz --score_factors=1,2,3,4,5 --save_score --visualize_score

bsub -R "rusage[mem=100000]" python create_dataset.py --seed 42 --num_timeseries 10500 --length 100 --kernel sinusoid --save_data --file_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k_100 --factors_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_100k_5k_100