#!/bin/bash

#mkdir -p models/gp_part2/base/len_5/same
#mkdir -p models/gp_part2/base/len_5/scaled
#mkdir -p models/gp_part2/base/len_10/same
#mkdir -p models/gp_part2/base/len_10/scaled
#mkdir -p models/gp_part2/ada/len_5/same
#mkdir -p models/gp_part2/ada/len_5/scaled



for n in {6..10}; do
  seed=$RANDOM
  # Base model

  # GP data, partial range
  bsub -o models/gp_full2/base/len_5/same/log_%J -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range2.npz \
  --exp_name n_"$n" --basedir models/gp_full2/base/len_5/same \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range2.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score

  bsub -o models/gp_full2/base/len_5/scaled/log_%J -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range2.npz \
  --exp_name n_"$n" --basedir models/gp_full2/base/len_5/scaled --len_init scaled --kernel_scales 16 \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range2.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score

  bsub -o models/gp_full2/base/len_10/same/log_%J -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 10 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range2.npz \
  --exp_name n_"$n" --basedir models/gp_full2/base/len_10/same \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range2.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score

  bsub -o models/gp_full2/base/len_10/scaled/log_%J -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 10 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range2.npz \
  --exp_name n_"$n" --basedir models/gp_full2/base/len_10/scaled --len_init scaled --kernel_scales 16 \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range2.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score


  # Ada extended model

  # GP data, part range
  bsub -o models/gp_full2/ada/len_5/same/log_%J -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type ada-gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range2.npz \
  --exp_name n_"$n" --basedir models/gp_full2/ada/len_5/same \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range2.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score

  bsub -o models/gp_full2/ada/len_5/scaled/log_%J -g /gpvae_disent \
  -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type ada-gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range2.npz \
  --exp_name n_"$n" --basedir models/gp_full2/ada/len_5/scaled --len_init scaled --kernel_scales 16 \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range2.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score

done