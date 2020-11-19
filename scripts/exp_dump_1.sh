#!/bin/bash

mkdir -p models/sin_resamp/base/len_5
mkdir -p models/sin_resamp/base/len_10
mkdir -p models/sin_resamp/ada/len_5
mkdir -p models/gp_full/base/len_5
mkdir -p models/gp_full/base/len_10
mkdir -p models/gp_full/ada/len_5
mkdir -p models/gp_part/base/len_5
mkdir -p models/gp_part/base/len_10
mkdir -p models/gp_part/ada/len_5



for n in {1..5}; do
  seed=$RANDOM
  # Base model
  # Sinusoid data, period resampled
#  bsub -o models/sin_resamp/base/len_5/log_%J -g /gpvae_disent \
#  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
#  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_sin_resample.npz \
#  --exp_name n_"$n" --basedir models/sin_resamp/base/len_5 \
#  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
#  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
#  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_sin_resample.npz \
#  --score_factors=1,2,3,4,5 --save_score --visualize_score

#  bsub -o models/sin_resamp/base/len_10/log_%J -g /gpvae_disent \
#  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 10 --testing --batch_size 32 \
#  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_sin_resample.npz \
#  --exp_name n_"$n" --basedir models/sin_resamp/base/len_10 \
#  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
#  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
#  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_sin_resample.npz \
#  --score_factors=1,2,3,4,5 --save_score --visualize_score

  # GP data, full range
  bsub -o models/gp_full/base/len_5/log_%J -g /gpvae_disent \
  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range1.npz \
  --exp_name n_"$n" --basedir models/gp_full/base/len_5 \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range1.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score

  bsub -o models/gp_full/base/len_10/log_%J -g /gpvae_disent \
  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 10 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range1.npz \
  --exp_name n_"$n" --basedir models/gp_full/base/len_10 \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range1.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score

  # GP data, partial range
  bsub -o models/gp_part/base/len_5/log_%J -g /gpvae_disent \
  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_part_range1.npz \
  --exp_name n_"$n" --basedir models/gp_part/base/len_5 \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_part_range1.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score

  bsub -o models/gp_part/base/len_10/log_%J -g /gpvae_disent \
  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 10 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_part_range1.npz \
  --exp_name n_"$n" --basedir models/gp_part/base/len_10 \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_part_range1.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score


  # Ada extended model
  # Sinusoid data, period resampled
#    bsub -o models/sin_resamp/ada/len_5/log_%J -g /gpvae_disent \
#  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#  python run_experiment.py --model_type ada-gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
#  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_sin_resample.npz \
#  --exp_name n_"$n" --basedir models/sin_resamp/ada/len_5 \
#  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
#  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
#  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_sin_resample.npz \
#  --score_factors=1,2,3,4,5 --save_score --visualize_score

  # GP data, full range
  bsub -o models/gp_full/ada/len_5/log_%J -g /gpvae_disent \
  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type ada-gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range1.npz \
  --exp_name n_"$n" --basedir models/gp_full/ada/len_5 \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range1.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score

  # GP data, partial range
  bsub -o models/gp_part/ada/len_5/log_%J -g /gpvae_disent \
  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type ada-gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_part_range1.npz \
  --exp_name n_"$n" --basedir models/gp_part/ada/len_5 \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
  --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_part_range1.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score
done