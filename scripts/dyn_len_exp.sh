#!/bin/bash

mkdir -p models/dyn_len_exp/base/len_1
mkdir -p models/dyn_len_exp/base/len_10
mkdir -p models/dyn_len_exp/base/len_5
mkdir -p models/dyn_len_exp/ada/len_1
mkdir -p models/dyn_len_exp/ada/len_10
mkdir -p models/dyn_len_exp/ada/len_5

for len in 1 5 10; do
  for n in {6..10}; do
    seed=$RANDOM
    # Base
    bsub -W 8:00 -o models/dyn_len_exp/base/len_"$len"/log_%J -g /gpvae_disent \
    -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
    python run_experiment.py --model_type gp-vae --data_type dsprites --time_len "$len" --testing --batch_size 32 \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k_100.npz \
    --exp_name n_"$n" --basedir models/dyn_len_exp/base/len_"$len"_old \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
    --num_epochs 5 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_100k_5k_100.npz \
    --score_factors=1,2,3,4,5 --save_score --visualize_score

    # Weak supervision
    bsub -W 8:00 -o models/dyn_len_exp/ada/len_"$len"/log_%J -g /gpvae_disent \
    -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
    python run_experiment.py --model_type ada-gp-vae --data_type dsprites --time_len "$len" --testing --batch_size 32 \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k_100.npz \
    --exp_name n_"$n" --basedir models/dyn_len_exp/ada/len_"$len"_old \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
    --num_epochs 5 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_100k_5k_100.npz \
    --score_factors=1,2,3,4,5 --save_score --visualize_score

  done
done

bsub -R "rusage[mem=65000]" python disentanglement_lib/disentanglement_lib/data/ground_truth/create_dataset.py --seed 42 --num_timeseries 10500 --length 100 --kernel sinusoid --periods=0,0,0,5,10,20 --resample_period --save_data --file_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_sin_resample --factors_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_sin_resample

bsub -R "rusage[mem=65000]" python disentanglement_lib/disentanglement_lib/data/ground_truth/create_dataset.py --seed 42 --num_timeseries 10500 --length 100 --kernel rbf --gp_weight 1 --periods=0,0,0,1.5,0.5,0.01 --debug --file_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_full_range1 --factors_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range1

bsub -R "rusage[mem=65000]" python disentanglement_lib/disentanglement_lib/data/ground_truth/create_dataset.py --seed 42 --num_timeseries 10500 --length 100 --kernel rbf --gp_weight 1 --univ_rescaling --periods=0,0,2,15.0,1.0,0.1 --file_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_part4 --factors_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_part4 --save_data


