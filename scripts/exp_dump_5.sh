#!/bin/bash

mkdir -p models/gp_part3/base/len_5/same
mkdir -p models/gp_part3/base/len_5/scaled
mkdir -p models/gp_part3/base/len_10/same
mkdir -p models/gp_part3/base/len_10/scaled
mkdir -p models/gp_part3/ada/len_5/same
mkdir -p models/gp_part3/ada/len_5/scaled

mkdir -p models/gp_full3/base/len_5/same
mkdir -p models/gp_full3/base/len_5/scaled
mkdir -p models/gp_full3/base/len_10/same
mkdir -p models/gp_full3/base/len_10/scaled
mkdir -p models/gp_full3/ada/len_5/same
mkdir -p models/gp_full3/ada/len_5/scaled

mkdir -p models/gp_part4/base/len_5/same
mkdir -p models/gp_part4/base/len_5/scaled
mkdir -p models/gp_part4/base/len_10/same
mkdir -p models/gp_part4/base/len_10/scaled
mkdir -p models/gp_part4/ada/len_5/same
mkdir -p models/gp_part4/ada/len_5/scaled

mkdir -p models/gp_full4/base/len_5/same
mkdir -p models/gp_full4/base/len_5/scaled
mkdir -p models/gp_full4/base/len_10/same
mkdir -p models/gp_full4/base/len_10/scaled
mkdir -p models/gp_full4/ada/len_5/same
mkdir -p models/gp_full4/ada/len_5/scaled



for n in {6..10}; do
  seed=$RANDOM
  for type in part full; do
    for count in 3 4; do
      # Base model
      # GP data, partial range
      bsub -o models/gp_"$type""$count"/base/len_5/same/log_%J -g /gpvae_disent \
      -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_"$type"_range"$count".npz \
      --exp_name n_"$n" --basedir models/gp_"$type""$count"/base/len_5/same \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_"$type"_range"$count".npz \
      --score_factors=1,2,3,4,5 --save_score --visualize_score

      bsub -o models/gp_"$type""$count"/base/len_5/scaled/log_%J -g /gpvae_disent \
      -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_"$type"_range"$count".npz \
      --exp_name n_"$n" --basedir models/gp_"$type""$count"/base/len_5/scaled --len_init scaled --kernel_scales 16 \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_"$type"_range"$count".npz \
      --score_factors=1,2,3,4,5 --save_score --visualize_score

      bsub -o models/gp_"$type""$count"/base/len_10/same/log_%J -g /gpvae_disent \
      -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 10 --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_"$type"_range"$count".npz \
      --exp_name n_"$n" --basedir models/gp_"$type""$count"/base/len_10/same \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_"$type"_range"$count".npz \
      --score_factors=1,2,3,4,5 --save_score --visualize_score

      bsub -o models/gp_"$type""$count"/base/len_10/scaled/log_%J -g /gpvae_disent \
      -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type gp-vae --data_type dsprites --time_len 10 --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_"$type"_range"$count".npz \
      --exp_name n_"$n" --basedir models/gp_"$type""$count"/base/len_10/scaled --len_init scaled --kernel_scales 16 \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_"$type"_range"$count".npz \
      --score_factors=1,2,3,4,5 --save_score --visualize_score


      # Ada extended model
      # GP data, part range
      bsub -o models/gp_"$type""$count"/ada/len_5/same/log_%J -g /gpvae_disent \
      -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type ada-gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_"$type"_range"$count".npz \
      --exp_name n_"$n" --basedir models/gp_"$type""$count"/ada/len_5/same \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_"$type"_range"$count".npz \
      --score_factors=1,2,3,4,5 --save_score --visualize_score

      bsub -o models/gp_"$type""$count"/ada/len_5/scaled/log_%J -g /gpvae_disent \
      -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
      python run_experiment.py --model_type ada-gp-vae --data_type dsprites --time_len 5 --testing --batch_size 32 \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_gp_"$type"_range"$count".npz \
      --exp_name n_"$n" --basedir models/gp_"$type""$count"/ada/len_5/scaled --len_init scaled --kernel_scales 16 \
      --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
      --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
      --num_epochs 1 --kernel cauchy --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_"$type"_range"$count".npz \
      --score_factors=1,2,3,4,5 --save_score --visualize_score
    done
  done
done
#
#bsub -o models/physionet/test0/log_%J -g /gpvae_norm \
#-W 8:00 -R "rusage[mem=20000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#python run_experiment.py --model_type gp-vae --data_type physionet --time_len 24 --testing --batch_size 64 \
#--exp_name n_0 --basedir models/physionet/test0 --len_init same \
#--data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/physionet/physionet_norm.npz \
#--seed 0 --banded_covar --latent_dim 32 --encoder_sizes=128,128 \
#--decoder_sizes=256,256 --window_size 24 --sigma 1.005 --length_scale 1.0 --beta 0.2 \
#--num_epochs 5 --kernel cauchy \
#--save_score --visualize_score --data_type_dci physionet --rescaling linear
#
#bsub -o models/physionet/test0/log_%J -g /gpvae_norm -W 8:00 -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python run_experiment.py --model_type gp-vae --data_type physionet --time_len 24 --testing --batch_size 64 --exp_name n_1 --basedir models/physionet/test0 --len_init same --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/physionet/physionet_norm.npz --seed 0 --banded_covar --latent_dim 32 --encoder_sizes=128,128 --decoder_sizes=256,256 --window_size 24 --sigma 1.005 --length_scale 1.0 --beta 0.2 --num_epochs 1 --kernel cauchy --save_score --visualize_score --data_type_dci physionet --rescaling linear
#
#
#
#--model_type
#gp-vae
#--data_type
#physionet
#--time_len
#24
#--testing
#--exp_name
#dsprites_debug_local0
#--basedir
#/Users/Simon/Documents/Uni/HiWi-Stellen/BMI/dgpvae/models/local_debug0
#--seed
#0
#--banded_covar
#--latent_dim
#32
#--encoder_sizes=128,128
#--decoder_sizes=256,256
#--window_size
#24
#--sigma
#1.005
#--length_scale
#1.0
#--beta
#0.2
#--num_steps
#1
#--print_interval
#1
#--kernel
#cauchy
#--kernel_scales
#16
#--c_path
#/Users/Simon/Documents/Uni/HiWi-Stellen/BMI/dgpvae/data/physionet/physionet_dci_features.npz
#--save_score
#--visualize_score
#--data_type_dci
#physionet
#--rescaling
#linear
#
#bsub -R "rusage[mem=65000]" python disentanglement_lib/disentanglement_lib/data/ground_truth/create_dataset.py --data_type smallnorb --seed 42 --num_timeseries 10500 --length 100 --kernel rbf --gp_weight 1 --periods=0,15,0.1,1 --file_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/norb_full1 --factors_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1
#
#bsub -R "rusage[mem=200000]" python disentanglement_lib/disentanglement_lib/data/ground_truth/create_dataset.py --data_type cars3d --seed 42 --num_timeseries 6500 --length 100 --kernel rbf --gp_weight 1 --periods=3,10,0 --file_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/cars_full5 --factors_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_cars_full5 --save_data
#
#bsub -R "rusage[mem=200000]" python disentanglement_lib/disentanglement_lib/data/ground_truth/create_dataset.py --data_type shapes3d --seed 42 --num_timeseries 6500 --length 100 --kernel rbf --gp_weight 1 --periods=0,0,0,5,0,1 --file_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/shapes_part1 --factors_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/factors_shapes_part1 --univ_rescaling --save_data