#!/bin/bash

#mkdir -p models/hirid/const_mix/dim_8/len_25/const_5/same
#mkdir -p models/hirid/const_mix/dim_8/len_25/const_5/scaled
#mkdir -p models/hirid/const_mix/dim_8/len_25/const_25/same
#mkdir -p models/hirid/const_mix/dim_8/len_25/const_25/scaled
#mkdir -p models/hirid/const_mix/dim_8/len_50/const_5/same
#mkdir -p models/hirid/const_mix/dim_8/len_50/const_5/scaled
#mkdir -p models/hirid/const_mix/dim_8/len_50/const_25/same
#mkdir -p models/hirid/const_mix/dim_8/len_50/const_25/scaled
#
#mkdir -p models/hirid/const_mix/dim_16/len_25/const_5/same
#mkdir -p models/hirid/const_mix/dim_16/len_25/const_5/scaled
#mkdir -p models/hirid/const_mix/dim_16/len_25/const_25/same
#mkdir -p models/hirid/const_mix/dim_16/len_25/const_25/scaled
#mkdir -p models/hirid/const_mix/dim_16/len_50/const_5/same
#mkdir -p models/hirid/const_mix/dim_16/len_50/const_5/scaled
#mkdir -p models/hirid/const_mix/dim_16/len_50/const_25/same
#mkdir -p models/hirid/const_mix/dim_16/len_50/const_25/scaled
#
#for n in {1..10}; do
#  seed=$RANDOM
#  for const_val in 5 25; do
#    for lat_dim in 8 16; do
##      for len in 5,3,10 25,12,4 50,25,4; do
#      for len in 25,12,4 50,25,4; do
#        IFS=',' read time_len window_size run_time <<< "${len}"
#        bsub -o models/hirid/const_mix/dim_"$lat_dim"/len_"$time_len"/const_"$const_val"/same/log_%J \
#        -g /gpvae_disent -W "$run_time":00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#        python run_experiment.py --model_type gp-vae --data_type hirid --time_len "$time_len" \
#        --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/const_mix/dim_"$lat_dim"/len_"$time_len"/const_"$const_val"/same \
#        --len_init same --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 \
#        --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 2.0 --beta 1.0 \
#        --num_epochs 1 --kernel cauchy_const --const_kernel_scale "$const_val"\
#        --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#        --visualize_score --save_score --eval_type dci \
#        --data_type_dci hirid --shuffle --dci_seed "$seed"
#
#        bsub -o models/hirid/const_mix/dim_"$lat_dim"/len_"$time_len"/const_"$const_val"/scaled/log_%J \
#        -g /gpvae_disent -W "$run_time":00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#        python run_experiment.py --model_type gp-vae --data_type hirid --time_len "$time_len" \
#        --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/const_mix/dim_"$lat_dim"/len_"$time_len"/const_"$const_val"/scaled \
#        --len_init scaled --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 --kernel_scales "$lat_dim"\
#        --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 20.0 --beta 1.0 \
#        --num_epochs 1 --kernel cauchy_const -const_kernel_scale "$const_val"\
#        --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#        --visualize_score --save_score --eval_type dci \
#        --data_type_dci hirid --shuffle --dci_seed "$seed"
#
#      done
#    done
#  done
#done

mkdir -p models/hirid/ada_const_mix/dim_8/len_25/const_5/same
mkdir -p models/hirid/ada_const_mix/dim_8/len_25/const_5/scaled
mkdir -p models/hirid/ada_const_mix/dim_8/len_25/const_25/same
mkdir -p models/hirid/ada_const_mix/dim_8/len_25/const_25/scaled
mkdir -p models/hirid/ada_const_mix/dim_8/len_50/const_5/same
mkdir -p models/hirid/ada_const_mix/dim_8/len_50/const_5/scaled
mkdir -p models/hirid/ada_const_mix/dim_8/len_50/const_25/same
mkdir -p models/hirid/ada_const_mix/dim_8/len_50/const_25/scaled

mkdir -p models/hirid/ada_const_mix/dim_16/len_25/const_5/same
mkdir -p models/hirid/ada_const_mix/dim_16/len_25/const_5/scaled
mkdir -p models/hirid/ada_const_mix/dim_16/len_25/const_25/same
mkdir -p models/hirid/ada_const_mix/dim_16/len_25/const_25/scaled
mkdir -p models/hirid/ada_const_mix/dim_16/len_50/const_5/same
mkdir -p models/hirid/ada_const_mix/dim_16/len_50/const_5/scaled
mkdir -p models/hirid/ada_const_mix/dim_16/len_50/const_25/same
mkdir -p models/hirid/ada_const_mix/dim_16/len_50/const_25/scaled

for n in {1..10}; do
  seed=$RANDOM
  for const_val in 5 25; do
    for lat_dim in 8 16; do
#      for len in 5,3,10 25,12,4 50,25,4; do
      for len in 25,12,4 50,25,4; do
        IFS=',' read time_len window_size run_time <<< "${len}"
        bsub -o models/hirid/ada_const_mix/dim_"$lat_dim"/len_"$time_len"/const_"$const_val"/same/log_%J \
        -g /gpvae_disent -W "$run_time":00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
        python run_experiment.py --model_type ada-gp-vae --data_type hirid --time_len "$time_len" \
        --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/ada_const_mix/dim_"$lat_dim"/len_"$time_len"/const_"$const_val"/same \
        --len_init same --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 \
        --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 2.0 --beta 1.0 \
        --num_epochs 1 --kernel cauchy_const --const_kernel_scale "$const_val"\
        --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
        --visualize_score --save_score --eval_type dci \
        --data_type_dci hirid --shuffle --dci_seed "$seed"

        bsub -o models/hirid/ada_const_mix/dim_"$lat_dim"/len_"$time_len"/const_"$const_val"/scaled/log_%J \
        -g /gpvae_disent -W "$run_time":00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
        python run_experiment.py --model_type ada-gp-vae --data_type hirid --time_len "$time_len" \
        --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/ada_const_mix/dim_"$lat_dim"/len_"$time_len"/const_"$const_val"/scaled \
        --len_init scaled --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 --kernel_scales "$lat_dim"\
        --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 20.0 --beta 1.0 \
        --num_epochs 1 --kernel cauchy_const -const_kernel_scale "$const_val"\
        --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
        --visualize_score --save_score --eval_type dci \
        --data_type_dci hirid --shuffle --dci_seed "$seed"

      done
    done
  done
done