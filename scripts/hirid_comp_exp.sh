#!/bin/bash

mkdir -p models/hirid/comp/base/dim_8/len_10/same
mkdir -p models/hirid/comp/base/dim_8/len_10/scaled
mkdir -p models/hirid/comp/base/dim_8/len_25/same
mkdir -p models/hirid/comp/base/dim_8/len_25/scaled
mkdir -p models/hirid/comp/base/dim_8/len_50/same
mkdir -p models/hirid/comp/base/dim_8/len_50/scaled
mkdir -p models/hirid/comp/base/dim_16/len_10/same
mkdir -p models/hirid/comp/base/dim_16/len_10/scaled
mkdir -p models/hirid/comp/base/dim_16/len_25/same
mkdir -p models/hirid/comp/base/dim_16/len_25/scaled
mkdir -p models/hirid/comp/base/dim_16/len_50/same
mkdir -p models/hirid/comp/base/dim_16/len_50/scaled

mkdir -p models/hirid/comp/ada/dim_8
mkdir -p models/hirid/comp/ada/dim_16

#for n in {1..10}; do
for n_seed in 1,22019 2,3412 3,13351 4,2677 5,6071 6,10843 7,16040 8,8088 9,22060 10,29153; do
#  seed=$RANDOM
  IFS=',' read n seed <<< "${n_seed}"
  for lat_dim in 8 16; do
#    for len in 10,5,24 25,12,4 50,25,4; do
#      IFS=',' read time_len window_size run_time <<< "${len}"
#      bsub -o models/hirid/comp/base/dim_"$lat_dim"/len_"$time_len"/same/log_%J \
#      -g /gpvae_disent -W "$run_time":00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#      python run_experiment.py --model_type gp-vae --data_type hirid --time_len "$time_len" \
#      --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/comp/base/dim_"$lat_dim"/len_"$time_len"/same \
#      --len_init same --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#      --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 \
#      --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 2.0 --beta 1.0 \
#      --num_epochs 1 --kernel cauchy \
#      --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#      --visualize_score --save_score --eval_type dci \
#      --data_type_dci hirid --shuffle --dci_seed "$seed"
#
#      bsub -o models/hirid/comp/base/dim_"$lat_dim"/len_"$time_len"/scaled/log_%J \
#      -g /gpvae_disent -W "$run_time":00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#      python run_experiment.py --model_type gp-vae --data_type hirid --time_len "$time_len" \
#      --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/comp/base/dim_"$lat_dim"/len_"$time_len"/scaled \
#      --len_init scaled --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#      --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 --kernel_scales "$lat_dim"\
#      --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 20.0 --beta 1.0 \
#      --num_epochs 1 --kernel cauchy \
#      --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#      --visualize_score --save_score --eval_type dci \
#      --data_type_dci hirid --shuffle --dci_seed "$seed"
#    done
    bsub -o models/hirid/comp/ada/dim_"$lat_dim"/log_%J -g /gpvae_norm -W 62:00 -R "rusage[mem=65000,ngpus_excl_p=1]" \
    python run_experiment.py --model_type ada-gp-vae \
    --data_type hirid --time_len 1 --testing --batch_size 64 --exp_name n_"$n"_1 \
    --basedir models/hirid/comp/ada/dim_"$lat_dim" --len_init same \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
    --seed "$seed" --latent_dim "$lat_dim" --encoder_sizes=128,128 --decoder_sizes=256,256 \
    --window_size 1 --sigma 1.005 --beta 1.0 --num_steps 404683 --kernel id \
    --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
    --visualize_score --save_score --eval_type dci --data_type_dci hirid --shuffle --dci_seed "$seed"
  done
done

#bsub -o models/hirid/comp/ada/dim_8/log_%J -g /gpvae_norm -W 75:00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python run_experiment.py --model_type ada-gp-vae --data_type hirid --time_len 1 --testing --batch_size 64 --exp_name n_83 --basedir models/hirid/comp/ada/dim_8 --len_init same --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz --seed 18112 --latent_dim 8 --encoder_sizes=128,128 --decoder_sizes=256,256 --window_size 1 --sigma 1.005 --beta 1.0 --num_steps 404683 --kernel id --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz --visualize_score --save_score --eval_type dci --data_type_dci hirid --shuffle --dci_seed 18112
#bsub -o models/hirid/comp/ada/dim_8/log_%J -g /gpvae_norm -W 75:00 -R "rusage[mem=65000,ngpus_excl_p=1]" python run_experiment.py --model_type ada-gp-vae --data_type hirid --time_len 1 --testing --batch_size 64 --exp_name n_xxx --basedir models/hirid/comp/ada/dim_8 --len_init same --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz --seed 18112 --latent_dim 8 --encoder_sizes=128,128 --decoder_sizes=256,256 --window_size 1 --sigma 1.005 --beta 1.0 --num_steps 404683 --kernel id --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz --visualize_score --save_score --eval_type dci --data_type_dci hirid --shuffle --dci_seed 18112