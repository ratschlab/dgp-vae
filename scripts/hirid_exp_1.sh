#!/bin/bash

mkdir -p models/hirid/std/dim_8/len_5/same
mkdir -p models/hirid/no_std/dim_8/len_5/same
mkdir -p models/hirid/std/dim_16/len_5/same
mkdir -p models/hirid/no_std/dim_16/len_5/same
mkdir -p models/hirid/std/dim_32/len_5/same
mkdir -p models/hirid/no_std/dim_32/len_5/same

mkdir -p models/hirid/std/dim_8/len_25/same
mkdir -p models/hirid/no_std/dim_8/len_25/same
mkdir -p models/hirid/std/dim_16/len_25/same
mkdir -p models/hirid/no_std/dim_16/len_25/same
mkdir -p models/hirid/std/dim_32/len_25/same
mkdir -p models/hirid/no_std/dim_32/len_25/same

mkdir -p models/hirid/std/dim_8/len_50/same
mkdir -p models/hirid/no_std/dim_8/len_50/same
mkdir -p models/hirid/std/dim_16/len_50/same
mkdir -p models/hirid/no_std/dim_16/len_50/same
mkdir -p models/hirid/std/dim_32/len_50/same
mkdir -p models/hirid/no_std/dim_32/len_50/same

mkdir -p models/hirid/std/dim_8/len_5/scaled
mkdir -p models/hirid/no_std/dim_8/len_5/scaled
mkdir -p models/hirid/std/dim_16/len_5/scaled
mkdir -p models/hirid/no_std/dim_16/len_5/scaled
mkdir -p models/hirid/std/dim_32/len_5/scaled
mkdir -p models/hirid/no_std/dim_32/len_5/scaled

mkdir -p models/hirid/std/dim_8/len_25/scaled
mkdir -p models/hirid/no_std/dim_8/len_25/scaled
mkdir -p models/hirid/std/dim_16/len_25/scaled
mkdir -p models/hirid/no_std/dim_16/len_25/scaled
mkdir -p models/hirid/std/dim_32/len_25/scaled
mkdir -p models/hirid/no_std/dim_32/len_25/scaled

mkdir -p models/hirid/std/dim_8/len_50/scaled
mkdir -p models/hirid/no_std/dim_8/len_50/scaled
mkdir -p models/hirid/std/dim_16/len_50/scaled
mkdir -p models/hirid/no_std/dim_16/len_50/scaled
mkdir -p models/hirid/std/dim_32/len_50/scaled
mkdir -p models/hirid/no_std/dim_32/len_50/scaled

for n in {1..5}; do
  seed=$RANDOM
  for sens_eval_type in std no_std; do
    for lat_dim in 8 16 32; do
      for len in 5,3,10 25,12,4 50,25,4; do
        IFS=',' read time_len window_size run_time <<< "${len}"
        bsub -o models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/same/log_%J \
        -g /gpvae_disent -W "$run_time":00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
        python run_experiment.py --model_type gp-vae --data_type hirid --time_len "$time_len" \
        --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/same \
        --len_init same --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_"$sens_eval_type".npz \
        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 \
        --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 2.0 --beta 1.0 \
        --num_epochs 1 --kernel cauchy --eval_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_sensitivity_eval_"$sens_eval_type".npz \
        --visualize --save --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_"$sens_eval_type".npz \
        --visualize_score --save_score --eval_type both --sens_eval_type "$sens_eval_type" \
        --data_type_dci hirid --shuffle

        bsub -o models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/scaled/log_%J \
        -g /gpvae_disent -W "$run_time":00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
        python run_experiment.py --model_type gp-vae --data_type hirid --time_len "$time_len" \
        --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/scaled \
        --len_init scaled --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_"$sens_eval_type".npz \
        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 --kernel_scales "$latent_dim"\
        --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 20.0 --beta 1.0 \
        --num_epochs 1 --kernel cauchy --eval_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_sensitivity_eval_"$sens_eval_type".npz \
        --visualize --save --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_"$sens_eval_type".npz \
        --visualize_score --save_score --eval_type both --sens_eval_type "$sens_eval_type" \
        --data_type_dci hirid --shuffle

      done
    done
  done
done