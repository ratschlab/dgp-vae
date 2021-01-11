#!/bin/bash

for n in {1..5}; do
  seed=$RANDOM
  for sens_eval_type in std no_std; do
    for lat_dim in 8 16 32; do
      for len in 5,3 25,12 50,25; do
        IFS=',' read time_len window_size <<< "${len}"
        bsub -o models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/same/log_%J \
        -g /gpvae_disent -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
        python run_experiment.py --model_type gp-vae --data_type hirid --time_len "$time_len" \
        --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/same \
        --len_init same --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 \
        --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 2.0 --beta 1.0 \
        --num_epochs 10 --kernel cauchy --eval_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_sensitivity_eval_std.npz \
        --visualize --save --eval_type sens --sens_eval_type "$sens_eval_type"

        bsub -o models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/scaled/log_%J \
        -g /gpvae_disent -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
        python run_experiment.py --model_type gp-vae --data_type hirid --time_len "$time_len" \
        --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/scaled \
        --len_init scaled --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 --kernel_scales "$latent_dim"\
        --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 20.0 --beta 1.0 \
        --num_epochs 10 --kernel cauchy --eval_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_sensitivity_eval_std.npz \
        --visualize --save --eval_type sens --sens_eval_type "$sens_eval_type"
      done
    done
  done
done