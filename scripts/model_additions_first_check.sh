#!/bin/bash

# AdaGVAE baseline
#bsub -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" \
#python baselines/train_baseline.py --base_dir test_dump --output_dir init_test_0 \
#--subset sin_rand --dim 64 --model adagvae --seed 0 --steps 15620

# AdaGPVAE
#bsub -g /gpvae_disent -R "rusage[mem=200000,ngpus_excl_p=1]" \
#-R "select[gpu_model0==GeForceGTX1080Ti]" python run_experiment.py \
#--model_type ada-gp-vae --data_type dsprites --testing \
#--data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
#--exp_name init_test_0 --basedir models/adagpvae/test_dump --seed 0 --banded_covar \
#--latent_dim 64 --encoder_sizes=32,256,256 --decoder_sizes=256,256,256 \
#--window_size 3 --sigma 1 --length_scale 2 --beta 1.0 --num_epochs 2 --kernel cauchy \
#--z_name factors_100k_5k.npz --save_score

# Lagging Inference
for n in {1..10}; do
  seed=$RANDOM
  bsub -g /gpvae_disent -R "rusage[mem=200000,ngpus_excl_p=1]" \
  -R "select[gpu_model0==GeForceGTX1080Ti]" python run_experiment.py \
  --model_type gp-vae --data_type dsprites --testing \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
  --exp_name init_test_"$n" --basedir models/lagging_inf/test_dump --seed "$seed" --banded_covar \
  --latent_dim 64 --encoder_sizes=32,256,256 --decoder_sizes=256,256,256 --lagging_inference \
  --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 --num_epochs 2 --kernel cauchy \
  --print_interval 500 --z_name factors_100k_5k.npz --save_score
done
