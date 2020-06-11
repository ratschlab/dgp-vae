#!/bin/bash

for epochs in 1 3 7 14 19; do
  mkdir -p models/gpvae/epoch_sweep0/epoch_"$epochs"
done

for n in {1..10}; do
  seed=$RANDOM
  for epochs in 1 3 7 14 19; do
    bsub -o models/gpvae/epoch_sweep0/epoch_"$epochs"/log_"$epochs"_"$n" -g /gpvae_disent \
    -R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
    python run_experiment.py --model_type gp-vae --data_type dsprites --testing\
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
    --exp_name dsprites_"$epochs"_n"$n" --basedir models/gpvae/epoch_sweep0/epoch_"$epochs" \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 \
    --beta 1.0 --num_epochs "$epochs" --kernel cauchy --z_name factors_100k_5k.npz --save_score
  done
done