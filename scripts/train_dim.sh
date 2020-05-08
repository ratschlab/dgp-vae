#!/bin/bash

# Create directory for each tested latent dimension
for dim in 6 16 32 64 128; do
  mkdir -p models/dim/dsprites_dimsweep_"$dim"_0
done

for n in {1..10}; do
  SEED=$RANDOM
  for dim in 6 16 32 64 128; do

  bsub -o models/dim/dsprites_dimsweep_"$dim"_0/log_"$dim"_$n -g /gpvae_disent \
  -R "rusage[mem=120000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"\
  python train_exp.py --model_type gp-vae --data_type dsprites --testing\
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_100k_5k.npz \
  --exp_name dsprites_"$dim"_n$n --basedir models/dim/dsprites_dimsweep_"$dim"_0 \
  --seed $SEED --banded_covar --latent_dim $dim --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 --num_epochs 1

  done
done
