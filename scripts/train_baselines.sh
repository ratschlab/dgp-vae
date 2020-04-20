#!/bin/bash

# Train all baselines
for n in 7; do
  SEED=$RANDOM
  bsub -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" \
  python baselines/betatcvae/train.py --output_dir betatcvae_n$n --seed $SEED
  bsub -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" \
  python baselines/factorvae/train.py --output_dir factorvae_n$n --seed $SEED
  # bsub -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" \
  # python baselines/dipvae_i/train.py --output_dir dipvae_i_n$n --seed $SEED
done
