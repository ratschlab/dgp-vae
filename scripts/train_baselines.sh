#!/bin/bash

# Train all baselines
for n in {1..10}; do
  SEED=$RANDOM
  bsub -o baselines/betatcvae/betatcvae_n"$n"/lsf_log_$n -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" \
  baselines/betatcvae/train.py --output_dir betatcvae_n$n --seed $SEED
  bsub -o baselines/betatcvae/factorvae_n"$n"/lsf_log_$n -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" \
  baselines/factorvae/train.py --output_dir factorvae_n$n --seed $SEED
  # bsub -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" \
  # baselines/dipvae_I/train.py --output_dir dipvae_I_n$n --seed $SEED
done
