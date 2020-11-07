#!/bin/bash

for n in {1..5}; do
  seed=$RANDOM
  bsub -g /disent_baseline -R "rusage[mem=16000,ngpus_excl_p=1]" \
  python baselines/train_baseline.py --base_dir ada_new_pairing_sin \
  --output_dir n_"$n" --dim 64 --model adagvae --seed "$seed" \
  --steps 15620 --subset sin_rand
done