#!/bin/bash

for n in {1..10}; do
  seed=$RANDOM
  for steps in 15620 50000 100000 200000 300000; do
    for model in adagvae annealedvae betavae betatcvae factorvae dipvae_i dipvae_ii; do
      bsub -g /disent_baseline -R "rusage[mem=16000,ngpus_excl_p=1]" \
      python baselines/train_baseline.py --base_dir exp_steps_full1/steps_"$steps" \
      --output_dir steps_"$steps"_n_"$n" --dim 64 --model "$model" --seed "$seed" \
      --steps "$steps"
    done
  done
done
