#!/bin/bash

for n in {1..10}; do
  seed=$RANDOM
  for steps in 15620 50000 100000 200000 300000; do
    bsub -R "rusage[mem=16000]" \
    python baselines/train_baseline.py --base_dir exp_steps_sub1/steps_"$steps" \
    --output_dir steps_"$steps"_n_"$n" --dim 64 --model adagvae --seed "$seed" \
    --steps "$steps" --subset sin_rand
  done
done