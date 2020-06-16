#!/bin/bash

for n in 1,2,3; do
  python baselines/train_baseline.py --base_dir test_dump --output_dir 300k_test_"$n" \
  --dim 64 --model adagvae --seed "$RANDOM" --steps 300000
done