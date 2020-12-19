#!/bin/bash

for n in 99; do
  seed=0
  for subset in norb_full1 cars_full1 shapes_full1 shapes_full2; do
    bsub -o baselines/adagvae/"$subset"/log_%J \
    -g /disent_baseline -R "rusage[mem=16000,ngpus_excl_p=1]" \
    python baselines/train_baseline.py --base_dir ada_"$subset" \
    --output_dir n_"$n" --dim 64 --model adagvae --seed 0 \
    --steps 15620 --subset "$subset"
  done
done