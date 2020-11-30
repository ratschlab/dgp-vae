#!/bin/bash

#for n in {1..5}; do
#  seed=$RANDOM
#  for subset in gp_full_1 gp_part_1 gp_full_2 gp_part_2 gp_full_const_1 gp_part_const_1; do
#    bsub -o baselines/adagvae/"$subset"/log_%J \
#    -g /disent_baseline -R "rusage[mem=16000,ngpus_excl_p=1]" \
#    python baselines/train_baseline.py --base_dir ada_"$subset" \
#    --output_dir n_"$n" --dim 64 --model adagvae --seed "$seed" \
#    --steps 15620 --subset "$subset"
#  done
#done

for n in {1..5}; do
  seed=$RANDOM
  for subset in gp_full_3 gp_part_3 gp_full_4 gp_part_4; do
    bsub -o baselines/adagvae/"$subset"/log_%J \
    -g /disent_baseline -R "rusage[mem=16000,ngpus_excl_p=1]" \
    python baselines/train_baseline.py --base_dir ada_"$subset" \
    --output_dir n_"$n" --dim 64 --model adagvae --seed "$seed" \
    --steps 15620 --subset "$subset"
  done
done