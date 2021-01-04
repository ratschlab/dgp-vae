#!/bin/bash

for n in {1..10}; do
  seed=$RANDOM
  for model in adagvae annealedvae betavae betatcvae factorvae dipvae_i dipvae_ii; do
#    for i in smallnorb,norb_full1 cars3d,cars_full1 shapes3d,shapes_full1 shapes3d,shapes_full2; do
    for i in dsprites_full,gp_full_3 dsprites_full,gp_part_3 dsprites_full,gp_full_4 dsprites_full,gp_part_4; do
      IFS=',' read data subset <<< "${i}"
      bsub -o baselines/adagvae/"$subset"/log_%J \
      -g /disent_baseline -R "rusage[mem=60000,ngpus_excl_p=1]" \
      python baselines/train_baseline.py --base_dir "$subset"_1 \
      --output_dir n_"$n" --dim 64 --model "$model" --seed "$seed" \
      --steps 15620 --data "$data" --subset "$subset"
    done
  done
done