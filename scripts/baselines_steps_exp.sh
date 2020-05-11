#!/bin/bash

#for steps in 15620 62480 93720 124960 156200; do
#  for n in {1..10}; do
#    seed=$RANDOM
#    for model in betatcvae factorvae dip_vae_i; do
#      bsub -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" \
#      python baselines/train_baseline.py --base_dir exp_steps0 \
#      --output_dir steps_"$steps"_n_"$n" --subset sin_rand --dim 64\
#      --model "$model" --seed "$seed" --steps "$steps"
#    done
#  done
#done

for model in betatcvae factorvae dip_vae_i; do
  echo -g /gpvae_disent -R "rusage[mem=16000,ngpus_excl_p=1]" \
  python baselines/train_baseline.py --base_dir exp_steps0 \
  --output_dir steps_"$steps"_n_"$n" --subset sin_rand --dim 64\
  --model "$model" --seed "$seed" --steps "$steps"
done
