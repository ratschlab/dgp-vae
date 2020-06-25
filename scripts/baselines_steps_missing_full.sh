#!/bin/bash

# missing adagvae steps
seeds=(3963 25513 10353 14012 19903 26616 32096 22381 23778 10513)
for n in ${!seeds[@]}; do
  let n+=1
  echo "$n"
  echo "${seeds[$n-1]}"
  for steps in 200000 300000; do
    bsub -W 13:00 -g /disent_baseline_24 -R "rusage[mem=16000,ngpus_excl_p=1]" \
    python baselines/train_baseline.py --base_dir exp_steps_full1/steps_"$steps" \
    --output_dir steps_"$steps"_n_"$n" --dim 64 --model adagvae --seed "${seeds[$n-1]}" \
    --steps "$steps"
  done
done

# missing runs for n=4
seed=14012
for steps in 100000; do
  for model in adagvae dipvae_i dipvae_ii; do
    bsub -g /disent_baseline -R "rusage[mem=16000,ngpus_excl_p=1]" \
    python baselines/train_baseline.py --base_dir exp_steps_full1/steps_"$steps" \
    --output_dir steps_"$steps"_n_4 --dim 64 --model "$model" --seed "$seed" \
    --steps "$steps"
  done
done

for steps in 200000 300000; do
  for model in annealedvae betavae betatcvae factorvae dipvae_i dipvae_ii; do
    bsub -g /disent_baseline -R "rusage[mem=16000,ngpus_excl_p=1]" \
    python baselines/train_baseline.py --base_dir exp_steps_full1/steps_"$steps" \
    --output_dir steps_"$steps"_n_4 --dim 64 --model "$model" --seed "$seed" \
    --steps "$steps"
  done
done