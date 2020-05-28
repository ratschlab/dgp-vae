#!/bin/bash

for steps in 15620 50000 100000 200000 300000; do
  for model in annealedvae betavae dipvae_ii; do
    python dci_aggregate.py --model "$model" --base_dir exp_steps1 \
    --exp_name steps_"$steps" --save
  done
done
