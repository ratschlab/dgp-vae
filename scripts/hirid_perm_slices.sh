#!/bin/bash

for n in {0..258}; do
  bsub -g /gpvae_eval -R "rusage[mem=10000]" python hirid_labels.py -n "$n"
done