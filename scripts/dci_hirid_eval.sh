#!/bin/bash

for n in {1..100}; do
  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/std/dim_8/len_50/scaled/log_%J \
  -g /gpvae_disent -R "rusage[mem=5000]" python dsprites_dci.py \
  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/std/dim_8/len_50/scaled/210114_n_3 \
  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$RANDOM"
done
