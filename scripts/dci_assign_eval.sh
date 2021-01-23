#!/bin/bash

for model_seed in 210120_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210121_n_7,16040 210121_n_8,8088 210121_n_9,22060 210121_n_10,29153; do
  IFS=',' read model seed <<< "${model_seed}"
  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_50/scaled/log_%J \
  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_50/scaled/"$model" \
  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
done