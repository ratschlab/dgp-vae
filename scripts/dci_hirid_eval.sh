#!/bin/bash

#for n in {1..100}; do
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/std/dim_8/len_50/scaled/log_%J \
#  -g /gpvae_disent -R "rusage[mem=5000]" python dsprites_dci.py \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/std/dim_8/len_50/scaled/210114_n_3 \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$RANDOM"
#done

for n in {200..210}; do
  bsub -o /cluster/home/bings/dgpvae/models/hirid/std/dim_8/len_50/scaled/log_%J \
  -g /gpvae_disent -R "rusage[mem=5000]" python dsprites_dci.py \
  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
  --model_name /cluster/home/bings/dgpvae/models/hirid/std/dim_8/len_50/scaled/210119_n_"$n" \
  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed 18112
done

bsub -o /cluster/home/bings/dgpvae/models/hirid/ada/std/dim_8/log_%J -g /gpvae_norm -R "rusage[mem=5000]" python dsprites_dci.py --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz --model_name /cluster/home/bings/dgpvae/models/hirid/ada/std/dim_8/210119_n_82 --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed 18112
