#!/bin/bash

for n in 201224_n_1 201224_n_2 201224_n_3 201224_n_4 201225_n_5 201225_n_6 201225_n_7 201225_n_8 201225_n_9 201225_n_10; do
  bsub -R "rusage[mem=15000]" python dsprites_dci.py \
  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
  --model_name /cluster/home/bings/dgpvae/models/norb_full1/base/len_5/same/"$n" --save_score
done

for n in 201229_n_1 201230_n_2 201230_n_3 201231_n_4 201231_n_5 201231_n_6 210101_n_7 210101_n_8 210101_n_9 210101_n_10; do
  bsub -R "rusage[mem=15000]" python dsprites_dci.py \
  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
  --model_name /cluster/home/bings/dgpvae/models/cars_part1/base/len_5/same/"$n" --save_score
done

for n in 210102_n_1 210102_n_2 210102_n_3 210102_n_4 210102_n_5 210102_n_6 210102_n_7 210102_n_8 210102_n_9 210103_n_10; do
  bsub -R "rusage[mem=15000]" python dsprites_dci.py \
  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
  --model_name /cluster/home/bings/dgpvae/models/shapes_part2/base/len_5/same/"$n" --save_score
done