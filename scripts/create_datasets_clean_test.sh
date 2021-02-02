#!/bin/bash

bsub -o create_dsprites_log_%J -g /gpvae_disent -R "rusage[mem=150000]" python data/create_dataset.py \
--data_type dsprites \
--factors_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range4.npz \
--out_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites

bsub -o create_norb_log_%J -g /gpvae_disent -R "rusage[mem=150000]" python data/create_dataset.py \
--data_type smallnorb \
--factors_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
--out_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb

bsub -o create_cars_log_%J -g /gpvae_disent -R "rusage[mem=150000]" python data/create_dataset.py \
--data_type cars3d \
--factors_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_cars_part1.npz \
--out_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d

bsub -o create_shapes_log_%J -g /gpvae_disent -R "rusage[mem=150000]" python data/create_dataset.py \
--data_type shapes3d \
--factors_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/factors_shapes_part2.npz \
--out_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d