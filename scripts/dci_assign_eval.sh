#!/bin/bash

# Our model

#for model_seed in 210121_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210121_n_7,16040 210122_n_8,8088 210122_n_9,22060 210122_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_10/same/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_10/same/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done
#
#for model_seed in 210121_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210122_n_7,16040 210122_n_8,8088 210122_n_9,22060 210122_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_10/scaled/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_10/scaled/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done
#
#for model_seed in 210120_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210121_n_7,16040 210121_n_8,8088 210121_n_9,22060 210121_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_25/same/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_25/same/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done
#
#for model_seed in 210120_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210121_n_7,16040 210121_n_8,8088 210121_n_9,22060 210121_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_25/scaled/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_25/scaled/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done
#
#for model_seed in 210120_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210121_n_7,16040 210121_n_8,8088 210121_n_9,22060 210121_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_50/same/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_50/same/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done
#
#for model_seed in 210120_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210121_n_7,16040 210121_n_8,8088 210121_n_9,22060 210121_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_50/scaled/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_8/len_50/scaled/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done
#
#for model_seed in 210121_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210122_n_7,16040 210122_n_8,8088 210122_n_9,22060 210122_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_10/same/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_10/same/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done

#for model_seed in 210121_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210122_n_7,16040 210122_n_8,8088 210122_n_9,22060 210122_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_10/scaled/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_10/scaled/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed 10843
#done

#for model_seed in 210120_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210121_n_7,16040 210121_n_8,8088 210121_n_9,22060 210121_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_25/same/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_25/same/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done
#
#for model_seed in 210120_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210121_n_7,16040 210121_n_8,8088 210121_n_9,22060 210121_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_25/scaled/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_25/scaled/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done
#
#for model_seed in 210120_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210121_n_7,16040 210121_n_8,8088 210121_n_9,22060 210121_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_50/same/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_50/same/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done
#
#for model_seed in 210121_n_1,22019 210121_n_2,3412 210121_n_3,13351 210121_n_4,2677 210121_n_5,6071 210121_n_6,10843 210121_n_7,16040 210121_n_8,8088 210121_n_9,22060 210121_n_10,29153; do
#  IFS=',' read model seed <<< "${model_seed}"
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_50/scaled/log_%J \
#  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
#  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
#  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/hirid/comp1/base/dim_16/len_50/scaled/"$model" \
#  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
#done

# AdaGVAE

for model_seed in 210122_n_1_noband,22019 210122_n_2_noband,3412 210122_n_4_noband,2677 210122_n_5_noband,6071 210122_n_6_noband,10843 210122_n_7_noband,16040 210122_n_8_noband,8088 210122_n_9_noband,22060 210122_n_10_noband,29153; do
  IFS=',' read model seed <<< "${model_seed}"
  bsub -o /models/hirid/comp/ada/dim_8/log_%J \
  -g /gpvae_eval -R "rusage[mem=5000]" python dsprites_dci.py \
  --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_1.npy \
  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
  --model_name /models/hirid/comp/ada/dim_8/"$model" \
  --data_type_dci hirid --save_score --visualize_score --shuffle --dci_seed "$seed"
done