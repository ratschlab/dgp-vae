#!/bin/bash

for n in {1..10}; do
  for class in svm rf; do
    bsub -g /gpvae_norm python classifier.py \
    --representation_path /cluster/home/bings/dgpvae/models/hirid/comp/base/dim_8/len_25/scaled/n_scales_4/210126_n_"$n" \
    --classifier "$class"
  done
done

for model in 210122_n_1_noband_1 210122_n_2_noband_1 210125_n_3_noband_1 210122_n_4_noband_1 210122_n_5_noband_1 210122_n_6_noband_1 210122_n_7_noband_1 210122_n_8_noband_1 210122_n_9_noband_1 210122_n_10_noband_1; do
  for class in svm rf; do
    bsub -g /gpvae_norm python classifier.py \
    --representation_path /cluster/home/bings/dgpvae/models/hirid/comp/ada/dim_8/"$model" \
    --classifier "$class"
  done
done