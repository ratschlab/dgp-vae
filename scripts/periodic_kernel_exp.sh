#!/bin/bash
mkdir -p models/periodic_kernel/base/len_5
mkdir -p models/periodic_kernel/base/len_10

for n in {1..5}; do
  seed=$RANDOM
  for len in 5 10; do
  bsub -o models/periodic_kernel/base/len_"$len"/log_%J -g /gpvae_disent \
  -R "rusage[mem=150000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
  python run_experiment.py --model_type gp-vae --data_type dsprites --time_len "$len" --testing --batch_size 32 \
  --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/dsprites_sin_resample.npz \
  --exp_name n_"$n" --basedir models/periodic_kernel/base/len_"%len" \
  --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 \
  --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 20 --beta 1.0 \
  --num_epochs 1 --kernel periodic --kernel_scales 16 --len_init scaled \
  --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_sin_resample.npz \
  --score_factors=1,2,3,4,5 --save_score --visualize_score
  done
done
