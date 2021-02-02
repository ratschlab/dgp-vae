# DGP-VAE: Disentangled GP-VAE
Tensorflow implementation of the DGP-VAE model. 
Accompanying code for our paper.

## Overview
We investigate the performance of a [GP-VAE](http://arxiv.org/abs/1907.04155) type model to learning
disentangled representations from time series by providing a slightly modified model: the DGP-VAE.

Our model learns disentangled representations from sequential data 
by modeling each independent latent channel with a Gaussian Process and employing
a structured variational distribution that can capture long-term dependencies in time.

We show the efficacy of our approach in two experiments:

1) A benchmark experiment comparing against state-of-the-art disentanglement models on synthetic data.

2) An experiment on real medical time series data, where we provide a detailed comparison 
with the state-of-the-art model for disentanglement that can exploit the structure of sequential data.

![DGP-VAE overview](figures/overview.png)

## Dependencies

* Python >= 3.6
* TensorFlow = 1.15
* disentanglement_lib (Available [here](https://github.com/google-research/disentanglement_lib))
* Some more packages: see `requirements.txt`

## Run
1. Clone or download this repo. `cd` yourself to its root directory.
2. Grab or build a working python enviromnent. [Anaconda](https://www.anaconda.com/) works fine.
3. Install dependencies, using `pip install -r requirements.txt`
4. Clone or download the disentanglement_lib and add its root directory to your `PYTHONPATH`.
5. Download data: TODO: Dump data and create download link.
6. Run the command `python run_experiment.py --model_type dgp-vae --data_type {dsprites, smallnorb, cars3d, shapes3d, hirid} 
--exp_name <your_name> ...`

   
   To see all available flags run: `python train.py --help`

## Reproducibility
The exact hyperparameters used for the experiments are reported in our paper.
To reproduce our final results run the following commands:

* dSprites, SmallNORB, Cars3D, Shapes3D: `python run_experiment.py --model_type dgp-vae 
--data_type {dsprites, smallnorb, cars3d, shapes3d} --time_len 5 --testing --batch_size 32 
--exp_name reproduce_{dsprites, smallnorb, cars3d, shapes3d} --seed $RANDOM --banded_covar 
--latent_dim 64 --encoder_sizes=32,256,256 --decoder_sizes=256,256,256 --window_size 3 
--sigma 1 --length_scale 2 --beta 1.0 --data_type_dci {dsprites, smallnorb, cars3d, shapes3d} 
--shuffle --save_score --visualize_score`
  
