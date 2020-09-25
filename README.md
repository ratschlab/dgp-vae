# DGP-VAE: Disentangled GP-VAE
Model to learn disentangled representations from time series. 

## Overview
Our model is an extension of the GP-VAE model used for time series imputation ([code](https://github.com/ratschlab/GP-VAE), [paper](http://arxiv.org/abs/1907.04155)).

The base GP-VAE model uses Variational Autoencoders in conjunction with a Gaussian Process prior to encode a latent time series from the time series in the feature space. 
We test a number of extensions to this model with the goal of further improving disentanglement.

* Weak Supervision: As the change over time of the underlying factors of variation is sparse in all
time series, we argue that most of the latent factors of paired time series will be shared. We explicitly enforce the sharing of some factors to exploit this assumption.

* Learnable Length Scales: The length scales of the GP kernel can now be learned to increase the expressive power of the model.

More extensions and improvements tbd. This is still a work in progress.

## Dependencies

* Python >= 3.6
* TensorFlow = 1.15
* Some more packages: see `requirements.txt`

## Run
1. Clone or download this repo. `cd` yourself to it's root directory.
2. Grab or build a working python enviromnent. [Anaconda](https://www.anaconda.com/) works fine.
3. Install dependencies, using `pip install -r requirements.txt`
4. Download data: TODO: Dump data and create download link.
5. Run command `python run_experiment.py ada-gp-vae --data_type dsprites --testing --exp_name <your_name> ...`

To run on ETH cluster prepend command with:
`-R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"`
   
   To see all available flags run: `python train.py --help`

## Reproducibility

TODO: add command for best run configuration.
  
