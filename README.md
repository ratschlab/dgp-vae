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

![A test image](figures/overview.png)

## Dependencies

* Python >= 3.6
* TensorFlow = 1.15
* disentanglement_lib (Available [here](https://github.com/google-research/disentanglement_lib))
* Some more packages: see `requirements.txt`

## Run
1. Clone or download this repo. `cd` yourself to its root directory.
2. Grab or build a working python enviromnent. [Anaconda](https://www.anaconda.com/) works fine.
3. Install dependencies, using `pip install -r requirements.txt`
4. Add the disentanglement_lib to your pythonpath.
4. Download data: TODO: Dump data and create download link.
5. Run command `python run_experiment.py ada-gp-vae --data_type dsprites --testing --exp_name <your_name> ...`

To run on ETH cluster prepend command with:
`-R "rusage[mem=200000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"`
   
   To see all available flags run: `python train.py --help`

## Reproducibility

TODO: add command for best run configuration.
  
