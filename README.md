# DL-ROM: Deep Learning for Reduced Order Modelling

![DL-ROM](Images/DL_POF_2.gif)

## Overview

This repository contains the implementation of DL-ROM: [Deep Learning for Reduced Order Modelling and Efficient Temporal Evolution of Fluid Simulations](http://arxiv.org/abs/2107.04556).

## About DL-ROM

Reduced Order Modeling (ROM) creates low-order, computationally inexpensive representations of higher-order dynamical systems. DL-ROM (Deep Learning - Reduced Order Modelling) uses neural networks for non-linear projections to reduced-order states, predicting future time steps efficiently using 3D Autoencoder and 3D U-Net architectures without ground truth supervision or solving expensive Navier-Stokes equations. It achieves significant computational savings while maintaining accuracy.

![Framework](Images/framework_nn.png)
**Framework for the transient Reduced order Model (DL-ROM). 10 snapshots of the previously solved CFD data are stacked and used as input to the model. The model then uses a 3D encoder architecture to reduce the high-dimensional CFD data to reduced order latent vector. This latent space is then deconvolved using a 3D-decoder to produce the higher order CFD prediction at timestep t+1.**

## Installation

Clone the repository:

```sh
git clone https://github.com/Shen-Ming-Hong/DL-ROM.git
cd DL-ROM/
```

## Download Datasets

Download the 5 datasets used for the evaluation of the DL-ROM model and place them in the `data` folder.

The data can be downloaded from the following [Google Drive Link](https://drive.google.com/drive/folders/1JI4jTBM1vE9AjkdxYce0GCDG9tCi5FtQ?usp=sharing).

## Running the Model

```sh
Usage: test_benchmark.py [-h] [-N NUM_EPOCHS] [-B BATCH_SIZE] [-d_set {SST,2d_plate,2d_cylinder_CFD,2d_sq_cyl,channel_flow}] [--train] [--transfer] [--test] [-test_epoch EPOCH] [--simulate]

Arguments:
  -h, --help            Show this help message and exit
  -N NUM_EPOCHS         Number of epochs for model training
  -B BATCH_SIZE         The batch size of the dataset (default: 16)
  -d_set DATASET_NAME   Name of the dataset to perform model training with
  --train               Run DL-ROM in training mode
  --transfer            Use pretrained weights to speed up model convergence (must be used with --train flag)
  --test                Run DL-ROM in testing mode
  -test_epoch EPOCH     Use the weight saved at epoch number specified by EPOCH
  --simulate            Run DL-ROM in simulation mode (used for in-the-loop prediction of future simulation timesteps)
```

### Example Usage in Colab

Link👉 [Google Colab](https://colab.research.google.com/drive/1Udy-rfNUtSZSVfG0gtq7CDoSv6PjqiuL?usp=sharing)

```sh
# Clone the DL-ROM GitHub repository
!git clone https://github.com/Shen-Ming-Hong/DL-ROM.git

# Change directory to the cloned DL-ROM directory
%cd DL-ROM/

# List the contents of the current directory
!ls

# Download the dataset from the specified Google Drive folder into the 'data' directory
!gdown --folder https://drive.google.com/drive/folders/1JI4jTBM1vE9AjkdxYce0GCDG9tCi5FtQ -O data

# List the contents of the 'data' directory to confirm the download
!ls data

# Change directory to the 'code' folder within the DL-ROM directory
%cd code/
```

### Training

```sh
!python main.py -N 100 -B 16 -d_set 2d_cylinder_CFD --train
```

### Testing

```sh
!python main.py --test -test_epoch 100 -d_set 2d_cylinder_CFD
```

### Transfer Learning

```sh
!python main.py --transfer -N 100 -B 16 -d_set 2d_cylinder_CFD --train
```

### Simulating

```sh
!python main.py --simulate -test_epoch 100 -d_set 2d_cylinder_CFD
```

## Visualizing Results

```sh
Usage: visualize.py [-mode {result, simulate}] [-d_set {SST,2d_plate,2d_cylinder_CFD,2d_sq_cyl,channel_flow}] [-freq] [--MSE] [--train_plot]

Arguments:
  -mode result:         Plots the results of the validation set of the selected dataset with the given frequency
  -mode simulate:       Creates a MSE lineplot for the validation set of the selected dataset and two animations: one for prediction and one for the groundtruth
  -d_set SST:           Dataset name (default: 2d_cylinder_CFD)
  -freq INT:            Frequency of saving images in the [-mode result] (default: 20)
  --MSE:                Creates a barplot for comparing the MSE achieved on all supported datasets using our approach
  --train_plot:         Plots the training and validation loss curve over epochs for the selected dataset
```

### Example

```sh
python visualize.py -mode results -d_set 2d_cylinder -freq 10
python visualize.py -mode simulate -d_set 2d_cylinder
```

## Results

<p align="center"><img src="Images/Architecture Finale.png" alt="">
<figcaption align="left"><b>Framework for the transient Reduced 3D Autoencoder based UNet Model Architecture for our framework DL-ROM. 10 timesteps are concatenated to generate temporal context as the input to the architecture. Each block represents the intermediate size of the data. The arrows represent the skip connections between encoder and decoder part of the architecture. The bottleneck represents a 1D vector of the Reduced Order States of the input.</b></figcaption></p>

<p align="center"><img src="Images/results.png" alt="">
<figcaption align="left"><b>Results obtained on the 5 datasets using our Deep Learning based approach for Reduced Order Modelling. Each dataset is split into training and validation subsets. The labels and the corresponding predictions presented are from the validation split which are not used for training. Progression of MSE with timesteps evaluated on the validation dataset. The DL-ROM model is provided with data for only the initial timestep. The simulation evolves using the predictions of the previous timesteps without supervision from the ground truth values. As expected the value of negative log MSE gradually decreases over time due to accumulation of errors. Note that decreasing lineplots of Negative Log MSE represent increasing MSE values.</b></figcaption></p>

<p align="center"><img src="Images/barplot_time.png" alt="">
<figcaption align="left"><b>Comparing average CPU runtime for one iteration of the simulation. Comparison has been made between CFD (solved on OpenFOAM using the PimpleFOAM solver) and the DL-ROM machine learning model. The DL-ROM outperforms the runtimes of CFD simulations by nearly 2 orders of magnitude.</b></figcaption></p>

[Paper](https://arxiv.org/abs/2107.04556) / [Code](https://github.com/pranshupant/DL-ROM)

```
@misc{pant2021deep,
      title={Deep Learning for Reduced Order Modelling and Efficient Temporal Evolution of Fluid Simulations}, 
      author={Pranshu Pant and Ruchit Doshi and Pranav Bahl and Amir Barati Farimani},
      year={2021},
      eprint={2107.04556},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn}
}
```
