# Seismic Fault Segmentation using 2D U-Net

This project is an implementation of a 2D U-Net convolutional neural network for seismic fault segmentation using synthetic data slices inspired by the [FaultSeg3D](https://github.com/xinwucwp/faultSeg) dataset.

## Project Summary

The goal of this project was to identify seismic faults from 2D slices of 3D seismic data volumes using a U-Net architecture implemented from scratch.


Key features:

* Uses synthetic seismic slices from the FaultSeg3D dataset for training and evaluation.
* Implements a custom U-Net with encoder-decoder structure for pixel-wise segmentation.
* Applies global normalization, slice-wise visualization, and data augmentation techniques.
* Evaluates the segmentation output using standard metrics (accuracy, Dice score, etc.).

## Model Architecture

* **U-Net-based** encoder-decoder with skip connections.
* Input: 2D grayscale seismic slices.
* Output: Binary fault mask per slice.
* Loss function: Binary Cross-Entropy.
* Optimizer: Adam.

## Dataset
2D Processed and normalized data - (https://www.kaggle.com/datasets/preetdaiict/processed-seismic-and-fault-img-xinwucwpfaultseg).

The dataset is based on the **FaultSeg3D** dataset introduced in the paper:

> **"FaultSeg3D: Using synthetic data sets to train an end-to-end convolutional neural network for 3D seismic fault segmentation"**
> *Xinming Wu et al., Geophysics, 2019*
> [Paper](https://doi.org/10.1190/geo2018-0646.1) | [GitHub Repo](https://github.com/xinwucwp/faultSeg)

### Folder structure:

* `train/seis/` and `train/fault/`: Training `.dat` volumes.
* `validation/`: Validation set (also in `.dat` format).
* Each `.dat` volume is sliced into 128 2D images.
* Processed images are stored as `.png` files.


---

### `.dat` to 2D Slice Conversion — `data_processing.ipynb`

This notebook loads 3D seismic and fault label volumes in `.dat` format (from the `train/` and `validation/` folders) and **slices each 3D volume into 128 2D images** along the depth axis.
Each `.dat` file produces 128 `.png` slices:

* Seismic slices are stored in `processed_seis/`
* Fault slices are stored in `processed_fault/`


---

### Normalization of Slices — `normalize_image.ipynb`

This script performs **global normalization** of all seismic and fault image slices:

* Each image is normalized by subtracting the mean and dividing by the standard deviation.
* Normalized images are saved to a new folder (e.g., `normalized_seis/`, `normalized_fault/`).
* This ensures consistent input range for model training.

---

### Reshape and Visualize Slices — `reshape_visualize.ipynb`

This notebook loads processed slices and:

* Reshapes them if necessary (e.g., for model compatibility).
* Visualizes both seismic and fault label slices using `matplotlib` for inspection.

Useful for verifying data quality before training.

---


## Getting Started

1. Clone this repo and download the FaultSeg3D dataset.
2. Preprocess the `.dat` files into 2D slices.
3. Normalize the data.
4. Train the U-Net model.
5. Evaluate and visualize the segmentation results.

## 🔧 Requirements

* Python ≥ 3.7
* TensorFlow / Keras
* NumPy, Matplotlib, scikit-image

## 📊 Results

The trained U-Net model shows effective segmentation of seismic faults in 2D slices, closely matching the ground truth masks from the synthetic dataset.
![image](https://github.com/user-attachments/assets/3b17f397-8e8b-4ff2-87a1-3e81b78d1149)

