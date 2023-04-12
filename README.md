# Paper-CUQIpy-1-Core

This contains the code for the paper "CUQIpy -- Part I: computational UQ for inverse problems in Python".

## Installation
Install the package using pip (assuming python is installed):
```
pip install cuqipy
```
Some examples require additional packages like the plugins for CIL and PyTorch. These can be installed following the instructions on:
* [CUQIpy-CIL](https://github.com/CUQI-DTU/CUQIpy-CIL)
* [CUQIpy-PyTorch](https://github.com/CUQI-DTU/CUQIpy-PyTorch)

## Running the examples
The examples are located folders for each case study. The examples are written in Jupyter notebooks. To run the examples, you need to install [Jupyter](https://jupyter.org/install). 

One can also simply view the notebooks on GitHub by clicking on the notebook files in the folders (links below).

## Case studies
The following case studies are included in this repository:

* [Section 1: Introductory motivating example (2D deconvolution)](intro/intro.ipynb) in the folder `intro`
* [Section 3: Software overview (1D deconvolution sinc phantom)](deconvolution1D/paper1_deconv1D.ipynb) in the folder `deconvolution1D`
* [Section 3: Software overview (1D deconvolution square phantom)](deconvolution1D/paper1_deconv1D_square.ipynb) in the folder `deconvolution1D`
* [Section 4: Gravity anomaly inversion](gravity/gravity.ipynb) in the folder `gravity`
* [Section 5: CT using CUQIpy-CIL](CT/CT.ipynb) in the folder `CT`
* [Section 6: Eight Schools using CUQIpy-PyTorch](eight_schools/eight_schools.ipynb) in the folder `eight_schools`

