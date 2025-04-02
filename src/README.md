# Fitness landscaep graph ruggedness

This is the official repository for the manuscript `Fitness Landscape Ruggedness Arises from Biophysical Complexity` and constains the source code to measure fitness landscape ruggedness as the timestep parameter of the heat diffusion kernel. Ipynbs are also provided in `figure_notebooks` to reproduce all data, analysis and figures presented in the manuscript main and supplementary texts. 

## Python environment
All scripts were run using Python 3.11.7 using the packages listed in `requirements.txt`. Fist, create a virtual python environment


`python3 -m venv .env`
`source .env/bin/activate`

And install the packages with

`pip install -r requirements.txt`


To reproduce analyses and figures from main and supplementary texts first retrieve the appropriate data and then run the provided figure notebooks. Open source data from other studies has been consolidated from the original publications and is hosted at the affiliated zenodo. 

Download this repository, and move it to `./data_files`.

Move the ipynb files from `./figure_notebooks` to `./src`, or update the ipynb import paths. 

## File overview

### GMRF - `./src/gaussian_markov_random_field.py` 
Source code to model fitness landscapes as a GMRF, with covariance defined by the heat diffusion kernel. 

### GFT - `./src/graph_fourier_transform.py`
Source code for the graph fourier transform used throughout the codebase and manuscript. 

### Graph constructrion and analysis - `./src/graph_ruggedness_de.py`
Source code to construct fitness landscape graphs and meausre quantities such as the (local) Dirichlet energy. 

### Utils - `./src/graph_utils.py`
Utility functions to visualise graphs with a signal cast over them. 

### NK Landscape simulations - `./src/nk_landscape.py`
Source code to produce NK landscapes at set values of `N`, `K` and alphabet sizes. 

### Sequence evolution simulations - `./src/sequence_evolution.py`
Source code to simulate sequence evolution over phylogenetic trees for sparse fitness landscape datasets. 

### Solution space simulations - `./src/solution_space_simulation.py`
Source code for spectral clustering to simulate fitness landscapes of different solution sizes.

### t parameter fitting - `./src/timestep_opt.py`
Source code to estimate the posterior distribution of t, fit the MAP t value to a fitness landscape and to perform Bayesian analyses (such as compute the Bayes factor).