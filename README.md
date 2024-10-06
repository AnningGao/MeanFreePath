# MeanFreePath

This is the code for measuring the mean free path of ionizing photons in IGM using QSO composite spectra.

This repo is organized as follows:

1. `mfp`: Code that contains necessary functions and can be installed as a package.
   - `model.py`: MFP models.
   - `fitting.py`: Fit the composite spectrum using MFP models. $\chi^2$ and MCMC methods are available.
   - `stack.py`: Generate the composite spectrum.
   - `utils.py`: Useful functions.
2. `nb`: Jupyter notebooks that explains the usage of `mfp` package and the core of our analysis.
   - `stack.ipynb`: Explains the stacking procedure. 
   - `fitting.ipynb`: Explains the fitting procedure.