import numpy as np
from scipy.interpolate import interp1d
import emcee
from mfp import utils
from tqdm import tqdm

from mfp.model import Fiducial
model_default = Fiducial()

def model_fit(wave, flux, error, zmed, flux_telfer, wvmin, wvmax,
              gamma_lyman=3.0, plot=None, method='mcmc', model=model_default):
    """
    Fit the stacked spectrum with the MFP model.

    ## Parameters
    wave: array
        Wavelength array of the stacked spectrum.
    flux: array
        Flux array of the stacked spectrum.
    error: array
        Error array of the stacked spectrum.
    zmed: float
        Median redshift of the stacked spectrum.
    flux_telfer: array
        Flux array of the Telfer spectrum.
    wvmin: float
        Minimum wavelength of the stacked spectrum.
    wvmax: float
        Maximum wavelength of the stacked spectrum.
    gamma_lyman: float, optional, default: 3.0
        Power law index of the Lyman series opacity.
    plot: str, optional, default: None
        Plot the fitting result and save the figure.
    method: str, optional, default: 'mcmc'
        Method to fit the stacked spectrum. Options are 'chi2' and 'mcmc'.
    model: mfp.model.Model, optional, default: mfp.models.Fiducial()
        Model to fit the stacked spectrum. You can design it yourself!

    ## Returns
    ### If method is 'chi2'
    theta_best: array
        Best fit parameters of the model.
    chi2_min: float
        Minimum chi2 value.
    chi2_matrix: array
        Chi2 matrix of the grid search.
    ### If method is 'mcmc'
    mle_sample: array
        Maximum likelihood estimate of the parameters.
    stds: array
        Standard deviations of the parameters (assuming Gaussian dist).
    sampler: emcee.EnsembleSampler
        Sampler of the MCMC fitting.
    """
    # Prepare data
    wLL = 911.7633  # Lyman limit in Angstrom
    gd_stk = (wave>wvmin) & (wave<wvmax)
    a_flux, a_wave, a_error = flux[gd_stk], wave[gd_stk], error[gd_stk]
    flux_tel = flux_telfer[gd_stk]  # Telfer spectrum in the stacked wavelength range

    # calculate Lyman series optical depth
    fLL_notilt = interp1d(wave, flux_telfer, kind='cubic')(wLL)
    fLL_data = np.median(flux[np.abs(wave-wLL)<10])
    tau_Lyman_0 = np.log(fLL_notilt/fLL_data)

    z912 = a_wave * (1+zmed)/wLL - 1

    model.get_args(a_flux, a_error, zmed, tau_Lyman_0, gamma_lyman, z912, flux_tel)

    if method == 'chi2':
        chi2_params = model.chi2_params

        # Create theta grid
        theta_space = []
        for i in range(model.ndim):
            if chi2_params["grid_method"][i]=="linear":
                theta_space.append(np.linspace(chi2_params["range"][i][0],
                                               chi2_params["range"][i][1],
                                               chi2_params["ngrid"][i]))
            elif chi2_params["grid_method"][i]=="log":
                theta_space.append(np.logspace(np.log10(chi2_params["range"][i][0]),
                                               np.log10(chi2_params["range"][i][1]),
                                               chi2_params["ngrid"][i]))
            else:
                raise ValueError("grid_method must be either 'linear' or 'log'")
        theta_space = np.stack(np.meshgrid(*theta_space, indexing='ij'), axis=-1)
        # Index for each group of theta
        indices = np.stack(np.meshgrid(*[np.arange(chi2_params["ngrid"][i]) for i in range(model.ndim)], indexing='ij'), axis=-1)

        # Calculate chi2
        # Time consuming part, maybe parallelize?
        chi2_matrix = np.zeros(theta_space.shape[:-1])
        for index in indices.reshape(-1, model.ndim):
            theta = theta_space[tuple(index)]
            continuum = model.get_continuum(theta)
            chi2 = np.sum((continuum - a_flux)**2 / a_error**2)
            chi2_matrix[tuple(index)] = chi2

        # Get minimum chi2
        chi2_min = np.min(chi2_matrix)
        coord = np.where(chi2_matrix == chi2_min)
        theta_best = theta_space[coord][0]

        # Plotting
        if plot is not None:
            continuum = model.get_continuum(theta_best)
            utils.plot_best_fit(a_wave, a_flux, continuum, plot)

        return theta_best, chi2_min, chi2_matrix

    if method == 'mcmc':
        # Set up MCMC
        nwalkers = model.mcmc_params["run_params"]["nwalkers"]
        nsteps = model.mcmc_params["run_params"]["nsteps"]
        nburn = model.mcmc_params["run_params"]["nburn"]
        ndim = model.ndim

        sampler = emcee.EnsembleSampler(nwalkers, ndim, model.log_probability)
        initial = np.array(model.mcmc_params["init"]) + 1e-4 * np.random.randn(nwalkers, ndim)

        # Run MCMC
        sampler.run_mcmc(initial, nsteps)

        # Get MLE result
        log_probs = sampler.get_log_prob()
        max_prob_idx = np.unravel_index(log_probs.argmax(), log_probs.shape)
        mle_sample = sampler.get_chain()[max_prob_idx]

        # Get uncertainties (assuming Gaussian dist)
        # TODO: consider uncertainty estimation from an asymmetric dist
        flat_samples = sampler.get_chain(discard=nburn, thin=1, flat=True)
        stds = np.std(flat_samples, axis=0)

        # Plotting
        if plot is not None:
            continuum = model.get_continuum(mle_sample)
            utils.plot_best_fit(a_wave, a_flux, continuum, plot)

        return mle_sample, stds, sampler

    else:
        raise ValueError("method must be either 'chi2' or 'mcmc'")
