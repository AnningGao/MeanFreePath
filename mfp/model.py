import numpy as np
from scipy.interpolate import interp1d
from astropy import cosmology
import emcee
from mfp import utils

cosmo_default = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

def model_fit(wave, flux, error, zmed, flux_telfer, wvmin, wvmax, 
              tilt, cosmo=cosmo_default, plot=None, method='MCMC',
              chi2_params=(30, 0.1, 100, (100, 1000)), 
              mcmc_params=(32, 2000, 200, (1, 200))):
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
    tilt: float
        Tilt of the continuum.
    norm_telf: float
        Normalization of the Telfer spectrum.
    cosmo: astropy.cosmology, optional, default: FlatLambdaCDM(H0=70, Om0=0.3)
        Cosmology used to calculate the mean free path.
    plot_name: str, optional, default: None
        Plot the fitting result and save the figure.
    method: str, optional, default: 'MCMC'
        Method to fit the stacked spectrum. Options are 'chi2' and 'MCMC'.
    mcmc_params: tuple, (nwalkers, nsteps, nburn, (norm_ini, kappa_ini)), optional, default: (32, 2000, 200, (1, 200))
        Parameters for the MCMC fitting. nwalkers is the number of walkers,
        nsteps is the number of steps, nburn is the number of burn-in steps,
        norm_ini is the initial value of the normalization, and kappa_ini is
        the initial value of kappa.
    chi2_params: tuple, (nnorm, norm_telf, ngrid, (kappa_min, kappa_max)), optional, default: (100, (100, 1000))
        Parameters for the chi2 fitting. nnorm is the number of normalization factor grid points,
        norm_telf is the half range of normalization factor (1-norm_telf, 1+norm_telf), 
        ngrid is the number of kappa grid points, kappa_min is the estimated minimum value of kappa, 
        and kappa_max is theestimated maximum value of kappa.

    ## Returns
    ### If method is 'chi2'
    mfp: astropy.units.quantity.Quantity
        Mean free path of the stacked spectrum.
    ### If method is 'MCMC'
    mfp: astropy.units.quantity.Quantity
        Best value of mean free path of the stacked spectrum.
    mfp_error: array
        Error of the mean free path of the stacked spectrum.
    sampler: emcee.EnsembleSampler
        Sampler of the MCMC fitting.
    """

    wLL = 911.7633  # Lyman limit in Angstrom
    gamma_lyman = 2.5  # slope of Lyman series optical depth evolution with redshift 
                       # (Worseck 2014 Eq4) 
    gd_stk = (wave>wvmin) & (wave<wvmax)
    ngpix = np.sum(gd_stk)  # number of pixels in the stacked spectrum
    a_flux, a_wave, a_error = flux[gd_stk], wave[gd_stk], error[gd_stk] 
    flux_tel = flux_telfer[gd_stk]  # Telfer spectrum in the stacked wavelength range

    #############################################
    # calculate Lyman series optical depth

    # No tilt first
    fLL_notilt = interp1d(wave, flux_telfer, kind='cubic')(wLL)
    fLL_data = interp1d(wave, flux, kind='cubic')(wLL)
    tau_Lyman_notilt = np.log(fLL_notilt/fLL_data)

    # Find the continnum correction
    conti_correct = (wLL/1450)**tilt
    tau_Lyman_correct = np.log(conti_correct*fLL_notilt/fLL_data)/tau_Lyman_notilt
    tau_Lyman_0 = tau_Lyman_notilt * tau_Lyman_correct

    # Just plain-jane now
    zeval = a_wave*(1+zmed)/wLL - 1.
    tau_Lyman = tau_Lyman_0 * ((1+zeval)/(1+zmed))**gamma_lyman

    #############################################
    # MFP model (really opacity model)

    z912 = a_wave * (1+zmed)/wLL - 1
    expon = 4.25

    if method == 'chi2':
        nnorm, norm_telf, ngrid, rngk = chi2_params

        #############################################
        # generate continuum model

        # overall continuum modification
        normv = (1-norm_telf) + 2*norm_telf*np.arange(0,step=1, stop=nnorm) / (nnorm-1)

        ltmp = a_wave.reshape(-1,1) @ np.ones((1,nnorm))
        ctmp = np.ones(ngpix).reshape(-1,1) @ np.array([normv]) 
        ttmp = flux_tel.reshape(-1,1) @ np.ones((1,nnorm))
        continuum = ctmp * ttmp * ((ltmp/1450)**tilt)  # column: continuum
                                                    # row: different normalization
        
        # Create the vector of kappa
        kvec = np.log10(rngk[0]) + \
            (np.log10(rngk[1]) - np.log10(rngk[0])) * np.arange(ngrid) / (ngrid - 1)
        kvec = 10**kvec

        # Fill up model tau grid
        kgrid = np.ones(ngpix).reshape(-1, 1) @ np.array([kvec])
        
        vecz = (1+z912)**2.75 * ( 1./(1+z912)**expon - 1./(1+zmed)**expon ) / expon 
        gridz = vecz.reshape(-1, 1) @ np.array([np.ones(ngrid)])
        tauLL_grid = kgrid * gridz

        # Set up
        exp_LL = np.exp(-tauLL_grid)
        exp_Lyman = np.exp(-tau_Lyman).reshape(-1, 1) @ np.ones((1, ngrid))

        # Create observational grids
        obs_grid = a_flux.reshape(-1, 1) @ np.ones((1, ngrid))
        error_grid = a_error.reshape(-1, 1) @ np.ones((1, ngrid))

        # chi^2 initialization
        chi2 = np.zeros((nnorm, ngrid))

        # Loop over continuum normalizations to calculate chi^2
        for ii in range(nnorm):
            conti = continuum[:, ii].reshape(-1, 1) @ np.ones((1, ngrid))
            model_flux = conti * exp_LL * exp_Lyman

            chi2_grid = np.sum((obs_grid - model_flux)**2 / error_grid**2, axis=0)
            chi2[ii,:] = chi2_grid

        # Find the minimum chi^2
        redchi2 = chi2 / (ngpix - 3)
        min_chi = np.min(redchi2)
        # Find the minimum chi^2 location
        imn = np.where(redchi2 == min_chi)
        coord =np.array([imn[0][0],imn[1][0]]) 
        # coord[0]: normalization index, coord[1]: kappa index

        #############################################
        # Plotting
        if plot is not None:
            conti = (continuum[:, coord[0]].reshape(-1, 1) @ np.ones((1, ngrid))) * exp_LL * exp_Lyman
            utils.plot_best_fit(a_wave, a_flux, conti[:, coord[1]], plot)
        
        return utils.mfp_calculation(kvec[imn[1]], zmed, a_wave, cosmo), redchi2

    elif method == 'MCMC':
        # Set up MCMC
        nwalkers, nsteps, nburn, (norm_ini, kappa_ini) = mcmc_params
        ndim = 2
        sampler = emcee.EnsembleSampler(nwalkers, ndim, utils.log_probability, 
                                        args=(a_flux, a_error, zmed, tau_Lyman_0, gamma_lyman, z912, flux_tel))
        initial = np.array([norm_ini, kappa_ini]) + 1e-4 * np.random.randn(nwalkers, ndim)

        # Run MCMC
        sampler.run_mcmc(initial, nsteps)
        flat_samples = sampler.get_chain(discard=nburn, thin=1, flat=True)
        mcmc_norm = np.percentile(flat_samples[:, 0], [16, 50, 84])
        mcmc_kappa = np.percentile(flat_samples[:, 1], [16, 50, 84])

        #############################################
        # Plotting
        if plot is not None:
            tau_LL = (1+z912)**2.75 * ( 1./(1+z912)**expon - 1./(1+zmed)**expon ) / expon
            continuum = mcmc_norm[1] * flux_tel * np.exp(-tau_Lyman) * np.exp(-mcmc_kappa[1] * tau_LL)
            utils.plot_best_fit(a_wave, a_flux, continuum, plot)
            
        mfps = np.array([utils.mfp_calculation(kappa, zmed, a_wave, cosmo) for kappa in mcmc_kappa])
        mfp_error = np.array([mfps[2]-mfps[1], mfps[0]-mfps[1]])
        return mfps[1], mfp_error, sampler

    else:
        raise ValueError("method must be either 'chi2' or 'MCMC'")
