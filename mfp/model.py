import numpy as np
import matplotlib.pyplot as plt
from linetools.spectra.xspectrum1d import XSpectrum1D
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.integrate import quad

def model_fit(wave, flux, error, zmed, flux_telfer, 
              wvmin, wvmax, tilt, norm_telf, ngrid, 
              rngk, plot=None):
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
    ngrid: int
        Number of grid points for the MFP model.
    rngk: array
        Range of the kappa values for the MFP model.
    plot_name: str
        Plot the fitting result and save the figure.
    """

    wLL = 911.7633  # Lyman limit in Angstrom
    gamma_lyman = 2.5  # slope of Lyman series optical depth evolution with redshift 
                       # (Worseck 2014 Eq4) 
    gd_stk = (wave>wvmin) & (wave<wvmax)
    ngpix = np.sum(gd_stk)  # number of pixels in the stacked spectrum
    a_flux, a_wave, a_error = flux[gd_stk], wave[gd_stk], error[gd_stk] 
    flux_tel = flux_telfer[gd_stk]  # Telfer spectrum in the stacked wavelength range

    #############################################
    # generate continuum model

    # overall continuum modification
    nnorm = 30
    normv = (1-norm_telf) + 2*norm_telf*np.arange(0,step=1, stop=nnorm) / (nnorm-1)

    ltmp = a_wave.reshape(-1,1) @ np.ones((1,nnorm))
    ctmp = np.ones(ngpix).reshape(-1,1) @ np.array([normv]) 
    ttmp = flux_tel.reshape(-1,1) @ np.ones((1,nnorm))
    continuum = ctmp * ttmp * ((ltmp/1450)**tilt)  # column: continuum
                                                   # row: different normalization

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

    # Create the vector of kappa
    kvec = np.log10(rngk[0]) + \
        (np.log10(rngk[1]) - np.log10(rngk[0])) * np.arange(ngrid) / (ngrid - 1)
    kvec = 10**kvec

    z912 = a_wave * (1+zmed)/wLL - 1
    # Fill up model tau grid
    kgrid = np.ones(ngpix).reshape(-1, 1) @ np.array([kvec])
    expon = 4.25
    vecz = (1+z912)**2.75 * ( 1./(1+z912)**expon - 1./(1+zmed)**expon ) / expon 
    gridz = vecz.reshape(-1, 1) @ np.array([np.ones(ngrid)])
    tauLL_grid = kgrid * gridz

    # Create observational grids
    obs_grid = a_flux.reshape(-1, 1) @ np.ones((1, ngrid))
    error_grid = a_error.reshape(-1, 1) @ np.ones((1, ngrid))

    # chi^2 initialization
    chi2 = np.zeros((nnorm, ngrid))

    # Set up
    exp_LL = np.exp(-tauLL_grid)
    exp_Lyman = np.exp(-tau_Lyman).reshape(-1, 1) @ np.ones((1, ngrid))

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
        plt.plot(a_wave, a_flux, 'k', label='data')
        plt.plot(a_wave, conti[:, coord[1]], c='r', linestyle='--', label='best fit')
        plt.title(plot)
        plt.xlabel('Wavelength (Angstrom)')
        plt.ylabel('Flux')
        plt.legend()
        plt.savefig(plot+'.png')
        plt.close()
            
    #############################################
    # Calculate the MFP

    best_k = kvec[imn[1]]
    dwv_fin = wave[1] - wave[0]
    # extend the wavelength grid to contain the tau=1 point
    z912_extend = -np.flip(np.arange(step=dwv_fin,start=-a_wave[-1], stop=-400)) * (1+zmed)/wLL - 1
    vecz_extend = (1+z912_extend)**2.75 * ( 1./(1+z912_extend)**expon - 1./(1+zmed)**expon ) / expon 
    tauLL_grid_extend = best_k * vecz_extend
    z912_best = interp1d(tauLL_grid_extend, z912_extend, kind='cubic')(1.0)
    mfp = proper_distance(z912_best, zmed, 0.3, 0.7, 0, 0, 0.7) # default cosmology

    return mfp


c = 2.99792458e10 # speed of light in cm/s
pc = 3.0856776e18 # parsec in cm
def H0(h100):
    """
    Hubble constant in units of cm/s/cm.
    """
    return 100 * h100 * 1e5 / (1e6 * pc)

def proper_distance(z_ini, z_end, Om, Ol, Or, Ok, h100):
    """
    Calculate the proper distance between two redshifts.

    ## Parameters
    z_ini: float
        Initial redshift.
    z_end: float
        Final redshift.
    Om: float
        Density parameter of matter.
    Ol: float
        Density parameter of dark energy.
    Or: float
        Density parameter of radiation.
    Ok: float
        Density parameter of curvature.
    h100: float
        Hubble constant in units of 100 km/s/Mpc.
    """
    # Calculate the integral
    integral = quad(lambda z: 1.0 / (1+z) / \
                    np.sqrt(Om * (1.0 + z)**3 + Or * (1.0 + z)**4 + \
                            Ok * (1.0 + z)**2 + Ol), z_ini, z_end)[0]
    # Calculate the proper distance
    d = c / H0(h100) * integral
    return d / (1e6 * pc) # Convert to Mpc


def telfer(wave_fin, wave_ori, flux_telfer_ori):
    """
    Rebin the Telfer spectrum to the same wavelength grid as the stacked data.
    
    ## Parameters
    wave: array
        Wavelength grid of the stacked data.
    flux_telfer_ori: array
        Flux of the Telfer spectrum.
    """
    wave_ori = wave_ori * u.angstrom
    flux = flux_telfer_ori / \
           np.median(flux_telfer_ori[(wave_ori.value > 1450) & (wave_ori.value < 1470)])
    tel =XSpectrum1D.from_tuple((wave_ori, flux))
    return tel.rebin(wave_fin*u.angstrom).data['flux'].data[0]
