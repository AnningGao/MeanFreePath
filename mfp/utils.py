import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy import units as u
from astropy import cosmology
from linetools.spectra.xspectrum1d import XSpectrum1D
import matplotlib.pyplot as plt
import extinction  # https://github.com/kbarbary/extinction

c = 2.99792458e10 # speed of light in cm/s
pc = 3.0856776e18 # parsec in cm

def proper_distance(z_ini, z_end, cosmo):
    """
    Calculate the proper distance between two redshifts.

    ## Parameters
    z_ini: float
        Initial redshift.
    z_end: float
        Final redshift.
    cosmo: astropy.cosmology
        Cosmology model.
    """
    distance = cosmo.lookback_distance(z_end) - cosmo.lookback_distance(z_ini)
    return np.fabs(distance).value # Mpc


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

def mfp_calculation(kappa, zmed, wave, cosmo):
    """
    Mean free path calculation given the opacity kappa.

    ## Parameters
    kappa: float
        Opacity.
    zmed: float
        Median redshift of the stacked spectrum.
    wave: array
        Wavelength array of the stacked spectrum.
    cosmo: astropy.cosmology
        Cosmology model.
    """

    dwv_fin = wave[1] - wave[0]
    expon = 4.25
    wLL = 911.7633
    # extend the wavelength grid to contain the tau=1 point
    z912_extend = -np.flip(np.arange(step=dwv_fin,start=-wave[-1], stop=-400)) * (1+zmed)/wLL - 1
    vecz_extend = (1+z912_extend)**2.75 * ( 1./(1+z912_extend)**expon - 1./(1+zmed)**expon ) / expon 
    tauLL_grid_extend = kappa * vecz_extend
    z912_best = interp1d(tauLL_grid_extend, z912_extend, kind='cubic')(1.0)
    mfp = proper_distance(z912_best, zmed, cosmo)
    return mfp

def deredden(wave, ebv, RV=3.1):
    """
    Generate the flux modification factor to correct reddening.

    ## Parameters
        wave : (N,) np.ndarray
            The wavelength array (without shifting by redshift)
            shared by all spectrum. N is the number of pixels.
        ebv : (M,) np.ndarray
            The E(B-V) array. M is the number of spectra.
        RV : float
            The R_V parameter in the extinction law.
            The default value is 3.1. 
    ## Returns
        dered : (M, N) np.ndarray
            The flux modification factor.
    """
    AV = ebv * RV
    Al = np.zeros((len(ebv), len(wave)))
    for i in range(len(ebv)):
        Al[i] = extinction.ccm89(wave, AV[i], RV)
    return 10**(Al/2.5)

def plot_best_fit(wave, flux_data, flux_model, filename):
    """
    Plot the best model fit.

    ## Parameters
    wave: array
        Wavelength array of the stacked spectrum.
    flux_data: array
        Flux of the stacked spectrum.
    flux_model: array
        Flux of the best model.
    filename: str
        Filename of the saved plot (without the extension).
    """
    plt.plot(wave, flux_data, 'k', label='data')
    plt.plot(wave, flux_model, c='r', linestyle='--', label='best fit')
    plt.title(filename)
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux')
    plt.legend()
    plt.savefig(filename+'.png')
    plt.close()

def log_likelihood(theta, flux, error, zmed, tau_Lyman_0, gamma_lyman, z912, flux_tel):
    """
    The log likelihood function for a contant Gaussian noise model.

    ## Parameters
    theta: (2,) array
        Fitted parameters of the model.
    flux: array
        Flux of the stacked spectrum.
    error: array
        Error of the stacked spectrum.
    zmed: float
        Median redshift of the stacked spectrum.
    tau_Lyman_0: float
        Lyman series opacity at 912A.
    gamma_lyman: float
        Power law index of the Lyman series opacity.
    z912: float
        Redshift of the Lyman series opacity.
    flux_tel: array
        Flux of the Telfer spectrum.
    """
    norm, kappa = theta
    tau_Lyman = tau_Lyman_0 * ((1+z912)/(1+zmed))**gamma_lyman
    expon = 4.25
    tau_LL = (1+z912)**2.75 * ( 1./(1+z912)**expon - 1./(1+zmed)**expon ) / expon
    continuum = norm * flux_tel * np.exp(-tau_Lyman) * np.exp(-kappa * tau_LL)
    return -0.5 * np.sum((flux - continuum)**2 / error**2)


def log_prior(theta):
    """
    Uniform prior
    
    ## Parameters
    theta: (2,) array
        Fitted parameters of the model.
    """
    norm, kappa = theta
    if 0. < norm < 2. and 100 < kappa < 1000.:
        return 0.0
    return -np.inf

def log_probability(theta, flux, error, zmed, tau_Lyman_0, gamma_lyman, z912, flux_tel):
    """
    Log probability function for the MCMC sampling. ( p(theta) * p(data|theta) )

    ## Parameters
    theta: (2,) array
        Fitted parameters of the model.
    flux: array
        Flux of the stacked spectrum.
    error: array
        Error of the stacked spectrum.
    zmed: float
        Median redshift of the stacked spectrum.
    tau_Lyman_0: float
        Lyman series opacity at 912A.
    gamma_lyman: float
        Power law index of the Lyman series opacity.
    z912: float
        Redshift of the Lyman series opacity.
    flux_tel: array
        Flux of the Telfer spectrum.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, flux, error, zmed, tau_Lyman_0, gamma_lyman, z912, flux_tel)
    