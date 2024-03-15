import numpy as np
from astropy import units as u
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
    plt.figure(dpi=500)
    plt.plot(wave, flux_data, 'k', label='data')
    plt.plot(wave, flux_model, c='r', linestyle='--', label='best fit')
    plt.xlabel('Wavelength (Angstrom)', fontsize=15)
    plt.ylabel('Flux', fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=13)
    plt.savefig(filename)
    plt.close()
