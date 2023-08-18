# This code stacks the quasar spectra in the rest frame

# It is based on the IDL code from Prof. J. X. Prochaska (UCSC)

# This code assumes the input spectra have nearly constant dlambda,
# i.e. the standard for a grism. It then calculates a final wavelength
# scale based on the median z_em and in input (default=1.5) size
# relative to the original pixel width.

import numpy as np
from tqdm import tqdm

def stack_qsos(wave, flux, error, zqso, z_interval, boot_size, boot_num, 
               snr_cut=0., width_scl=None, dwv_fin=None, norm_range=(1450, 1470)):
    """
    Generate the stacked spectrum and the bootstrap error.

    ## Parameters
        wave : (N,) np.ndarray
            The wavelength array (without shifting by redshift) 
            shared by all spectrum. N is the number of pixels.
        flux : (M, N) np.ndarray 
            The flux array. M is the number of spectra.
        error : (M, N) np.ndarray
            The error array.
        zqso : (M,) np.ndarray: 
            The redshift array.
        z_interval : tuple 
            The redshift interval to stack, e.g. (2.5, 2.8).
        boot_size : int
            The number of spectra to use in each bootstrap.
        boot_num : int
            The number of bootstrap.
        snr_cut : float, optional, default:0
            The signal-to-noise ratio cut. The spectra with SNR lower than
            this value will be excluded from the stack.
        width_scl : float, optional, default:None
            The width of the final pixel relative to the original.
            This parameter must be larger than 1 if given.
        dwv_fin : float, optional, default:None
            The width of the final pixel in restframe.            
        norm_range : tuple, optional, default:(1450, 1470)
            The wavelength range to normalize the spectra.
    ## Returns
        wave_fin : (N,) np.ndarray
            The final wavelength array.
        flux_tot : (N,) np.ndarray
            The stacked spectrum.
        flux_boot : (boot_num, N) np.ndarray
            The bootstrap flux array. We can estimate the error from it.
        num : int
            The number of spectra used to generate the stack.
        navg : (N,) np.ndarray
            The number of spectra used to generate each pixel.
        mask_avg : (N,) np.ndarray
            The mask array indicating whether a pixel is stacked from 
            the full set of spectra in the redshift interval.
    ## Notes
        The parameter ``width_scl`` and ``dwv_fin`` are mutually exclusive. 
        You must and only need to give one of them.
    """
    mask_z = (zqso > z_interval[0]) & (zqso < z_interval[1])

    # exclude the spectra with low SNR
    if snr_cut > 0:
        snr = np.zeros((len(flux)))
        for i in range(len(flux)):
            mask_snr = (wave/(1+zqso[i]) > 1260) & (wave/(1+zqso[i]) < 1280)
            snr[i] = np.median(flux[i][mask_snr] / error[i][mask_snr])
        mask_z = mask_z & (snr >= snr_cut)
    
    num = np.sum(mask_z)


    # get the final wavelength array
    z_median = np.median(zqso[mask_z])
    dwv_median = np.median(wave[1:]-wave[:-1])

    if width_scl is None and dwv_fin is None:
        raise ValueError("Must specify either width_scl or dwv_fin.")
    elif width_scl is not None and dwv_fin is not None:
        raise ValueError("Cannot specify both width_scl and dwv_fin.")
    elif dwv_fin is None:
        dwv_fin = dwv_median/(1+z_median) * width_scl
        npix = int(np.round(len(wave) / width_scl)) + 100
    elif width_scl is None:
        npix = int(np.round(len(wave) / (dwv_fin/dwv_median))) + 100
    else:
        raise ValueError("Something is wrong with width_scl and dwv_fin.")

    wave_fin = np.linspace(start=wave[0]/(1+np.max(zqso[mask_z])), \
                        stop=wave[0]/(1+np.max(zqso[mask_z])) + (npix-1)*dwv_fin, \
                        num=npix)
    
    # get the wavelength interval for each pixel
    wave_upper = (wave_fin + np.roll(wave_fin, -1))/2
    wave_upper[-1] = wave_upper[-2] + dwv_fin
    wave_lower = (wave_fin + np.roll(wave_fin, 1))/2
    wave_lower[0] = wave_lower[1] - dwv_fin


    # **generate the stack spectra**

    # initialize the stack array
    flux_stack = np.zeros((num, npix))
    error_stack = np.zeros((num, npix))
    mask_stack = np.zeros((num, npix))

    # get the flux and redshift array for the selected redshift interval
    flux_use = flux[mask_z]
    error_use = error[mask_z]
    zqso_use = zqso[mask_z]
    for i, flux_this in tqdm(enumerate(flux_use)):
        error_this = error[i]
        zqso_this = zqso_use[i]
        wave_rest = wave / (1+zqso_this)

        # normalization
        norm_factor = np.median(flux_this[(wave_rest > norm_range[0]) 
                                          & (wave_rest < norm_range[1])])
        
        # brute force method
        if norm_factor > 1e-3:
            for j in range(npix):
                gd_pix = (wave_rest >= wave_lower[j]) & (wave_rest <= wave_upper[j])
                if np.sum(gd_pix) > 0:
                    flux_stack[i,j] = np.median(flux_this[gd_pix]) / norm_factor
                    error_stack[i,j] = np.median(error_this[gd_pix]) / norm_factor
                    mask_stack[i,j] = 1
        else:
            for j in range(npix):
                gd_pix = (wave_rest >= wave_lower[j]) & (wave_rest <= wave_upper[j])
                if np.sum(gd_pix) > 0:
                    flux_stack[i,j] = np.median(flux_this[gd_pix]) + 1
                    error_stack[i,j] = np.median(error_this[gd_pix]) + 1/snr[i] # maintain S/N
                    mask_stack[i,j] = 1
        
        # double check
        fill = np.where(mask_stack[i] > 0)[0]
        if fill[-1] - fill[0] != len(fill) - 1:
            raise ValueError(f"The mask is not continuous!  i={i}")

    navg = np.sum(mask_stack, axis=0) # number of stacked spectra in each pixel
    mask_avg = navg == num 
    flux_tot = np.sum(flux_stack, axis=0)
    error_tot = np.sum(error_stack, axis=0)
    flux_tot[navg>1] = flux_tot[navg>1] / navg[navg>1]
    error_tot[navg>1] = error_tot[navg>1] / navg[navg>1]

    # **bootstrap**

    # initialize the bootstrap array
    flux_boot = np.zeros((boot_num, npix))
    navg_boot = np.zeros((boot_num, npix))
    boot_size = np.min([boot_size, num])  # ensure boot_size <= num

    # generate the random index for bootstrap
    all_bindx = np.around(np.random.random((boot_num, boot_size)) * num)
    while np.sum(all_bindx >= num) > 0:
        all_bindx[all_bindx >= num] = np.around(np.random.random(np.sum(all_bindx >= num)) * num)

    for i in range(boot_num):
        bindx = all_bindx[i].astype('int')
        flux_boot[i] = np.sum(flux_stack[bindx], axis=0)
        navg_boot[i] = np.sum(mask_stack[bindx], axis=0)
        flux_boot[i][navg_boot[i]>1] = flux_boot[i][navg_boot[i]>1] / navg_boot[i][navg_boot[i]>1]


    return (wave_fin, flux_tot, error_tot, flux_boot, num, navg, mask_avg)

