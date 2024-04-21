# This code stacks the quasar spectra in the rest frame

# It is based on the IDL code from Prof. J. X. Prochaska (UCSC)

# This code assumes the input spectra have nearly constant dlambda,
# i.e. the standard for a grism. It then calculates a final wavelength
# scale based on the median z_em and in input (default=1.5) size
# relative to the original pixel width.

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

def stack_qsos(wave, flux, error, zqso, z_interval, boot_size, boot_num, 
               zerr=None, snr=None, wave_max=np.inf, snr_cut=0., width_scl=None, 
               dwv_fin=None, norm_range=(1450, 1470)):
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
        zerr: (M,) np.ndarray, optional, default:None
            The redshift error array. If given, the code will perform
            bootstrap with redshift error.
        snr : (M,) np.ndarray, optional, default:None
            The signal-to-noise ratio array. If not given, the code will
            calculate it from the spectra.
        wave_max : float, optional, default:np.inf
            The maximum wavelength to stack. Designed to accelerate the 
            stacking by neglecting the pixels beyond this value.
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
        error_tot: (N,) np.ndarray
            The error of the stacked spectrum.
        z_median : float
            The median redshift of the stacked spectra.
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
        if snr is None:
            snr = np.zeros((len(flux)))
            for i, (flux_this, error_this, zqso_this) in enumerate(zip(flux, error, zqso)):
                mask_snr = (wave/(1+zqso_this) > 1450) & (wave/(1+zqso_this) < 1470)
                snr[i] = np.median(flux_this[mask_snr] / error_this[mask_snr])
        mask_z = mask_z & (snr >= snr_cut)

    num = np.sum(mask_z)


    # get the final wavelength array
    z_median = np.median(zqso[mask_z])
    dwv_median = np.median(wave[1:]-wave[:-1])

    if width_scl is None and dwv_fin is None:
        raise ValueError("Must specify either width_scl or dwv_fin.")
    if width_scl is not None and dwv_fin is not None:
        raise ValueError("Cannot specify both width_scl and dwv_fin.")
    if dwv_fin is None:
        dwv_fin = dwv_median/(1+z_median) * width_scl
        npix = int(np.round(len(wave) / width_scl)) + 100
    else: # width_scl is None
        npix = int(np.round(len(wave) / (dwv_fin/dwv_median))) + 100

    wave_fin = np.linspace(start=wave[0]/(1+np.max(zqso[mask_z])),
                           stop=wave[0]/(1+np.max(zqso[mask_z]))+(npix-1)*dwv_fin,
                           num=npix)
    wave_fin = wave_fin[(wave_fin < wave_max)]
    npix = len(wave_fin)

    # get the wavelength interval for each pixel
    wave_upper = (wave_fin + np.roll(wave_fin, -1))/2
    wave_upper[-1] = wave_upper[-2] + dwv_fin
    wave_lower = (wave_fin + np.roll(wave_fin, 1))/2
    wave_lower[0] = wave_lower[1] - dwv_fin


    # **generate the stack spectra**

    # initialize the stack array
    flux_stack = np.zeros((num, npix))
    var_stack = np.zeros((num, npix))
    mask_stack = np.zeros((num, npix))

    # get the flux and redshift array for the selected redshift interval
    flux_use = flux[mask_z]
    error_use = error[mask_z]
    zqso_use = zqso[mask_z]
    mask_pix = (error<1e5)[mask_z]

    stack_single_partial = partial(stack_single, wave=wave, norm_range=norm_range,
                                   npix=npix, wave_lower=wave_lower, wave_upper=wave_upper)

    with Pool(8) as p:
        results = p.map(stack_single_partial, 
                        tqdm(zip(flux_use, error_use, zqso_use, mask_pix), total=num))

    for i, result in enumerate(results):
        flux_stack[i], var_stack[i], mask_stack[i] = result

    navg = np.sum(mask_stack, axis=0) # number of stacked spectra in each pixel
    flux_tot = np.sum(flux_stack, axis=0)
    error_tot = np.sqrt(np.sum(var_stack, axis=0))
    flux_tot[navg>1] = flux_tot[navg>1] / navg[navg>1]
    error_tot[navg>1] = error_tot[navg>1] / navg[navg>1]

    mask_avg = navg == num
    min_idx, max_idx = np.min(np.where(mask_avg)[0]), np.max(np.where(mask_avg)[0])
    mask_avg[min_idx:max_idx+1] = True

    # **bootstrap**

    # initialize the bootstrap array
    flux_boot = np.zeros((boot_num, npix))
    navg_boot = np.zeros((boot_num, npix))
    boot_size = np.min([boot_size, num])  # ensure boot_size <= num

    # generate the random index for bootstrap
    all_bindx = np.array([np.random.choice(np.arange(num), size=boot_size, replace=True)
                          for _ in range(boot_num)])

    if zerr is None:
        for i in range(boot_num):
            bindx = all_bindx[i]
            flux_boot[i] = np.sum(flux_stack[bindx], axis=0)
            navg_boot[i] = np.sum(mask_stack[bindx], axis=0)
            flux_boot[i][navg_boot[i]>1] = flux_boot[i][navg_boot[i]>1] / navg_boot[i][navg_boot[i]>1]
    else:
        iter_bar = tqdm(range(boot_num))
        iter_bar.set_description('Bootstrapping with redshift error...')
        stack_single_boot = partial(stack_single, wave=wave, norm_range=norm_range, npix=npix,
                                    wave_lower=wave_lower, wave_upper=wave_upper, bootstrap=True)
        for i in iter_bar:
            bindx = all_bindx[i]
            zerr_boot = np.random.normal(0, zerr[bindx])
            zqso_boot = zqso_use[bindx] + zerr_boot
            mask_pix_boot = mask_pix[bindx]
            flux_stack_boot = np.zeros((boot_size, npix))
            mask_stack_boot = np.zeros((boot_size, npix))
            with Pool(8) as p:
                results = p.map(stack_single_boot,
                                zip(flux_use[bindx], range(boot_size), zqso_boot, mask_pix_boot))
            for j, result in enumerate(results):
                flux_stack_boot[j], mask_stack_boot[j] = result
            flux_boot[i] = np.sum(flux_stack_boot, axis=0)
            navg_boot[i] = np.sum(mask_stack_boot, axis=0)
            flux_boot[i][navg_boot[i]>1] = flux_boot[i][navg_boot[i]>1] / navg_boot[i][navg_boot[i]>1]

    return wave_fin, flux_tot, error_tot, flux_boot, z_median, num, navg, mask_avg


def stack_single(args, wave, norm_range, npix, wave_lower, wave_upper, bootstrap=False):
    """
    Stack a single spectrum. Designed for parallel processing.
    """
    flux, error, zqso, mask_pixel = args
    flux = flux[mask_pixel]
    wave_rest = (wave / (1+zqso)) [mask_pixel] # de-redshift

    # normalize
    norm_factor = np.median(flux[(wave_rest > norm_range[0]) & (wave_rest < norm_range[1])])
    flux, error = flux / norm_factor, error / norm_factor

    # criterion for stacking
    gd_pix = [(wave_rest >= wave_lower[j]) & (wave_rest <= wave_upper[j]) for j in range(npix)]
    sums = [np.sum(g) for g in gd_pix]

    flux_stack_this = np.zeros(npix)
    mask_stack_this = np.zeros(npix)

    if bootstrap:
        for j in range(npix):
            if sums[j] > 0:
                flux_stack_this[j] = np.mean(flux[gd_pix[j]])
                mask_stack_this[j] = 1
        return flux_stack_this, mask_stack_this
    else:
        error = error[mask_pixel] / norm_factor
        var_stack_this = np.zeros(npix)
        for j in range(npix):
            if sums[j] > 0:
                flux_stack_this[j] = np.mean(flux[gd_pix[j]])
                var_stack_this[j] = np.sum(error[gd_pix[j]]**2) / sums[j]**2
                mask_stack_this[j] = 1
        return flux_stack_this, var_stack_this, mask_stack_this
