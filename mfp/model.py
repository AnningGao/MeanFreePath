import numpy as np
from scipy.interpolate import interp1d
from astropy import cosmology
from mfp.utils import proper_distance

cosmo_default = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

class Model:
    def __init__(self, cosmo=cosmo_default):
        self.flux = None
        self.error = None
        self.zmed = None
        self.tau_Lyman_0 = None
        self.gamma_lyman = None
        self.z912 = None
        self.flux_tel = None
        self.cosmo = cosmo
        self.mcmc_params = None
        self.chi2_params = None

    def get_args(self, *args):
        self.flux = args[0]
        self.error = args[1]
        self.zmed = args[2]
        self.tau_Lyman_0 = args[3]
        self.gamma_lyman = args[4]
        self.z912 = args[5]
        self.flux_tel = args[6]

    def get_continuum(self, *args):
        raise NotImplementedError

    def log_probability(self, *args):
        raise NotImplementedError

    def mfp_calculation(self, *args):
        raise NotImplementedError


class Fiducial(Model):
    def __init__(self, mcmc_params={"init":(1, 200), "prior":((0., 3.), (0., 1000.)),
                                    "run_params":{"nwalkers":32, "nsteps":2000, "nburn":200}},
                 chi2_params={"range":((0.9, 1.1), (100, 1000)), "ngrid":(30, 100), 
                              "grid_method":("linear", "log")}):
        """
        mcmc_params: dict
            Parameters for the MCMC fitting. init is the intitial values, prior
            is the allowed range, the nwalkers is the number of walkers, nsteps
            is the number of steps, nburn is the number of burn-in steps.
        chi2_params: dict
            Parameters for the chi2 fitting. range is the allowed range, ngrid
            is the number of grid points, grid_method is the method to sample the
            grid points.
        """
        super().__init__()
        self.ndim = 2
        self.mcmc_params = mcmc_params
        self.chi2_params = chi2_params

    def get_continuum(self, theta):
        """
        Get the continuum of the stacked spectrum.

        ## Parameters
        theta: (2,) array
            Fitted parameters of the model.
        """
        norm, kappa = theta
        tau_Lyman = self.tau_Lyman_0 * ((1+self.z912)/(1+self.zmed))**self.gamma_lyman
        expon = 4.25
        tau_LL = (1+self.z912)**2.75 * ( 1./(1+self.z912)**expon - 1./(1+self.zmed)**expon ) / expon
        continuum = norm * self.flux_tel * np.exp(-tau_Lyman) * np.exp(-kappa * tau_LL)
        return continuum

    def log_likelihood(self, theta):
        continuum = self.get_continuum(theta)
        return -0.5 * np.sum((self.flux - continuum)**2 / self.error**2)

    def log_prior(self, theta):
        for i, theta_i in enumerate(theta):
            if not self.mcmc_params["prior"][i][0] < theta_i < self.mcmc_params["prior"][i][1]:
                return -np.inf
        return 0.

    def log_probability(self, theta):
        """
        Probability function for the MCMC sampling. ( p(theta) * p(data|theta) )
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def mfp_calculation(self, theta):
        """
        Calculate the MFP.
        """
        _, kappa = theta
        wLL, expon = 911.7633, 4.25
        wave = (self.z912 + 1) / (self.zmed + 1) * wLL
        dwv = wave[1] - wave[0]
        z912_extend = -np.flip(np.arange(step=dwv, start=-wave[-1], stop=-400)) * (1+self.zmed)/wLL - 1
        vecz_extend = (1+z912_extend)**2.75 * ( 1./(1+z912_extend)**expon - 1./(1+self.zmed)**expon ) / expon
        tauLL_grid_extend = kappa * vecz_extend
        z912_best = interp1d(tauLL_grid_extend, z912_extend, kind='cubic')(1.0)
        mfp = proper_distance(z912_best, self.zmed, self.cosmo)
        return mfp

class KappaEvo(Model):
    def __init__(self, mcmc_params={"init":(1, 200, 1), "prior":((0., 3.), (0., 1000.), (0., np.inf)),
                                    "run_params":{"nwalkers":32, "nsteps":2000, "nburn":200}},
                 chi2_params={"range":((0.9, 1.1), (100, 1000), (0, 10)), "ngrid":(30, 100, 100),
                              "grid_method":("linear", "log", "linear")}):
        super().__init__()
        self.ndim = 3
        self.mcmc_params = mcmc_params
        self.chi2_params = chi2_params

    def get_continuum(self, theta):
        norm, kappa, gamma = theta
        tau_Lyman = self.tau_Lyman_0 * ((1+self.z912)/(1+self.zmed))**self.gamma_lyman
        expon = 4.25 - gamma
        tau_LL = (1+self.z912)**2.75 * (1 + self.zmed)**(-gamma) * ( 1./(1+self.z912)**expon - 1./(1+self.zmed)**expon ) / expon
        continuum = norm * self.flux_tel * np.exp(-tau_Lyman) * np.exp(-kappa * tau_LL)
        return continuum

    def log_likelihood(self, theta):
        continuum = self.get_continuum(theta)
        return -0.5 * np.sum((self.flux - continuum)**2 / self.error**2)

    def log_prior(self, theta):
        for i, theta_i in enumerate(theta):
            if not self.mcmc_params["prior"][i][0] < theta_i < self.mcmc_params["prior"][i][1]:
                return -np.inf
        return 0.

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def mfp_calculation(self, theta):
        _, kappa, gamma = theta
        wLL, expon = 911.7633, 4.25 - gamma
        wave = (self.z912 + 1) / (self.zmed + 1) * wLL
        dwv = wave[1] - wave[0]
        z912_extend = -np.flip(np.arange(step=dwv, start=-wave[-1], stop=-400)) * (1+self.zmed)/wLL - 1
        vecz_extend = (1+z912_extend)**2.75 * (1+self.zmed)**(-gamma) * ( 1./(1+z912_extend)**expon - 1./(1+self.zmed)**expon ) / expon
        tauLL_grid_extend = kappa * vecz_extend
        z912_best = interp1d(tauLL_grid_extend, z912_extend, kind='cubic')(1.0)
        mfp = proper_distance(z912_best, self.zmed, self.cosmo)
        return mfp
