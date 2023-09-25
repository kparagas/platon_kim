import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import time

from .constants import h, c, k_B, R_jup, M_jup, R_sun
from ._atmosphere_solver import AtmosphereSolver

from astropy.io import ascii
import pandas as pd
import copy 

temps = [950.717697, 979.65893 , 917.114564, 936.569297, 919.868872,
       845.786244, 911.630666, 947.871021, 956.436723]

model_surface_spectra_table = ascii.read('/Users/kimparagas/desktop/jwst_project/code/current/products/model_fluxes_platon.csv')

# def planck_function(wl, T):
#     """
#     wl: array of wavelengths in meters
#     T: effective temp of object in Kelvin 
#     """
#     flux = np.array([])
 
#     l = (2 * h * c**2) / (wl**5)
#     #         print(l)
#     r = 1 / np.expm1((h * c) / (wl * k_B * T))
#     #         print(np.expm1((h * c) / (wl * k * T)))
#     #         print(l*r)
# #     integral = np.sum(r*l) / 2
# #     #         print(integral)
#     flux = np.append(flux, r*l)
#     return flux





class EclipseDepthCalculator:
    def __init__(self, include_condensation=True, method="xsec"):
        '''
        All physical parameters are in SI.

        Parameters
        ----------
        include_condensation : bool
            Whether to use equilibrium abundances that take condensation into
            account.
        num_profile_heights : int
            The number of zones the atmosphere is divided into
        ref_pressure : float
            The planetary radius is defined as the radius at this pressure
        method : string
            "xsec" for opacity sampling, "ktables" for correlated k
        '''
        self.atm = AtmosphereSolver(include_condensation=include_condensation, method=method)

        # scipy.special.expn is slow when called on millions of values, so
        # use interpolator to speed it up
        tau_cache = np.logspace(-6, 3, 1000)
        self.exp3_interpolator = scipy.interpolate.interp1d(
            tau_cache,
            scipy.special.expn(3, tau_cache),
            bounds_error=False,
            fill_value=(0.5, 0))

        
    def change_wavelength_bins(self, bins):        
        '''Same functionality as :func:`~platon.transit_depth_calculator.TransitDepthCalculator.change_wavelength_bins`'''
        self.atm.change_wavelength_bins(bins)


    def _get_binned_depths(self, depths, stellar_spectrum, n_gauss=10):
        #Step 1: do a first binning if using k-coeffs; first binning is a
        #no-op otherwise
        if self.atm.method == "ktables":
            #Do a first binning based on ktables
            points, weights = scipy.special.roots_legendre(n_gauss)
            percentiles = 100 * (points + 1) / 2
            weights /= 2
            assert(len(depths) % n_gauss == 0)
            num_binned = int(len(depths) / n_gauss)
            intermediate_lambdas = np.zeros(num_binned)
            intermediate_depths = np.zeros(num_binned)
            intermediate_stellar_spectrum = np.zeros(num_binned)

            for chunk in range(num_binned):
                start = chunk * n_gauss
                end = (chunk + 1 ) * n_gauss
                intermediate_lambdas[chunk] = np.median(self.atm.lambda_grid[start : end])
                intermediate_depths[chunk] = np.sum(depths[start : end] * weights)
                intermediate_stellar_spectrum[chunk] = np.median(stellar_spectrum[start : end])
        elif self.atm.method == "xsec":
            intermediate_lambdas = self.atm.lambda_grid
            # intermediate_atm_depths = atmosphere_depths
            # intermediate_surface_depths = surface_depths
            intermediate_depths = depths
            intermediate_stellar_spectrum = stellar_spectrum
        else:
            assert(False)

        
        if self.atm.wavelength_bins is None:
            return intermediate_lambdas, intermediate_depths, intermediate_lambdas, intermediate_depths
        
        binned_wavelengths = []
        binned_atm_depths = []
        binned_surface_depths = []
        binned_depths = []
        for (start, end) in self.atm.wavelength_bins:
            cond = np.logical_and(
                intermediate_lambdas >= start,
                intermediate_lambdas < end)
            binned_wavelengths.append(np.mean(intermediate_lambdas[cond]))
            # binned_atm_depth = np.average(intermediate_atm_depths[cond],
            #                           weights=intermediate_stellar_spectrum[cond])
            # binned_surface_depth = np.average(intermediate_surface_depths[cond],
            #                           weights=intermediate_stellar_spectrum[cond])
            binned_depth = np.average(intermediate_depths[cond],
                                      weights=intermediate_stellar_spectrum[cond])
            # binned_atm_depths.append(binned_atm_depth)
            # binned_surface_depths.append(binned_surface_depth)
            binned_depths.append(binned_depth)
            
        return intermediate_lambdas, intermediate_depths, np.array(binned_wavelengths), np.array(binned_depths)#, np.array(binned_atm_depths), np.array(binned_surface_depths)

    def _get_photosphere_radii(self, taus, radii):
        intermediate_radii = 0.5 * (radii[0:-1] + radii[1:])
        photosphere_radii = np.array([np.interp(1, t, intermediate_radii) for t in taus])
        return photosphere_radii
    
    def compute_depths(self, t_p_profile, star_radius, planet_mass,
                       planet_radius, T_star, logZ=0, CO_ratio=0.53,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       T_spot=None, spot_cov_frac=None,
                       ri = None, frac_scale_height=1,number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       stellar_blackbody=False, surface_P = 100,
                       full_output=False, surface_name = None):
        '''Most parameters are explained in :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths`

        Parameters
        ----------
        t_p_profile : Profile
            A Profile object from TP_profile
        '''
        T_profile = t_p_profile.temperatures
        P_profile = t_p_profile.pressures
        atm_info = self.atm.compute_params(
            star_radius, planet_mass, planet_radius, P_profile, T_profile,
            logZ, CO_ratio, add_gas_absorption, add_H_minus_absorption, add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, cloudtop_pressure, custom_abundances,
            T_star, T_spot, spot_cov_frac,
            ri, frac_scale_height, number_density, part_size, part_size_std,
            P_quench)

        assert(np.max(atm_info["P_profile"]) <= cloudtop_pressure)
        absorption_coeff = atm_info["absorption_coeff_atm"]
        intermediate_coeff = 0.5 * (absorption_coeff[0:-1] + absorption_coeff[1:])
        intermediate_T = 0.5 * (atm_info["T_profile"][0:-1] + atm_info["T_profile"][1:])
        dr = atm_info["dr"]
        d_taus = intermediate_coeff.T * dr
        taus = np.cumsum(d_taus, axis=1)

        lambda_grid = self.atm.lambda_grid

        reshaped_lambda_grid = lambda_grid.reshape((-1, 1))
        planck_function = 2*h*c**2/reshaped_lambda_grid**5/(np.exp(h*c/reshaped_lambda_grid/k_B/intermediate_T) - 1)
        #padded_taus: ensures 1st layer has 0 optical depth
        padded_taus = np.zeros((taus.shape[0], taus.shape[1] + 1))
        padded_taus[:, 1:] = taus
        integrand = planck_function * np.diff(scipy.special.expn(3, padded_taus), axis=1)
        fluxes = -2 * np.pi * np.sum(integrand, axis=1)
        fluxes_pre_surface = copy.deepcopy(fluxes)
                
        if not np.isinf(cloudtop_pressure):
            max_taus = np.max(taus, axis=1)
            fluxes_from_bb = -np.pi * planck_function[:, -1] * (max_taus**2 * scipy.special.expi(-max_taus) + max_taus * np.exp(-max_taus) - np.exp(-max_taus))
            fluxes_atm = fluxes_pre_surface + fluxes_from_bb
            fluxes_from_cloud = -model_surface_spectra_table['Metal-rich'] * (max_taus**2 * scipy.special.expi(-max_taus) + max_taus * np.exp(-max_taus) - np.exp(-max_taus)) 
            fluxes += fluxes_from_cloud   
        # fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 6), sharex = True, constrained_layout = True)
        # ax1.plot(lambda_grid, fluxes_pre_surface, color = 'k', label = 'flux without surface')
        # ax1.set_title(surface_name)
        # ax2.plot(lambda_grid, fluxes_from_cloud, color = 'k', label = 'flux from surface')
        # ax1.legend()
        # ax2.legend()
        # plt.savefig('/Users/kimparagas/desktop/jwst_project/code/current/products/dble_plots/' + str(surface_name) + '.png', dpi = 300)
        # plt.close() 
        stellar_photon_fluxes, _ = self.atm.get_stellar_spectrum(
            lambda_grid, T_star, T_spot, spot_cov_frac, stellar_blackbody)
        d_lambda = self.atm.d_ln_lambda * lambda_grid
        photon_fluxes = fluxes * d_lambda / (h * c / lambda_grid)

        photosphere_radii = self._get_photosphere_radii(taus, atm_info["radii"])
        eclipse_depths = photon_fluxes / stellar_photon_fluxes * (photosphere_radii/star_radius)**2
        if not np.isinf(cloudtop_pressure):
            atmosphere_photon_fluxes = fluxes_atm * d_lambda / (h * c / lambda_grid)
            atmosphere_depths = atmosphere_photon_fluxes / stellar_photon_fluxes * (photosphere_radii/star_radius)**2
            surface_photon_fluxes = fluxes_from_cloud * d_lambda / (h * c / lambda_grid)
            surface_depths = surface_photon_fluxes / stellar_photon_fluxes * (photosphere_radii/star_radius)**2
        # plt.plot(reshaped_lambda_grid, eclipse_depths*1e6)
        # plt.ylabel('eclipse depths')
        # plt.show()

        #For correlated k, eclipse_depths has n_gauss points per wavelength, while unbinned_depths has 1 point per wavelength
        unbinned_wavelengths, unbinned_depths, binned_wavelengths, binned_depths = self._get_binned_depths(eclipse_depths, stellar_photon_fluxes)
        # if not np.isinf(cloudtop_pressure):
        #     unbinned_wavelengths, unbinned_depths, binned_wavelengths, binned_depths, binned_atm_depths, binned_surface_depths = self._get_binned_depths(atmosphere_depths, surface_depths, eclipse_depths, stellar_photon_fluxes)

        if full_output:
            atm_info["stellar_spectrum"] = stellar_photon_fluxes
            atm_info["planet_spectrum"] = fluxes
            atm_info["unbinned_wavelengths"] = unbinned_wavelengths
            atm_info["unbinned_eclipse_depths"] = unbinned_depths
            atm_info["taus"] = taus
            atm_info["contrib"] = -integrand / fluxes[:, np.newaxis]
            return binned_wavelengths, binned_depths, atm_info
        return binned_wavelengths, binned_depths
            


