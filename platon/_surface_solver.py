from functools import total_ordering
from matplotlib.transforms import Bbox
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
from astropy.io import fits
from astropy.io import ascii
from lmfit import Model
from scipy.stats import binned_statistic
import copy

#define constants in SI units
k = 1.381e-23 
h = 6.626e-34 
c = 3e8 
# T_star = 3036 
R_sun = 6.957e+8 #m
R_star = 0.19 * R_sun #ExoMAST
a = 0.006 # AU; ExoMAST
a_m = a * 1.496e+11 #m
R_planet = 0.116 #Rj 
R_planet = R_planet * 69.9111e6 #m
A_bond_conversion = (3/2) 

class SurfaceSolver:
    def __init__(self, crust_type = 'Metal-rich'):        
        self.crust_type = crust_type
        self.crust_emission_flux = ascii.read('/Users/kimparagas/Desktop/jwst_project/code/from_renyu/data/Crust_EmissionFlux.dat', delimiter = '\t')
        self.geoa = ascii.read('/Users/kimparagas/Desktop/jwst_project/code/from_renyu/data/GeoA.dat', delimiter = '\t', header_start = 1, data_start = 3)
        self.stellar_flux_txt = ascii.read('/Users/kimparagas/Desktop/jwst_project/code/from_renyu/lhs3844_sf.txt')

        self.surface_names = list(self.geoa.columns)[1:]
        self.surface_dict = {'Metal-rich': 0,
                             'Ultramafic': 1,
                             'Feldspathic': 2,
                             'Basaltic': 3,
                             'Granitoid': 4,
                             'Clay': 5,
                             'Ice-rich silicate': 6,
                             'Fe-oxidized': 7}


        self.wavelengths = self.geoa['Wavelength'][1:] * 10**(-6) 
        self.geo_albedos = np.array([self.geoa[name][1:] for name in self.surface_names])
    

        
        self.wl_stellar_flux = self.stellar_flux_txt['col1']
        self.stellar_flux = self.stellar_flux_txt['col2']
        # self.T_star = T_star

        diff = np.diff(self.wavelengths)
        bins = self.wavelengths[1:] - diff/2
        bins_tg = np.concatenate(([self.wavelengths[0] + (bins[0] - self.wavelengths[1])], bins, [self.wavelengths[-1] - (bins[-1] - self.wavelengths[-1])]))
        self.bins = np.zeros([(len(bins_tg) -1), 2])
        for i in np.arange(len(bins_tg) -1):
        #     print(i)
            self.bins[i] = [bins_tg[i], bins_tg[i+1]]
        binned_a, _, _ = binned_statistic(self.wavelengths, self.geo_albedos, statistic = 'mean', bins = np.unique(self.bins))
        # print(binned_a)
        self.binned_a_specified_crust = binned_a[self.surface_dict[crust_type]]
        print(self.binned_a_specified_crust)
        print(np.shape(self.binned_a_specified_crust))

    
    def bin_arr(self, x, y, bins): 
        binned_arr, _, _ = binned_statistic(x, y, statistic='mean', bins=np.unique(bins))
    
        return binned_arr


    def solve_for_rhs(self, flux):
        """
    Ag: geometric albedo as a function of wavelength
    flux: stellar flux as a function of wavelength
    """

        As = (3/2) * self.binned_a_specified_crust
        integral = (np.pi / 2) * (np.trapz(y = flux * ((R_star / a_m)**2) * (1-As), x = self.wavelengths))

        return integral

    # def compute_stellar_fluxes(self, bins):
    #     self.flux_s = self.planck_function_sum(bins, self.T_star) 
    #     return self.flux_s
    
    def solve_for_temp(self, flux_s):
    #mean albedo for each crust and avged fluxes in each bin 
        # rebinned_albedo = self.bin_arr(self.wavelengths, self.geo_albedos, bins)
        temps = np.array([])
    
        irrad = self.solve_for_rhs(flux_s)
        temp = self.crust_emission_flux['Temperature [K]'][np.abs(self.crust_emission_flux[self.crust_type].value - irrad).argmin()]
        temps = np.append(temps, temp)
        print(f'{self.crust_type}: {irrad:.2f} W/m², {temp:.1f} K')

        return temps

    def planck_function(self, wl, T):
        """
        wl: array of wavelengths in meters
        T: effective temp of object in Kelvin 
        """
        flux = np.array([])
    
        l = (2 * h * c**2) / (wl**5)
        r = 1 / np.expm1((h * c) / (wl * k * T))
        flux = np.append(flux, r*l)
        return flux
        
    def compute_emitted_flux_p(self, temps):
        emitted_fluxes_p = []
        for T in temps:
        #     print(T)
            emi = 1 - self.binned_a_specified_crust
            flux_e = emi * self.planck_function(self.wavelengths, T)
            emitted_fluxes_p += [flux_e]
        return emitted_fluxes_p

    def compute_reflected_flux_p(self, flux_s):
        refl_fluxes_p = []
        flux_r = flux_s * ((R_star / a_m)**2 * self.binned_a_specified_crust)
        refl_fluxes_p += [flux_r]
        return refl_fluxes_p
    
    def compute_total_flux_p(self, emitted_fluxes_p, refl_fluxes_p):
        total_fluxes = []
        for flux_e, flux_refl in zip(emitted_fluxes_p, refl_fluxes_p):
            flux_tot = (flux_e + flux_refl) 
            total_fluxes += [flux_tot]
        return total_fluxes
    
    def compute_depths(self, total_fluxes, refl_fluxes_p, emitted_fluxes_p, flux_s):
        depths = ((total_fluxes) / (flux_s)) * (R_planet / R_star)**2
        refl_depths  = ((refl_fluxes_p) / (flux_s)) * (R_planet / R_star)**2
        emitted_depths = ((emitted_fluxes_p) / (flux_s)) * (R_planet / R_star)**2

        return depths, emitted_depths, refl_depths

    def compute_surface_flux(self, platon_wls, flux_s):
                # print(rebinned_albedos_new)
        binned_flux_s = self.bin_arr(platon_wls, flux_s, self.bins)
        temps = self.solve_for_temp(binned_flux_s)
        emitted_fluxes = self.compute_emitted_flux_p(temps)
        refl_fluxes = self.compute_reflected_flux_p(binned_flux_s)
        total_fluxes = self.compute_total_flux_p(emitted_fluxes, refl_fluxes)
        return emitted_fluxes, refl_fluxes, total_fluxes


