from unittest import skip
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import time

from .constants import h, c, k_B, R_jup, M_jup, R_sun, AU
from ._atmosphere_solver import AtmosphereSolver

from astropy.io import ascii
import pandas as pd
import copy 
from scipy.stats import binned_statistic
import astropy.units as u
import astropy.constants as const
from scipy import interpolate
import sys

class SurfaceCalculator:
    def __init__(self, T_star, R_star, a, R_planet, planet_name, path_to_own_stellar_spectrum):
        self.T_star = T_star 
        self.R_star = R_star# * R_sun #ExoMAST
        self.a = a # AU; ExoMAST
        # a = 0.001
        self.a = self.a  #m
        # R_planet = 0.116 #Rj 
        self.R_planet = R_planet# * R_jup #m
        self.planet_name = planet_name
        self.A_bond_conversion = (3/2) 

        # self.geoa = ascii.read('/Users/kimparagas/Desktop/jwst_project/code/from_renyu/GeoA.dat', delimiter = '\t', header_start = 1, data_start = 3)
        self.geoa = pd.read_csv('/Users/kimparagas/Desktop/jwst_project/new_GeoA.csv', sep = '\t')
        self.geoa['Wavelength'] = self.geoa['Wavelength']# * 1e-6
        self.geoa_columns = list(self.geoa.columns)

        self.wavelengths = self.geoa['Wavelength'].to_numpy()#[1:]
    
        self.mask = np.full(len(self.wavelengths), True)
        wavelengths = self.wavelengths
        diff = np.diff(wavelengths)
        print(wavelengths)
        bins = wavelengths[1:] - diff/2
        bins_tg = np.concatenate(([wavelengths[0] + (bins[0] - wavelengths[1])], bins, [wavelengths[-1] - (bins[-1] - wavelengths[-1])]))
        self.bins = np.zeros([(len(bins_tg) -1), 2])
        for i in np.arange(len(bins_tg) -1):
            self.bins[i] = [bins_tg[i], bins_tg[i+1]]

        self.surfaces = ['Metal-rich', 'Ultramafic', 'Feldspathic',
        'Basaltic', 'Granitoid', 'Clay', 'Ice-rich silicate',
        'Fe-oxidized']

        self.crust_emission_flux = ascii.read('/Users/kimparagas/Desktop/jwst_project/code/from_renyu/Crust_EmissionFlux.dat', delimiter = '\t')
        self.crust_emission_emissivity = ascii.read('/Users/kimparagas/desktop/jwst_project/notebooks/old/Crust_EmissionEmissivity.dat', delimiter = '\t')
        self.TD = ascii.read('/Users/kimparagas/Desktop/jwst_project/code/from_renyu/Result/TD_T1b.dat', header_start = 1, data_start = 3, delimiter = '\t')

        # self.new_albedos = pd.DataFrame(columns = self.geoa_columns)
        
        Teq = (1/4)**(1/4) * T_star * np.sqrt(self.R_star / self.a)
        f_poly_coeffs = pd.read_csv('/Users/kimparagas/Desktop/f_results/f_poly_coeffs.csv', sep = '\t', index_col = 0)
        poly_models = []
        for surface in self.surfaces:
            coeffs = f_poly_coeffs.loc[surface]
            for i, (c, name) in enumerate(zip(coeffs, coeffs.keys())):
                if np.isnan(c) == True:
                    coeffs[name] = 0
        poly_models = []
        for surface in self.surfaces:
            coeffs = f_poly_coeffs.loc[surface].to_numpy()
            coeffs = np.flip(coeffs)
    
            p = np.poly1d(coeffs)
            
            poly_models += [p]
                    
        factors = []
        if Teq <= 1064 and Teq >= 300:
            for i,surface in enumerate(self.surfaces):
                factors += [poly_models[i](Teq)]
            
        else:
            if Teq < 300:
                print('Full redistribution equilibrium temperature is colder than 300 K (the lower boundary of our grid), but will use factors corresponding to 300 K.')
                Teq = 300
                for i,surface in enumerate(self.surfaces):
                    factors += [poly_models[i](Teq)]
                
            if Teq > 1064:
                print(f'WARNING: Full redistribution equilibrium temperature {Teq:.2f} K is hotter than 1064 K.\nDayside may be (partially) molten in the corresponding 2D models.\nWill use factors corresponding to 1064 K (ensuring a non-molten dayside).')
                Teq = 1064
                for i,surface in enumerate(self.surfaces):
                    factors += [poly_models[i](Teq)]
        
        self.factors = np.array(factors)
        self.model_fluxes = pd.DataFrame(columns = self.geoa_columns)
        self.temperatures = {'Metal-rich': 0, 'Ultramafic': 0, 'Feldspathic': 0,
        'Basaltic': 0, 'Granitoid': 0, 'Clay': 0, 'Ice-rich silicate': 0,
        'Fe-oxidized': 0}
        self.model_depths = pd.DataFrame(columns = self.geoa_columns)
        
        self.path_to_own_stellar_spectrum = path_to_own_stellar_spectrum

    def bin_arr(x, y, bins): 
        binned_arr, _, _ = binned_statistic(x, y, statistic='mean', bins=np.unique(bins))
    
        return binned_arr
    
    def calc_new_albedos(self, plot = False): 
        self.new_albedos = pd.DataFrame(columns = self.geoa_columns)
        for i in np.arange(len(self.geoa_columns[1:])):
            platon_albedo = np.interp(self.wavelengths, self.geoa['Wavelength'], self.geoa[self.geoa_columns[i+1]])
            self.new_albedos[self.geoa_columns[i+1]] = platon_albedo
        self.new_albedos['Wavelength'] = self.wavelengths

        if plot:
            for i in np.arange(len(self.geoa_columns[1:])):    
                plt.plot(self.geoa['Wavelength'] * 1e6, self.geoa[self.geoa_columns[i+1]], color = 'red', label = 'old')
            
                plt.plot(self.wavelengths * 1e6, self.new_albedos[self.geoa_columns[i+1]], 'k--', label = 'new')
                plt.title(f'{self.surfaces[i]} - changed')
                plt.xlabel('wavelength')
                plt.ylabel('albedo') 
                plt.legend()   
                plt.show()
                plt.close()  
    
    def planck_function_sum(self, bins, T):
        """
        wl: array of wavelengths in meters
        T: effective temp of object in Kelvin 
        """
        flux = np.array([])
        for b in bins: 
            l = (2 * h * c**2) / (b**5)
            r = 1 / np.expm1((h * c) / (b * k_B * T))
            integral = np.sum(r*l) / 2
            flux = np.append(flux, integral)
        return flux
    
    def planck_function(self, wl, T):
        """
        wl: array of wavelengths in meters
        T: effective temp of object in Kelvin 
        """
        flux = np.array([])

        l = (2 * h * c**2) / (wl**5)
        r = 1 / np.expm1((h * c) / (wl * k_B * T))
        flux = np.append(flux, r*l)
        
        return flux   

    def calc_fluxes_and_depths(self, x, emis, temp, binned_stellar_flux, albedo_arrays, i):
        emitted_flux_p = np.pi * emis[i] * self.planck_function(x, temp)
            
        refl_flux_p = binned_stellar_flux * ((self.R_star / self.a)**2) * albedo_arrays[i]


        total_flux = (emitted_flux_p + refl_flux_p) 

        depths = ((total_flux) / (binned_stellar_flux)) * (self.R_planet / self.R_star)**2
        
        # refl_depths  = ((refl_flux_p) / (binned_stellar_flux)) * (R_planet / R_star)**2
        # emitted_depths = ((emitted_flux_p) / (binned_stellar_flux)) * (R_planet / R_star)**2
        return total_flux, depths  

    def calc_surface_fluxes(self, albedo_df, new_model_fluxes = None, skip_temp_calc = True, stellar_blackbody = False,
                            plot_stellar_spectum = False, plot_surface_spectra = False, save_stellar_spectrum = False):
        # lambda_grid = self.wavelengths
        # print(skip_temp_calc)
        # if skip_temp_calc:
            # new_model_fluxes = pd.DataFrame(columns = self.geoa_columns)
            # new_model_fluxes['Wavelength'] = self.wavelengths
        if stellar_blackbody == True:
            diff = np.diff(self.wavelengths)
            bins_temp = self.wavelengths[1:] - diff/2
            bins_tg = np.concatenate(([self.wavelengths[0] + (bins_temp[0] - self.wavelengths[1])], bins_temp, [self.wavelengths[-1] - (bins_temp[-1] - self.wavelengths[-1])]))
            bins_temp = np.zeros([(len(bins_tg) -1), 2])
            for i in np.arange(len(bins_tg) -1):
                bins_temp[i] = [bins_tg[i], bins_tg[i+1]]
            binned_stellar_flux = np.pi * self.planck_function_sum(bins_temp, self.T_star) 
            stellar_flux = np.pi * self.planck_function(self.wavelengths, self.T_star) 
            
            f = interpolate.interp1d(self.wavelengths, stellar_flux)

            # self.wavelengths = self.wavelengths[self.mask]
            binned_stellar_flux = f(self.wavelengths)
        
        
        ########################### UNCOMMENT THIS FOR NON BB STELLAR SPECTRA ###########################
        #################################################################################################
        if self.path_to_own_stellar_spectrum is not None:
            stellar_flux = ascii.read('/Users/kimparagas/Desktop/jwst_project/code/from_renyu/lhs3844_sf.txt')
            wl = stellar_flux['col1']
            # wl = wl[wl > 3.026708e-07]
            
            # print(np.array(wavelengths))
            stellar_flux = stellar_flux['col2']
            wl = wl * u.m
            stellar_flux = stellar_flux * u.W / u.m**3
            
            wl = wl.si.value
            stellar_flux = stellar_flux.to(u.W / u.m**3).value
            # spectrum = pd.read_csv(self.path_to_own_stellar_spectrum, sep = '\t')
            # wl = (spectrum['wavelength [m]'].to_numpy() * u.m)
            # wl = wl.to(u.m).value
            # stellar_flux = (spectrum['stellar flux [W/m^2/um]'].to_numpy() * u.W / u.m**2 / u.um)
            # stellar_flux = stellar_flux.to(u.W/u.m**3).value
            # wl = wl
            # stellar_flux = stellar_flux
            print('stellar flux inherit bounds')
            print(wl.min(), wl.max())
            
            # plt.plot(wl * 1e6, stellar_flux, color = 'k', lw = 2)
            plt.plot(wl * 1e6, stellar_flux, color = 'crimson', lw = 1, ls = '--')
            plt.xlim(3,13)
            plt.show()
            plt.close()
        
       
            
            if skip_temp_calc == False:
                self.mask = np.where(((self.wavelengths >= wl[0]) & (self.wavelengths <= wl[-1])))[0]
                self.wavelengths = self.wavelengths[self.mask]
                self.calc_new_albedos()
                albedo_df = self.new_albedos
                
    
        # self.wavelengths = self.wavelengths[self.wavelengths < 24.09e-6]
        # mask = np.where(self.wavelengths < 24.09e-6)[0]
        # self.wavelengths = self.wavelengths[self.mask]
            if skip_temp_calc:
                new_model_fluxes['Wavelength'] = self.wavelengths
                
            print(self.wavelengths[0], self.wavelengths[-1])
            binned_stellar_flux = np.interp(self.wavelengths, wl, stellar_flux) #interpolate.interp1d(wl, stellar_flux)
        
        
        
        
        
        # print(len(binned_stellar_flux))
        ###################################################### END ######################################################
        #################################################################################################################

        # cond = np.any([
        #     np.logical_and(self.wavelengths > start, self.wavelengths < end) \
        #     for (start, end) in bins_temp], axis=0)


        
        # binned_stellar_flux_exp = binned_stellar_flux * u.W / u.m**3
        # binned_stellar_flux_exp = binned_stellar_flux_exp.to(u.erg / u.cm**2 / u.s / u.angstrom)

        # if save_stellar_spectrum == True:
        #     starflux = f'/Users/kimparagas/Desktop/jwst_project/retrieval_attempt/surface_only/data/other_planets/{self.planet_name}/stellar_model.dat'
        #     with open(starflux, 'w') as sf:
        #         for w,f in zip((self.wavelengths * 100), binned_stellar_flux_exp.value):
        #             sf.write(f'{w:.15f}   {f:.15e}\n')

        if plot_stellar_spectum:
            plt.plot(self.wavelengths * 1e6, binned_stellar_flux, 'k')
            plt.xlabel(r'wavelength [$\mu$m]')
            plt.ylabel('stellar flux [W/m^3]')
            plt.savefig('/Users/kimparagas/desktop/stellar_spectrum.pdf')
            plt.show()
            plt.close()
            # sys.exit()
        
        # plt.plot(self.wavelengths * 1e6, binned_stellar_flux, 'k')
        # plt.xlabel(r'wavelength [$\mu$m]')
        # plt.ylabel('stellar flux [W/m^3]')
        # plt.savefig('/Users/kimparagas/desktop/stellar_spectrum.pdf')
        # plt.show()
        # plt.close()

        albedo_arrays = []
        for name in self.geoa_columns[1:]:
            albedo_arrays += [np.array(albedo_df[name])]
        self.albedo_arrays = np.array(albedo_arrays)

        emis = np.zeros((8, len(self.albedo_arrays[0])))

        for i,name in enumerate((self.geoa_columns)[1:]):
            emis[i] = (1 - (self.albedo_arrays[i]))

        def calc_temps(x, redist_factor, i):
            Ag = np.mean(self.albedo_arrays[i])
            As = (3/2) * Ag
            irrad = redist_factor * (np.trapz(y = (binned_stellar_flux) * ((self.R_star / self.a)**2) * (1-As), x = x)) #do not need pi in this if the stellar spectrum is used vs the planck function 
            # temp = self.crust_emission_flux['Temperature [K]'][np.abs(self.crust_emission_flux[self.surfaces[i]].value - irrad).argmin()] #jwst_project
            temp = self.crust_emission_flux['Temperature [K]'][np.abs(self.crust_emission_flux[self.surfaces[i]].data - irrad).argmin()] #clr
            print(f'{self.surfaces[i]}: {irrad:.2f} W/m², {temp:.1f} K')
            # print(temp)
            return temp

        model_depths = np.zeros((8, len(self.wavelengths)))
        total_fluxes = np.zeros((8, len(self.wavelengths)))
        temps_meana = np.zeros(8)
        # print(f'HERE!!! {len(self.wavelengths)}')
        for i in np.arange(len(self.geoa_columns[1:])):
            if skip_temp_calc:
                temps_meana[i] = self.temperatures[self.surfaces[i]]
            else:
                temps_meana[i] = calc_temps(self.wavelengths, self.factors[i], i)
                # print(self.geoa_columns[1:][i], temps_meana[i])
            # print(temps_meana)
            total_fluxes[i], model_depths[i] = self.calc_fluxes_and_depths(self.wavelengths, emis, temps_meana[i], binned_stellar_flux, albedo_arrays, i)
            
        for i in np.arange(len(self.geoa_columns[1:])):
            if skip_temp_calc == False:
                self.model_fluxes[self.surfaces[i]] = total_fluxes[i]
                self.temperatures[self.surfaces[i]] = temps_meana[i]
                self.model_depths[self.surfaces[i]] = model_depths[i]
        
            if skip_temp_calc:
                # print(len(model_depths[i]))
                new_model_fluxes[self.surfaces[i]] = total_fluxes[i]
                self.model_depths[self.surfaces[i]] = model_depths[i]
                # print(len(self.model_depths[self.surfaces[i]]))
        # print(self.model_depths)
        # sys.exit()
        
        if skip_temp_calc == False:
            self.model_fluxes['Wavelength'] = self.wavelengths
            self.model_depths['Wavelength'] = self.model_fluxes['Wavelength']
        if skip_temp_calc:
            self.model_fluxes = new_model_fluxes
            print('end model fluxes wl grid bounds')
            print(self.model_fluxes['Wavelength'].to_numpy().min(), self.model_fluxes['Wavelength'].to_numpy().max())
        # print(self.model_depths)
        # sys.exit()

        if plot_surface_spectra:
            model_surface_spectra_table = ascii.read('/Users/kimparagas/desktop/jwst_project/code/current/products/model_fluxes_platon_stellar_bb.csv')
            for i in np.arange(len(self.geoa_columns[1:])):
                plt.plot(model_surface_spectra_table['Wavelength']*1e6, model_surface_spectra_table[self.surfaces[i]], 'k')
                plt.plot(self.model_fluxes['Wavelength'] * 1e6, total_fluxes[i], 'r--')
                # plt.xlim(5, 20)
                plt.xlabel(r'wavelength [$\mu$m]')
                plt.ylabel('fluxes [si]')
                plt.title(self.surfaces[i])
                plt.show()
                plt.close()

    def read_in_temps(self, names, temps):
        for i,name in enumerate(names):
            self.temperatures[name] = temps[i]
        # print(self.temperatures)
        # sys.exit()

    def calc_initial_spectra(self, skip_temp_calc = False):
        self.calc_surface_fluxes(albedo_df = self.geoa, skip_temp_calc = skip_temp_calc, plot_stellar_spectum = False, plot_surface_spectra = False)
        # sys.exit()
        
    def change_spectra(self, wavelengths):
        # self.mask = np.where(((self.wavelengths >= wavelengths[0]) & (self.wavelengths <= wavelengths[-1])))[0]
        self.wavelengths = wavelengths
        print('change spectra to these wavelength bounds')
        print(wavelengths[0], wavelengths[-1])
        self.calc_new_albedos()
        self.model_depths = pd.DataFrame(columns = self.geoa_columns)
        new_model_fluxes = pd.DataFrame(columns = self.geoa_columns)
        self.model_depths[self.geoa_columns[0]] = self.wavelengths
        self.calc_surface_fluxes(albedo_df = self.new_albedos, new_model_fluxes = new_model_fluxes)
        

    # def calc_model_depths(self, stellar_spectrum):
        

