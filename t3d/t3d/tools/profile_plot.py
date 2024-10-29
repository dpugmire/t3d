import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from netCDF4 import Dataset
import adios2
import itertools
from os import getcwd

from t3d import Profiles, Grid


class plotter:

    def __init__(self):

        # for backwards compat to before t3d packaging
        sys.modules['Profiles'] = Profiles
        sys.modules['Grid'] = Grid

        np.set_printoptions(linewidth=500)

        self.grid_lines = False
        self.savefig = False

        # Dictionary of plotting functions for the legacy panel plot
        self.panel_plots = {
            "density": self.plot_density_profiles,
            "temperature": self.plot_temperature_profiles,
            "pressure": self.plot_pressure_profiles,
            "power_balance": self.plot_power_balance,
            "heat_flux": self.plot_heat_flux,
            "particle_balance": self.plot_particle_balance,
            "particle_flux": self.plot_particle_flux,
            "fusion_power_density": self.plot_fusion_power_density,
            "radiation": self.plot_radiation_profiles,
            "exchange": self.plot_exchange,
            "power_source": self.plot_auxiliary_power_source,
            "q": self.plot_q,
            "particle_source": self.plot_auxiliary_particle_source,
            "gamma": self.plot_gamma,
        }

        # Dictionary of all available plotting functions
        self.plotting_functions = self.panel_plots.copy()
        self.plotting_functions["state_profiles"] = self.plot_state_profiles
        self.plotting_functions["flux"] = self.plot_flux
        self.plotting_functions["source_total"] = self.plot_source_total
        self.plotting_functions["sink_terms"] = self.plot_sink_terms

    def list_available_plots(self):
        ''' Function to list available plots '''
        print("\nList of available plots:")
        for plot in self.plotting_functions:
            print(f"  {plot}")

    def read_data(self, fin):
        ''' Function to read the Trinity3D npy file '''

        # Load the pickle file
        data = np.load(fin, allow_pickle=True).tolist()

        self.t_ref = data['norms']['t_ref']
        try:
            self.rho_star = data['norms']['rho_star']
        except:
            self.rho_star = 1 / data['norms']['gyro_scale']
        self.time = np.array(data['t']) * self.t_ref
        self.tags = data['species_tags']
        self.ion_tag = data['bulk_ion_tag']

        self.N = len(self.time)

        # Read the species profile data
        self.n = np.asarray(data['n_e'])
        self.pi = np.asarray(data['p_' + self.ion_tag])
        self.pe = np.asarray(data['p_e'])
        self.Ti = np.asarray(data['T_' + self.ion_tag])
        self.Te = np.asarray(data['T_e'])
        self.Gamma = {}
        self.Qi = {}
        self.Qe = {}
        self.Gamma['tot'] = np.asarray(data['pflux_' + self.ion_tag])
        self.Qi['tot'] = np.asarray(data['qflux_' + self.ion_tag])
        self.Qe['tot'] = np.asarray(data['qflux_e'])
        self.flux_model_labels = data['flux_model_labels']
        for label in self.flux_model_labels:
            self.Gamma[label] = np.asarray(data['pflux_' + label + '_' + self.ion_tag])
            self.Qi[label] = np.asarray(data['qflux_' + label + '_' + self.ion_tag])
            self.Qe[label] = np.asarray(data['qflux_' + label + '_e'])
        self.aLn = np.asarray(data['aLn_' + self.ion_tag])
        self.aLpi = np.asarray(data['aLp_' + self.ion_tag])
        self.aLpe = np.asarray(data['aLp_e'])
        self.aLTi = np.asarray(data['aLT_' + self.ion_tag])
        self.aLTe = np.asarray(data['aLT_e'])

        # Convert pressure sources from code units to MW/m^3 by dividing by p_source_scale
        self.P_ref_MWm3 = data['norms']['P_ref_MWm3']
        self.Sn_ref_SI20 = data['norms']['Sn_ref_SI20']
        try:
            self.a_minor = data['norms']['a_minor']
        except:
            self.a_minor = data['norms']['a_ref']

        # Density sources, in code units
        self.Sn_aux_i = np.asarray(data['Sn_aux_' + self.ion_tag])
        self.Sn_aux_e = np.asarray(data['Sn_aux_e'])

        # Pressure sources, in code units, multiply by P_ref_MWm3 to get MW/m^3
        self.Sp_aux_i = np.asarray(data['Sp_aux_' + self.ion_tag])
        self.Sp_aux_e = np.asarray(data['Sp_aux_e'])
        self.Sp_alpha_i = np.asarray(data['Sp_alpha_' + self.ion_tag])
        self.Sp_alpha_e = np.asarray(data['Sp_alpha_e'])
        self.Sp_rad_i = np.asarray(data['Sp_rad_' + self.ion_tag])
        self.Sp_rad_e = np.asarray(data['Sp_rad_e'])
        if len(self.Sp_rad_i) == 0:  # old files use Sp_brem
            self.Sp_rad_i = np.asarray(data['Sp_brem_' + self.ion_tag])
            self.Sp_rad_e = np.asarray(data['Sp_brem_e'])
        self.Sp_heating_i = np.asarray(data['Sp_heating_' + self.ion_tag])
        self.Sp_heating_e = np.asarray(data['Sp_heating_e'])
        self.Sp_coll_i = np.asarray(data['Sp_coll_' + self.ion_tag])
        self.Sp_coll_e = np.asarray(data['Sp_coll_e'])
        self.Sp_tot_i = np.asarray(data['Sp_tot_' + self.ion_tag])
        self.Sp_tot_e = np.asarray(data['Sp_tot_e'])

        self.Sp_aux_int_MW_i = np.asarray(data['Sp_aux_int_MW_' + self.ion_tag])
        self.Sp_aux_int_MW_e = np.asarray(data['Sp_aux_int_MW_e'])
        self.Sp_aux_int_MW_tot = self.Sp_aux_int_MW_i + self.Sp_aux_int_MW_e
        self.Sp_alpha_int_MW_i = np.asarray(data['Sp_alpha_int_MW_' + self.ion_tag])
        self.Sp_alpha_int_MW_e = np.asarray(data['Sp_alpha_int_MW_e'])
        self.Sp_rad_int_MW_i = np.asarray(data['Sp_rad_int_MW_' + self.ion_tag])
        self.Sp_rad_int_MW_e = np.asarray(data['Sp_rad_int_MW_e'])
        self.Prad_tot = np.asarray(data['Sp_rad_int_MW_e'])
        self.Sp_heating_int_MW_i = np.asarray(data['Sp_heating_int_MW_' + self.ion_tag])
        self.Sp_heating_int_MW_e = np.asarray(data['Sp_heating_int_MW_e'])
        self.Sp_coll_int_MW_i = np.asarray(data['Sp_coll_int_MW_' + self.ion_tag])
        self.Sp_coll_int_MW_e = np.asarray(data['Sp_coll_int_MW_e'])

        self.Sn_tot_int_SI20_i = np.asarray(data['Sn_tot_int_SI20_' + self.ion_tag])
        self.Sn_tot_int_SI20_e = np.asarray(data['Sn_tot_int_SI20_e'])
        self.Sn_tot_cumint_SI20_i = np.asarray(data['Sn_tot_cumint_SI20_' + self.ion_tag])
        self.Sn_tot_cumint_SI20_e = np.asarray(data['Sn_tot_cumint_SI20_e'])
        if len(self.Sn_tot_cumint_SI20_i) == 0:  # backwards compat
            self.Sn_tot_cumint_SI20_i = np.asarray(data['Sn_tot_int_SI20_' + self.ion_tag])
            self.Sn_tot_cumint_SI20_e = np.asarray(data['Sn_tot_int_SI20_e'])

        self.Sp_aux_cumint_MW_i = np.asarray(data['Sp_aux_cumint_MW_' + self.ion_tag])
        self.Sp_aux_cumint_MW_e = np.asarray(data['Sp_aux_cumint_MW_e'])
        self.Sp_alpha_cumint_MW_i = np.asarray(data['Sp_alpha_cumint_MW_' + self.ion_tag])
        self.Sp_alpha_cumint_MW_e = np.asarray(data['Sp_alpha_cumint_MW_e'])
        self.Sp_rad_cumint_MW_i = np.asarray(data['Sp_rad_cumint_MW_' + self.ion_tag])
        self.Sp_rad_cumint_MW_e = np.asarray(data['Sp_rad_cumint_MW_e'])
        self.Sp_heating_cumint_MW_i = np.asarray(data['Sp_heating_cumint_MW_' + self.ion_tag])
        self.Sp_heating_cumint_MW_e = np.asarray(data['Sp_heating_cumint_MW_e'])
        self.Sp_coll_cumint_MW_i = np.asarray(data['Sp_coll_cumint_MW_' + self.ion_tag])
        self.Sp_coll_cumint_MW_e = np.asarray(data['Sp_coll_cumint_MW_e'])
        self.Sp_tot_cumint_MW_i = np.asarray(data['Sp_tot_cumint_MW_' + self.ion_tag])
        self.Sp_tot_cumint_MW_e = np.asarray(data['Sp_tot_cumint_MW_e'])
        if len(self.Sp_tot_cumint_MW_i) == 0:  # backwards compat
            self.Sp_tot_cumint_MW_i = np.asarray(data['Sp_tot_int_MW_' + self.ion_tag])
            self.Sp_tot_cumint_MW_e = np.asarray(data['Sp_tot_int_MW_e'])

        self.Palpha_MWm3 = np.asarray(data['Palpha_MWm3'])
        self.Palpha_int_MW = data['Palpha_int_MW']

        # power balance
        self.Q_MW_i = {}
        self.Q_MW_e = {}
        self.Q_MW_i['tot'] = np.asarray(data['Q_MW_' + self.ion_tag])
        self.Q_MW_e['tot'] = np.asarray(data['Q_MW_e'])
        for label in self.flux_model_labels:
            self.Q_MW_i[label] = np.asarray(data['Q_' + label + '_MW_' + self.ion_tag])
            self.Q_MW_e[label] = np.asarray(data['Q_' + label + '_MW_e'])

        # particle balance
        self.Gam_SI20_i = {}
        self.Gam_SI20_e = {}
        self.Gam_SI20_i['tot'] = np.asarray(data['Gam_SI20_' + self.ion_tag])
        self.Gam_SI20_e['tot'] = np.asarray(data['Gam_SI20_e'])
        for label in self.flux_model_labels:
            self.Gam_SI20_i[label] = np.asarray(data['Gam_' + label + '_SI20_' + self.ion_tag])
            self.Gam_SI20_e[label] = np.asarray(data['Gam_' + label + '_SI20_e'])

        self.grid = data['grid']
        self.grid_obj = data['grid_obj']
        self.geo_obj = data['geo_obj']
        self.N_rho = self.grid['N_radial']
        try:
            self.rho = self.grid['rho']
        except LookupError:
            self.rho = self.grid['rho_axis']  # For backward compatability for old output files
        try:
            self.midpoints = self.grid['midpoints']
        except LookupError:
            self.rho = self.grid['mid_axis']  # For backward compatability for old output files
        try:
            self.flux_label = self.grid['flux_label']
        except LookupError:
            self.flux_label = 'rminor'
        if self.flux_label == 'rminor':
            self.rad_label = r'$r/a$'
        elif self.flux_label == 'torflux':
            self.rad_label = r'$\rho_{tor}$'

        geo = data['geo_obj']
        self.area = geo.area.profile
        self.grho = geo.grho.profile
        self.area_grid = geo.area.toGridProfile().profile
        self.grho_grid = geo.grho.toGridProfile().profile
        self.G_fac = geo.G_fac.profile

        try:
            self.Btor = geo.Btor
        except:
            # backward compatible to JET runs
            self.Btor = data['geo']['B_ref']

        self.grid = data['grid_obj']

        self.dtau = data['time']['dtau']
        self.geo = geo

        # form a path for plotting output
        if fin[-7:] == "log.npy":
            froot = fin[:-7]
        elif fin[-3:] == "npy":
            froot = fin[:-3]
        else:
            froot = fin
        self.fileout = getcwd() + '/' + froot

        self.filename = data['trinity_infile']
        self.data = data

    def read_netcdf(self, fin):
        ''' Function to read the Trinity3D NetCDF file '''

        # Open the NetCDF file
        ncf = Dataset(fin, "r", format="NETCDF4")

        # form a path for plotting output
        if fin[-2:] == "nc":
            froot = fin[:-2]
        else:
            froot = fin
        self.fileout = getcwd() + '/' + froot
        self.filename = ncf.infile

        # Read norms and parameters
        self.t_ref = ncf['norms'].variables['t_ref'][0]
        self.rho_star = ncf['norms'].variables['rho_star'][0]
        self.P_ref_MWm3 = ncf['norms'].variables['P_ref_MWm3'][0]
        self.Sn_ref_SI20 = ncf['norms'].variables['Sn_ref_SI20'][0]
        self.flux_label = ncf['norms'].variables['flux_label'][0]
        self.a_minor = ncf['norms'].variables['a_minor'][0]

        # Read and dimensionalize the time array
        self.N = ncf.dimensions['n_steps'].size
        self.time = ncf['time'].variables['t'][:]
        self.time *= self.t_ref

        # Read the species profile data
        self.ion_tag = ncf['species'].variables['tags'].bulk_ion_tag
        self.n = ncf['species'].variables['n_e'][:, :]
        self.pe = ncf['species'].variables['p_e'][:, :]
        self.Te = ncf['species'].variables['T_e'][:, :]
        self.pi = ncf['species'].variables['p_' + self.ion_tag][:, :]
        self.Ti = ncf['species'].variables['T_' + self.ion_tag][:, :]
        self.Gamma = {}
        self.Qi = {}
        self.Qe = {}
        self.Gamma['tot'] = ncf['species'].variables['pflux_' + self.ion_tag][:, :]
        self.Qi['tot'] = ncf['species'].variables['qflux_' + self.ion_tag][:, :]
        self.Qe['tot'] = ncf['species'].variables['qflux_e'][:, :]
        try:
            self.flux_model_labels = ncf['species'].variables['flux_model_labels'][:]
        except:
            self.flux_model_labels = ['turb', 'neo']  # backwards compat
        for label in self.flux_model_labels:
            self.Gamma[label] = ncf['species'].variables['pflux_' + label + '_' + self.ion_tag][:, :]
            self.Qi[label] = ncf['species'].variables['qflux_' + label + '_' + self.ion_tag][:, :]
            self.Qe[label] = ncf['species'].variables['qflux_' + label + '_e'][:, :]
        self.aLn = ncf['species'].variables['aLn_' + self.ion_tag][:, :]
        self.aLpi = ncf['species'].variables['aLp_' + self.ion_tag][:, :]
        self.aLpe = ncf['species'].variables['aLp_e'][:, :]
        self.aLTi = ncf['species'].variables['aLT_' + self.ion_tag][:, :]
        self.aLTe = ncf['species'].variables['aLT_e'][:, :]

        # Density sources, in code units
        self.Sn_aux_i = ncf['species'].variables['Sn_aux_' + self.ion_tag][:, :]
        self.Sn_aux_e = ncf['species'].variables['Sn_aux_e'][:, :]

        # Pressure sources, in code units, multiply by P_ref_MWm3 to get MW/m^3
        self.Sp_aux_i = ncf['species'].variables['Sp_aux_' + self.ion_tag][:, :]
        self.Sp_aux_e = ncf['species'].variables['Sp_aux_e'][:, :]
        self.Sp_alpha_i = ncf['species'].variables['Sp_alpha_' + self.ion_tag][:, :]
        self.Sp_alpha_e = ncf['species'].variables['Sp_alpha_e'][:, :]
        self.Sp_rad_i = ncf['species'].variables['Sp_rad_' + self.ion_tag][:, :]
        self.Sp_rad_e = ncf['species'].variables['Sp_rad_e'][:, :]
        self.Sp_heating_i = ncf['species'].variables['Sp_heating_' + self.ion_tag][:, :]
        self.Sp_heating_e = ncf['species'].variables['Sp_heating_e'][:, :]
        self.Sp_coll_i = ncf['species'].variables['Sp_coll_' + self.ion_tag][:, :]
        self.Sp_coll_e = ncf['species'].variables['Sp_coll_e'][:, :]
        self.Sp_tot_i = ncf['species'].variables['Sp_tot_' + self.ion_tag][:, :]
        self.Sp_tot_e = ncf['species'].variables['Sp_tot_e'][:, :]

        self.Sp_aux_int_MW_i = ncf['species'].variables['Sp_aux_int_MW_' + self.ion_tag][:]
        self.Sp_aux_int_MW_e = ncf['species'].variables['Sp_aux_int_MW_e'][:]
        self.Sp_aux_int_MW_tot = self.Sp_aux_int_MW_i + self.Sp_aux_int_MW_e
        self.Sp_alpha_int_MW_i = ncf['species'].variables['Sp_alpha_int_MW_' + self.ion_tag][:]
        self.Sp_alpha_int_MW_e = ncf['species'].variables['Sp_alpha_int_MW_e'][:]
        self.Sp_rad_int_MW_i = ncf['species'].variables['Sp_rad_int_MW_' + self.ion_tag][:]
        self.Sp_rad_int_MW_e = ncf['species'].variables['Sp_rad_int_MW_e'][:]
        self.Prad_tot = ncf['species'].variables['Sp_rad_int_MW_e'][:]
        self.Sp_heating_int_MW_i = ncf['species'].variables['Sp_heating_int_MW_' + self.ion_tag][:]
        self.Sp_heating_int_MW_e = ncf['species'].variables['Sp_heating_int_MW_e'][:]
        self.Sp_coll_int_MW_i = ncf['species'].variables['Sp_coll_int_MW_' + self.ion_tag][:]
        self.Sp_coll_int_MW_e = ncf['species'].variables['Sp_coll_int_MW_e'][:]

        self.Sn_tot_cumint_SI20_i = ncf['species'].variables['Sn_tot_cumint_SI20_' + self.ion_tag][:, :]
        self.Sn_tot_cumint_SI20_e = ncf['species'].variables['Sn_tot_cumint_SI20_e'][:, :]

        self.Sp_aux_cumint_MW_i = ncf['species'].variables['Sp_aux_cumint_MW_' + self.ion_tag][:]
        self.Sp_aux_cumint_MW_e = ncf['species'].variables['Sp_aux_cumint_MW_e'][:]
        self.Sp_alpha_cumint_MW_i = ncf['species'].variables['Sp_alpha_cumint_MW_' + self.ion_tag][:]
        self.Sp_alpha_cumint_MW_e = ncf['species'].variables['Sp_alpha_cumint_MW_e'][:]
        self.Sp_rad_cumint_MW_i = ncf['species'].variables['Sp_rad_cumint_MW_' + self.ion_tag][:]
        self.Sp_rad_cumint_MW_e = ncf['species'].variables['Sp_rad_cumint_MW_e'][:]
        self.Sp_heating_cumint_MW_i = ncf['species'].variables['Sp_heating_cumint_MW_' + self.ion_tag][:]
        self.Sp_heating_cumint_MW_e = ncf['species'].variables['Sp_heating_cumint_MW_e'][:]
        self.Sp_coll_cumint_MW_i = ncf['species'].variables['Sp_coll_cumint_MW_' + self.ion_tag][:]
        self.Sp_coll_cumint_MW_e = ncf['species'].variables['Sp_coll_cumint_MW_e'][:]
        self.Sp_tot_cumint_MW_i = ncf['species'].variables['Sp_tot_cumint_MW_' + self.ion_tag][:, :]
        self.Sp_tot_cumint_MW_e = ncf['species'].variables['Sp_tot_cumint_MW_e'][:, :]

        # Alpha heating
        self.Palpha_MWm3 = ncf['species'].variables['Palpha_MWm3'][:, :]
        self.Palpha_int_MW = ncf['species'].variables['Palpha_int_MW'][:]

        # Power balance
        self.Q_MW_i = {}
        self.Q_MW_e = {}
        self.Q_MW_i['tot'] = ncf['species'].variables['Q_MW_' + self.ion_tag][:, :]
        self.Q_MW_e['tot'] = ncf['species'].variables['Q_MW_e'][:, :]
        for label in self.flux_model_labels:
            self.Q_MW_i[label] = ncf['species'].variables['Q_' + label + '_MW_' + self.ion_tag][:, :]
            self.Q_MW_e[label] = ncf['species'].variables['Q_' + label + '_MW_e'][:, :]

        # Particle balance
        self.Gam_SI20_i = {}
        self.Gam_SI20_e = {}
        self.Gam_SI20_i['tot'] = ncf['species'].variables['Gam_SI20_' + self.ion_tag][:, :]
        self.Gam_SI20_e['tot'] = ncf['species'].variables['Gam_SI20_e'][:, :]
        for label in self.flux_model_labels:
            self.Gam_SI20_i[label] = ncf['species'].variables['Gam_' + label + '_SI20_' + self.ion_tag][:, :]
            self.Gam_SI20_e[label] = ncf['species'].variables['Gam_' + label + '_SI20_e'][:, :]

        # Grid
        self.N_rho = ncf.dimensions['n_radial'].size
        try:
            self.rho = ncf['grid'].variables['rho'][:]
        except LookupError:
            self.rho = ncf['grid'].variables['rho_axis'][:]  # For backward compatability for old output files
        try:
            self.midpoints = ncf['grid'].variables['midpoints'][:]
        except LookupError:
            self.midpoints = ncf['grid'].variables['mid_axis'][:]  # For backward compatability for old output files
        if self.flux_label == 'rminor':
            self.rad_label = r'$r/a$'
        elif self.flux_label == 'torflux':
            self.rad_label = r'$\rho_{tor}$'

        # Close the NetCDF file
        ncf.close()

    def read_adios2(self, fin):
        ''' Function to read the Trinity3D ADIOS2 output '''

        with adios2.FileReader(fin) as fh:

            # varname = 'species/T_H'
            # T_H = fh.read(varname)
            # print(T_H)

            # form a path for plotting output
            if fin[-2:] == "bp":
                froot = fin[:-2]
            elif fin[-3:] == "bp/":
                froot = fin[:-3]
            else:
                froot = fin
            self.fileout = getcwd() + '/' + froot
            self.filename = fh.read_attribute('infile')

            # Read norms and parameters
            self.t_ref = fh.read_attribute('norms/t_ref')
            self.rho_star = fh.read_attribute('norms/rho_star')
            self.P_ref_MWm3 = fh.read_attribute('norms/P_ref_MWm3')
            self.Sn_ref_SI20 = fh.read_attribute('norms/Sn_ref_SI20')
            self.flux_label = fh.read_attribute('norms/flux_label')
            self.a_minor = fh.read_attribute('norms/a_minor')

            # Read and dimensionalize the time array
            self.N = fh.read_attribute('n_steps')
            self.time = fh.read('time')
            self.time *= self.t_ref

            # Read the species profile data
            self.ion_tag = fh.read_attribute('species/bulk_ion_tag')
            self.n = fh.read('species/n_e')
            self.pe = fh.read('species/p_e')
            self.Te = fh.read('species/T_e')
            self.pi = fh.read('species/p_' + self.ion_tag)
            self.Ti = fh.read('species/T_' + self.ion_tag)
            self.Gamma = {}
            self.Qi = {}
            self.Qe = {}
            self.Gamma['tot'] = fh.read('species/pflux_' + self.ion_tag)
            self.Qi['tot'] = fh.read('species/qflux_' + self.ion_tag)
            self.Qe['tot'] = fh.read('species/qflux_e')
            try:
                self.flux_model_labels = fh.read_attribute('species/flux_model_labels')
            except:
                self.flux_model_labels = ['turb', 'neo']  # backwards compat
            for label in self.flux_model_labels:
                self.Gamma[label] = fh.read('species/pflux_' + label + '_' + self.ion_tag)
                self.Qi[label] = fh.read('species/qflux_' + label + '_' + self.ion_tag)
                self.Qe[label] = fh.read('species/qflux_' + label + '_e')
            self.aLn = fh.read('species/aLn_' + self.ion_tag)
            self.aLpi = fh.read('species/aLp_' + self.ion_tag)
            self.aLpe = fh.read('species/aLp_e')
            self.aLTi = fh.read('species/aLT_' + self.ion_tag)
            self.aLTe = fh.read('species/aLT_e')

            # Density sources, in code units
            self.Sn_aux_i = fh.read('species/Sn_aux_' + self.ion_tag)
            self.Sn_aux_e = fh.read('species/Sn_aux_e')

            # Pressure sources, in code units, multiply by P_ref_MWm3 to get MW/m^3
            self.Sp_aux_i = fh.read('species/Sp_aux_' + self.ion_tag)
            self.Sp_aux_e = fh.read('species/Sp_aux_e')
            self.Sp_alpha_i = fh.read('species/Sp_alpha_' + self.ion_tag)
            self.Sp_alpha_e = fh.read('species/Sp_alpha_e')
            self.Sp_rad_i = fh.read('species/Sp_rad_' + self.ion_tag)
            self.Sp_rad_e = fh.read('species/Sp_rad_e')
            self.Sp_heating_i = fh.read('species/Sp_heating_' + self.ion_tag)
            self.Sp_heating_e = fh.read('species/Sp_heating_e')
            self.Sp_coll_i = fh.read('species/Sp_coll_' + self.ion_tag)
            self.Sp_coll_e = fh.read('species/Sp_coll_e')
            self.Sp_tot_i = fh.read('species/Sp_tot_' + self.ion_tag)
            self.Sp_tot_e = fh.read('species/Sp_tot_e')

            self.Sp_aux_int_MW_i = fh.read('species/Sp_aux_int_' + self.ion_tag)
            self.Sp_aux_int_MW_e = fh.read('species/Sp_aux_int_e')
            self.Sp_aux_int_MW_tot = self.Sp_aux_int_MW_i + self.Sp_aux_int_MW_e
            self.Sp_alpha_int_MW_i = fh.read('species/Sp_alpha_int_' + self.ion_tag)
            self.Sp_alpha_int_MW_e = fh.read('species/Sp_alpha_int_e')
            self.Sp_rad_int_MW_i = fh.read('species/Sp_rad_int_' + self.ion_tag)
            self.Sp_rad_int_MW_e = fh.read('species/Sp_rad_int_e')
            self.Prad_tot = fh.read('species/Sp_rad_int_e')
            self.Sp_heating_int_MW_i = fh.read('species/Sp_heating_int_' + self.ion_tag)
            self.Sp_heating_int_MW_e = fh.read('species/Sp_heating_int_e')
            self.Sp_coll_int_MW_i = fh.read('species/Sp_coll_int_' + self.ion_tag)
            self.Sp_coll_int_MW_e = fh.read('species/Sp_coll_int_e')

            self.Sn_tot_cumint_SI20_i = fh.read('species/Sn_tot_cumint_' + self.ion_tag)
            self.Sn_tot_cumint_SI20_e = fh.read('species/Sn_tot_cumint_e')

            self.Sp_aux_cumint_MW_i = fh.read('species/Sp_aux_cumint_' + self.ion_tag)
            self.Sp_aux_cumint_MW_e = fh.read('species/Sp_aux_cumint_e')
            self.Sp_alpha_cumint_MW_i = fh.read('species/Sp_alpha_cumint_' + self.ion_tag)
            self.Sp_alpha_cumint_MW_e = fh.read('species/Sp_alpha_cumint_e')
            self.Sp_rad_cumint_MW_i = fh.read('species/Sp_rad_cumint_' + self.ion_tag)
            self.Sp_rad_cumint_MW_e = fh.read('species/Sp_rad_cumint_e')
            self.Sp_heating_cumint_MW_i = fh.read('species/Sp_heating_cumint_' + self.ion_tag)
            self.Sp_heating_cumint_MW_e = fh.read('species/Sp_heating_cumint_e')
            self.Sp_coll_cumint_MW_i = fh.read('species/Sp_coll_cumint_' + self.ion_tag)
            self.Sp_coll_cumint_MW_e = fh.read('species/Sp_coll_cumint_e')
            self.Sp_tot_cumint_MW_i = fh.read('species/Sp_tot_cumint_' + self.ion_tag)
            self.Sp_tot_cumint_MW_e = fh.read('species/Sp_tot_cumint_e')

            # Alpha heating
            self.Palpha_MWm3 = fh.read('species/Palpha')
            self.Palpha_int_MW = fh.read('species/Palpha_int')

            # Power balance
            self.Q_MW_i = {}
            self.Q_MW_e = {}
            self.Q_MW_i['tot'] = fh.read('species/Q_' + self.ion_tag)
            self.Q_MW_e['tot'] = fh.read('species/Q_e')
            for label in self.flux_model_labels:
                self.Q_MW_i[label] = fh.read('species/Q_' + label + '_' + self.ion_tag)
                self.Q_MW_e[label] = fh.read('species/Q_' + label + '_e')

            # Particle balance
            self.Gam_SI20_i = {}
            self.Gam_SI20_e = {}
            self.Gam_SI20_i['tot'] = fh.read('species/Gam_' + self.ion_tag)
            self.Gam_SI20_e['tot'] = fh.read('species/Gam_e')
            for label in self.flux_model_labels:
                self.Gam_SI20_i[label] = fh.read('species/Gam_' + label + '_' + self.ion_tag)
                self.Gam_SI20_e[label] = fh.read('species/Gam_' + label + '_e')

            # Grid
            self.N_rho = fh.read_attribute('n_radial')
            self.rho = fh.read('grid/rho')
            self.midpoints = fh.read('grid/midpoints')
            if self.flux_label == 'rminor':
                self.rad_label = r'$r/a$'
            elif self.flux_label == 'torflux':
                self.rad_label = r'$\rho_{tor}$'

    def list_profile_times(self):
        ''' Function to list the available profile times and index number that are not from Newton iterations '''
        for i in range(self.N):
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            print(f'Profile #{i} at time {t_curr:.6e} s')

    def plot_density_profiles(self, axs):
        ''' Function to plot the density profiles '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            # plot profiles
            if i == 0 or i == self.tidx[-1]:
                axs.plot(self.rho, self.n[i], '.-', color=self.green_map[i],
                         label=f'$n_e$, $t =${t_curr:.2f} s')
            else:
                axs.plot(self.rho, self.n[i], '.-', color=self.green_map[i])
        nmax = np.max(self.n)
        axs.set_ylim(bottom=0, top=1.5 * nmax)
        axs.set_xlim(left=0.0)
        axs.set_xlabel(self.rad_label)
        axs.set_title(r'density [10$^{20}$ m$^{-3}$]')
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_temperature_profiles(self, axs):
        ''' Function to plot the temperature profiles '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            # plot profiles
            if i == 0:
                axs.plot(self.rho, self.Ti[i], '.-', color=self.warm_map[i], label=f'$T_i$, $t =${t_curr:.2f} s')
                axs.plot(self.rho, self.Te[i], '.-', color=self.cool_map[i], label=f'$T_e$, $t =${t_curr:.2f} s')
            elif i == self.tidx[-1]:
                axs.plot(self.rho, self.Ti[i], '.-', color=self.warm_map[i], label=f'$T_i$, $t =${t_curr:.2f} s')
                if self.plot_electrons:
                    axs.plot(self.rho, self.Te[i], '.-', color=self.cool_map[i], label=f'$T_e$, $t =${t_curr:.2f} s')
            else:
                axs.plot(self.rho, self.Ti[i], '.-', color=self.warm_map[i])
                if self.plot_electrons:
                    axs.plot(self.rho, self.Te[i], '.-', color=self.cool_map[i])
        axs.set_ylim(bottom=0)
        axs.set_xlim(left=0.0)
        axs.set_xlabel(self.rad_label)
        axs.set_title('temperature [keV]')
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_pressure_profiles(self, axs):
        ''' Function to plot the pressure profiles '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            # plot profiles
            axs.plot(self.rho, self.pi[i], '.-', color=self.warm_map[i])
            if self.plot_electrons:
                axs.plot(self.rho, self.pe[i], '.-', color=self.cool_map[i])
        axs.set_ylim(bottom=0)
        axs.set_xlim(left=0.0)
        axs.set_xlabel(self.rad_label)
        axs.set_title(r'pressure [10$^{20}$m$^{-3}$ keV]')
        axs.grid(self.grid_lines)

    def plot_power_balance(self, axs):
        ''' Function to plot the power balance '''
        i = self.tidx[-1]
        axs.plot(self.midpoints, self.Q_MW_i['tot'][i], 'o-', color=self.warm_map[i], fillstyle='none', label='$Q_i\\cdot A_{{surf}}$ (total)')
        marker = itertools.cycle(('d', 's', 'v', 'P', '*', 'X'))
        for label in self.flux_model_labels:
            axs.plot(self.midpoints, self.Q_MW_i[label][i], '-', marker=next(marker), color=self.warm_map[i], fillstyle='none', label=f'$Q_i\\cdot A_{{surf}}$ ({label})')
        axs.plot(self.midpoints, self.Sp_tot_cumint_MW_i[i][:-1], 'o:', color=self.warm_map[i], label=r'$\int S_{p,i}$')
        axs.plot(self.midpoints, self.Q_MW_e['tot'][i], 'o-', color=self.cool_map[i], fillstyle='none', label='$Q_e\\cdot A_{{surf}}$ (total)')
        marker = itertools.cycle(('d', 's', 'v', 'P', '*', 'X'))
        for label in self.flux_model_labels:
            axs.plot(self.midpoints, self.Q_MW_e[label][i], '-', marker=next(marker), color=self.cool_map[i], fillstyle='none', label=f'$Q_e\\cdot A_{{surf}}$ ({label})')
        axs.plot(self.midpoints, self.Sp_tot_cumint_MW_e[i][:-1], 'o:', color=self.cool_map[i], label=r'$\int S_{p,e}$')
        _, top = axs.get_ylim()
        axs.set_ylim(bottom=0, top=1.6 * top)
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        axs.set_title('power balance [MW]')
        numitems = len(list(axs.get_legend_handles_labels()[1]))
        nrows = int(numitems/2)
        ncols = int(np.ceil(numitems / float(nrows)))
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=ncols, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_heat_flux(self, axs):
        ''' Function to plot the heat flux '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            axs.plot(self.midpoints, self.Qi['tot'][i], 'x-', color=self.warm_map[i])
            axs.plot(self.midpoints, self.Qe['tot'][i], 'x-', color=self.cool_map[i])
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        axs.set_title('heat flux [GB]')
        axs.grid(self.grid_lines)

    def plot_particle_balance(self, axs):
        ''' Function to plot the particle balance '''
        i = self.tidx[-1]
        axs.plot(self.midpoints, self.Gam_SI20_e['tot'][i], 'o-', color=self.green_map[i], fillstyle='none', label=r'$\Gamma_e\cdot A_{{surf}}$')
        marker = itertools.cycle(('d', 's', 'v', 'P', '*', 'X'))
        for label in self.flux_model_labels:
            axs.plot(self.midpoints, self.Gam_SI20_e[label][i], '-', marker=next(marker), color=self.green_map[i], fillstyle='none', label=f'$\\Gamma_e\\cdot A_{{surf}}$ ({label})')
        axs.plot(self.midpoints, self.Sn_tot_cumint_SI20_e[i][:-1], 'o:', color=self.green_map[i], label=r'$\int S_{ne}$')
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        l, u = axs.get_ylim()
        if l > -10:
            axs.set_ylim(bottom=-10)
        if u < 10:
            axs.set_ylim(top=10)
        axs.set_title('particle balance [10$^{20}$/s]')
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_particle_flux(self, axs):
        ''' Function to plot the particle flux '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            axs.plot(self.midpoints, self.Gamma['tot'][i], 'x-', color=self.green_map[i])
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        axs.set_title('particle flux [GB]')
        axs.grid(self.grid_lines)

    def plot_fusion_power_density(self, axs):
        ''' Function fusion power density '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            if i == self.tidx[-1]:
                axs.plot(self.rho, self.Sp_alpha_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label='$S_{p,i}$')
                axs.plot(self.rho, self.Sp_alpha_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label='$S_{p,e}$')
                axs.plot(self.rho, self.Palpha_MWm3[i], '.-', color=self.green_map[i], label='$S_{p,{tot}}$')
            else:
                axs.plot(self.rho, self.Sp_alpha_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i])
                axs.plot(self.rho, self.Sp_alpha_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i])
                axs.plot(self.rho, self.Palpha_MWm3[i], '.-', color=self.green_map[i])
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        axs.set_title('fusion power density \n [MW/m$^{3}$]')
        title = (r'$P_{\alpha} = $' + f'{self.Palpha_int_MW[i]:.2f} MW\n' +
                 r'$P_{fus} = $' + f'{5.03 * self.Palpha_int_MW[i]:.2f} MW')
        leg = axs.legend(loc='best', title=title, fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_radiation_profiles(self, axs):
        ''' Function to plot the radiation profiles '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            if i == self.tidx[-1]:
                axs.plot(self.rho, self.Sp_rad_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label='$S_{p,i}$')
                axs.plot(self.rho, self.Sp_rad_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label='$S_{p,e}$')
            else:
                axs.plot(self.rho, self.Sp_rad_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i])
                axs.plot(self.rho, self.Sp_rad_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i])
        axs.set_title('radiation [MW/m$^{3}$]')
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        try:
            leg = axs.legend(loc='best', title=r'$P_{rad} = $' + f'{self.Prad_tot[-1]:.2f} MW', fancybox=False, shadow=False, ncol=1, fontsize=8)
            leg.get_frame().set_edgecolor('k')
            leg.get_frame().set_linewidth(0.65)
        except:
            pass
        axs.grid(self.grid_lines)

    def plot_exchange(self, axs):
        ''' Function to plot the collisional and turbulent exchange terms '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            if i == self.tidx[-1]:
                axs.plot(self.rho, self.Sp_coll_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label=r'$S^\mathrm{coll}_{p,i}$')
                axs.plot(self.rho, self.Sp_coll_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label=r'$S^\mathrm{coll}_{p,e}$')
                axs.plot(self.rho, self.Sp_heating_i[i] * self.P_ref_MWm3, '.:', color=self.warm_map[i], label=r'$S^\mathrm{turb}_{p,i}$')
                axs.plot(self.rho, self.Sp_heating_e[i] * self.P_ref_MWm3, '.:', color=self.cool_map[i], label=r'$S^\mathrm{turb}_{p,e}$')
            else:
                axs.plot(self.rho, self.Sp_coll_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i])
                axs.plot(self.rho, self.Sp_coll_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i])
                axs.plot(self.rho, self.Sp_heating_i[i] * self.P_ref_MWm3, '.:', color=self.warm_map[i])
                axs.plot(self.rho, self.Sp_heating_e[i] * self.P_ref_MWm3, '.:', color=self.cool_map[i])
        axs.set_title('collisional and turbulent \n exchange [MW/m$^{3}$]')
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        leg = axs.legend(loc='best', title=r'$P^{coll}_i = $' + f'{self.Sp_coll_int_MW_i[-1]:.2f} MW', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_auxiliary_power_source(self, axs):
        ''' Function to plot auxiliary power source, convert from Trinity units to MW/m3 '''
        i = self.tidx[-1]
        axs.plot(self.rho, self.Sp_aux_i[i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label='$S_{p,i}$')
        axs.plot(self.rho, self.Sp_aux_e[i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label='$S_{p,e}$')
        _, top = axs.get_ylim()
        axs.set_ylim(bottom=0, top=1.5 * top)
        axs.set_xlim(left=0.0)
        axs.set_title('auxiliary power source \n [MW/m$^{3}$]')
        axs.set_xlabel(self.rad_label)
        leg = axs.legend(loc='best', title='$P_{aux} =$' + f'{self.Sp_aux_int_MW_tot[i]:.2f} MW', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_q(self, axs):
        ''' Function to plot q '''
        r = self.ridx
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            axs.plot(self.aLpi[i][r] - self.aLn[i][r], self.Qi['tot'][i][r], '.', color=self.warm_map[i])
            axs.plot(self.aLpe[i][r] - self.aLn[i][r], self.Qe['tot'][i][r], '.', color=self.cool_map[i])
        axs.set_title(r'$Q(L_T)$ [GB]')
        axs.set_xlabel('$a/L_T$')
        # leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        # leg.get_frame().set_edgecolor('k')
        # leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_auxiliary_particle_source(self, axs):
        ''' Function to plot the auxiliary particle source '''
        i = self.N - 1
        axs.plot(self.rho, self.Sn_aux_e[i] * self.Sn_ref_SI20, '.-', color=self.green_map[i], label='$S_{n}$')
        _, top = axs.get_ylim()
        axs.set_ylim(bottom=0, top=1.5 * top)
        axs.set_xlim(left=0.0)
        axs.set_xlabel(self.rad_label)
        axs.set_title('auxiliary particle source \n [10$^{20}$/(m$^{3}$ s)]')
        axs.grid(self.grid_lines)

    def plot_gamma(self, axs):
        ''' Function to plot gamma '''
        r = self.ridx
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            axs.plot(self.aLn[i][r], self.Gamma['tot'][i][r], '.', color=self.green_map[i])
        axs.set_xlabel('$a/L_n$')
        axs.set_title(r'$\Gamma(L_n)$ [GB]')
        axs.grid(self.grid_lines)

    def plot_state_profiles(self, axs, profile='Ti', t_stop=-1):
        ''' Function to plot the density or temperature profiles '''

        if t_stop < 0:
            t_stop = self.N

        t_old = -1
        for ti in np.arange(t_stop):
            t_curr = self.time[ti]
            if (t_curr == t_old):
                # skip majority of prints
                continue
            else:
                t_old = t_curr
                t = ti
            # plot profiles
            if profile == 'Ti':
                if t == 0 or t == t_stop-1:
                    axs.plot(self.rho, self.Ti[t], '.-', color=self.warm_map[t], label=f'$T_i$, $t =${self.time[t]:.2f} s')
                else:
                    axs.plot(self.rho, self.Ti[t], '.-', color=self.warm_map[t])
                axs.set_title('temperature [keV]')

            if profile == 'Te':
                if t == 0 or t == t_stop-1:
                    axs.plot(self.rho, self.Te[t], '.-', color=self.cool_map[t], label=f'$T_e$, $t =${self.time[t]:.2f} s')
                else:
                    axs.plot(self.rho, self.Te[t], '.-', color=self.cool_map[t])
                axs.set_title('temperature [keV]')

            if profile == 'ne':
                if t == 0 or t == t_stop-1:
                    axs.plot(self.rho, self.n[t], '.-', color=self.green_map[t], label=f'$n_e$, $t =${self.time[t]:.2f} s')
                else:
                    axs.plot(self.rho, self.n[t], '.-', color=self.green_map[t])
                axs.set_title(r'density [10$^{20}$ m$^{-3}$]')

        _, top = axs.get_ylim()
        axs.set_ylim(bottom=0, top=1.5*top)
        axs.set_xlim(left=0.0)
        axs.set_xlabel(self.rad_label)
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_flux(self, axs, profile='Ti', t_stop=-1):
        '''
        This function used to plot the flux from GX
        now it plots the total flux

        A future version could break out parts in a FOR loop on label
        '''

        if t_stop < 0:
            t_stop = self.N

        t_old = -1
        for ti in np.arange(t_stop):
            t_curr = self.time[ti]
            if (t_curr == t_old):
                # skip majority of prints
                continue
            else:
                t_old = t_curr
                t = ti

            if profile == 'Ti':
                axs.plot(self.aLpi[t] - self.aLn[t], self.Qi['tot'][t], '.', color=self.warm_map[t])
                axs.set_title(r'$Q(L_{T_i})$ [GB]')
                axs.set_xlabel('$a/L_{{T_i}}$')

            if profile == 'Te':
                axs.plot(self.aLpe[t] - self.aLn[t], self.Qe['tot'][t], '.', color=self.cool_map[t])
                axs.set_title(r'$Q(L_{T_e})$ [GB]')
                axs.set_xlabel('$a/L_{{T_e}}$')

            if profile == 'ne':
                axs.plot(self.aLn[t], self.Gamma['tot'][t], '.', color=self.green_map[t])
                axs.set_xlabel('$a/L_n$')
                axs.set_title(r'$\Gamma(L_n)$ [GB]')
        # leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        # leg.get_frame().set_edgecolor('k')
        # leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_source_total(self, axs, profile='Ti', t=-1):
        ''' Plots total source and sink terms, for each profile '''

        if t < 0:
            t = self.N - 1
        elif t == 0:
            return
        else:
            t = t-1

        if profile == 'Ti':

            Spi_aux = self.Sp_aux_i[t] * self.P_ref_MWm3
            Spi_tot = self.Sp_tot_int_MW_i[t]  # for some reason Sp_aux is excluded?
            axs.plot(self.rho, Spi_tot + Spi_aux, 'x--', color=self.warm_map[t], label=r'$S_{p,i}^{tot}$')

            axs.plot(self.midpoints, self.Q_MW_i[t], 'o:', color=self.warm_map[t], fillstyle='none', label='$-Q_i^{tot}$')
            axs.set_title('ion total source [MW]')

        if profile == 'Te':
            Spe_aux = self.Sp_aux_e[t] * self.P_ref_MWm3
            Spe_tot = self.Sp_tot_int_MW_e[t]  # for some reason Sp_aux is excluded?
            axs.plot(self.rho, Spe_tot + Spe_aux, 'x--', color=self.cool_map[t], label=r'$S_{p,e}^{tot}$')

            axs.plot(self.midpoints, self.Q_MW_e[t], 'o:', color=self.cool_map[t], fillstyle='none', label='$-Q_e^{tot}$')
            axs.set_title('electron total source [MW]')

        if profile == 'ne':
            axs.plot(self.midpoints, self.Sn_tot_int_SI20_e[t][:-1], 'x--', color=self.green_map[t], label=r'$S_n$')
            axs.plot(self.midpoints, self.Gam_SI20_e[t], 'o:', color=self.green_map[t], fillstyle='none', label=r'$-\Gamma$')
            axs.set_title('ne total source [10$^{20}$/s]')

        _, top = axs.get_ylim()
        axs.set_ylim(bottom=0, top=2*top)
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_sink_terms(self, axs, profile='Ti', t=-1):
        ''' Plots source and sink term-by-term, for each profile '''

        if t < 0:
            t = self.N - 1
        elif t == 0:
            return
        else:
            t = t-1

        if profile == 'Ti':
            axs.plot(self.midpoints, self.Q_turb_MW_i[t], 's--', color=self.warm_map[t], fillstyle='full', label=r'$Q_i^{\mathrm{turb}}$')
            if np.any(self.Q_neo_MW_i[t] > 1e-15):
                axs.plot(self.midpoints, self.Q_neo_MW_i[t], 'd--', color=self.warm_map[t], fillstyle='full', label=r'$Q_i^{\mathrm{neo}}$')

            if np.any(self.Q_alpha_int_MW_i[t] > 1e-15):
                axs.plot(self.rho, self.Q_alpha_int_MW_i[t], 'H--', color=self.warm_map[t], fillstyle='none', label=r'$-S_{p,i}^{\mathrm{alpha}}$')

            if np.any(self.Q_rad_int_MW_i[t] > 1e-15):
                axs.plot(self.rho, self.Q_rad_int_MW_i[t], '^--', color=self.warm_map[t], fillstyle='full', label=r'$Q_{p,i}^{\mathrm{rad}}$')

            if np.mean(self.Q_coll_int_MW_i[t] > 0):
                axs.plot(self.rho, self.Q_coll_int_MW_i[t], 'v--', color=self.cool_map[t], fillstyle='none', label=r'$-S_{p,i}^{\mathrm{coll}}$')
            elif np.mean(self.Q_coll_int_MW_i[t] < 0):
                axs.plot(self.rho, -self.Q_coll_int_MW_i[t], 'v--', color=self.cool_map[t], fillstyle='full', label=r'$Q_{p,i}^{\mathrm{coll}}$')

            axs.plot(self.rho, self.Sp_aux_i[t] * self.P_ref_MWm3, 'x--', color=self.warm_map[t], label=r'$-S_{p,i}^{\mathrm{aux}}$')

            axs.set_title('ion sink terms [MW]')

        if profile == 'Te':
            axs.plot(self.midpoints, self.Q_neo_MW_e[t], 'd--', color=self.cool_map[t], fillstyle='full', label=r'$Q_e^{\mathrm{neo}}$')

            if np.any(self.Q_rad_int_MW_e[t] != 0):
                axs.plot(self.rho, -self.Q_rad_int_MW_e[t], '^--', color=self.cool_map[t], fillstyle='full', label=r'$Q_{p,e}^{\mathrm{rad}}$')

            if np.any(self.Q_alpha_int_MW_e[t] != 0):
                axs.plot(self.rho, self.Q_alpha_int_MW_e[t], 'H--', color=self.cool_map[t], fillstyle='none', label=r'$-S_{p,e}^{\mathrm{alpha}}$')

            if np.any(self.Q_coll_int_MW_e[t] > 1e-15):
                axs.plot(self.rho, self.Q_coll_int_MW_e[t], 'v--', color=self.cool_map[t], fillstyle='none', label=r'$-S_{p,e}^{\mathrm{coll}}$')

            axs.plot(self.rho, self.Sp_aux_e[t] * self.P_ref_MWm3, 'x--', color=self.cool_map[t], label=r'$-S_{p,e}^{\mathrm{aux}}$')

            axs.set_title('electron sink terms [MW]')

        if profile == 'ne':
            axs.plot(self.midpoints, self.Gam_SI20_e[t], 'o-', color=self.green_map[t], fillstyle='none', label=r'$\Gamma^{\mathrm{turb}}$')
            axs.set_title('ne sink terms [10$^{20}$/s]')

        _, top = axs.get_ylim()
        axs.set_xlabel(self.rad_label)
        # axs.set_ylim(bottom=0, top=2*top)
        axs.set_xlim(left=0.0)
        leg = axs.legend(loc='upper left', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def make_plots(self, plots):
        ''' Create requested plots '''

        self.plot_electrons = True

        # set up color maps
        if len(self.tidx) == 1:
            self.warm_map = pylab.cm.autumn(np.linspace(0.25, 0.25, self.N))
            self.cool_map = pylab.cm.Blues(np.linspace(1, 1, self.N))
            self.green_map = pylab.cm.YlGn(np.linspace(1, 1, self.N))
        else:
            self.warm_map = pylab.cm.autumn(np.linspace(1, 0.25, self.N))
            self.cool_map = pylab.cm.Blues(np.linspace(0.25, 1, self.N))
            self.green_map = pylab.cm.YlGn(np.linspace(0.25, 1, self.N))

        for plot in plots:
            _, axs = plt.subplots(1, 1, figsize=(4,4))
            plotting_function = self.plotting_functions.get(plot.lower())
            plotting_function(axs)

            if self.savefig:
                figname = self.fileout + plot + '.pdf'
                print("Saving figure: "+figname)
                plt.savefig(figname)

        if not self.savefig:
            plt.show()

    def plot_panel(self):
        ''' Create the legacy plot panel '''

        rlabel = rf'[{self.filename}]'
        _, axs = plt.subplots(2, 7, figsize=(65, 8))

        plt.suptitle(rlabel)

        self.plot_electrons = True

        # set up color maps
        if len(self.tidx) == 1:
            self.warm_map = pylab.cm.autumn(np.linspace(0.25, 0.25, self.N))
            self.cool_map = pylab.cm.Blues(np.linspace(1, 1, self.N))
            self.green_map = pylab.cm.YlGn(np.linspace(1, 1, self.N))
            # purple_map = pylab.cm.Purples(np.linspace(1, 1, self.N))
        else:
            self.warm_map = pylab.cm.autumn(np.linspace(1, 0.25, self.N))
            self.cool_map = pylab.cm.Blues(np.linspace(0.25, 1, self.N))
            self.green_map = pylab.cm.YlGn(np.linspace(0.25, 1, self.N))
            # purple_map = pylab.cm.Purples(np.linspace(0.25, 1, self.N))

        for i, (_, plot) in enumerate(self.panel_plots.items()):
            plot(axs[int(i / 7), i % 7])

        # Create the plot panel
        plt.tight_layout()
        plt.subplots_adjust(left=0.05,
                            bottom=0.1,
                            right=0.95,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

        if self.savefig:
            figname = self.fileout + 'panel.pdf'
            print("Saving figure: "+figname)
            plt.savefig(figname)
        else:
            plt.show()


def main():

    import argparse

    plots = plotter()

    description = "This program creates plots from the T3D pickle file .npy, NetCDF4 .nc file, or ADIOS2 .bp output"
    epilog = '''\
Usage: to create the plot panel
  t3d-plot [trinity.log.npy]
  t3d-plot [trinity.nc]
  t3d-plot [trinity.bp]

Usage: to create indvidual plots
  t3d-plot [trinity.log.npy] -p density temperature etc
  t3d-plot [trinity.nc] -p density temperature etc
  t3d-plot [trinity.bp] -p density temperature etc
'''
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filename",
                        help="Specify plot file ")
    parser.add_argument("-p", "--plots",
                        nargs="+",
                        help="Space separated list of plots",
                        default=None)
    parser.add_argument("-t", "--tidx", nargs="+",
                        help="Space separated list of time indices",
                        default=None)
    parser.add_argument("-r", "--ridx", nargs="+",
                        help="Space separated list of rho indices",
                        default=None)
    parser.add_argument("-l", "--list",
                        help="List available plot routines",
                        action="store_true")
    parser.add_argument("-g", "--grid",
                        help="Include grid lines in plot",
                        action="store_true")
    parser.add_argument("-s", "--savefig",
                        help="Save plots to file rather than interactive display",
                        action="store_true")
    parser.add_argument("-i", "--show_newton_iterations", "--show-newton-iterations",
                        help="Plot intermediate Newton iterations",
                        action="store_true", default=False)
    parser.add_argument("--list-times",
                        help="List available profile times and numbers",
                        action="store_true")

    args = parser.parse_args()

    if args.list:
        plots.list_available_plots()
        exit(0)

    if args.grid:
        plots.grid_lines = True
    if args.savefig:
        plots.savefig = True

    if args.filename.split('.')[-1] == "npy":
        plots.read_data(args.filename)
    elif args.filename.split('.')[-1] == "nc":
        plots.read_netcdf(args.filename)
    elif (args.filename.split('.')[-1] == "bp" or
          args.filename.split('.')[-1] == "bp/"):
        plots.read_adios2(args.filename)
    else:
        print(f"ERROR: Output format not recognized {args.filename.split('.')[-1]}")
        exit(1)

    if args.tidx:
        plots.tidx = list(map(int, args.tidx))
    else:
        plots.tidx = np.arange(len(plots.time))

    if args.ridx:
        plots.ridx = list(map(int, args.ridx))
    else:
        plots.ridx = slice(None)

    plots.show_newton_iterations = args.show_newton_iterations

    if args.list_times:
        plots.list_profile_times()
        exit(0)

    if args.plots is None:
        plots.plot_panel()
    else:
        plots.make_plots(args.plots)


if __name__ == '__main__':
    '''
        This program plots Trinity3D output read from the .npy output
        file and stored in a python dictionary.
    '''

    main()
