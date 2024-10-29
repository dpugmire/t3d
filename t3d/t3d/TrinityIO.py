import os
import numpy as np
from collections import defaultdict
import copy
from netCDF4 import Dataset
import adios2
from datetime import date
import getpass
from t3d.Logbook import info, errr

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        errr("Please install tomli or upgrade to Python >= 3.11")
        exit()


def bool2str(val):
    # turn bool into lower case string
    if val:
        return 'true'
    else:
        return 'false'


def writeLine(s,pair,f):

    key,val = pair
    if type(val) is str:
        val = f"'{val}'"  # add quotes

    if type(val) is bool:
        val = bool2str(val)

    if type(val) is dict:
        out = "{"
        for k,v in val.items():

            if type(v) is bool:
                v = bool2str(v)

            out += "%s = %s, " % (k,v)

        out = out[:-2]  # get rid of trailing commas
        out += "}"

        # one problems: bool

        val = out
    print(s.format(key, val), file=f)


class Input:
    """
    This class handles Trinity input files and parameters
    Parameters are set from input file if they exist, otherwise use defaults
    """

    def __init__(self, fin):
        inputs = self.read_input(fin)

        self.input_dict = inputs

    def read_input(self, fin):
        """
        Read TOML input file
        """

        try:
            with open(fin, mode="rb") as f:
                obj = tomllib.load(f)
        except:
            obj = tomllib.loads(fin)

        return obj

    def write(self, fout="temp.in", overwrite=False):

        # do not overwrite
        if os.path.exists(fout) and not overwrite:
            info(f"  input exists, skipping write{fout}")
            return

        with open(fout, "w") as f:
            for item in self.input_dict.items():

                # if list (eg [[species]]), loop
                if type(item[1]) is list:
                    for nest in item[1]:
                        print(f"\n[[{item[0]}]]", file=f)
                        longest_key = max(nest.keys(), key=len)
                        N_space = len(longest_key)
                        for pair in nest.items():
                            s = "  {:%i}  =  {}" % N_space
                            writeLine(s,pair,f)
                    continue

                # if not list and not dict, write elements (headerless like 'debug')
                if type(item[1]) is not dict:
                    s = "  %s = %s "
                    writeLine(s,item,f)
                    continue

                # if dict, write the header, then write elements
                header, nest = item
                print("\n[%s]" % header, file=f)

                longest_key = max(nest.keys(), key=len)
                N_space = len(longest_key)
                for pair in nest.items():
                    s = "  {:%i}  =  {}" % N_space
                    writeLine(s,pair,f)

        info(f"  wrote input: {fout}")

    def pretty_print(self, entry=""):
        """
        dumps out current input data, in GX input format
        if entry, where entry is one of the GX input headers
        only print the inputs nested under entry
        """

        for item in self.inputs.items():
            # a catch for the debug input, which has no header
            if type(item[1]) is not dict:
                if entry == "":
                    print("  %s = %s " % item)
                    continue

                header, nest = item

                # special case
                if entry != "":
                    header = entry
                    nest = self.inputs[entry]

                print("\n[%s]" % header)

                longest_key = max(nest.keys(), key=len)
                N_space = len(longest_key)
                for pair in nest.items():
                    s = "  {:%i}  =  {}" % N_space
                    print(s.format(*pair))

                # special case
                if entry != "":
                    break


class Logger:
    '''
    This class writes the log (.npy) file that saves outputs from Trinity.
    '''

    def __init__(self, infile, inputs, grid, time, species, geo, physics, norms):

        # read log parameters from input file
        log_parameters = inputs.get('log', {})
        self.f_save = log_parameters.get('f_save', infile[:-3]+'.log.npy')
        self.nc_save = log_parameters.get('f_save', infile[:-3]+'.nc')
        self.bp_save = log_parameters.get('f_save', infile[:-3]+'.bp')
        self.output_netcdf = log_parameters.get('output_netcdf', False)
        self.output_adios2 = log_parameters.get('output_adios2', True)

        # saves TRINITY data as a nested dictionary
        log = defaultdict(list)  # note: log IS the saved object in .npy output

        log['trinity_infile'] = infile

        log['species_types'] = [s.type for s in species.get_species_list()]
        log['species_tags'] = [s.type_tag for s in species.get_species_list()]
        log['bulk_ion_tag'] = species.bulk_ion.type_tag
        log['n_evolve_species'] = species.n_evolve_keys
        log['T_evolve_species'] = species.T_evolve_keys

        self.log = log
        self.geo = geo
        self.grid = grid
        self.norms = norms
        self.physics = physics
        self.species = species

        self.auxiliary_heating = False
        for s in species.get_species_list():
            if s.Sp_aux_height > 0 or s.Sp_aux_int or s.Sp_aux_import:
                self.auxiliary_heating = True

        self.store_system(grid, time, geo, norms)

    def save(self, species, solver, time, final=False):

        # save time markers
        self.log['t'].append(time.time)
        self.log['t_iter_idx'].append(time.iter_idx)
        self.log['t_step_idx'].append(time.step_idx)
        self.log['t_rms'].append(time.rms)

        self.log['species'].append(copy.deepcopy(species))

        W_tot = 0
        Pheat_tot = 0
        Ploss_tot = 0
        Paux_tot = 0
        beta_vol = 0

        for s in species.get_species_list():
            tag = s.type_tag
            self.log['n_'+tag].append(s.n().profile.copy())
            self.log['p_'+tag].append(s.p().profile.copy())
            self.log['T_'+tag].append(s.T().profile.copy())
            self.log['pflux_'+tag].append(s.pflux)
            self.log['qflux_'+tag].append(s.qflux)
            self.log['heat_'+tag].append(s.heat)
            flux_model_labels = list(s.pflux_labeled.keys())
            for label in flux_model_labels:
                self.log['pflux_'+label+'_'+tag].append(s.pflux_labeled[label])
                self.log['qflux_'+label+'_'+tag].append(s.qflux_labeled[label])
                self.log['heat_'+label+'_'+tag].append(s.heat_labeled[label])

            aLn, aLT, aLp = s.get_fluxgrads()
            self.log['aLn_'+tag].append(aLn)
            self.log['aLT_'+tag].append(aLT)
            self.log['aLp_'+tag].append(aLp)

            source_model_labels = list(s.Sn_labeled.keys())
            for label in source_model_labels:
                self.log['Sn_'+label+'_'+tag].append(s.Sn_labeled[label])
                self.log['Sn_'+label+'_int_SI20_'+tag].append(self.geo.volume_integrate(s.Sn_labeled[label])*self.norms.Sn_ref_SI20)
                self.log['Sn_'+label+'_cumint_SI20_'+tag].append(self.geo.volume_integrate(s.Sn_labeled[label], profile=True)*self.norms.Sn_ref_SI20)
            self.log['Sn_tot_'+tag].append(s.Sn)
            self.log['Sn_tot_int_SI20_'+tag].append(self.geo.volume_integrate(s.Sn)*self.norms.Sn_ref_SI20)
            self.log['Sn_tot_cumint_SI20_'+tag].append(self.geo.volume_integrate(s.Sn, profile=True)*self.norms.Sn_ref_SI20)

            source_model_labels = list(s.Sp_labeled.keys())
            for label in source_model_labels:
                self.log['Sp_'+label+'_'+tag].append(s.Sp_labeled[label])
                self.log['Sp_'+label+'_int_MW_'+tag].append(self.geo.volume_integrate(s.Sp_labeled[label])*self.norms.P_ref_MWm3)
                self.log['Sp_'+label+'_cumint_MW_'+tag].append(self.geo.volume_integrate(s.Sp_labeled[label], profile=True)*self.norms.P_ref_MWm3)
            self.log['Sp_tot_'+tag].append(s.Sp)
            self.log['Sp_tot_int_MW_'+tag].append(self.geo.volume_integrate(s.Sp)*self.norms.P_ref_MWm3)
            self.log['Sp_tot_cumint_MW_'+tag].append(self.geo.volume_integrate(s.Sp, profile=True)*self.norms.P_ref_MWm3)

            # power balance
            VprimeQ_s = 1.5*s.Fp_plus(species.ref_species, self.geo)*self.norms.P_ref_MWm3*self.norms.a_meters**3  # MW
            self.log['Q_MW_'+tag].append(VprimeQ_s)
            for label in flux_model_labels:
                self.log['Q_'+label+'_MW_'+tag].append(1.5*s.Fp_plus(species.ref_species, self.geo, label=label)*self.norms.P_ref_MWm3*self.norms.a_meters**3)

            # particle balance
            self.log['Gam_SI20_'+tag].append(s.Fn_plus(species.ref_species, self.geo)*self.norms.Sn_ref_SI20*self.norms.a_meters**3)
            for label in flux_model_labels:
                self.log['Gam_'+label+'_SI20_'+tag].append(s.Fn_plus(species.ref_species, self.geo, label=label)*self.norms.Sn_ref_SI20*self.norms.a_meters**3)

            # various integrated quantities
            # stored energy
            W_s = self.geo.volume_integrate(1.5*s.p())*self.norms.p_ref/1e6  # MJ
            self.log['W_MJ_'+tag].append(W_s)
            W_tot += W_s

            Paux_s = self.log['Sp_aux_int_MW_'+tag][-1]
            Palpha_s = self.log['Sp_alpha_int_MW_'+tag][-1]
            Prad_s = self.log['Sp_rad_int_MW_'+tag][-1]
            Pexch_s = self.log['Sp_heating_int_MW_'+tag][-1]
            Pcoll_s = self.log['Sp_coll_int_MW_'+tag][-1]
            Pheat_s = Paux_s + Palpha_s + Pcoll_s + Pexch_s + Prad_s
            self.log['Pheat_MW_'+tag].append(Pheat_s)
            Ploss_s = VprimeQ_s[-1]
            self.log['Ploss_MW_'+tag].append(Ploss_s)
            Pheat_tot += Pheat_s
            Ploss_tot += Ploss_s
            Paux_tot += Paux_s

            beta_vol += self.geo.volume_average(s.beta(self.geo.Btor))

            n_volavg_s = self.geo.volume_average(s.n())  # 10^20/m^3
            self.log['n_volavg_'+tag].append(n_volavg_s)
            n_lineavg_s = np.average(s.n().profile)  # 10^20/m^3
            if s.type == "electron":
                ne_volavg = n_volavg_s
                ne_avg = n_lineavg_s
                Prad = Prad_s
            self.log['n_lineavg_'+tag].append(n_lineavg_s)
            npeak_s = s.n()[0]/n_volavg_s
            self.log['npeak_'+tag].append(npeak_s)

            T_volavg_s = self.geo.volume_average(s.T())  # keV
            self.log['T_volavg_'+tag].append(T_volavg_s)
            T_lineavg_s = np.average(s.T().profile)  # keV
            self.log['T_lineavg_'+tag].append(T_lineavg_s)
            Tpeak_s = s.T()[0]/T_volavg_s
            self.log['Tpeak_'+tag].append(Tpeak_s)

        self.log['Palpha_MWm3'].append(self.physics.compute_total_alpha_heating(species))
        self.log['Palpha_cumint_MW'].append(self.geo.volume_integrate(self.physics.compute_total_alpha_heating(species), profile=True))
        Palpha_tot = self.geo.volume_integrate(self.physics.compute_total_alpha_heating(species))
        Pfus = 5.03*Palpha_tot
        self.log['Palpha_int_MW'].append(Palpha_tot)
        self.log['Pfus_MW'].append(Pfus)

        # fusion metrics
        self.log['Wtot_MJ'].append(W_tot)
        self.log['Pheat_MW'].append(Pheat_tot)
        self.log['Ploss_MW'].append(Ploss_tot)
        self.log['Paux_MW'].append(Paux_tot)
        self.log['beta_vol'].append(beta_vol)
        Qfus = Pfus / Paux_tot if Paux_tot != 0 else 0
        self.log['Qfus'].append(Qfus)

        # Compute energy confinement time and related metrics only
        # when alpha or external heating sources are present
        if self.physics.alpha_heating or self.auxiliary_heating:
            tauE = self.physics.compute_tauE(species)
            self.log['tauE'].append(tauE)
            if final:
                info("\n Final confinement time and related metrics:", color='yellow')
                info(f"   Wtot = {W_tot:.3f} MJ", color='yellow')
                info(f"   Pheat = {Pheat_tot:.3f} MW", color='yellow')
                info(f"   Ploss = {Ploss_tot:.3f} MW", color='yellow')
                info(f"   Palpha = {self.log['Palpha_int_MW'][-1]:.3f} MW", color='yellow')
                info(f"   Q_fus = {self.log['Palpha_int_MW'][-1]*5.03/self.log['Paux_MW'][-1]:.3f}", color='yellow')
                info(f"   tauE = {tauE:.3e} s", color='yellow')
                info(f"   beta_vol = {beta_vol*100:.2f} %", color='yellow')

            if self.geo.geo_option == "vmec" or self.geo.geo_option == "desc" or self.geo.geo_option == 'basic':
                R = self.geo.AspectRatio*self.geo.a_minor
                a = self.geo.a_minor
                P = Pheat_tot
                n19 = ne_avg*10
                B = np.abs(self.geo.B0)
                iota = np.abs(self.geo.iota23)

                tauE_ISS04 = self.physics.compute_ISS04(species)
                self.log['tauE_ISS04'].append(tauE_ISS04)
                self.log['f_renorm'].append(tauE/tauE_ISS04)

                n_sudo = np.sqrt(P*B/(16*R*a**2))  # 10^20/m^3
                self.log['n_sudo'].append(n_sudo)
                f_sudo_lineavg = ne_avg/n_sudo
                self.log['f_sudo_line'].append(f_sudo_lineavg)
                f_sudo_vol = ne_volavg/n_sudo
                self.log['f_sudo_vol'].append(f_sudo_vol)
                f_sudo_edge = species['electron'].n()[-1]/n_sudo
                self.log['f_sudo_edge'].append(f_sudo_edge)
                if final:
                    info(f"tauE_ISS04 = {tauE_ISS04:.3e} s, f_renorm = {tauE/tauE_ISS04:.3f}, n_sudo = {n_sudo:.3f} 10^20/m^3, f_sudo_line = {f_sudo_lineavg:.3f}, f_sudo_vol = {f_sudo_vol:.3f}, f_sudo_edge = {f_sudo_edge:.3f}")

        # compute alpha pressure
        p_alpha = self.physics.compute_alpha_pressure(species)
        beta_alpha = 4.03e-11*p_alpha*1e17/(self.geo.Btor*1e4)**2
        self.log['p_alpha'].append(p_alpha)
        self.log['beta_alpha'].append(beta_alpha)

        if self.physics.Zeff_prof:
            self.log['Zeff'].append(self.physics.Zeff_prof)

        try:
            self.log['solver_jacobian'].append(solver.jacob)
            self.log['solver_rhs'].append(solver.rhs)
        except:
            pass

        self.log['flux_model_labels'] = flux_model_labels

    def store_system(self, grid, time, geo, norms):
        ''' Saves run settings for reproducing runs '''

        # time step info lives here
        time_settings = {}
        time_settings['alpha'] = time.alpha
        time_settings['dtau'] = time.dtau
        time_settings['N_steps'] = time.N_steps
        time_settings['max_newton_iter'] = time.max_newton_iter
        self.log['time'] = time_settings

        # grid info lives here
        grid_settings = {}
        grid_settings['N_radial'] = grid.N_radial
        grid_settings['rho_edge'] = grid.rho_edge
        grid_settings['rho_inner'] = grid.rho_inner
        grid_settings['rho'] = grid.rho
        grid_settings['midpoints'] = grid.midpoints
        grid_settings['drho'] = grid.drho
        grid_settings['flux_label'] = grid.flux_label
        self.log['grid'] = grid_settings
        self.log['grid_obj'] = grid

        geo_settings = {}
        geo_settings['area'] = geo.area.profile
        geo_settings['grho'] = geo.grho.profile
        geo_settings['area_grid'] = geo.area.toGridProfile().profile
        geo_settings['grho_grid'] = geo.grho_grid.profile
        geo_settings['Btor'] = geo.Btor
        geo_settings['B0'] = geo.B0
        geo_settings['a_minor'] = geo.a_minor
        self.log['geo'] = geo_settings
        self.log['geo_obj'] = geo

        norm_settings = {}
        norm_settings['p_ref'] = norms.p_ref
        norm_settings['Btor'] = norms.Btor
        norm_settings['a_minor'] = norms.a_meters
        norm_settings['vT_ref'] = norms.vT_ref
        norm_settings['rho_star'] = norms.rho_star
        norm_settings['t_ref'] = norms.t_ref
        norm_settings['P_ref_MWm3'] = norms.P_ref_MWm3
        norm_settings['Sn_ref_SI20'] = norms.Sn_ref_SI20
        self.log['norms'] = norm_settings

    def export(self, f_save=None):
        if f_save is None:
            f_save = self.f_save
        np.save(f_save, self.log)

    def write_netCDF4(self, infile, f_save=None):
        ''' This function writes the output to a netCDF4 file '''

        # Exit immediately if NetCDF4 output is not needed
        if not self.output_netcdf:
            return

        # Set the output filename
        if f_save is None:
            f_save = self.nc_save

        # Open the netCDF4 file
        ncf = Dataset(f_save, mode='w', format='NETCDF4')

        # Attributes
        ncf.title = "T3D netCDF4 output file"
        ncf.history = "Created " + date.today().strftime("%Y/%m/%d") + " by " + getpass.getuser()

        # Copy the input file to the output NetCDF4 file
        ncf.infile = infile
        with open(infile) as f:
            inputs = f.read()
            ncf.inputs = inputs  # This dumps the entire T3D input file

        # Global dimensions
        _ = ncf.createDimension('n_steps', len(self.log['t']))
        _ = ncf.createDimension('n_radial', self.log['grid']['N_radial'])
        _ = ncf.createDimension('n_radial_m_1', self.log['grid']['N_radial'] - 1)

        # Add scalars and norms
        norms = ncf.createGroup('norms')

        t_ref = norms.createVariable('t_ref', 'f8')
        t_ref.units = "[s]"
        t_ref.description = "Reference time given by the minor radius divided by the reference thermal velocity / rhostar**2"
        norms['t_ref'][0] = self.log['norms']['t_ref']

        rho_star = norms.createVariable('rho_star', 'f8')
        rho_star.units = "-"
        rho_star.description = "rho*, the ratio of the gyroradius to the minor radius"
        norms['rho_star'][0] = self.log['norms']['rho_star']

        P_ref_MWm3 = norms.createVariable('P_ref_MWm3', 'f8')
        P_ref_MWm3.units = "[MW m^-3]"
        P_ref_MWm3.description = "Reference pressure source"
        norms['P_ref_MWm3'][0] = self.log['norms']['P_ref_MWm3']

        Sn_ref_SI20 = norms.createVariable('Sn_ref_SI20', 'f8')
        Sn_ref_SI20.units = "[10^20 m^-3 s]"
        Sn_ref_SI20.description = "Reference particle source"
        norms['Sn_ref_SI20'][0] = self.log['norms']['Sn_ref_SI20']

        flux_label = norms.createVariable('flux_label', str)
        flux_label.units = "-"
        flux_label.description = "String to denote the flux label: rminor = r/a or torflux = sqrt(toroidal_flux/toroidal_flux_LCFS)"
        norms['flux_label'][0] = self.log['grid']['flux_label']

        a_minor = norms.createVariable('a_minor', 'f8')
        a_minor.units = "[m]"
        a_minor.description = "Minor radius"
        norms['a_minor'][0] = self.log['geo']['a_minor']

        R_major = norms.createVariable('R_major', 'f8')
        R_major.units = "[m]"
        R_major.description = "Major radius"
        norms['R_major'][0] = self.geo.AspectRatio*self.geo.a_minor

        # Create time group
        time = ncf.createGroup('time')
        times = time.createVariable('t', 'f8', ('n_steps',))
        times.units = "[t_ref]"
        times.description = "Normalized time array, denormalize with t_ref"
        time['t'][:] = self.log['t']

        # Create grid group
        grid = ncf.createGroup('grid')

        rho = grid.createVariable('rho', 'f8', ('n_radial',))
        rho.units = "-"
        rho.description = "Flux surface grid"
        grid['rho'][:] = self.log['grid']['rho']

        midpoints = grid.createVariable('midpoints', 'f8', ('n_radial_m_1',))
        midpoints.units = "-"
        midpoints.description = "Midpoints of the flux surface grid, where flux models are evaluated"
        grid['midpoints'][:] = self.log['grid']['midpoints']

        # Create geo group
        geo = ncf.createGroup('geo')

        B0 = geo.createVariable('B0', 'f8')
        B0.units = "[T]"
        B0.description = "Magnetic field"
        geo['B0'][0] = self.log['geo']['B0']

        area = geo.createVariable('area', 'f8', ('n_radial_m_1',))
        area.units = "-"
        area.description = "Surface area of flux surfaces on the midpoints grid, normalized by the minor radius squared"
        geo['area'][:] = self.log['geo']['area']

        grho = geo.createVariable('grho', 'f8', ('n_radial_m_1',))
        grho.units = "-"
        grho.description = "<|grad rho|> on the midpoints grid, normalized by the minor radius"
        geo['grho'][:] = self.log['geo']['grho']

        area_grid = geo.createVariable('area_grid', 'f8', ('n_radial',))
        area_grid.units = "-"
        area_grid.description = "Surface area of flux surfaces on the rho grid, normalized by the minor radius squared"
        geo['area_grid'][:] = self.log['geo']['area_grid']

        grho_grid = geo.createVariable('grho_grid', 'f8', ('n_radial',))
        grho_grid.units = "-"
        grho_grid.description = "<|grad rho|> on the rho grid, normalized by the minor radius"
        geo['grho_grid'][:] = self.log['geo']['grho_grid']

        # Create species and profiles group
        species = ncf.createGroup('species')
        _ = species.createDimension('n_species', len(self.log['species_types']))

        _ = species.createVariable('types', str, 'n_species')
        species_tags = species.createVariable('tags', str, 'n_species')
        for n in range(len(self.log['species_types'])):
            species['types'][n] = self.log['species_types'][n]
            species['tags'][n] = self.log['species_tags'][n]
        species_tags.bulk_ion_tag = self.log['bulk_ion_tag']

        _ = species.createDimension('n_flux_models', len(self.log['flux_model_labels']))
        flux_model_labels = species.createVariable('flux_model_labels', str, 'n_flux_models')
        for n in range(len(self.log['flux_model_labels'])):
            species['flux_model_labels'][n] = self.log['flux_model_labels'][n]

        _ = species.createDimension('num_n_evolve_species', len(self.log['n_evolve_species']))
        v = species.createVariable('n_evolve_species', str, 'num_n_evolve_species')
        v.description = "List of species for which the density is being evolved"
        for n in range(len(self.log['n_evolve_species'])):
            species['n_evolve_species'][n] = self.log['n_evolve_species'][n]

        _ = species.createDimension('num_T_evolve_species', len(self.log['T_evolve_species']))
        v = species.createVariable('T_evolve_species', str, 'num_T_evolve_species')
        v.description = "List of species for which the temperature is being evolved"
        for n in range(len(self.log['T_evolve_species'])):
            species['T_evolve_species'][n] = self.log['T_evolve_species'][n]

        # Density, pressure, and temperature profiles
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            varname = 'n_' + tag
            n_profile = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            n_profile.units = "[10^20 m^-3]"
            n_profile.description = f"{s} density profile on rho grid"
            species[varname][:, :] = self.log[varname]

            varname = 'p_' + tag
            p_profile = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            p_profile.units = "[10^20 m^-3 keV]"
            p_profile.description = f"{s} pressure profile on rho grid"
            species[varname][:, :] = self.log[varname]

            varname = 'T_' + tag
            T_profile = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            T_profile.units = "[keV]"
            T_profile.description = f"{s} temperature profile on rho grid"
            species[varname][:, :] = self.log[varname]

        # Flux profiles
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            varname = 'pflux_' + tag
            pflux = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
            pflux.units = "[GB]"
            pflux.description = f"GyroBohm-normalized total {s} particle flux profile on midpoints grid"
            species[varname][:, :] = self.log[varname]

            varname = 'qflux_' + tag
            qflux = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
            qflux.units = "[GB]"
            qflux.description = f"GyroBohm-normalized total {s} heat flux profile on midpoints grid"
            species[varname][:, :] = self.log[varname]

            varname = 'heat_' + tag
            heat = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
            heat.units = "[GB]"
            heat.description = f"GyroBohm-normalized total {s} collisional heating profile on midpoints grid"
            species[varname][:, :] = self.log[varname]

            for label in self.log['flux_model_labels']:
                varname = 'pflux_' + label + '_' + tag
                pflux = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
                pflux.units = "[GB]"
                pflux.description = f"GyroBohm-normalized {s} particle flux profile from {label} on midpoints grid"
                species[varname][:, :] = self.log[varname]

                varname = 'qflux_' + label + '_' + tag
                qflux = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
                qflux.units = "[GB]"
                qflux.description = f"GyroBohm-normalized {s} heat flux profile from {label} on midpoints grid"
                species[varname][:, :] = self.log[varname]

                varname = 'heat_' + label + '_' + tag
                heat = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
                heat.units = "[GB]"
                heat.description = f"GyroBohm-normalized {s} collisional heating profile from {label} on midpoints grid"
                species[varname][:, :] = self.log[varname]

        # Density, pressure, temperature gradient profiles
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            varname = 'aLn_' + tag
            aLn = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
            aLn.units = "-"
            aLn.description = f"Normalized {s} density gradient, a/L_n = 1/n dn/drho, on midpoints grid"
            species[varname][:, :] = self.log[varname]

            varname = 'aLp_' + tag
            aLp = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
            aLp.units = "-"
            aLp.description = f"Normalized {s} pressure gradient, a/L_p = 1/p dp/drho, on midpoints grid"
            species[varname][:, :] = self.log[varname]

            varname = 'aLT_' + tag
            aLT = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
            aLT.units = "-"
            aLp.description = f"Normalized {s} temperature gradient, a/L_T = 1/T dT/drho, on midpoints grid"
            species[varname][:, :] = self.log[varname]

        # Source terms
        Sn_aux_i = 0
        Sp_aux_i = 0
        Sp_alpha_i = 0
        Sp_rad_i = 0
        Sp_coll_i = 0
        Sp_heating_i = 0
        Sp_tot_i = 0
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            varname = 'Sn_aux_' + tag
            Sn_aux = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sn_aux.units = "[Sn_ref_SI20]"
            Sn_aux.description = f"Normalized {s} auxiliary particle source on rho grid"
            species[varname][:, :] = self.log[varname]
            if s != "electron":
                Sn_aux_i += np.array(self.log[varname])

            varname = 'Sp_aux_' + tag
            Sp_aux = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_aux.units = "[P_ref_MWm3]"
            Sp_aux.description = f"Normalized {s} auxiliary power source on rho grid"
            species[varname][:, :] = self.log[varname]
            if s != "electron":
                Sp_aux_i += np.array(self.log[varname])

            varname = 'Sp_alpha_' + tag
            Sp_alpha = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_alpha.units = "[P_ref_MWm3]"
            Sp_alpha.description = f"Normalized {s} power source from fusion alphas on rho grid"
            species[varname][:, :] = self.log[varname]
            if s != "electron":
                Sp_alpha_i += np.array(self.log[varname])

            varname = 'Sp_rad_' + tag
            Sp_rad = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_rad.units = "[P_ref_MWm3]"
            Sp_rad.description = f"Normalized {s} power source (sink) from radiation on rho grid"
            species[varname][:, :] = self.log[varname]
            if s != "electron":
                Sp_rad_i += np.array(self.log[varname])

            varname = 'Sp_heating_' + tag
            Sp_heating = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_heating.units = "[P_ref_MWm3]"
            Sp_heating.description = f"Normalized {s} power source from turbulent heating on rho grid"
            species[varname][:, :] = self.log[varname]
            if s != "electron":
                Sp_heating_i += np.array(self.log[varname])

            varname = 'Sp_coll_' + tag
            Sp_coll = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_coll.units = "[P_ref_MWm3]"
            Sp_coll.description = f"Normalized {s} power source (sink) from collisional equilibration on rho grid"
            species[varname][:, :] = self.log[varname]
            if s != "electron":
                Sp_coll_i += np.array(self.log[varname])

            varname = 'Sp_tot_' + tag
            Sp_tot = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_tot.units = "[P_ref_MWm3]"
            Sp_tot.description = f"Normalized total {s} power source on rho grid"
            species[varname][:, :] = self.log[varname]
            if s != "electron":
                Sp_tot_i += np.array(self.log[varname])

        # write ion-summed source profiles
        tag = 'i'
        varname = 'Sn_aux_' + tag
        Sn_aux = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
        Sn_aux.units = "[Sn_ref_SI20]"
        Sn_aux.description = "Normalized ion-summed auxiliary particle source on rho grid"
        species[varname][:, :] = Sn_aux_i

        varname = 'Sp_aux_' + tag
        Sp_aux = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
        Sp_aux.units = "[P_ref_MWm3]"
        Sp_aux.description = "Normalized ion-summed auxiliary power source on rho grid"
        species[varname][:, :] = Sp_aux_i

        varname = 'Sp_tot_' + tag
        Sp_tot = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
        Sp_tot.units = "[P_ref_MWm3]"
        Sp_tot.description = "Normalized ion-summed total power source on rho grid"
        species[varname][:, :] = Sp_tot_i

        varname = 'Sp_coll_' + tag
        Sp_coll = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
        Sp_coll.units = "[P_ref_MWm3]"
        Sp_coll.description = "Normalized ion-summed power source (sink) from collisional equilibration on rho grid"
        species[varname][:, :] = Sp_coll_i

        varname = 'Sp_heating_' + tag
        Sp_heating = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
        Sp_heating.units = "[P_ref_MWm3]"
        Sp_heating.description = "Normalized ion-summed power source from turbulent heating on rho grid"
        species[varname][:, :] = Sp_heating_i

        varname = 'Sp_rad_' + tag
        Sp_rad = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
        Sp_rad.units = "[P_ref_MWm3]"
        Sp_rad.description = "Normalized ion-summed power source (sink) from radiation on rho grid"
        species[varname][:, :] = Sp_rad_i

        varname = 'Sp_alpha_' + tag
        Sp_alpha = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
        Sp_alpha.units = "[P_ref_MWm3]"
        Sp_alpha.description = "Normalized ion-summed power source from fusion alphas on rho grid"
        species[varname][:, :] = Sp_alpha_i

        # Source terms volume integrated along the profile
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            varname = 'Sn_tot_int_SI20_' + tag
            Sn_tot_int_SI20 = species.createVariable(varname, 'f8', ('n_steps', ))
            Sn_tot_int_SI20.units = "[10^20]"
            Sn_tot_int_SI20.description = f"Integrated {s} particle source"
            species[varname][:] = self.log[varname]

            varname = 'Sp_aux_int_MW_' + tag
            Sp_aux_int_MW = species.createVariable(varname, 'f8', ('n_steps', ))
            Sp_aux_int_MW.units = "[MW]"
            Sp_aux_int_MW.description = f"Integrated {s} auxillary power"
            species[varname][:] = self.log[varname]

            varname = 'Sp_alpha_int_MW_' + tag
            Sp_alpha_int_MW = species.createVariable(varname, 'f8', ('n_steps', ))
            Sp_alpha_int_MW.units = "[MW]"
            Sp_alpha_int_MW.description = f"Integrated {s} alpha heating"
            species[varname][:] = self.log[varname]

            varname = 'Sp_rad_int_MW_' + tag
            Sp_rad_int_MW = species.createVariable(varname, 'f8', ('n_steps', ))
            Sp_rad_int_MW.units = "[MW]"
            Sp_rad_int_MW.description = f"Integrated {s} radiation"
            species[varname][:] = self.log[varname]

            varname = 'Sp_heating_int_MW_' + tag
            Sp_rad_int_MW = species.createVariable(varname, 'f8', ('n_steps', ))
            Sp_rad_int_MW.units = "[MW]"
            Sp_rad_int_MW.description = f"Integrated {s} turbulent heating"
            species[varname][:] = self.log[varname]

            varname = 'Sp_coll_int_MW_' + tag
            Sp_rad_int_MW = species.createVariable(varname, 'f8', ('n_steps', ))
            Sp_rad_int_MW.units = "[MW]"
            Sp_rad_int_MW.description = f"Integrated {s} collisional equilibration"
            species[varname][:] = self.log[varname]

        # Source terms cumulatively volume integrated over the profile
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            varname = 'Sn_tot_cumint_SI20_' + tag
            Sn_tot_cumint_SI20 = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sn_tot_cumint_SI20.units = "[10^20]"
            Sn_tot_cumint_SI20.description = f"Integrated {s} particle source interior to each flux surface on rho grid"
            species[varname][:, :] = self.log[varname]

            varname = 'Sp_aux_cumint_MW_' + tag
            Sp_aux_cumint_MW = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_aux_cumint_MW.units = "[MW]"
            Sp_aux_cumint_MW.description = f"Integrated {s} auxillary power interior to each flux surface on rho grid"
            species[varname][:] = self.log[varname]

            varname = 'Sp_alpha_cumint_MW_' + tag
            Sp_alpha_cumint_MW = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_alpha_cumint_MW.units = "[MW]"
            Sp_alpha_cumint_MW.description = f"Integrated {s} alpha heating power interior to each flux surface on rho grid"
            species[varname][:] = self.log[varname]

            varname = 'Sp_rad_cumint_MW_' + tag
            Sp_rad_cumint_MW = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_rad_cumint_MW.units = "[MW]"
            Sp_rad_cumint_MW.description = f"Integrated {s} radiation power interior to each flux surface on rho grid"
            species[varname][:] = self.log[varname]

            varname = 'Sp_heating_cumint_MW_' + tag
            Sp_rad_cumint_MW = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_rad_cumint_MW.units = "[MW]"
            Sp_rad_cumint_MW.description = f"Integrated {s} turbulent heating power interior to each flux surface on rho grid"
            species[varname][:] = self.log[varname]

            varname = 'Sp_coll_cumint_MW_' + tag
            Sp_rad_cumint_MW = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_rad_cumint_MW.units = "[MW]"
            Sp_rad_cumint_MW.description = f"Integrated {s} collisional equilibration power interior to each flux surface on rho grid"
            species[varname][:] = self.log[varname]

            varname = 'Sp_tot_cumint_MW_' + tag
            Sp_tot_cumint_MW = species.createVariable(varname, 'f8', ('n_steps', 'n_radial'))
            Sp_tot_cumint_MW.units = "[MW]"
            Sp_tot_cumint_MW.description = f"Integrated {s} total power source interior to each flux surface on rho grid"
            species[varname][:, :] = self.log[varname]

        # Heat fluxes in physical units
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            varname = 'Q_MW_' + tag
            Q_MW = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
            Q_MW.units = "[MW]"
            Q_MW.description = f"{s} total heat flux in MW on midpoints grid"
            species[varname][:, :] = self.log[varname]

            for label in self.log['flux_model_labels']:
                varname = 'Q_' + label + '_MW_' + tag
                Q_MW = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
                Q_MW.units = "[MW]"
                Q_MW.description = f"{s} total heat flux from {label} in MW on midpoints grid"
                species[varname][:, :] = self.log[varname]

        # Particle fluxes in physical units
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            varname = 'Gam_SI20_' + tag
            Gam_SI20 = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
            Gam_SI20.units = "[10^20 / s]"
            Gam_SI20.description = f"{s} particle flux in 10^20/s"
            species[varname][:, :] = self.log[varname]

            for label in self.log['flux_model_labels']:
                varname = 'Gam_' + label + '_SI20_' + tag
                Gam_SI20 = species.createVariable(varname, 'f8', ('n_steps', 'n_radial_m_1'))
                Gam_SI20.units = "[10^20 / s]"
                Gam_SI20.description = f"{s} particle flux from {label} in 10^20/s"
                species[varname][:, :] = self.log[varname]

        # Alpha heating
        Palpha_MWm3 = species.createVariable('Palpha_MWm3', 'f8', ('n_steps', 'n_radial'))
        Palpha_MWm3.units = "[MW m^-3]"
        Palpha_MWm3.description = "Total alpha heating power on rho grid"
        species['Palpha_MWm3'][:] = self.log['Palpha_MWm3']

        Palpha_int_MW = species.createVariable('Palpha_int_MW', 'f8', ('n_steps', ))
        Palpha_int_MW.units = "[MW]"
        Palpha_int_MW.description = "Volume-integrated total alpha heating power"
        species['Palpha_int_MW'][:] = self.log['Palpha_int_MW']

        p_alpha = species.createVariable('p_alpha', 'f8', ('n_steps', 'n_radial'))
        p_alpha.units = "[10^20 m^-3 keV]"
        p_alpha.description = "Alpha particle pressure profile"
        species['p_alpha'][:, :] = self.log['p_alpha']

        beta_alpha = species.createVariable('beta_alpha', 'f8', ('n_steps', 'n_radial'))
        beta_alpha.units = "-"
        beta_alpha.description = "Alpha particle beta profile"
        species['beta_alpha'][:, :] = self.log['beta_alpha']

        if self.physics.Zeff_prof:
            Zeff_prof = species.createVariable('Zeff', 'f8', ('n_steps', 'n_radial'))
            Zeff_prof.units = "-"
            Zeff_prof.description = "Effective charge, Z_effective"
            species['Zeff'][:, :] = self.log['Zeff']

        # Create zerod group
        zerod = ncf.createGroup('zeroD')

        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            varname = 'n_volavg_' + tag
            n_volavg = zerod.createVariable(varname, 'f8', ('n_steps'))
            n_volavg.units = "[10^20 m^-3]"
            n_volavg.description = f"Volume-averaged {s} density"
            zerod[varname][:] = self.log[varname]

            varname = 'n_lineavg_' + tag
            n_lineavg = zerod.createVariable(varname, 'f8', ('n_steps'))
            n_lineavg.units = "[10^20 m^-3]"
            n_lineavg.description = f"Line-averaged {s} density"
            zerod[varname][:] = self.log[varname]

            varname = 'npeak_' + tag
            n_lineavg = zerod.createVariable(varname, 'f8', ('n_steps'))
            n_lineavg.units = "-"
            n_lineavg.description = f"{s} density peaking factor, f = n_0/<n>_vol"
            zerod[varname][:] = self.log[varname]

            varname = 'T_volavg_' + tag
            T_volavg = zerod.createVariable(varname, 'f8', ('n_steps'))
            T_volavg.units = "[keV]"
            T_volavg.description = f"Volume-averaged {s} temperature"
            zerod[varname][:] = self.log[varname]

            varname = 'T_lineavg_' + tag
            T_lineavg = zerod.createVariable(varname, 'f8', ('n_steps'))
            T_lineavg.units = "[keV]"
            T_lineavg.description = f"Line-averaged {s} temperature"
            zerod[varname][:] = self.log[varname]

            varname = 'Tpeak_' + tag
            T_lineavg = zerod.createVariable(varname, 'f8', ('n_steps'))
            T_lineavg.units = "-"
            T_lineavg.description = f"{s} temperature peaking factor, f = T_0/<T>_vol"
            zerod[varname][:] = self.log[varname]

            varname = 'W_' + tag
            W = zerod.createVariable(varname, 'f8', ('n_steps'))
            W.units = "[MJ]"
            W.description = f"Volume integral of {s} pressure"
            zerod[varname][:] = self.log['W_MJ_'+tag]

            varname = 'Pheat_' + tag
            Pheat = zerod.createVariable(varname, 'f8', ('n_steps'))
            Pheat.units = "[MW]"
            Pheat.description = f"Volume integral of {s} auxiliary, alpha, collisional, and turbulent exchange heating"
            zerod[varname][:] = self.log['Pheat_MW_'+tag]

            varname = 'Ploss_' + tag
            Ploss = zerod.createVariable(varname, 'f8', ('n_steps'))
            Ploss.units = "[MW]"
            Ploss.description = f"Volume integral of {s} losses due to transport and radiation"
            zerod[varname][:] = self.log['Ploss_MW_'+tag]

        Wtot = zerod.createVariable('Wtot', 'f8', ('n_steps'))
        Wtot.units = "[MJ]"
        Wtot.description = "Volume integral of species-summed pressure"
        zerod['Wtot'][:] = self.log['Wtot_MJ']

        beta_vol = zerod.createVariable('beta_vol', 'f8', ('n_steps'))
        beta_vol.units = "-"
        beta_vol.description = "Volume averaged total beta"
        zerod['beta_vol'][:] = self.log['beta_vol']

        Pheat = zerod.createVariable('Pheat', 'f8', ('n_steps'))
        Pheat.units = "[MW]"
        Pheat.description = "Volume integral of species-summed auxiliary and alpha heating"
        zerod['Pheat'][:] = self.log['Pheat_MW']

        Ploss = zerod.createVariable('Ploss', 'f8', ('n_steps'))
        Ploss.units = "[MW]"
        Ploss.description = "Volume integral of species-summed losses due to transport and radiation"
        zerod['Ploss'][:] = self.log['Ploss_MW']

        Paux = zerod.createVariable('Paux', 'f8', ('n_steps'))
        Paux.units = "[MW]"
        Paux.description = "Volume integral of species-summed auxes due to transport and radiation"
        zerod['Paux'][:] = self.log['Paux_MW']

        Palpha = zerod.createVariable('Palpha', 'f8', ('n_steps'))
        Palpha.units = "[MW]"
        Palpha.description = "Volume-integrated species-summed alpha heating power"
        zerod['Palpha'][:] = self.log['Palpha_int_MW']

        Pfus = zerod.createVariable('Pfus', 'f8', ('n_steps'))
        Pfus.units = "[MW]"
        Pfus.description = "Volume-integrated species-summed fusion power"
        zerod['Pfus'][:] = self.log['Pfus_MW']

        Qfus = zerod.createVariable('Qfus', 'f8', ('n_steps'))
        Qfus.units = "-"
        Qfus.description = "Fusion gain"
        zerod['Qfus'][:] = self.log['Qfus']

        # Store energy confinement time and related metrics only
        # when alpha or external heating sources are present
        if self.physics.alpha_heating or self.auxiliary_heating:
            tauE = zerod.createVariable('tauE', 'f8', ('n_steps'))
            tauE.units = "[s]"
            tauE.description = "Energy confinement time (tauE = Wtot/Pheat), assuming steady-state"
            zerod['tauE'][:] = self.log['tauE']

            if self.geo.geo_option == "vmec" or self.geo.geo_option == "desc" or self.geo.geo_option == "basic":
                tauE_ISS04 = zerod.createVariable('tauE_ISS04', 'f8', ('n_steps'))
                tauE_ISS04.units = "[s]"
                tauE_ISS04.description = "Prediction of energy confinement time according to tauE_ISS04"
                zerod['tauE_ISS04'][:] = self.log['tauE_ISS04']

                f_renorm = zerod.createVariable('f_renorm', 'f8', ('n_steps'))
                f_renorm.units = "-"
                f_renorm.description = "f = tauE / tauE_ISS04, assuming steady-state"
                zerod['f_renorm'][:] = self.log['f_renorm']

                n_sudo = zerod.createVariable('n_sudo', 'f8', ('n_steps'))
                n_sudo.units = "[10^20 m^-3]"
                n_sudo.description = "Sudo density limit"
                zerod['n_sudo'][:] = self.log['n_sudo']

                f_sudo_line = zerod.createVariable('f_sudo_line', 'f8', ('n_steps'))
                f_sudo_line.units = "-"
                f_sudo_line.description = "f = <n_e>_line/n_sudo"
                zerod['f_sudo_line'][:] = self.log['f_sudo_line']

                f_sudo_vol = zerod.createVariable('f_sudo_vol', 'f8', ('n_steps'))
                f_sudo_vol.units = "-"
                f_sudo_vol.description = "f = <n_e>_vol/n_sudo"
                zerod['f_sudo_vol'][:] = self.log['f_sudo_vol']

                f_sudo_edge = zerod.createVariable('f_sudo_edge', 'f8', ('n_steps'))
                f_sudo_edge.units = "-"
                f_sudo_edge.description = "f = n_edge/n_sudo"
                zerod['f_sudo_edge'][:] = self.log['f_sudo_edge']

        # Close the NetCDF file
        ncf.close()

    def write_adios2(self, infile, f_save=None):
        """This function writes T3D output to ADIOS2 bp format

        Arguments:
        infile -- t3d input file name which gets dumped to the adios2 output
        f_save -- name of the adios2 output
        """

        def fix_labels(s: str) -> str:
            """Function to strip characters from labels that adios2 does not like"""
            a = s.replace(" ", "_")
            chars_to_remove = ["(", ")"]
            for c in chars_to_remove:
                a = a.replace(c, "")
            return a

        # Exit immediately if ADIOS2 output is not needed
        if not self.output_adios2:
            return

        # Set the output filename
        if f_save is None:
            f_save = self.bp_save

        # ADIOS output stream
        with adios2.Stream(f_save, "w") as fh:

            # Output the metadata
            fh.write_attribute('title', 'T3D ADIOS2 output')
            fh.write_attribute('history', f'Created {date.today().strftime("%Y/%m/%d")} by {getpass.getuser()}')

            # Output the input file location and contents
            fh.write_attribute('infile', infile)
            with open(infile) as f:
                inputs = f.read()
                fh.write_attribute('inputs', inputs)  # This dumps the entire T3D input file

            # Output dimensions
            fh.write_attribute('n_steps', len(self.log['t']))
            fh.write_attribute('n_radial', self.log['grid']['N_radial'])
            fh.write_attribute('n_radial_m_1', self.log['grid']['N_radial'] - 1)

            # Add scalars and norms
            fh.write_attribute('norms/t_ref', self.log['norms']['t_ref'])
            fh.write_attribute('norms/t_ref/units', 's')
            fh.write_attribute('norms/t_ref/description', 'Reference time given by the minor radius divided by the reference thermal velocity / rhostar**2')

            fh.write_attribute('norms/rho_star', self.log['norms']['rho_star'])
            fh.write_attribute('norms/rho_star/units', '-')
            fh.write_attribute('norms/rho_star/description', 'rho*, the ratio of the gyroradius to the minor radius')

            fh.write_attribute('norms/P_ref_MWm3', self.log['norms']['P_ref_MWm3'])
            fh.write_attribute('norms/P_ref_MWm3/units', '-')
            fh.write_attribute('norms/P_ref_MWm3/description', 'Reference pressure source')

            fh.write_attribute('norms/Sn_ref_SI20', self.log['norms']['Sn_ref_SI20'])
            fh.write_attribute('norms/Sn_ref_SI20/units', '10^20 m^-3 s')
            fh.write_attribute('norms/Sn_ref_SI20/description', 'Reference particle source')

            fh.write_attribute('norms/flux_label', self.log['grid']['flux_label'])
            fh.write_attribute('norms/flux_label/units', '-')
            fh.write_attribute('norms/flux_label/description', 'String to denote the flux label: rminor = r/a or torflux = sqrt(toroidal_flux/toroidal_flux_LCFS)')

            fh.write_attribute('norms/a_minor', self.log['geo']['a_minor'])
            fh.write_attribute('norms/a_minor/units', 'm')
            fh.write_attribute('norms/a_minor/description', 'Minor radius')

            fh.write_attribute('norms/R_major', self.geo.AspectRatio*self.geo.a_minor)
            fh.write_attribute('norms/R_major/units', 'm')
            fh.write_attribute('norms/R_major/description', 'Major radius')

            # Time
            values = np.array(self.log['t'])
            fh.write('time', values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute('time/units', 't_ref')
            fh.write_attribute('time/description', 'Normalized time array, denormalize with t_ref')

            # Grid
            values = np.array(self.log['grid']['rho'])
            fh.write('grid/rho', values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute('grid/rho/units', '-')
            fh.write_attribute('grid/rho/description', 'Flux surface grid')

            values = np.array(self.log['grid']['midpoints'])
            fh.write('grid/midpoints', values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute('grid/midpoints/units', '-')
            fh.write_attribute('grid/midpoints/description', 'Midpoints of the flux surface grid, where flux models are evaluated')

            # Geometry
            values = np.array(self.log['geo']['B0'])
            if values:
                fh.write('geo/B0', values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute('geo/B0/units', 'T')
                fh.write_attribute('geo/B0/description', 'Magnetic field')

            values = np.array(self.log['geo']['area'])
            fh.write('geo/area', values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute('geo/area/units', '-')
            fh.write_attribute('geo/area/description', 'Surface area of flux surfaces on the midpoints grid, normalized by the minor radius squared')

            values = np.array(self.log['geo']['grho'])
            fh.write('geo/grho', values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute('geo/grho/units', '-')
            fh.write_attribute('geo/grho/description', '<|grad rho|> on the midpoints grid, normalized by the minor radius')

            values = np.array(self.log['geo']['area_grid'])
            fh.write('geo/area_grid', values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute('geo/area_grid/units', '-')
            fh.write_attribute('geo/area_grid/description', 'Surface area of flux surfaces on the rho grid, normalized by the minor radius squared')

            values = np.array(self.log['geo']['grho_grid'])
            fh.write('geo/grho_grid', values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute('geo/grho_grid/units', '-')
            fh.write_attribute('geo/grho_grid/description', '<|grad rho|> on the rho grid, normalized by the minor radius')

            # Species and profiles
            group_name = 'species/'
            fh.write_attribute(group_name + 'types', self.log['species_types'])
            fh.write_attribute(group_name + 'types/description', 'List of specie types')
            fh.write_attribute(group_name + 'tags', self.log['species_tags'])
            fh.write_attribute(group_name + 'tags/description', 'List of specie tags')
            fh.write_attribute(group_name + 'bulk_ion_tag', self.log['bulk_ion_tag'])

            labels = []
            for label in self.log['flux_model_labels']:
                labels.append(fix_labels(label))
            fh.write_attribute(group_name + 'flux_model_labels', labels)

            fh.write_attribute(group_name + 'n_evolve_species', self.log['n_evolve_species'])
            fh.write_attribute(group_name + 'n_evolve_species/description', 'List of species for which the density is being evolved')
            fh.write_attribute(group_name + 'T_evolve_species', self.log['T_evolve_species'])
            fh.write_attribute(group_name + 'T_evolve_species/description', 'List of species for which the temperature is being evolved')

            # Density, pressure, temperature profiles
            for s, tag in zip(self.log['species_types'], self.log['species_tags']):
                varname = 'n_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '10^20 m^-3')
                fh.write_attribute(group_name + varname + '/description', f'{s} density profile on rho grid')

                varname = 'p_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '10^20 m^-3 keV')
                fh.write_attribute(group_name + varname + '/description', f'{s} pressure profile on rho grid')

                varname = 'T_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'keV')
                fh.write_attribute(group_name + varname + '/description', f'{s} temperature profile on rho grid')

            # Flux profiles
            for s, tag in zip(self.log['species_types'], self.log['species_tags']):
                varname = 'pflux_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'GB')
                fh.write_attribute(group_name + varname + '/description', f'GyroBohm-normalized total {s} particle flux profile on midpoints grid')

                varname = 'qflux_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'GB')
                fh.write_attribute(group_name + varname + '/description', f'GyroBohm-normalized total {s} heat flux profile on midpoints grid')

                varname = 'heat_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'GB')
                fh.write_attribute(group_name + varname + '/description', f'GyroBohm-normalized total {s} collisional heating profile on midpoints grid')

                for label in self.log['flux_model_labels']:
                    varname = 'pflux_' + label + '_' + tag
                    values = np.array(self.log[varname])
                    varname = fix_labels(varname)
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', 'GB')
                    fh.write_attribute(group_name + varname + '/description', f'GyroBohm-normalized {s} particle flux profile from {label} on midpoints grid')

                    varname = 'qflux_' + label + '_' + tag
                    values = np.array(self.log[varname])
                    varname = fix_labels(varname)
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', 'GB')
                    fh.write_attribute(group_name + varname + '/description', f'GyroBohm-normalized {s} heat flux profile from {label} on midpoints grid')

                    varname = 'heat_' + label + '_' + tag
                    values = np.array(self.log[varname])
                    varname = fix_labels(varname)
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', 'GB')
                    fh.write_attribute(group_name + varname + '/description', f'GyroBohm-normalized {s} collisional heating profile from {label} on midpoints grid')

            # Density, pressure, temperature gradient profiles
            for s, tag in zip(self.log['species_types'], self.log['species_tags']):
                varname = 'aLn_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '-')
                fh.write_attribute(group_name + varname + '/description', f'Normalized {s} density gradient, a/L_n = 1/n dn/drho, on midpoints grid')

                varname = 'aLp_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '-')
                fh.write_attribute(group_name + varname + '/description', f'Normalized {s} pressure gradient, a/L_p = 1/p dp/drho, on midpoints grid')

                varname = 'aLT_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '-')
                fh.write_attribute(group_name + varname + '/description', f'Normalized {s} temperature gradient, a/L_T = 1/T dT/drho, on midpoints grid')

            # Source terms
            Sn_aux_i = 0
            Sp_aux_i = 0
            Sp_alpha_i = 0
            Sp_rad_i = 0
            Sp_coll_i = 0
            Sp_heating_i = 0
            Sp_tot_i = 0
            for s, tag in zip(self.log['species_types'], self.log['species_tags']):
                varname = 'Sn_aux_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '[Sn_ref_SI20]')
                fh.write_attribute(group_name + varname + '/description', f'Normalized {s} auxiliary particle source on rho grid')
                if s != "electron":
                    Sn_aux_i += np.array(self.log[varname])

                varname = 'Sp_aux_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
                fh.write_attribute(group_name + varname + '/description', f'Normalized {s} auxiliary power source on rho grid')
                if s != "electron":
                    Sp_aux_i += np.array(self.log[varname])

                varname = 'Sp_alpha_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
                fh.write_attribute(group_name + varname + '/description', f'Normalized {s} power source from fusion alphas on rho grid')
                if s != "electron":
                    Sp_alpha_i += np.array(self.log[varname])

                varname = 'Sp_rad_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
                fh.write_attribute(group_name + varname + '/description', f'Normalized {s} power source (sink) from radiation on rho grid')
                if s != "electron":
                    Sp_rad_i += np.array(self.log[varname])

                varname = 'Sp_heating_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
                fh.write_attribute(group_name + varname + '/description', f'Normalized {s} power source from turbulent heating on rho grid')
                if s != "electron":
                    Sp_heating_i += np.array(self.log[varname])

                varname = 'Sp_coll_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
                fh.write_attribute(group_name + varname + '/description', f'Normalized {s} power source (sink) from collisional equilibration on rho grid')
                if s != "electron":
                    Sp_coll_i += np.array(self.log[varname])

                varname = 'Sp_tot_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
                fh.write_attribute(group_name + varname + '/description', f'Normalized total {s} power source on rho grid')
                if s != "electron":
                    Sp_tot_i += np.array(self.log[varname])

            # write ion-summed source profiles
            tag = 'i'
            varname = 'Sn_aux_' + tag
            values = np.array(Sn_aux_i)
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', '[Sn_ref_SI20]')
            fh.write_attribute(group_name + varname + '/description', 'Normalized ion-summed auxiliary particle source on rho grid')

            varname = 'Sp_aux_' + tag
            values = np.array(Sp_aux_i)
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
            fh.write_attribute(group_name + varname + '/description', 'Normalized ion-summed auxiliary power source on rho grid')

            varname = 'Sp_tot_' + tag
            values = np.array(Sp_tot_i)
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
            fh.write_attribute(group_name + varname + '/description', 'Normalized ion-summed total power source on rho grid')

            varname = 'Sp_coll_' + tag
            values = np.array(Sp_coll_i)
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
            fh.write_attribute(group_name + varname + '/description', 'Normalized ion-summed power source (sink) from collisional equilibration on rho grid')

            varname = 'Sp_heating_' + tag
            values = np.array(Sp_heating_i)
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
            fh.write_attribute(group_name + varname + '/description', 'Normalized ion-summed power source from turbulent heating on rho grid')

            varname = 'Sp_rad_' + tag
            values = np.array(Sp_rad_i)
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
            fh.write_attribute(group_name + varname + '/description', 'Normalized ion-summed power source (sink) from radiation on rho grid')

            varname = 'Sp_alpha_' + tag
            values = np.array(Sp_alpha_i)
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', '[P_ref_MWm3]')
            fh.write_attribute(group_name + varname + '/description', 'Normalized ion-summed power source from fusion alphas on rho grid')

            # Source terms volume integrated along the profile
            for s, tag in zip(self.log['species_types'], self.log['species_tags']):
                varname = 'Sn_tot_int_SI20_' + tag
                values = np.array(self.log[varname])
                varname = 'Sn_tot_int_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '10^20')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} particle source')

                varname = 'Sp_aux_int_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_aux_int_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} auxillary power')

                varname = 'Sp_alpha_int_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_alpha_int_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} alpha heating')

                varname = 'Sp_rad_int_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_rad_int_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} radiation')

                varname = 'Sp_heating_int_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_heating_int_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} turbulent heating')

                varname = 'Sp_coll_int_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_coll_int_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} collisional equilibration')

            # Source terms cumulatively volume integrated over the profile
            for s, tag in zip(self.log['species_types'], self.log['species_tags']):
                varname = 'Sn_tot_cumint_SI20_' + tag
                values = np.array(self.log[varname])
                varname = 'Sn_tot_cumint_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '10^20')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} particle source interior to each flux surface on rho grid')

                varname = 'Sp_aux_cumint_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_aux_cumint_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} auxillary power interior to each flux surface on rho grid')

                varname = 'Sp_alpha_cumint_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_alpha_cumint_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} alpha heating power interior to each flux surface on rho grid')

                varname = 'Sp_rad_cumint_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_rad_cumint_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} radiation power interior to each flux surface on rho grid')

                varname = 'Sp_heating_cumint_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_heating_cumint_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} turbulent heating power interior to each flux surface on rho grid')

                varname = 'Sp_coll_cumint_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_coll_cumint_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} collisional equilibration power interior to each flux surface on rho grid')

                varname = 'Sp_tot_cumint_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Sp_tot_cumint_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Integrated {s} total power source interior to each flux surface on rho grid')

            # Heat fluxes in physical units
            for s, tag in zip(self.log['species_types'], self.log['species_tags']):
                varname = 'Q_MW_' + tag
                values = np.array(self.log[varname])
                varname = 'Q_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'{s} total heat flux on midpoints grid')

                for label in self.log['flux_model_labels']:
                    varname = 'Q_' + label + '_MW_' + tag
                    values = np.array(self.log[varname])
                    varname = 'Q_' + fix_labels(label) + '_' + tag
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', 'MW')
                    fh.write_attribute(group_name + varname + '/description', f'{s} total heat flux from {label} on midpoints grid')

            # Particle fluxes in physical units
            for s, tag in zip(self.log['species_types'], self.log['species_tags']):
                varname = 'Gam_SI20_' + tag
                values = np.array(self.log[varname])
                varname = 'Gam_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '10^20 / s')
                fh.write_attribute(group_name + varname + '/description', f'{s} particle flux')

                for label in self.log['flux_model_labels']:
                    varname = 'Gam_' + label + '_SI20_' + tag
                    values = np.array(self.log[varname])
                    varname = 'Gam_' + fix_labels(label) + '_' + tag
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', '10^20 / s')
                    fh.write_attribute(group_name + varname + '/description', f'{s} particle flux from {label}')

            # Alpha heating
            values = np.array(self.log['Palpha_MWm3'])
            varname = 'Palpha'
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', 'MW m^-3')
            fh.write_attribute(group_name + varname + '/description', 'Total alpha heating power on rho grid')

            values = np.array(self.log['Palpha_int_MW'])
            varname = 'Palpha_int'
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', 'MW')
            fh.write_attribute(group_name + varname + '/description', 'Volume-integrated total alpha heating power')

            varname = 'p_alpha'
            values = np.array(self.log[varname])
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', '10^20 m^-3 keV')
            fh.write_attribute(group_name + varname + '/description', 'Alpha particle pressure profile')

            varname = 'beta_alpha'
            values = np.array(self.log[varname])
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', '-')
            fh.write_attribute(group_name + varname + '/description', 'Alpha particle beta profile')

            if self.physics.Zeff_prof:
                varname = 'Zeff'
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '-')
                fh.write_attribute(group_name + varname + '/description', 'Effective charge, Z_effective')

            # ZeroD output
            group_name = 'zeroD/'

            for s, tag in zip(self.log['species_types'], self.log['species_tags']):
                varname = 'n_volavg_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '10^20 m^-3')
                fh.write_attribute(group_name + varname + '/description', f'Volume-averaged {s} density')

                varname = 'n_lineavg_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '10^20 m^-3')
                fh.write_attribute(group_name + varname + '/description', f'Line-averaged {s} density')

                varname = 'npeak_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '-')
                fh.write_attribute(group_name + varname + '/description', f'{s} density peaking factor, f = n_0/<n>_vol')

                varname = 'T_volavg_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'keV')
                fh.write_attribute(group_name + varname + '/description', f'Volume-averaged {s} temperature')

                varname = 'T_lineavg_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'keV')
                fh.write_attribute(group_name + varname + '/description', f'Line-averaged {s} temperature')

                varname = 'Tpeak_' + tag
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', '-')
                fh.write_attribute(group_name + varname + '/description', f'{s} temperature peaking factor, f = T_0/<T>_vol')

                values = np.array(self.log['W_MJ_' + tag])
                varname = 'W_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MJ')
                fh.write_attribute(group_name + varname + '/description', f'Volume integral of {s} pressure')

                values = np.array(self.log['Pheat_MW_' + tag])
                varname = 'Pheat_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Volume integral of {s} auxiliary, alpha, collisional, and turbulent exchange heating')

                values = np.array(self.log['Ploss_MW_' + tag])
                varname = 'Ploss_' + tag
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 'MW')
                fh.write_attribute(group_name + varname + '/description', f'Volume integral of {s} losses due to transport and radiation')

            values = np.array(self.log['Wtot_MJ'])
            varname = 'Wtot'
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', 'MJ')
            fh.write_attribute(group_name + varname + '/description', 'Volume integral of species-summed pressure')

            varname = 'beta_vol'
            values = np.array(self.log[varname])
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', 'MJ')
            fh.write_attribute(group_name + varname + '/description', 'Volume averaged total beta')

            values = np.array(self.log['Pheat_MW'])
            varname = 'Pheat'
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', 'MW')
            fh.write_attribute(group_name + varname + '/description', 'Volume integral of species-summed auxiliary and alpha heating')

            values = np.array(self.log['Ploss_MW'])
            varname = 'Ploss'
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', 'MW')
            fh.write_attribute(group_name + varname + '/description', 'Volume integral of species-summed losses due to transport and radiation')

            values = np.array(self.log['Paux_MW'])
            varname = 'Paux'
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', 'MW')
            fh.write_attribute(group_name + varname + '/description', 'Volume integral of species-summed auxes due to transport and radiation')

            values = np.array(self.log['Palpha_int_MW'])
            varname = 'Palpha'
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', 'MW')
            fh.write_attribute(group_name + varname + '/description', 'Volume-integrated species-summed alpha heating power')

            values = np.array(self.log['Pfus_MW'])
            varname = 'Pfus'
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', 'MW')
            fh.write_attribute(group_name + varname + '/description', 'Volume-integrated species-summed fusion power')

            varname = 'Qfus'
            values = np.array(self.log[varname])
            fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
            fh.write_attribute(group_name + varname + '/units', '-')
            fh.write_attribute(group_name + varname + '/description', 'Fusion gain')

            # Store energy confinement time and related metrics only
            # when alpha or external heating sources are present
            if self.physics.alpha_heating or self.auxiliary_heating:
                varname = 'tauE'
                values = np.array(self.log[varname])
                fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                fh.write_attribute(group_name + varname + '/units', 's')
                fh.write_attribute(group_name + varname + '/description', 'Energy confinement time (tauE = Wtot/Pheat), assuming steady-state')

                if self.geo.geo_option == "vmec" or self.geo.geo_option == "desc" or self.geo.geo_option == "basic":
                    varname = 'tauE_ISS04'
                    values = np.array(self.log[varname])
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', 's')
                    fh.write_attribute(group_name + varname + '/description', 'Prediction of energy confinement time according to tauE_ISS04')

                    varname = 'f_renorm'
                    values = np.array(self.log[varname])
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', '-')
                    fh.write_attribute(group_name + varname + '/description', 'f = tauE / tauE_ISS04, assuming steady-state')

                    varname = 'n_sudo'
                    values = np.array(self.log[varname])
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', '10^20 m^-3')
                    fh.write_attribute(group_name + varname + '/description', 'Sudo density limit')

                    varname = 'f_sudo_line'
                    values = np.array(self.log[varname])
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', '-')
                    fh.write_attribute(group_name + varname + '/description', 'f = <n_e>_line/n_sudo')

                    varname = 'f_sudo_vol'
                    values = np.array(self.log[varname])
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', '-')
                    fh.write_attribute(group_name + varname + '/description', 'f = <n_e>_vol/n_sudo')

                    varname = 'f_sudo_edge'
                    values = np.array(self.log[varname])
                    fh.write(group_name + varname, values, values.shape, [0]*values.ndim, values.shape)
                    fh.write_attribute(group_name + varname + '/units', '-')
                    fh.write_attribute(group_name + varname + '/description', 'f = n_edge/n_sudo')
