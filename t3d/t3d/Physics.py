import numpy as np
from t3d.Profiles import GridProfile
from t3d.Species import Species
from t3d.Logbook import info
from scipy.constants import e
import importlib.resources
import re


class Physics():

    def __init__(self, inputs, grid, geo, norms, imported=None):

        physics_parameters = inputs.get('physics', inputs.get('debug', {}))
        self.collisions = physics_parameters.get('collisions', True)
        self.alpha_heating = physics_parameters.get('alpha_heating', True)
        self.radiation = physics_parameters.get('radiation', False)
        self.bremsstrahlung = physics_parameters.get('bremsstrahlung', False)
        self.bremsstrahlung = physics_parameters.get('bremstrahlung', self.bremsstrahlung)  # allow common typo in input
        self.turbulent_exchange = physics_parameters.get('turbulent_exchange', False)
        self.f_tritium = physics_parameters.get('f_tritium', 0.5)  # assumed fraction of deuterium density to be used as tritium for alpha heating calculation
        self.alpha_efficiency = physics_parameters.get('alpha_efficiency', 1.0)  # efficiency of alpha heating, which can be degraded by fast ion losses
        self.scale_density_to = physics_parameters.get('scale_density_to', None)

        def parse_string(input_str):
            # Define a regular expression pattern to match the numeric value and the unit
            pattern = r'(\d+(\.\d+)?)\s*\*\s*(\w+)'

            # Use re.match to search for the pattern in the input string
            match = re.match(pattern, input_str)
            match2 = re.match(r'\w+', input_str)  # handle case with no multiplier

            if match:
                # Extract the numeric value and unit from the matched groups
                value = float(match.group(1))
                unit = match.group(3)

                return [value, unit]
            elif match2:
                value = 1.0
                unit = match2.group(0)
                return [value, unit]
            else:
                assert False, f"Error parsing string \"{input_str}\""

        if self.scale_density_to is not None:
            (self.scale_density_frac, self.scale_density_to) = parse_string(self.scale_density_to)

        self.grid = grid
        self.geo = geo
        self.norms = norms

        # time dependent trinity parameters
        dynamics_parameters = inputs.get('time_dependence', {})
        self.update_source = dynamics_parameters.get('update_source', False)
        self.source_Q_threshold_low = dynamics_parameters.get('Q_threshold_low', 2)
        self.source_Q_threshold_high = dynamics_parameters.get('Q_threshold_high', 10)
        self.source_p_up_multiplier = dynamics_parameters.get('p_up_multiplier', 1.01)
        self.source_p_down_multiplier = dynamics_parameters.get('p_down_multiplier', 0.9)
        self.source_p_aux_max_MW = dynamics_parameters.get('p_aux_max_MW', 100)
        self.source_p_aux_min_MW = dynamics_parameters.get('p_aux_min_MW', 0.01)
        self.update_edge = dynamics_parameters.get('update_edge', False)
        self.edge_p_ratio = dynamics_parameters.get('p_threshold_ratio', 4)
        self.edge_p_multiplier = dynamics_parameters.get('p_edge_multiplier', 1.05)
        self.edge_T_max = dynamics_parameters.get('T_edge_max', 40)
        self.edge_n_ratio = dynamics_parameters.get('n_threshold_ratio', 4)
        self.edge_n_multiplier = dynamics_parameters.get('n_edge_multiplier', 1.05)
        self.edge_n_max = dynamics_parameters.get('n_edge_max', 40)

        # parameters for artificial Zeff profile. only used for bremsstrahlung model.
        Zeff_parameters = physics_parameters.get('Zeff', {})
        if isinstance(Zeff_parameters, (float, int)):  # const Zeff
            self.Zeff_prof = GridProfile(Zeff_parameters, grid)
            self.Zeff_from_species = False
        elif isinstance(Zeff_parameters, dict) and len(Zeff_parameters) == 0:
            self.Zeff_prof = None  # this will get set later
            self.Zeff_from_species = True  # if Zeff not specified, default to from_species = True
        else:
            self.Zeff_shape = Zeff_parameters.get('shape', 'parabolic')
            # PARABOLIC PARAMETERS
            self.Zeff_core = Zeff_parameters.get('core', 0)
            self.Zeff_volavg = Zeff_parameters.get('volavg', None)
            self.Zeff_lineavg = Zeff_parameters.get('lineavg', None)
            self.Zeff_alpha = Zeff_parameters.get('alpha', 1)
            self.Zeff_alpha1 = Zeff_parameters.get('alpha1', 2)
            self.Zeff_edge = Zeff_parameters.get('edge', 0)
            self.Zeff_sep = Zeff_parameters.get('sep', 0)
            # GAUSSIAN
            self.Zeff_height = Zeff_parameters.get('height', 0)
            self.Zeff_int = Zeff_parameters.get('integrated', None)
            self.Zeff_width = Zeff_parameters.get('width', 0.1)
            self.Zeff_center = Zeff_parameters.get('center', 0)
            # EXPONENTIAL
            self.Zeff_coefs = Zeff_parameters.get('coefs', [0,0,0,0])
            if len(self.Zeff_coefs) < 4:  # pad with zeros
                self.Zeff_coefs.extend([0] * (4 - len(self.Zeff_coefs)))
            # IMPORT
            self.Zeff_import = Zeff_parameters.get('import', False)
            self.Zeff_import_key = Zeff_parameters.get('key', None)
            # MANUAL
            self.Zeff_profstr = Zeff_parameters.get('profstr', None)
            # FROM SPECIES
            self.Zeff_from_species = Zeff_parameters.get('from_species', False)

            if self.Zeff_from_species:
                self.Zeff_prof = None  # this will get set later
            elif self.Zeff_import:
                assert imported.imported is not None, 'Error: must specify [import] section to use import = true for Zeff'
                Zeff = imported.get_Zeff(key=self.Zeff_import_key)
                self.Zeff_prof = GridProfile(Zeff, grid)
            # use analytic profile for Zeff
            elif self.Zeff_shape == 'parabolic':
                if self.Zeff_edge is None:
                    self.Zeff_edge = 0.5  # default value

                if self.Zeff_lineavg:
                    Zeff_prof = GridProfile((1 - (grid.rho/grid.rho_edge)**self.Zeff_alpha1)**self.Zeff_alpha, grid)
                    self.Zeff_prof = (self.Zeff_lineavg - self.Zeff_edge) / np.average(Zeff_prof) * Zeff_prof + self.Zeff_edge
                elif self.Zeff_volavg:
                    Zeff_prof = GridProfile((1 - (grid.rho/grid.rho_edge)**self.Zeff_alpha1)**self.Zeff_alpha, grid)
                    self.Zeff_prof = (self.Zeff_volavg - self.Zeff_edge) / self.geo.volume_average(Zeff_prof) * Zeff_prof + self.Zeff_edge
                elif self.Zeff_core and self.Zeff_sep:
                    self.Zeff_prof = GridProfile((self.Zeff_core - self.Zeff_sep)*(1 - (grid.rho)**self.Zeff_alpha1)**self.Zeff_alpha + self.Zeff_sep, grid)
                elif self.Zeff_core is not None:
                    self.Zeff_prof = GridProfile((self.Zeff_core - self.Zeff_edge)*(1 - (grid.rho/grid.rho_edge)**self.Zeff_alpha1)**self.Zeff_alpha + self.Zeff_edge, grid)
            elif self.Zeff_shape == 'gaussian':
                Gaussian = np.vectorize(self.Gaussian)
                info(f'Zeff is Gaussian with width {self.Zeff_width}, center {self.Zeff_center}, height {self.Zeff_height}, int {self.Zeff_int}')
                Zeff_prof = GridProfile(Gaussian(grid.rho, A=1, sigma=self.Zeff_width, x0=self.Zeff_center), grid)  # Trinity units
                if self.Zeff_int:
                    info(f'Zeff using target int {self.Zeff_int}')
                    target = self.Zeff_int
                    volume_integral = self.geo.volume_integrate(Zeff_prof, normalized=False, interpolate=True)
                    self.Zeff_prof = (target / volume_integral) * Zeff_prof
                else:
                    info(f'Zeff using height {self.Zeff_height}')
                    self.Zeff_prof = self.Zeff_height * Zeff_prof
            elif self.Zeff_shape == 'exponential':
                coefs = self.Zeff_coefs
                x = grid.rho
                self.Zeff_prof = GridProfile(coefs[0]*np.exp(-coefs[1]*x**2 - coefs[2]*x**4 - coefs[3]*x**6), grid)
            elif self.Zeff_shape == 'manual':
                from scipy.interpolate import PchipInterpolator
                profXY = np.array(np.mat(self.Zeff_profstr))
                profX,profY = profXY
                self.Zeff_prof = GridProfile(PchipInterpolator(profX,profY)(grid.rho/grid.rho_edge), grid)
            else:
                raise ValueError('Invalid Zeff profile shape in Trinity input')

        if self.alpha_heating:
            self.xs = {}
            for xs_id, xs_name in xs_names.items():
                self.xs[xs_id] = Xsec.read_xsec(xs_name)
            # Total D + D fusion cross section is due to equal contributions from the above two processes.
            self.xs['D-D'] = self.xs['D-D_a'] + self.xs['D-D_b']
            self.xs['T-3He'] = self.xs['T-3He_a'] + self.xs['T-3He_b']

    def compute_bremsstrahlung(self, s, species):
        if self.Zeff_from_species:
            Zeff = self.Zeff_prof = species.Zeff()
        else:
            Zeff = self.Zeff_prof

        if s.type == 'electron':
            P_MWm3 = 5.35e3*s.n()**2*Zeff*s.T()**0.5/1e6
            # convert to code units
            return P_MWm3 / self.norms.P_ref_MWm3
        else:
            return 0.0*s.n()

    def compute_radiation(self, s, species):
        '''
        compute radiation due to impurities
        includes line, bremsstrahlung, and recombination radiation
        polynomial fits  taken from
        "Improved fits of coronal radiative cooling rates forhigh-temperature plasmas,"
        A. A. Mavrin, Radiation Effects and Defects in Solids, 173:5-6, 388-398
        '''
        from numpy.polynomial.polynomial import polyval

        def Lrad_helium(Te):
            if Te < 0.1:
                return 0.
            elif Te < 100:
                A = [-3.5551e1, 3.1469e-1, 1.0156e-1, -9.373e-2, 2.502e-2]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        def Lrad_lithium(Te):
            if Te < 0.1:
                return 0.
            elif Te < 100:
                A = [-3.5115e1, 1.9475e-1, 2.5082e-1, -1.607e-1, 3.519e-2]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        def Lrad_beryllium(Te):
            if Te < 0.1:
                return 0.
            elif Te < 100:
                A = [-3.4765e1, 3.727e-2, 3.8363e-1, -2.1384e-1, 4.169e-2]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        def Lrad_carbon(Te):
            if Te < 0.1:
                return 0.
            elif Te < 0.5:
                A = [-3.4738e1, -5.0085, -1.2788e1, -1.6637e1, -7.2904]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 100:
                A = [-3.4174e1, -3.6687e-1, 6.8856e-1, -2.9191e-1, 4.447e-2]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        def Lrad_nitrogen(Te):
            if Te < 0.1:
                return 0.
            elif Te < 0.5:
                A = [-3.4065e1, -2.3614, -6.0605, -1.157e1, -6.9621]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 2:
                A = [-3.3899e1, -5.9668e-1, 7.6272e-1, -1.716e-1, 5.877e-2]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 100:
                A = [-3.3913e1, -5.2628e-1, 7.0047e-1, -2.279e-1, 2.835e-2]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        def Lrad_oxygen(Te):
            if Te < 0.1:
                return 0.
            elif Te < 0.3:
                A = [-3.7257e1, -1.5635e1, -1.7141e1, -5.3765, 0.]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 100:
                A = [-3.364e1, -7.6211e-1, 7.9655e-1, -2.085e-1, 1.436e-2]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        def Lrad_neon(Te):
            if Te < 0.1:
                return 0.
            elif Te < 0.7:
                A = [-3.3132e1, 1.7309, 1.523e1, 2.8939e1, 1.5648e1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 5:
                A = [-3.329e1, -8.775e-1, 8.6842e-1, -3.9544e-1, 1.7244e-1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 100:
                A = [-3.341e1, -4.5345e-1, 2.9731e-1, 4.396e-2, -2.693e-2]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        def Lrad_argon(Te):
            if Te < 0.1:
                return 0.
            elif Te < 0.6:
                A = [-3.2155e1, 6.5221, 3.0769e1, 3.9161e1, 1.5353e1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 3:
                A = [-3.253e1, 5.449e-1, 1.5389, -7.6887, 4.9806]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 100:
                A = [-3.1853e1, -1.6674, 6.1339e-1, 1.748e-1, -8.226e-2]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        def Lrad_krypton(Te):
            if Te < 0.1:
                return 0.
            elif Te < 0.447:
                A = [-3.4512e1, -2.1484e1, -4.4723e1, -4.0133e1, -1.3564e1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 2.364:
                A = [-3.1399e1, -5.0091e-1, 1.9148, -2.5865, -5.2704]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 100:
                A = [-2.9954e1, -6.3683, 6.6831, -2.9674, 4.8356e-1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        def Lrad_xenon(Te):
            if Te < 0.1:
                return 0.
            elif Te < 0.5:
                A = [-2.9303e1, 1.4351e1, 4.7081e1, 5.958e1, 2.5615e1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 2.5:
                A = [-3.1113e1, 5.9339e-1, 1.2808, -1.1628e1, 1.0748e1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 10:
                A = [-2.5813e1, -2.7526e1, 4.8614e1, -3.6885e1, 1.0069e1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 100:
                A = [-2.2138e1, -2.2592e1, 1.9619e1, -7.5181, 1.0858]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        def Lrad_tungsten(Te):
            if Te < 0.1:
                return 0.
            elif Te < 1.5:
                A = [-3.0374e1, 3.8304e-1, -9.5126e-1, -1.0311, -1.0103e-1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 4:
                A = [-3.0238e1, -2.9208, 2.2824e1, -6.3303e1, 5.1849e1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            elif Te < 100:
                A = [-3.2153e1, 5.2499, -6.274, 2.6627, -3.6759e-1]
                x = np.log10(Te)
                return 10**polyval(x, A)
            else:
                return 0.

        Lrad = {
            'helium': Lrad_helium,
            'lithium': Lrad_lithium,
            'beryllium': Lrad_beryllium,
            'carbon': Lrad_carbon,
            'nitrogen': Lrad_nitrogen,
            'oxygen': Lrad_oxygen,
            'neon': Lrad_neon,
            'argon': Lrad_argon,
            'krypton': Lrad_krypton,
            'xenon': Lrad_xenon,
            'tungsten': Lrad_tungsten
        }

        Prad_Wm3 = 0.0*s.n()
        if s.type == 'electron':
            Te = s.T().profile
            ne = s.n().profile
            for imp in species.get_species_list():
                # loop over impurity species, summing radiation contribution
                if imp.Z > 1 and imp.type in Lrad.keys():
                    nZ = imp.n().profile
                    for i in range(len(Te)):
                        L = Lrad[imp.type](Te[i])
                        Prad_Wm3[i] = Prad_Wm3[i] + (1e20*nZ[i])*(1e20*ne[i])*L
        else:
            pass

        return Prad_Wm3 * 1e-6 / self.norms.P_ref_MWm3

    def compute_total_alpha_heating(self, species):
        '''
        compute total alpha heating in MW/m^3 (assumes deuterium bulk ions)
        '''

        try:
            n_D = species['deuterium'].n()
        except:
            n_D = 0.0*species.ref_species.n()

        if not self.alpha_heating:
            return 0.0*n_D

        try:
            n_T = species['tritium'].n()
        except:
            # if tritium is not an evolved species, assume the deuterium species is actually partially tritium
            n_T = self.f_tritium*n_D
            n_D = (1-self.f_tritium)*n_D

        # compute reaction rate sigv (cm^3/s)
        sigv = self.get_reactivity(self.xs['D-T'], species['deuterium'].T())

        # compute total alpha power (W/cm^3 = MW/m^3)
        P_tot_MWm3 = self.alpha_efficiency*5.6e-13*sigv*(1e14*n_D)*(1e14*n_T)
        return P_tot_MWm3

    def compute_alpha_heating(self, s, species):
        '''
        compute alpha heating to deuterium and electron species (assumes deuterium bulk ions)
        '''
        P_tot_MWm3 = self.compute_total_alpha_heating(species)

        ion_frac = self.calc_alpha_ion_heating_fraction(species)

        if s.type == 'electron':
            return (1-ion_frac)*P_tot_MWm3/self.norms.P_ref_MWm3
        else:  # heat all thermal ions, weighted by density
            # compute sum_i n_i Z_i**2/m_i
            z2_ov_m = 0
            for sp in species.get_species_list():
                if sp.type != 'electron':
                    z2_ov_m += sp.n()*sp.Z**2/sp.mass/species['electron'].n()
            return ion_frac*(s.n()/species['electron'].n()*s.Z**2/s.mass)/z2_ov_m*P_tot_MWm3/self.norms.P_ref_MWm3

    def calc_alpha_ion_heating_fraction(self, species):

        # compute sum_i n_i Z_i**2/m_i
        z2_ov_m = 0
        for s in species.get_species_list():
            if s.type != 'electron':
                z2_ov_m += s.n()*s.Z**2/s.mass/species['electron'].n()

        E_crit = 4*14.8*species['electron'].T()*z2_ov_m**(2./3.)
        x = 3.5e3/E_crit.profile
        y = x**0.5
        frac = np.log((1.+y**3)/(1.+y)**3)/(3.*x) + 2.*np.arctan2(2.*y-1., np.sqrt(3.))/(np.sqrt(3.)*x) \
            - 2.*np.arctan2(-1., np.sqrt(3.))/(np.sqrt(3.)*x)
        return frac

    def get_reactivity(self, xs, T):
        """Return reactivity, <sigma.v> in cm3.s-1 for temperature T in keV."""

        T = T[:, None]

        fac = 4 / np.sqrt(2 * np.pi * xs.mr * u)
        fac /= (1000 * T * e)**1.5
        fac *= (1000 * e)**2
        func = fac * xs.xs * xs.Egrid * np.exp(-xs.Egrid / T)
        Ireact = np.trapz(func, xs.Egrid, axis=1)
        # Convert from m3.s-1 to cm3.s-1
        return Ireact * 1.e6

    def compute_alpha_pressure(self, species):
        """ Return alpha particle pressure in T3D units."""

        E_alpha = 3.5e3  # keV
        # compute sum_i n_i Z_i**2/m_i
        z2_ov_m = 0
        for s in species.get_species_list():
            if s.type != 'electron':
                z2_ov_m += s.n()*s.Z**2/s.mass/species['electron'].n()
        E_crit = 4*14.8*species['electron'].T()*z2_ov_m**(2./3.)  # keV
        x = E_alpha/E_crit.profile
        y = x**0.5  # == v_alpha/v_crit
        A = -np.pi/6/np.sqrt(3) + y**2/2 - np.arctan2(2*y-1, np.sqrt(3))/np.sqrt(3) + np.log(1+y)/3 - np.log(1-y+y**2)/6  # an integral
        Palpha = self.compute_total_alpha_heating(species)/self.norms.P_ref_MWm3
        alpha = Species({'mass': 4, 'Z': 2, 'type':"alpha"}, self.grid)
        alpha.init_profiles(self.grid, None, self.norms)
        nu_slow = alpha.nu_equil(species['electron'])

        p_alpha = 2/(3*x)*Palpha/nu_slow*A  # in T3D units
        return p_alpha

    def scale_to_sudo_density(self, species):
        assert self.geo.geo_option == "vmec" or self.geo.geo_option == "desc" or self.geo.geo_option == "basic", "using scale_density_to = 'sudo' requires vmec or desc geometry"
        Palpha = self.geo.volume_integrate(self.compute_total_alpha_heating(species))
        Paux = self.geo.volume_integrate(species.Paux*self.norms.P_ref_MWm3)
        Prad = 0
        if self.radiation:  # this radiation model includes bremsstrahlung, along with line and recombination radiation
            Prad = -self.geo.volume_integrate(self.compute_radiation(species['electron'], species))*self.norms.P_ref_MWm3
        elif self.bremsstrahlung:  # just bremsstrahlung
            Prad = -self.geo.volume_integrate(self.compute_bremsstrahlung(species['electron'], species))*self.norms.P_ref_MWm3
        R = self.geo.AspectRatio*self.geo.a_minor
        a = self.geo.a_minor
        B = abs(self.geo.B0)
        V = 16*R*a**2
        if self.scale_density_to[5:] == 'vol':
            ne = self.geo.volume_average(species['electron'].n())
        elif self.scale_density_to[5:] == 'line':
            ne = np.average(species['electron'].n())
        f_sudo = self.scale_density_frac
        n_sudo = np.sqrt(Paux*B/V/(1 - B/V*(Palpha+Prad)*f_sudo**2/ne**2))

        scale = f_sudo*n_sudo/ne
        for s in species.get_species_list():
            s.set_n(scale*s.n(), fixed_T=True)

    def compute_Pheat(self, species):
        P = 0
        for s in species.get_species_list():
            Paux_s = self.geo.volume_integrate(s.Sp_labeled['aux'])*self.norms.P_ref_MWm3
            Palpha_s = self.geo.volume_integrate(s.Sp_labeled['alpha'])*self.norms.P_ref_MWm3
            Prad_s = self.geo.volume_integrate(s.Sp_labeled['rad'])*self.norms.P_ref_MWm3
            P_s = Paux_s + Palpha_s + Prad_s
            P += P_s
        return P

    def compute_tauE(self, species):
        W = 0
        P = 0
        for s in species.get_species_list():
            W_s = self.geo.volume_integrate(1.5*s.p())*self.norms.p_ref/1e6  # MJ
            W += W_s

            Paux_s = self.geo.volume_integrate(s.Sp_labeled['aux'])*self.norms.P_ref_MWm3
            Palpha_s = self.geo.volume_integrate(s.Sp_labeled['alpha'])*self.norms.P_ref_MWm3
            Prad_s = self.geo.volume_integrate(s.Sp_labeled['rad'])*self.norms.P_ref_MWm3
            P_s = Paux_s + Palpha_s + Prad_s
            P += P_s

        return W/P

    def compute_ISS04(self, species):
        R = self.geo.AspectRatio*self.geo.a_minor
        a = self.geo.a_minor
        Pheat = 0
        for s in species.get_species_list():
            Paux_s = self.geo.volume_integrate(s.Sp_labeled['aux'])*self.norms.P_ref_MWm3
            Palpha_s = self.geo.volume_integrate(s.Sp_labeled['alpha'])*self.norms.P_ref_MWm3
            Prad_s = self.geo.volume_integrate(s.Sp_labeled['rad'])*self.norms.P_ref_MWm3
            Pheat_s = Paux_s + Palpha_s + Prad_s
            Pheat += Pheat_s
        P = Pheat
        ne_avg = np.average(species['electron'].n().profile)  # 10^20/m^3
        n19 = ne_avg*10
        B = np.abs(self.geo.B0)
        iota = np.abs(self.geo.iota23)

        tauE_ISS04 = 0.134*(a**2.28)*(R**0.64)*(P**-0.61)*(n19**0.54)*(B**0.84)*(iota**0.41)
        return tauE_ISS04

    def compute_sources(self, species, geometry, norms):
        if self.scale_density_to is not None and self.scale_density_to[:4] == 'sudo':
            self.scale_to_sudo_density(species)

        for k in species.n_evolve_keys:
            s = species[k]
            s.Sn = s.Sn_labeled['aux']

        for k in species.T_evolve_keys:
            s = species[k]
            s.Sp = s.Sp_labeled['aux']

            if self.alpha_heating:
                s.Sp_labeled['alpha'] = self.compute_alpha_heating(s, species)
                s.Sp += s.Sp_labeled['alpha']

            if self.radiation:  # this radiation model includes bremsstrahlung, along with line and recombination radiation
                s.Sp_labeled['rad'] = -self.compute_radiation(s, species)
                s.Sp += s.Sp_labeled['rad']
            elif self.bremsstrahlung:  # just bremsstrahlung
                s.Sp_labeled['rad'] = -self.compute_bremsstrahlung(s, species)
                s.Sp += s.Sp_labeled['rad']

            if self.turbulent_exchange:
                s.Sp_labeled['heating'] = species.ref_species.p()**2.5/species.ref_species.n()**1.5/self.geo.Btor**2*(s.heat.toGridProfile())
                s.Sp += s.Sp_labeled['heating']

            if self.collisions:
                s.Sp_labeled['coll'] = 1.5*s.coll_eq(species)
                s.Sp += s.Sp_labeled['coll']


'''
This file hosts the Xsec class, which calculates fusion cross-sections based on (n,T) profiles.
See https://scipython.com/blog/nuclear-fusion-cross-sections/.
'''

# Reactant masses in atomic mass units (u).
u = 1.66053906660e-27
masses = {'D': 2.014, 'T': 3.016, '3He': 3.016, '11B': 11.009305167,
          'p': 1.007276466620409}

# Define a dictionary of available Fusion Cross Sections
xs_names = {'D-T': 'D_T_-_a_n.txt',              # D + T -> a + n
            'D-D_a': 'D_D_-_T_p.txt',            # D + D -> T + p
            'D-D_b': 'D_D_-_3He_n.txt',          # D + D -> 3He + n
            'D-3He': 'D_3He_-_4He_p.txt',        # D + 3He -> a + p
            'p-B': 'p_11B_-_3a.txt',             # p + 11B -> 3a
            'T-T': 'T_T_-_4He_n_n.txt',          # T + T -> 4He + 2n
            'T-3He_a': 'T_3He_-_n_p_4He.txt',    # T + 3He -> 4He + n + p
            'T-3He_b': 'T_3He_-_D_4He.txt',      # T + 3He -> 4He + D
            '3He-3He': '3He_3He_-_p_p_4He.txt'}  # 3He + 3He -> 4He + 2p

xs_labels = {'D-T': r'$\mathrm{D-T}$',
             'D-D': r'$\mathrm{D-D}$',
             'D-3He': r'$\mathrm{D-^3He}$',
             'p-B': r'$\mathrm{p-^{11}B}$',
             'T-T': r'$\mathrm{T-T}$',
             'T-3He': r'$\mathrm{T-^3He}$',
             '3He-3He': r'$\mathrm{^3He-^3He}$'}


class Xsec:
    def __init__(self, m1, m2, xs):
        self.m1, self.m2, self.xs = m1, m2, xs
        self.mr = self.m1 * self.m2 / (self.m1 + self.m2)
        # Energy grid, 1 - 1000 keV, evenly spaced in log-space.
        self.Egrid = np.logspace(0, 5, 1000)

    @classmethod
    def read_xsec(cls, filename, CM=True):
        """
        Read in cross section from filename and interpolate to energy grid.

        """

        data_file = importlib.resources.open_text('t3d.Fusion_cross_sections', filename)

        E, xs = np.genfromtxt(data_file, comments='#', skip_footer=2,
                              unpack=True)
        if CM:
            collider, target = filename.split('_')[:2]
            m1, m2 = masses[target], masses[collider]
            E *= m1 / (m1 + m2)

        Egrid = np.logspace(0, 5, 1000)

        xs = np.interp(Egrid, E*1.e3, xs*1.e-28)
        return cls(m1, m2, xs)

    def __add__(self, other):
        return Xsec(self.m1, self.m2, self.xs + other.xs)

    def __mul__(self, n):
        return Xsec(self.m1, self.m2, n * self.xs)
    __rmul__ = __mul__

    def __getitem__(self, i):
        return self.xs[i]

    def __len__(self):
        return len(self.xs)
