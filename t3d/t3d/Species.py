import numpy as np
from t3d.Profiles import FluxProfile, GridProfile
from collections import OrderedDict
import re
import copy
from t3d.Logbook import info

m_proton_cgs = 1.67e-24  # mass of proton in grams
e_cgs = 1.602e-19  # Coulomb


class SpeciesDict():

    def __init__(self, inputs, grid):
        species_list_params = inputs.get('species', {})
        self.N_species = len(species_list_params)
        self.N_radial = grid.N_radial
        self.grid = grid

        # create dictionary of species objects, keyed by species type (e.g. 'deuterium', 'electron', etc)
        self.species_dict = OrderedDict()
        self.has_adiabatic_species = False
        reference_species_count = 0
        adiabatic_species_count = 0
        bulk_ion_count = 0
        impurity_count = 0
        impurity_type = None
        first_ion_type = None
        last_ion_type = None
        last_evolving_ion_type = None

        self.N_dens_profiles = 0
        self.N_temp_profiles = 0
        self.N_profiles = 0
        self.n_evolve_keys = []
        self.T_evolve_keys = []
        self.n_equalto_keys = []
        self.T_equalto_keys = []

        for idx, sp in enumerate(species_list_params):
            # initialize a species object
            s = Species(sp, grid)
            # store species object in dictionary keyed by species type (e.g. 'deuterium', 'electron', etc)
            self.species_dict[s.type] = s

            # check for adiabatic species
            if s.is_adiabatic:
                adiabatic_species_count = adiabatic_species_count + 1
                adiabatic_type = s.type
                s.evolve_density = False
                s.evolve_temperature = False

            # check for reference species
            if s.use_as_reference:
                reference_species_count = reference_species_count + 1
                reference_type = s.type

            # check for bulk ion species
            if s.bulk_ion:
                bulk_ion_count = bulk_ion_count + 1
                bulk_ion_type = s.type
                assert s.type != 'electron', "Error: electrons cannot be used as bulk ion species"

            # check for impurity species
            if s.impurity:
                impurity_count = impurity_count + 1
                impurity_type = s.type

            # save the type of the first listed ion species
            if first_ion_type is None and s.type != 'electron':
                first_ion_type = s.type

            if s.density_equal_to is not None:
                s.evolve_density = False
                self.n_equalto_keys.append(s.type)
            if s.evolve_density:
                self.N_dens_profiles = self.N_dens_profiles + 1
                # make a list of species types that will evolve density
                self.n_evolve_keys.append(s.type)
            if s.temperature_equal_to is not None:
                s.evolve_temperature = False
                self.T_equalto_keys.append(s.type)
            if s.evolve_temperature:
                self.N_temp_profiles = self.N_temp_profiles + 1
                # make a list of species types that will evolve temperature
                self.T_evolve_keys.append(s.type)

            # save the type of the last ion species
            if s.type != 'electron':
                last_ion_type = s.type
                if s.evolve_density:
                    last_evolving_ion_type = s.type
            else:
                self.sidx_electron = idx

        if impurity_type is not None:
            qneut_type = impurity_type
        elif last_evolving_ion_type is not None:
            qneut_type = last_evolving_ion_type
        else:
            qneut_type = last_ion_type
        self.qneut_species = self.species_dict[qneut_type]

        # the qneut_type density profile can be set by quasineutrality,
        # so don't include it in count of evolved profiles
        if self.N_dens_profiles > 0 and self.qneut_species.evolve_density:  # last_evolving_ion_type is not None:
            self.N_dens_profiles = self.N_dens_profiles - 1
            self.species_dict[qneut_type].evolve_density = False
            self.n_evolve_keys.remove(qneut_type)

        # number of evolved profiles
        self.N_profiles = self.N_dens_profiles + self.N_temp_profiles

        # sanity checks
        assert adiabatic_species_count <= 1, "Error: cannot have more than one adiabatic species specified in [species] section. Specify adiabatic species in [model] instead."
        assert reference_species_count <= 1, "Error: cannot have more than one species set as reference species"
        assert bulk_ion_count <= 1, "Error: cannot have more than one species set as bulk ions"
        for t in self.n_equalto_keys:
            assert self.species_dict[t].density_equal_to in self.species_dict.keys(), f"Error: cannot set '{t}' density equal to non-existent species '{self.species_dict[t].density_equal_to}'"
        for t in self.T_equalto_keys:
            assert self.species_dict[t].temperature_equal_to in self.species_dict.keys(), f"Error: cannot set '{t}' temperature equal to non-existent species '{self.species_dict[t].temperature_equal_to}'"

        # create member labeling adiabatic species
        if adiabatic_species_count == 1:
            self.adiabatic_species = self.species_dict[adiabatic_type]
            self.has_adiabatic_species = True

        # create member labeling reference species
        if reference_species_count == 0:
            self.ref_species = self.species_dict[first_ion_type]
        else:
            self.ref_species = self.species_dict[reference_type]

        # create member labeling electron species
        self.electron = self.species_dict['electron']

        # create member labeling bulk ion species
        if bulk_ion_count == 0:
            # default bulk to first ion species in input file
            self.bulk_ion = self.species_dict[first_ion_type]
            bulk_ion_type = first_ion_type
        else:
            self.bulk_ion = self.species_dict[bulk_ion_type]
        self.other_ions_dict = OrderedDict(self.species_dict)
        del self.other_ions_dict['electron']
        del self.other_ions_dict[bulk_ion_type]

        # create member labeling impurity species
        if impurity_count == 1:
            self.impurity = self.species_dict[impurity_type]
        else:
            self.impurity = None

        self.first_ion_type = first_ion_type

        self.evolve_count = len(self.n_evolve_keys) + len(self.T_evolve_keys)

        self.clear_fluxes()
        self.clear_sources()

        self.init_psi_matrices()

    def __getitem__(self, s_type):
        '''
        accessor that allows a particular element (Species) of the species_dict dictionary to be accessed via species[s_type] == species.species_dict[s_type]
        s_type is a String
        '''
        return self.species_dict[s_type]

    def remove(self, s_type, in_place=False):
        '''
        removes an element (Species) from the SpeciesDict
        s_type is a String
        if in_place == False, a new SpeciesDict object will be returned and the calling SpeciesDict will not be modified
        if in_place == True, the calling SpeciesDict will be modified
        '''
        if in_place:
            new_species = self
        else:
            new_species = copy.deepcopy(self)

        del new_species.species_dict[s_type]
        new_species.N_species -= 1
        if s_type in new_species.n_evolve_keys:
            new_species.n_evolve_keys.remove(s_type)
        if s_type in new_species.T_evolve_keys:
            new_species.T_evolve_keys.remove(s_type)

        return new_species

    def get_species_list(self):
        '''
        return list of all elements of species_dict. each element is a Species class member
        '''

        return self.species_dict.values()

    def print_info(self):
        info("  Species Information")
        info(f"    This calculation contains {[s for s in self.species_dict]} species.")
        info(f"    The '{self.bulk_ion.type}' species is treated as the bulk ions.")
        if self.impurity:
            info(f"    The '{self.impurity.type}' species is treated as the impurity.")
        info(f"    Evolving densities: {self.n_evolve_keys}")
        for t in self.n_equalto_keys:
            info(f"    The '{t}' density will be kept equal to {self.species_dict[t].density_equal_to_frac} * {self.species_dict[t].density_equal_to} density.")
        if self.qneut_species:
            info(f"    The '{self.qneut_species.type}' density will be set by quasineutrality.")
        info(f"    Zeff = {self.Zeff().profile}")
        info(f"    Evolving temperatures: {self.T_evolve_keys}")
        for t in self.T_equalto_keys:
            info(f"    The '{t}' temperature will be kept equal to {self.species_dict[t].temperature_equal_to_frac} * {self.species_dict[t].temperature_equal_to} temperature.")

        if self.has_adiabatic_species:
            info(f"    The '{self.adiabatic_species.type}' species will be treated adiabatically.")

        info(f"    Using '{self.ref_species.type}' as the reference species for turbulence calculations.")

        info(f"    Total number of (parallelizable) flux tube calculations per step = {(self.N_radial-1)*(1 + self.evolve_count)}.")

    def init_profiles_and_sources(self, geo, norms, imported=None):
        for s in self.get_species_list():
            s.init_profiles(self.grid, geo, norms, imported)

        # enforce equal_to densities and temperatures
        self.enforce_equal_to_density(fixed_T=True, init=True)
        self.enforce_equal_to_temperature(init=True)

        # if impurity species and Zeff_target, enforce
        if self.impurity is not None and self.impurity.Zeff_target is not None:
            self.enforce_Zeff_target(fixed_T=True)

            # re-enforce equal_to densities and temperatures, now scaled to give desired Zeff
            self.enforce_equal_to_density(fixed_T=True, init=True)
            self.enforce_equal_to_temperature(init=True)

        # ensure that initial profiles satisfy quasineutrality
        self.enforce_quasineutrality(fixed_T=True)

        if self.evolve_count > 0:
            self.nt_vec = self.get_vec_from_profs()

        self.norms = norms

        self.Paux = GridProfile(0, self.grid)
        for s in self.get_species_list():
            s.init_aux_sources(self.grid, geo, norms, imported)
            self.Paux += s.Sp_labeled['aux']

    def get_vec_from_profs(self):
        '''
        copy and concatenate values from evolving species profiles into a numpy vector
        '''
        nt_vec = []

        for t in self.n_evolve_keys:
            nt_vec.append(self.species_dict[t].n().profile)

        for t in self.T_evolve_keys:
            nt_vec.append(self.species_dict[t].p().profile)

        nt_vec = np.concatenate(nt_vec)

        return nt_vec

    def enforce_quasineutrality(self, fixed_T=False):
        charge_density = np.zeros(self.N_radial)

        # compute charge density without the quasineutral species
        for s in self.get_species_list():
            if s.type != self.qneut_species.type:
                charge_density = charge_density + s.Z*s.n()

        # use quasineutrality to set density of last evolved species.
        # this needs to happen outside the above loop to ensure charge_density
        # has been computed correctly.
        # note that the charge_density does not include a contribution from the 'quasineutral' species,
        # so that we can compute the following
        if self.qneut_species:
            T = self.qneut_species.T()
            self.qneut_species.set_n(-charge_density/self.qneut_species.Z)

            # keep temperature fixed
            if fixed_T:
                self.qneut_species.set_p(self.qneut_species.n()*T)

    def enforce_Zeff_target(self, fixed_T=False):
        # make sure non-impurity ions are initialized quasineutral
        charge_density = 0
        for s in self.get_species_list():
            if s.type != 'electron' and s.type != self.impurity.type:
                charge_density = charge_density + s.Z*s.n()
        assert np.allclose(charge_density, self.electron.n()), "When specifying Zeff for an impurity species in [[species]], other ion species densities must be initialized such that quasineutrality is satisfied with no impurities present (ne = sum_j Z_j n_j for non-impurity ions j)."

        Zeff = self.impurity.Zeff_target
        ZI = self.impurity.Z
        T = self.impurity.T()
        self.impurity.set_n((Zeff-1)/(ZI*(ZI-1)))
        if fixed_T:
            self.impurity.set_p(self.impurity.n()*T)

        ion_density_scale = 1 - (Zeff-1)/(ZI-1)
        for s in self.get_species_list():
            if s.type != 'electron' and s.type != self.impurity.type:
                T = s.T()
                s.set_n(s.n()*ion_density_scale)

                if s.density_equal_to == 'electron':
                    s.density_equal_to_frac *= ion_density_scale

                if s.density_init_to == 'electron':
                    s.density_init_to_frac *= ion_density_scale

                if fixed_T:
                    s.set_p(s.n()*T)

    def enforce_equal_to_density(self, fixed_T=False, init=False):
        for s in self.get_species_list():
            if s.density_equal_to is not None:
                T = s.T()
                n_eq = s.density_equal_to_frac * self[s.density_equal_to].n()
                s.n_prof = n_eq
                if fixed_T:
                    s.p_prof = s.n_prof * T

        if init:
            for s in self.get_species_list():
                if s.density_init_to is not None:
                    T = s.T()
                    n_eq = s.density_init_to_frac * self[s.density_init_to].n()
                    s.n_prof = n_eq
                    if fixed_T:
                        s.p_prof = s.n_prof * T

    def enforce_equal_to_temperature(self, init=False):
        for s in self.get_species_list():
            if s.temperature_equal_to is not None:
                p = s.p()
                T_eq = s.temperature_equal_to_frac * self[s.temperature_equal_to].T()
                s.p_prof = T_eq * s.n()
        if init:
            for s in self.get_species_list():
                if s.temperature_init_to is not None:
                    p = s.p()
                    T_eq = s.temperature_init_to_frac * self[s.temperature_init_to].T()
                    s.p_prof = T_eq * s.n()

    def get_profs_from_vec(self, nt_vec):
        '''
        copy values from nt_vec vector (only contains evolving species) into the species profiles
        '''
        offset = 0
        ndens = 0
        for s in self.get_species_list():
            if s.evolve_density:
                if ndens < self.N_dens_profiles:
                    s.n().profile[:-1] = nt_vec[offset:offset+self.N_radial-1]
                    offset = offset + self.N_radial
                ndens = ndens + 1
            else:
                s.n_prof = s.n_init

        self.enforce_equal_to_density()

        self.enforce_quasineutrality()

        for s in self.get_species_list():
            if s.evolve_temperature:
                s.p().profile[:-1] = nt_vec[offset:offset+self.N_radial-1]
                offset = offset + self.N_radial
            else:
                s.p_prof = s.T_init * s.n()

        self.enforce_equal_to_temperature()

    def get_profiles_on_grid(self, normalize=False, pert_n=None, pert_T=None, pert_idx=slice(None), rel_step=0.1):

        ns = np.zeros((self.N_species, self.N_radial))
        Ts = np.zeros((self.N_species, self.N_radial))
        df = np.zeros(self.N_radial)

        if normalize:
            # normalize to reference species values
            n_ref = self.ref_species.n()  # 10^20 m^-3
            T_ref = self.ref_species.T()  # keV

        for i, s in enumerate(self.get_species_list()):
            n = s.n()
            T = s.T()

            if normalize:
                n = n/n_ref
                T = T/T_ref

            # perturb density at fixed pressure
            if pert_n == s.type:
                df[pert_idx] = rel_step*n[pert_idx]
                p = n*T
                n += df
                T = p/n
            if pert_T == s.type:
                df[pert_idx] = rel_step*T[pert_idx]
                T += df

            ns[i,:] = n
            Ts[i,:] = T

        # if perturbing density, maintain quasineutrality by modifying density of species evolved by quasineutrality
        for i, s in enumerate(self.get_species_list()):
            for j in np.arange(len(n)):
                if self.qneut_species and s.type == self.qneut_species.type and pert_n is not None:
                    # perturb density at fixed pressure
                    ns[i,j] = n[j] - self[pert_n].Z*df[j]/s.Z
                    Ts[i,j] = T[j] + self[pert_n].Z*df[j]/s.Z

        return ns, Ts, df[pert_idx]

    def get_profiles_on_flux_grid(self, normalize=False, a_minor=1.0):

        ns = np.zeros((self.N_species, self.N_radial-1))
        Ts = np.zeros((self.N_species, self.N_radial-1))
        nus = np.zeros((self.N_species, self.N_radial-1))

        if normalize:
            # reference values for flux tube calculation
            n_ref = self.ref_species.n().toFluxProfile()  # 10^20 m^-3
            T_ref = self.ref_species.T().toFluxProfile()  # keV
            vt_ref = self.ref_species.norms.vT_ref*T_ref**0.5  # m/s

        for i, s in enumerate(self.get_species_list()):
            n = s.n().toFluxProfile()
            T = s.T().toFluxProfile()
            nu_ss = s.collision_frequency(s).toFluxProfile()

            if normalize:
                n = n/n_ref
                T = T/T_ref
                nu_ss = nu_ss*a_minor/vt_ref

            ns[i,:] = n
            Ts[i,:] = T
            nus[i,:] = nu_ss

        return ns, Ts, nus

    def get_grads_on_flux_grid(self, pert_n=None, pert_T=None, rel_step=0.5, abs_step=0.5):

        kns = np.zeros((self.N_species, self.N_radial-1))
        kts = np.zeros((self.N_species, self.N_radial-1))
        kps = np.zeros((self.N_species, self.N_radial-1))
        dkap = np.zeros(self.N_radial-1)

        for i, s in enumerate(self.get_species_list()):
            kn, kt, kp = s.get_fluxgrads()
            for j in np.arange(len(kn)):
                kns[i,j] = kn[j]
                kts[i,j] = kt[j]
                kps[i,j] = kp[j]
                if pert_n == s.type:
                    # perturb density gradient at fixed pressure gradient
                    dkap[j] = -max(rel_step*abs(kn[j]), abs_step)
                    kns[i,j] = kn[j] + dkap[j]
                    kts[i,j] = kt[j] - dkap[j]
                if pert_T == s.type:
                    dkap[j] = max(rel_step*abs(kt[j]), abs_step)
                    kts[i,j] = kt[j] + dkap[j]
                    kps[i,j] = kts[i,j] + kns[i,j]

        # if perturbing density, maintain quasineutrality of density gradient in species evolved by quasineutrality
        for i, s in enumerate(self.get_species_list()):
            kn, kt, kp = s.get_fluxgrads()
            for j in np.arange(len(kn)):
                if self.qneut_species and s.type == self.qneut_species.type and pert_n is not None:
                    # perturb density gradient at fixed pressure gradient
                    kns[i,j] = kn[j] - self[pert_n].Z*dkap[j]/s.Z
                    kts[i,j] = kt[j] + self[pert_n].Z*dkap[j]/s.Z

        return kns, kts, kps, dkap

    def get_pressure_SI(self):
        '''
        return Trinity pressure, summed over species, in SI
        This is used for launching Equilibrium updates
        '''
        p_ref = self.norms.p_ref
        species_list = self.get_species_list()
        p_SI = np.sum([s.p_prof for s in species_list], axis=0) * p_ref

        return p_SI

    def get_masses(self, normalize=False, drop=None):
        '''
        return a numpy array of species masses. masses will be normalized to reference species mass if normalize=True
        '''
        if drop is not None:
            ms = np.zeros(self.N_species-1)
        else:
            ms = np.zeros(self.N_species)
        if normalize:
            m_ref = self.ref_species.mass
        else:
            m_ref = 1.0

        i = 0
        for s in self.get_species_list():
            if s.type != drop:
                ms[i] = s.mass/m_ref
                i += 1

        return ms

    def get_charges(self, normalize=False, drop=None):
        '''
        return a numpy array of species charges. charges will be normalized to reference species charge if normalize=True
        '''
        if drop is not None:
            Zs = np.zeros(self.N_species-1)
        else:
            Zs = np.zeros(self.N_species)
        if normalize:
            Z_ref = self.ref_species.Z
        else:
            Z_ref = 1.0

        i = 0
        for s in self.get_species_list():
            if s.type != drop:
                Zs[i] = s.Z/Z_ref
                i += 1

        return Zs

    def Zeff(self):
        Zeff = 0
        for s in self.get_species_list():
            if s.type != 'electron':
                Zeff += s.Z**2*s.n()
        Zeff = Zeff / self['electron'].n()
        return Zeff

    def get_types_ion_electron(self):
        ie_type = []
        for s in self.get_species_list():
            if s.type == "electron":
                ie_type.append("electron")
            else:
                ie_type.append("ion")

        return ie_type

    def clear_fluxes(self):
        for s in self.get_species_list():
            s.pflux = FluxProfile(1e-16, self.grid)
            s.qflux = FluxProfile(1e-16, self.grid)
            s.heat = FluxProfile(1e-16, self.grid)
            for t in self.get_species_list():
                s.dpflux_dkn[t.type] = FluxProfile(1e-16, self.grid)
                s.dpflux_dkT[t.type] = FluxProfile(1e-16, self.grid)
                s.dqflux_dkn[t.type] = FluxProfile(1e-16, self.grid)
                s.dqflux_dkT[t.type] = FluxProfile(1e-16, self.grid)
                s.dheat_dkn[t.type] = FluxProfile(1e-16, self.grid)
                s.dheat_dkT[t.type] = FluxProfile(1e-16, self.grid)

    def add_flux(self, pflux_sj, qflux_sj, heat_sj=None, label=None):
        for i, s in enumerate(self.get_species_list()):
            s.pflux += FluxProfile(pflux_sj[i, :], self.grid)
            s.qflux += FluxProfile(qflux_sj[i, :], self.grid)
            if heat_sj is not None:
                s.heat += FluxProfile(heat_sj[i, :], self.grid)

            if label is not None:
                s.pflux_labeled[label] = FluxProfile(pflux_sj[i, :], self.grid)
                s.qflux_labeled[label] = FluxProfile(qflux_sj[i, :], self.grid)
                if heat_sj is not None:
                    s.heat_labeled[label] = FluxProfile(heat_sj[i, :], self.grid)
                else:
                    s.heat_labeled[label] = FluxProfile(0.0, self.grid)

    def add_dflux_dkn(self, stype, dpflux_dkn_sj, dqflux_dkn_sj, dheat_dkn_sj=None):
        for i, s in enumerate(self.get_species_list()):
            s.dpflux_dkn[stype] += FluxProfile(dpflux_dkn_sj[i, :], self.grid)
            s.dqflux_dkn[stype] += FluxProfile(dqflux_dkn_sj[i, :], self.grid)
            if dheat_dkn_sj is not None:
                s.dheat_dkn[stype] += FluxProfile(dheat_dkn_sj[i, :], self.grid)

    def add_dflux_dkT(self, stype, dpflux_dkT_sj, dqflux_dkT_sj, dheat_dkT_sj=None):
        for i, s in enumerate(self.get_species_list()):
            s.dpflux_dkT[stype] += FluxProfile(dpflux_dkT_sj[i, :], self.grid)
            s.dqflux_dkT[stype] += FluxProfile(dqflux_dkT_sj[i, :], self.grid)
            if dheat_dkT_sj is not None:
                s.dheat_dkT[stype] += FluxProfile(dheat_dkT_sj[i, :], self.grid)

    def clear_sources(self):
        for s in self.get_species_list():
            s.Sn = GridProfile(0, self.grid)
            s.Sp = GridProfile(0, self.grid)
            for t in self.get_species_list():
                s.dSn_dn[t.type] = GridProfile(0, self.grid)
                s.dSn_dT[t.type] = GridProfile(0, self.grid)
                s.dSp_dn[t.type] = GridProfile(0, self.grid)
                s.dSp_dT[t.type] = GridProfile(0, self.grid)

    def add_source(self, Sn_sj, Sp_sj, label=None):
        for i, s in enumerate(self.get_species_list()):
            s.Sn += GridProfile(Sn_sj[i, :], self.grid)
            s.Sp += GridProfile(Sp_sj[i, :], self.grid)

            if label is not None:
                s.Sn_labeled[label] = GridProfile(Sn_sj[i, :], self.grid)
                s.Sp_labeled[label] = GridProfile(Sp_sj[i, :], self.grid)

    def add_dS_dn(self, stype, dSn_dn_sj, dSp_dn_sj):
        for i, s in enumerate(self.get_species_list()):
            s.dSn_dn[stype] += GridProfile(dSn_dn_sj[i, :], self.grid)
            s.dSp_dn[stype] += GridProfile(dSp_dn_sj[i, :], self.grid)

    def add_dS_dT(self, stype, dSn_dT_sj, dSp_dT_sj):
        for i, s in enumerate(self.get_species_list()):
            s.dSn_dT[stype] += GridProfile(dSn_dT_sj[i, :], self.grid)
            s.dSp_dT[stype] += GridProfile(dSp_dT_sj[i, :], self.grid)

    def compute_psi_n_Fn(self, geo):
        # this is Barnes notes Eq (92), pflux (n equation) index
        # geo.F_fac = area/Btor**2 as FluxProfile
        # geo.G_fac = grho/area as GridProfile
        drho = self.grid.drho
        c_plus = self.c_plus
        c_minus = self.c_minus
        d_plus = self.d_plus
        d_minus = self.d_minus
        for s in self.get_species_list():
            s.psi_n_Fn = {}
            for sprime in self.n_evolve_keys:
                K = s.pflux
                dK_dkn = s.dpflux_dkn[sprime]
                Fplus = s.Fn_plus(self.ref_species, geo)
                Fminus = s.Fn_minus(self.ref_species, geo)
                kap_n, _, _ = self[sprime].get_fluxgrads()
                n = self[sprime].n()

                delta_sr = int(sprime == self.ref_species.type)

                s.psi_n_Fn[sprime] = np.zeros((self.N_radial, self.N_radial))

                A_fac = -0.5

                # loop over j radial grid points, not including last (fixed) grid point
                for j in np.arange(self.N_radial-1):
                    F_plus = Fplus[j]
                    F_minus = Fminus[j]
                    K_plus = K.plus()[j]
                    K_minus = K.minus()[j]
                    kap_n_plus = kap_n.plus()[j]
                    kap_n_minus = kap_n.minus()[j]
                    dK_dkn_plus = dK_dkn.plus()[j]
                    dK_dkn_minus = dK_dkn.minus()[j]
                    n_plus = n.plus()[j]
                    n_minus = n.minus()[j]

                    for k in range(max(0,j-2), min(j+4,self.N_radial)):
                        s.psi_n_Fn[sprime][j,k] = \
                            geo.G_fac[j]/drho*(
                                F_plus/K_plus*(delta_sr/n_plus*c_plus[j,k-j]*A_fac*K_plus - (kap_n_plus*c_plus[j,k-j] + d_plus[j,k-j]/drho)*dK_dkn_plus/n_plus) -
                                F_minus/K_minus*(delta_sr/n_minus*c_minus[j,k-j]*A_fac*K_minus - (kap_n_minus*c_minus[j,k-j] + d_minus[j,k-j]/drho)*dK_dkn_minus/n_minus))

    def compute_psi_p_Fn(self, geo):
        # this is Barnes notes Eq (93), pflux (n equation) index
        # geo.F_fac = area/Btor**2 as FluxProfile
        # geo.G_fac = grho/area as GridProfile
        drho = self.grid.drho
        c_plus = self.c_plus
        c_minus = self.c_minus
        d_plus = self.d_plus
        d_minus = self.d_minus
        for s in self.get_species_list():
            s.psi_p_Fn = {}
            for sprime in self.T_evolve_keys:
                K = s.pflux
                dK_dkT = s.dpflux_dkT[sprime]
                Fplus = s.Fn_plus(self.ref_species, geo)
                Fminus = s.Fn_minus(self.ref_species, geo)
                _, _, kap_p = self[sprime].get_fluxgrads()
                p = self[sprime].p()

                delta_sr = int(sprime == self.ref_species.type)

                s.psi_p_Fn[sprime] = np.zeros((self.N_radial, self.N_radial))

                B_fac = 1.5

                # loop over j radial grid points, not including last (fixed) grid point
                for j in np.arange(self.N_radial-1):
                    F_plus = Fplus[j]
                    F_minus = Fminus[j]
                    K_plus = K.plus()[j]
                    K_minus = K.minus()[j]
                    kap_p_plus = kap_p.plus()[j]
                    kap_p_minus = kap_p.minus()[j]
                    dK_dkT_plus = dK_dkT.plus()[j]
                    dK_dkT_minus = dK_dkT.minus()[j]
                    p_plus = p.plus()[j]
                    p_minus = p.minus()[j]

                    for k in range(max(0,j-2), min(j+4,self.N_radial)):
                        s.psi_p_Fn[sprime][j,k] = \
                            geo.G_fac[j]/drho*(
                                F_plus/K_plus*(delta_sr/p_plus*c_plus[j,k-j]*B_fac*K_plus - (kap_p_plus*c_plus[j,k-j] + d_plus[j,k-j]/drho)*dK_dkT_plus/p_plus) -
                                F_minus/K_minus*(delta_sr/p_minus*c_minus[j,k-j]*B_fac*K_minus - (kap_p_minus*c_minus[j,k-j] + d_minus[j,k-j]/drho)*dK_dkT_minus/p_minus))

    def compute_psi_n_Fp(self, geo, physics):
        # this is Barnes notes Eq (92), qflux (p equation) index
        # geo.F_fac = area/Btor**2 as FluxProfile
        # geo.G_fac = grho/area as GridProfile
        drho = self.grid.drho
        c_plus = self.c_plus
        c_minus = self.c_minus
        d_plus = self.d_plus
        d_minus = self.d_minus
        for s in self.get_species_list():
            s.psi_n_Fp = {}
            for sprime in self.n_evolve_keys:
                K = s.qflux
                H = s.heat.toGridProfile()
                dK_dkn = s.dqflux_dkn[sprime]
                dH_dkn = s.dheat_dkn[sprime].toGridProfile()
                Fplus = s.Fp_plus(self.ref_species, geo)
                Fminus = s.Fp_minus(self.ref_species, geo)
                kap_n, _, _ = self[sprime].get_fluxgrads()
                n = self[sprime].n()

                delta_sr = int(sprime == self.ref_species.type)

                s.psi_n_Fp[sprime] = np.zeros((self.N_radial, self.N_radial))
                if physics.collisions:
                    dcolleq_dn = s.dcolleq_dn(self[sprime], self)
                if physics.alpha_heating:
                    dPalpha_dn = s.dPalpha_dn(sprime, self, physics)
                if physics.radiation or physics.bremsstrahlung:
                    dPrad_dn = s.dPrad_dn(sprime, self, physics)

                A_fac = -1.5

                # loop over j radial grid points, not including last (fixed) grid point
                for j in np.arange(self.N_radial-1):
                    F_plus = Fplus[j]
                    F_minus = Fminus[j]
                    K_plus = K.plus()[j]
                    K_minus = K.minus()[j]
                    kap_n_plus = kap_n.plus()[j]
                    kap_n_minus = kap_n.minus()[j]
                    dK_dkn_plus = dK_dkn.plus()[j]
                    dK_dkn_minus = dK_dkn.minus()[j]
                    n_plus = n.plus()[j]
                    n_minus = n.minus()[j]
                    nref_j = self.ref_species.n()[j]
                    pref_j = self.ref_species.p()[j]
                    Btor = geo.Btor

                    for k in range(max(0,j-2), min(j+4,self.N_radial)):
                        s.psi_n_Fp[sprime][j,k] = \
                            geo.G_fac[j]/drho*(
                                F_plus/K_plus*(delta_sr/n_plus*c_plus[j,k-j]*A_fac*K_plus - (kap_n_plus*c_plus[j,k-j] + d_plus[j,k-j]/drho)*dK_dkn_plus/n_plus) -
                                F_minus/K_minus*(delta_sr/n_minus*c_minus[j,k-j]*A_fac*K_minus - (kap_n_minus*c_minus[j,k-j] + d_minus[j,k-j]/drho)*dK_dkn_minus/n_minus))
                        # need to add collision source terms (Barnes eq 99)
                        if physics.collisions and j == k:
                            s.psi_n_Fp[sprime][j,k] += -dcolleq_dn[j]
                        if physics.turbulent_exchange:
                            if j == k:
                                delta_jk = 1
                            else:
                                delta_jk = 0
                            s.psi_n_Fp[sprime][j,k] += -2./3.*pref_j**2.5/nref_j**1.5/Btor**2*(
                                1/n[j]*(delta_jk*kap_n[j] + self.b_coef[j,k-j]/drho)*dH_dkn[j] - 1.5*delta_jk*delta_sr*H[j]/nref_j)
                        if physics.alpha_heating and j == k:
                            s.psi_n_Fp[sprime][j,k] += -2./3.*dPalpha_dn[j]
                        if (physics.radiation or physics.bremsstrahlung) and j == k:
                            s.psi_n_Fp[sprime][j,k] += -2./3.*dPrad_dn[j]

    def compute_psi_p_Fp(self, geo, physics):
        # this is Barnes notes Eq (93), qflux (p equation) index
        # geo.F_fac = area/Btor**2 as FluxProfile
        # geo.G_fac = grho/area as GridProfile
        drho = self.grid.drho
        c_plus = self.c_plus
        c_minus = self.c_minus
        d_plus = self.d_plus
        d_minus = self.d_minus
        for s in self.get_species_list():
            s.psi_p_Fp = {}
            for sprime in self.T_evolve_keys:
                K = s.qflux
                H = s.heat.toGridProfile()
                dK_dkT = s.dqflux_dkT[sprime]
                dH_dkT = s.dheat_dkT[sprime].toGridProfile()
                Fplus = s.Fp_plus(self.ref_species, geo)
                Fminus = s.Fp_minus(self.ref_species, geo)
                _, _, kap_p = self[sprime].get_fluxgrads()
                p = self[sprime].p()

                delta_sr = int(sprime == self.ref_species.type)

                s.psi_p_Fp[sprime] = np.zeros((self.N_radial, self.N_radial))
                if physics.collisions:
                    dcolleq_dp = s.dcolleq_dp(self[sprime], self)
                if physics.alpha_heating:
                    dPalpha_dp = s.dPalpha_dp(sprime, self, physics)
                if physics.radiation or physics.bremsstrahlung:
                    dPrad_dp = s.dPrad_dp(sprime, self, physics)

                B_fac = 2.5

                # loop over j radial grid points, not including last (fixed) grid point
                for j in np.arange(self.N_radial-1):
                    F_plus = Fplus[j]
                    F_minus = Fminus[j]
                    K_plus = K.plus()[j]
                    K_minus = K.minus()[j]
                    kap_p_plus = kap_p.plus()[j]
                    kap_p_minus = kap_p.minus()[j]
                    dK_dkT_plus = dK_dkT.plus()[j]
                    dK_dkT_minus = dK_dkT.minus()[j]
                    p_plus = p.plus()[j]
                    p_minus = p.minus()[j]
                    nref_j = self.ref_species.n()[j]
                    pref_j = self.ref_species.p()[j]
                    Btor = geo.Btor

                    for k in range(max(0,j-2), min(j+4,self.N_radial)):
                        s.psi_p_Fp[sprime][j,k] = \
                            geo.G_fac[j]/drho*(
                                F_plus/K_plus*(delta_sr/p_plus*c_plus[j,k-j]*B_fac*K_plus - (kap_p_plus*c_plus[j,k-j] + d_plus[j,k-j]/drho)*dK_dkT_plus/p_plus) -
                                F_minus/K_minus*(delta_sr/p_minus*c_minus[j,k-j]*B_fac*K_minus - (kap_p_minus*c_minus[j,k-j] + d_minus[j,k-j]/drho)*dK_dkT_minus/p_minus))

                        # need to add collision source terms (Barnes eq 100)
                        if physics.collisions and j == k:
                            s.psi_p_Fp[sprime][j,k] += -dcolleq_dp[j]
                        if physics.turbulent_exchange:
                            if j == k:
                                delta_jk = 1
                            else:
                                delta_jk = 0
                            s.psi_p_Fp[sprime][j,k] += -2./3.*pref_j**2.5/nref_j**1.5/Btor**2*(
                                1/p[j]*(delta_jk*kap_p[j] + self.b_coef[j,k-j]/drho)*dH_dkT[j] + 2.5*delta_jk*delta_sr*H[j]/pref_j)
                        if physics.alpha_heating and j == k:
                            s.psi_p_Fp[sprime][j,k] += -2./3.*dPalpha_dp[j]
                        if (physics.radiation or physics.bremsstrahlung) and j == k:
                            s.psi_p_Fp[sprime][j,k] += -2./3.*dPrad_dp[j]

    def init_psi_matrices(self):
        # set up coefficient matrices used for calculating the psi matrices
        nrad = self.N_radial
        self.c_plus = np.zeros((self.N_radial-1, 6))
        self.c_minus = np.zeros((self.N_radial-1, 6))
        self.d_plus = np.zeros((self.N_radial-1, 6))
        self.d_minus = np.zeros((self.N_radial-1, 6))
        self.b_coef = np.zeros((self.N_radial-1, 6))

        # see Barnes notes, Table 1
        self.c_plus[1:nrad-2, -1] = -0.0625
        self.c_plus[nrad-2,   -1] = -0.125
        self.c_plus[0,         0] = 0.375
        self.c_plus[1:nrad-2,  0] = 0.5625
        self.c_plus[nrad-2,    0] = 0.75
        self.c_plus[0,         1] = 0.75
        self.c_plus[1:nrad-2,  1] = 0.5625
        self.c_plus[nrad-2,    1] = 0.375
        self.c_plus[0,         2] = -0.125
        self.c_plus[1:nrad-2,  2] = -0.0625

        self.c_minus[2:nrad-1, -2] = -0.0625
        self.c_minus[1,        -1] = 0.375
        self.c_minus[2:nrad-1, -1] = 0.5625
        self.c_minus[1,         0] = 0.75
        self.c_minus[2:nrad-1,  0] = 0.5625
        self.c_minus[1,         1] = -0.125
        self.c_minus[2:nrad-1,  1] = -0.0625

        # see Barnes notes, Table 2
        self.d_plus[nrad-2,    -2] = 1./24.
        self.d_plus[1:nrad-2,  -1] = 1./24.
        self.d_plus[nrad-2,    -1] = -0.125
        self.d_plus[0,          0] = -23./24.
        self.d_plus[1:nrad-2,   0] = -27./24.
        self.d_plus[nrad-2,     0] = -21./24.
        self.d_plus[0,          1] = 21./24.
        self.d_plus[1:nrad-2,   1] = 27./24.
        self.d_plus[nrad-2,     1] = 23./24.
        self.d_plus[0,          2] = 0.125
        self.d_plus[1:nrad-2,   2] = -1./24.
        self.d_plus[0,          3] = -1./24.

        self.d_minus[2:nrad-1,  -2] = 1./24.
        self.d_minus[1,         -1] = -23./24.
        self.d_minus[2:nrad-1,  -1] = -27./24.
        self.d_minus[1,          0] = 21./24.
        self.d_minus[2:nrad-1,   0] = 27./24.
        self.d_minus[1,          1] = 0.125
        self.d_minus[2:nrad-1,   1] = -1./24.
        self.d_minus[1,          2] = -1./24.

        # see Barnes notes, Table 3
        self.b_coef[2:nrad-2, -2] = 1./12.
        self.b_coef[nrad-2,   -2] = 1./16.
        self.b_coef[1,        -1] = -1./3.
        self.b_coef[2:nrad-2, -1] = -3./4.
        self.b_coef[nrad-2,   -1] = -1.
        self.b_coef[0,         0] = -11./6.
        self.b_coef[1,         0] = -1./2.
        self.b_coef[nrad-2,    0] = 1./2.
        self.b_coef[0,         1] = 3.
        self.b_coef[1,         1] = 1.
        self.b_coef[2:nrad-2,  1] = 3./4.
        self.b_coef[nrad-2,    1] = 1./3.
        self.b_coef[0,         2] = -3./2.
        self.b_coef[1,         2] = -1./6.
        self.b_coef[2:nrad-2,  2] = -1./12.
        self.b_coef[0,         3] = 1./3.

    def update_edge(self, physics):
        p_ratio_threshold = physics.edge_p_ratio
        p_edge_multiplier = physics.edge_p_multiplier
        n_ratio_threshold = physics.edge_n_ratio
        n_edge_multiplier = physics.edge_n_multiplier
        T_edge_max = physics.edge_T_max
        n_edge_max = physics.edge_n_max

        for s in self.get_species_list():
            p = s.p()
            n = s.n()

            # increase pressure (and temperature) at fixed density
            # activate if pressure peak exceeds threshold
            if p[0] / p[-1] > p_ratio_threshold:
                # stop if edge temp is too high
                if s.T()[-1] < T_edge_max:
                    s.p_prof[-1] *= p_edge_multiplier
                    # print("\n\n Changing T edge \n\n")

            # increase density at fixed temperature
            # activate if density peak exceeds threshold
            if n[0] / n[-1] > n_ratio_threshold:
                if n[-1] < n_edge_max:
                    s.n_prof[-1] *= n_edge_multiplier
                    s.p_prof[-1] *= n_edge_multiplier
                    # print("\n\n Changing n edge \n\n")

        # enforce equal_to temperature
        self.enforce_equal_to_temperature()

        # enforce equal_to density -at fixed temperature-
        self.enforce_equal_to_density(fixed_T=True)

    def update_source(self, physics, geometry):

        # if Q < 1 increase P_aux, if Q > 2 decrease P_aux

        # load options
        Q_threshold_low = physics.source_Q_threshold_low
        Q_threshold_high = physics.source_Q_threshold_high
        p_source_up_multiplier = physics.source_p_up_multiplier
        p_source_down_multiplier = physics.source_p_down_multiplier
        p_aux_max_MW = physics.source_p_aux_max_MW
        p_aux_min_MW = physics.source_p_aux_min_MW

        # load profiles
        species = self.get_species_list()
        grid = self.grid
        Sp_aux_MW = 0
        Sp_alpha_MW = 0
        for s in self.get_species_list():
            Sp_aux_MW += geometry.volume_integrate(s.Sp_labeled['aux'])*self.norms.P_ref_MWm3
            Sp_alpha_MW += geometry.volume_integrate(s.Sp_labeled['alpha'])*self.norms.P_ref_MWm3

        # compute Q
        P_fus = 5.03*Sp_alpha_MW
        Q = P_fus / Sp_aux_MW
        info(f"P_aux = {Sp_aux_MW:.2f} MW, P_fus = {P_fus:.2f} MW, Q = {Q:.2f}, n0 = {self['electron'].n_prof[0]:.2f}")

        # update
        for s in species:
            if Q < Q_threshold_low:
                if Sp_aux_MW < p_aux_max_MW:
                    # print("\n\n Aux heat increase\n\n")
                    s.Sp_labeled['aux'] *= p_source_up_multiplier

            if Q > Q_threshold_high:
                if Sp_aux_MW > p_aux_min_MW:
                    s.Sp_labeled['aux'] *= p_source_down_multiplier
                    # print("\n\n Aux heat decrease\n\n")


class Species():

    def __init__(self, sp, grid):

        self.type = sp.get('type', "deuterium")
        self.is_adiabatic = sp.get('adiabatic', False)
        self.bulk_ion = sp.get('bulk', False)
        self.use_as_reference = sp.get('use_as_reference', self.bulk_ion)
        self.Zeff_target = sp.get('Zeff', None)
        self.impurity = sp.get('impurity', self.Zeff_target is not None)
        self.qneut = sp.get('qneut_species', False)
        density_parameters = sp.get('density', {})
        temperature_parameters = sp.get('temperature', {})

        # flags controlling density and temperature evolution
        self.evolve_density = density_parameters.get('evolve', True)
        self.evolve_temperature = temperature_parameters.get('evolve', True)
        self.density_equal_to = density_parameters.get('equal_to', None)
        self.temperature_equal_to = temperature_parameters.get('equal_to', None)
        self.density_init_to = density_parameters.get('init_to', None)
        self.temperature_init_to = temperature_parameters.get('init_to', None)
        # note: the density_equal_to, temperature_equal_to, density_init_to, temperature_init_to parameters can be
        # either strings corresponding to a species, e.g.
        # equal_to = "electron"
        # or a string containing a multiplicative fraction times the species, e.g.
        # equal_to = "0.5 * electron"
        # the following function parses the string to handle both cases

        # thank you ChatGPT for helping write this function...
        def parse_string(input_str):
            # Define a regular expression pattern to match the numeric value and the unit
            pattern = r'(\d+(\.\d+)?([eE][+-]?\d+)?)\s*\*\s*(\w+)'

            # Use re.match to search for the pattern in the input string
            match = re.match(pattern, input_str)
            match2 = re.match(r'\w+', input_str)  # handle case with no multiplier

            if match:
                # Extract the numeric value and unit from the matched groups
                value = float(match.group(1))
                unit = match.group(4)

                return [value, unit]
            elif match2:
                value = 1.0
                unit = match2.group(0)
                return [value, unit]
            else:
                assert False, f"Error parsing string \"{input_str}\""

        # the init_to parameter can be used to initialize a profile
        # as some fraction of another profile. after initialization
        # the profiles will evolve independently
        if self.density_init_to is not None:
            (self.density_init_to_frac, self.density_init_to) = parse_string(self.density_init_to)
        if self.temperature_init_to is not None:
            (self.temperature_init_to_frac, self.temperature_init_to) = parse_string(self.temperature_init_to)

        # the equal_to parameter can be used to maintain a profile
        # as some fraction of another profile, even as the system evolves
        if self.density_equal_to is not None:
            (self.density_equal_to_frac, self.density_equal_to) = parse_string(self.density_equal_to)
        if self.temperature_equal_to is not None:
            (self.temperature_equal_to_frac, self.temperature_equal_to) = parse_string(self.temperature_equal_to)

        self.type_tag = sp.get('tag') or self.get_type_tag()  # for diagnostic output

        # physical parameters
        self.mass = sp.get('mass') or self.get_mass()  # mass in units of proton mass
        self.Z = sp.get('Z') or self.get_charge()      # charge in units of e

        # initial profiles
        # density parameters in units of 10^20 m^-3
        self.n_shape = density_parameters.get('shape', 'parabolic')
        # PARABOLIC PARAMETERS
        self.n_core = density_parameters.get('core', 4)
        self.n_volavg = density_parameters.get('volavg', None)
        self.n_lineavg = density_parameters.get('lineavg', None)
        self.n_alpha = density_parameters.get('alpha', 1)
        self.n_alpha1 = density_parameters.get('alpha1', 2)
        self.n_edge = density_parameters.get('edge', None)
        self.n_sep = density_parameters.get('sep', None)
        # GAUSSIAN
        self.n_height = density_parameters.get('height', 0)
        self.n_int = density_parameters.get('integrated', None)
        self.n_width = density_parameters.get('width', 0.1)
        self.n_center = density_parameters.get('center', 0)
        # EXPONENTIAL
        self.n_coefs = density_parameters.get('coefs', [0,0,0,0])
        if len(self.n_coefs) < 4:  # pad with zeros
            self.n_coefs.extend([0] * (4 - len(self.n_coefs)))
        # IMPORT
        self.n_import = density_parameters.get('import', False)
        self.n_import_key = density_parameters.get('key', None)
        # MANUAL
        self.n_profstr = density_parameters.get('profstr', None)
        # CUSTOM
        if self.n_shape == 'custom':
            # this allows a python function 'func' to be specified in the input file.
            # for example,
            # [[species]]
            # ...
            # density = {shape = 'custom', func = '''
            # def init(rho):
            #     return (1-rho**2)
            # ''', evolve = true}
            # ...
            script = density_parameters.get('func')
            namespace = {}
            exec(script, namespace)
            self.n_init_func = namespace.get('init')

        # temperature parameters in units of keV
        self.T_shape = temperature_parameters.get('shape', 'parabolic')
        # PARABOLIC
        self.T_core = temperature_parameters.get('core', 8)
        self.T_volavg = temperature_parameters.get('volavg', None)
        self.T_lineavg = temperature_parameters.get('lineavg', None)
        self.T_alpha = temperature_parameters.get('alpha', 1)
        self.T_alpha1 = temperature_parameters.get('alpha1', 2)
        self.T_edge = temperature_parameters.get('edge', None)
        self.T_sep = temperature_parameters.get('sep', None)
        # GAUSSIAN
        self.T_height = temperature_parameters.get('height', 0)
        self.T_int = temperature_parameters.get('integrated', None)
        self.T_width = temperature_parameters.get('width', 0.1)
        self.T_center = temperature_parameters.get('center', 0)
        # EXPONENTIAL
        self.T_coefs = temperature_parameters.get('coefs', [0,0,0,0])
        if len(self.T_coefs) < 4:  # pad with zeros
            self.T_coefs.extend([0] * (4 - len(self.T_coefs)))
        # IMPORT
        self.T_import = temperature_parameters.get('import', False)
        self.T_import_key = temperature_parameters.get('key', None)
        # MANUAL
        self.T_profstr = temperature_parameters.get('profstr', None)
        # CUSTOM
        if self.T_shape == 'custom':
            # this allows a python function 'func' to be specified in the input file.
            # for example,
            # [[species]]
            # ...
            # temperature = {shape = 'custom', func = '''
            # def init(rho):
            #     return (1-rho**2)
            # ''', evolve = true}
            # ...
            script = temperature_parameters.get('func')
            namespace = {}
            exec(script, namespace)
            self.T_init_func = namespace.get('init')

        # sources
        # density sources in units of 10^20 m^-3 s^-1
        # can be specified in input file as 'aux_particle_source' or 'density_source'
        density_source = sp.get('aux_particle_source', {})
        density_source = sp.get('density_source', density_source)  # deprecate?
        self.Sn_aux_import = density_source.get('import', False)
        self.Sn_aux_import_keys = density_source.get('keys', None)
        # set some smart defaults
        self.Sn_aux_shape = density_source.get('shape', 'gaussian')
        # parameters for gaussian shape
        self.Sn_aux_height = density_source.get('height', 0)  # units of 10^20 m^-3 s^-1
        self.Sn_aux_int = density_source.get('integrated', None)  # units of 10^20 s^-1
        self.Sn_aux_width = density_source.get('width', 0.1)
        self.Sn_aux_center = density_source.get('center', 0)
        # pressure sources in units of MW/m^3
        # can be specified in input file as 'aux_power_source' or 'pressure_source'
        pressure_source = sp.get('aux_power_source', {})
        pressure_source = sp.get('pressure_source', pressure_source)  # deprecate?
        self.Sp_aux_import = pressure_source.get('import', False)
        self.Sp_aux_import_keys = pressure_source.get('keys', None)
        self.Sp_aux_shape = pressure_source.get('shape', 'gaussian')
        self.Sp_aux_height = pressure_source.get('height', 0)  # units of MW/m^3
        self.Sp_aux_int = pressure_source.get('integrated', None)  # units of MW
        self.Sp_aux_width = pressure_source.get('width', 0.1)
        self.Sp_aux_center = pressure_source.get('center', 0)

        # parameters for ReLU flux model only
        n_relu = sp.get('density_relu_flux', {})
        self.n_relu_critical_gradient0 = n_relu.get('critical_gradient', 1.0)
        self.n_relu_critical_gradient0 = n_relu.get('critical_gradient0', self.n_relu_critical_gradient0)
        self.n_relu_critical_gradient1 = n_relu.get('critical_gradient1', 0.0)
        self.n_relu_flux_slope = n_relu.get('slope', 0.5)
        self.n_relu_noise_frac = n_relu.get('noise_frac', 0.0)
        p_relu = sp.get('pressure_relu_flux', {})
        self.p_relu_critical_gradient0 = p_relu.get('critical_gradient', 2.0)
        self.p_relu_critical_gradient0 = p_relu.get('critical_gradient0', self.p_relu_critical_gradient0)
        self.p_relu_critical_gradient1 = p_relu.get('critical_gradient1', 0.0)
        self.p_relu_flux_slope = p_relu.get('slope', 0.5)
        self.p_relu_flux_slope0 = p_relu.get('slope0', self.p_relu_flux_slope)
        self.p_relu_flux_slope1 = p_relu.get('slope1', 0.0)
        self.p_relu_noise_frac = p_relu.get('noise_frac', 0.0)

        # init flux profiles with zeros. these will be set by SpeciesDict.add_flux
        self.pflux = FluxProfile(0, grid)
        self.qflux = FluxProfile(0, grid)
        self.heat = FluxProfile(0, grid)

        # init flux jacobians as empty dictionaries. these will be set by SpeciesDict.add_dflux_dk*
        self.dpflux_dkn = {}
        self.dqflux_dkn = {}
        self.dheat_dkn = {}
        self.dpflux_dkT = {}
        self.dqflux_dkT = {}
        self.dheat_dkT = {}

        # labeled flux profiles for diagnostics. initialize as empty dictionaries. these will be set by SpeciesDict.add_fluxes
        self.pflux_labeled = {}
        self.qflux_labeled = {}
        self.heat_labeled = {}

        # init source profiles with zeros. these will be set by SpeciesDict.add_sources
        self.Sn = GridProfile(0, grid)
        self.Sp = GridProfile(0, grid)

        self.Sn_labeled = {}
        self.Sn_labeled['aux'] = GridProfile(0, grid)

        self.Sp_labeled = {}
        self.Sp_labeled['aux'] = GridProfile(0, grid)
        self.Sp_labeled['alpha'] = GridProfile(0, grid)
        self.Sp_labeled['rad'] = GridProfile(0, grid)
        self.Sp_labeled['heating'] = GridProfile(0, grid)
        self.Sp_labeled['coll'] = GridProfile(0, grid)

        # init source jacobians as empty dictionaries
        self.dSn_dn = {}
        self.dSn_dT = {}
        self.dSp_dn = {}
        self.dSp_dT = {}

    def init_profiles(self, grid, geo, norms, imported=None):
        # import initial density from file
        if self.n_import:
            assert imported.imported is not None, 'Error: must specify [import] section to use import = true for density'
            n = imported.get_density(key=self.n_import_key, species_mass=self.mass, species_charge=self.Z, tag=self.type_tag, n_edge=self.n_edge, norms=norms)
            self.n_prof = GridProfile(n, grid)
        # use analytic profile for initial density
        elif self.n_shape == 'parabolic':
            if self.n_edge is None:
                self.n_edge = 0.5  # default value

            if self.n_lineavg:
                n_prof = GridProfile((1 - (grid.rho/grid.rho_edge)**self.n_alpha1)**self.n_alpha, grid)
                self.n_prof = (self.n_lineavg - self.n_edge) / np.average(n_prof) * n_prof + self.n_edge
            elif self.n_volavg:
                n_prof = GridProfile((1 - (grid.rho/grid.rho_edge)**self.n_alpha1)**self.n_alpha, grid)
                self.n_prof = (self.n_volavg - self.n_edge) / geo.volume_average(n_prof) * n_prof + self.n_edge
            elif self.n_core and self.n_sep:
                self.n_prof = GridProfile((self.n_core - self.n_sep)*(1 - (grid.rho)**self.n_alpha1)**self.n_alpha + self.n_sep, grid)
            elif self.n_core:
                self.n_prof = GridProfile((self.n_core - self.n_edge)*(1 - (grid.rho/grid.rho_edge)**self.n_alpha1)**self.n_alpha + self.n_edge, grid)
        elif self.n_shape == 'gaussian':
            Gaussian = np.vectorize(self.Gaussian)
            info(f'Density is Gaussian with width {self.n_width}, center {self.n_center}, height {self.n_height}, int {self.n_int}')
            # particle source
            n_prof = GridProfile(Gaussian(grid.rho, A=1, sigma=self.n_width, x0=self.n_center), grid)  # Trinity units
            if self.n_int:
                info(f'Density using target int {self.n_int}')
                target = self.n_int
                volume_integral = geo.volume_integrate(n_prof, normalized=False, interpolate=False)
                self.n_prof = (target / volume_integral) * n_prof
            else:
                info(f'Density using height {self.n_height}')
                self.n_prof = self.n_height * n_prof
        elif self.n_shape == 'exponential':
            coefs = self.n_coefs
            x = grid.rho
            self.n_prof = GridProfile(coefs[0]*np.exp(-coefs[1]*x**2 - coefs[2]*x**4 - coefs[3]*x**6), grid)
            if self.n_edge:
                self.n_prof = self.n_prof - self.n_prof[-1] + self.n_edge
        elif self.n_shape == 'manual':
            from scipy.interpolate import PchipInterpolator
            profXY = np.array(np.mat(self.n_profstr))
            profX,profY = profXY
            self.n_prof = GridProfile(PchipInterpolator(profX,profY)(grid.rho/grid.rho_edge), grid)
        elif self.n_shape == 'custom':
            self.n_prof = GridProfile(self.n_init_func(grid.rho), grid)
            del self.n_init_func  # this is necessary to allow Species class to be pickled

            # scale line-average or volume-average
            # note: n_edge (if present) will not be scaled
            if self.n_edge is None:
                self.n_edge = 0.0  # default value
            if self.n_lineavg:
                self.n_prof = (self.n_lineavg - self.n_edge) / np.average(self.n_prof) * self.n_prof + self.n_edge
            elif self.n_volavg:
                self.n_prof = (self.n_volavg - self.n_edge) / geo.volume_average(self.n_prof) * self.n_prof + self.n_edge
        else:
            raise ValueError('Invalid n profile shape in Trinity input')

        # import initial temperature from file
        if self.T_import:
            assert imported.imported is not None, 'Error: must specify [import] section to use import = true for temperature'
            T = imported.get_temperature(key=self.T_import_key, species_mass=self.mass, species_charge=self.Z, tag=self.type_tag, T_edge=self.T_edge, norms=norms)
            T_prof = GridProfile(T, grid)
        # use analytic profile for initial temperature
        elif self.T_shape == 'parabolic':
            if self.T_edge is None:
                self.T_edge = 2  # default value

            if self.T_lineavg:
                T_prof = GridProfile((1 - (grid.rho/grid.rho_edge)**self.T_alpha1)**self.T_alpha, grid)
                T_prof = (self.T_lineavg - self.T_edge) / np.average(T_prof) * T_prof + self.T_edge
            elif self.T_volavg:
                T_prof = GridProfile((1 - (grid.rho/grid.rho_edge)**self.T_alpha1)**self.T_alpha, grid)
                T_prof = (self.T_volavg - self.T_edge) / geo.volume_average(T_prof) * T_prof + self.T_edge
            elif self.T_core and self.T_sep:
                T_prof = GridProfile((self.T_core - self.T_sep)*(1 - (grid.rho)**self.T_alpha1)**self.T_alpha + self.T_sep, grid)
            elif self.T_core:
                T_prof = GridProfile((self.T_core - self.T_edge)*(1 - (grid.rho/grid.rho_edge)**self.T_alpha1)**self.T_alpha + self.T_edge, grid)
        elif self.T_shape == 'gaussian':
            Gaussian = np.vectorize(self.Gaussian)
            info(f'Temperature is Gaussian with width {self.T_width}, center {self.T_center}, height {self.T_height}, int {self.T_int}')
            # particle source
            T_prof = GridProfile(Gaussian(grid.rho, A=1, sigma=self.T_width, x0=self.T_center), grid)  # Trinity units
            if self.T_int:
                info(f'Temperature using target int {self.T_int}')
                target = self.T_int
                volume_integral = geo.volume_integrate(T_prof, normalized=False, interpolate=False)
                T_prof = (target / volume_integral) * T_prof
            else:
                info(f'Temperature using height {self.T_height}')
                T_prof = self.T_height * T_prof
        elif self.T_shape == 'exponential':
            coefs = self.T_coefs
            x = grid.rho
            T_prof = GridProfile(coefs[0]*np.exp(-coefs[1]*x**2 - coefs[2]*x**4 - coefs[3]*x**6), grid)
            if self.T_edge:
                T_prof = T_prof - T_prof[-1] + self.T_edge
        elif self.T_shape == 'manual':
            from scipy.interpolate import PchipInterpolator
            profXY = np.array(np.mat(self.T_profstr))
            profX,profY = profXY
            T_prof = GridProfile(PchipInterpolator(profX,profY)(grid.rho/grid.rho_edge), grid)
        elif self.T_shape == 'custom':
            T_prof = GridProfile(self.T_init_func(grid.rho), grid)
            del self.T_init_func  # this is necessary to allow Species class to be pickled

            # scale line-average or volume-average
            # note: T_edge (if present) will not be scaled
            if self.T_edge is None:
                self.T_edge = 0.0  # default value.
            if self.T_lineavg:
                T_prof = (self.T_lineavg - self.T_edge) / np.average(T_prof) * T_prof + self.T_edge
            elif self.T_volavg:
                T_prof = (self.T_volavg - self.T_edge) / geo.volume_average(T_prof) * T_prof + self.T_edge
        else:
            raise ValueError('Invalid T profile shape in Trinity input')

        # pressure is quantity that is actually stored (not temperature)
        self.p_prof = self.n_prof * T_prof

        # save a copy of initial profiles separately
        self.n_init = self.n_prof
        self.T_init = T_prof
        self.p_init = self.p_prof

        self.norms = norms

    def n(self):
        return self.n_prof

    def p(self):
        return self.p_prof

    def T(self):
        return self.p_prof / self.n_prof

    def beta(self, Btor):
        # compute beta
        # p_prof is in units of 10^20 m^-3*keV
        # Btor is in units of T

        # convert p and B to cgs
        p_cgs = self.p_prof*1e17
        B_cgs = Btor*1e4
        return 4.03e-11*p_cgs/(B_cgs*B_cgs)

    def nu_equil(self, other):
        s = self
        u = other
        nu_su = self.collision_frequency(u)
        return self.norms.t_ref*8./(3*np.sqrt(np.pi))*nu_su*(u.mass/s.mass)**0.5*(u.mass/s.mass + u.T()/s.T())**-1.5

    def collision_frequency(self, other):
        # recall:
        # n_prof is in units of 10^20 m^-3
        # T_prof is in units of keV

        Z_s = self.Z
        m_s = self.mass
        n_s = self.n()  # 10^20 m^-3
        T_s = self.T()  # keV

        Z_u = other.Z
        m_u = other.mass
        n_u = other.n()  # 10^20 m^-3
        T_u = other.T()  # keV

        logLambda = self.logLambda(other)

        # nu_su = 4*pi*(q_s*q_u)**2*n_u*logLambda / (m_s**0.5*(2*T_s)**1.5)
        nu = 2.85e2*(Z_s*Z_u)**2/m_s**0.5*logLambda*n_u/T_s**1.5  # 1/s

        return nu

    def logLambda(self, other):
        # recall:
        # n_prof is in units of 10^20 m^-3
        # T_prof is in units of keV
        # the below formula is from the NRL formulary, which is in cgs units (except for temperatures in eV)

        Z_s = self.Z
        m_s = self.mass*m_proton_cgs
        n_s = self.n().profile*1e14  # 10^20 m^-3 -> cm^-3
        T_s = self.T().profile*1e3  # keV -> eV

        Z_u = other.Z
        m_u = other.mass*m_proton_cgs
        n_u = other.n().profile*1e14  # 10^20 m^-3 -> cm^-3
        T_u = other.T().profile*1e3  # keV -> eV

        if self.type == "electron" and other.type == "electron":  # ee
            lamb = 23.5 - np.log(n_s**0.5 / T_s**1.25) \
                - np.sqrt(1e-5 + (np.log(T_s) - 2)**2 / 16)
        elif self.type == "electron":  # ei
            lamb = 24.0 - np.log(n_s**0.5 / T_s)
        elif other.type == "electron":  # ie
            lamb = 24.0 - np.log(n_u**0.5 / T_u)
        else:  # ii
            lamb = 23.0 - np.log(
                (Z_s * Z_u) * (m_s + m_u) /
                (m_s * T_u + m_u * T_s) *
                (n_s * Z_s**2 / T_s + n_u * Z_u**2 / T_u)**0.5)
        return lamb

    def set_n(self, n, fixed_T=False):
        T = self.T()
        self.n_prof = n
        if fixed_T:
            self.p_prof = self.n_prof * T

    def set_T(self, T):
        self.p_prof = self.n_prof*T

    def set_p(self, p):
        self.p_prof = p

    def get_fluxgrads(self):
        kap_n_grid = -1*self.n_prof.log_gradient()
        kap_p_grid = -1*self.p_prof.log_gradient()
        kap_T_grid = kap_p_grid - kap_n_grid

        return kap_n_grid.toFluxProfile(), kap_T_grid.toFluxProfile(), kap_p_grid.toFluxProfile()

    # flux vector F_n (normalized particle flux)
    def Fn_plus(self, ref_species, geo, label=None):
        # returns FluxProfile with fluxes evaluated at j+1/2
        if label is not None:
            pflux = self.pflux_labeled[label]
        else:
            pflux = self.pflux
        if self.evolve_density:
            return (geo.area/geo.Btor**2).plus()*(ref_species.p()**1.5/ref_species.n()**0.5).plus()*pflux.plus()
        else:
            return 0*pflux

    def Fn_minus(self, ref_species, geo, label=None):
        # returns FluxProfile with fluxes evaluated at j-1/2
        if label is not None:
            pflux = self.pflux_labeled[label]
        else:
            pflux = self.pflux
        if self.evolve_density:
            return (geo.area/geo.Btor**2).minus()*(ref_species.p()**1.5/ref_species.n()**0.5).minus()*pflux.minus()
        else:
            return 0*pflux

    def Fn_grid(self, ref_species, geo, label=None):
        # returns GridProfile with fluxes evaluated at j
        if label is not None:
            pflux = self.pflux_labeled[label]
        else:
            pflux = self.pflux
        if self.evolve_density:
            return (geo.area/geo.Btor**2*pflux).toGridProfile()*(ref_species.p()**1.5/ref_species.n()**0.5)
        else:
            return 0*pflux.toGridProfile()

    # flux vector F_p (normalized heat flux)
    def Fp_plus(self, ref_species, geo, label=None):
        # returns FluxProfile with fluxes evaluated at j+1/2
        if label is not None:
            qflux = self.qflux_labeled[label]
        else:
            qflux = self.qflux
        if self.evolve_temperature:
            return 2./3.*(geo.area/geo.Btor**2).plus()*(ref_species.p().plus()**2.5/ref_species.n().plus()**1.5)*qflux.plus()
        else:
            return 0*qflux

    def Fp_minus(self, ref_species, geo, label=None):
        # returns FluxProfile with fluxes evaluated at j-1/2
        if label is not None:
            qflux = self.qflux_labeled[label]
        else:
            qflux = self.qflux
        if self.evolve_temperature:
            return 2./3.*(geo.area/geo.Btor**2).minus()*(ref_species.p()**2.5/ref_species.n()**1.5).minus()*qflux.minus()
        else:
            return 0*qflux

    def Fp_grid(self, ref_species, geo, label=None):
        # returns GridProfile with fluxes evaluated at j
        if label is not None:
            qflux = self.qflux_labeled[label]
        else:
            qflux = self.qflux
        if self.evolve_temperature:
            return 2./3.*(geo.area/geo.Btor**2*qflux).toGridProfile()*(ref_species.p()**2.5/ref_species.n()**1.5)
        else:
            return 0*qflux.toGridProfile()

    def coll_eq(self, species):
        s = self
        coll = 0*s.n()
        for u in species.get_species_list():
            coll += s.p()*s.nu_equil(u)*(s.n()*u.p()/(u.n()*s.p()) - 1)
        return coll

    def dcolleq_dn(self, sprime, species):
        s = self
        dcoll = s.nu_equil(sprime)*sprime.T()*s.n()/sprime.n()*(
            1.5/(sprime.mass/s.mass + sprime.T()/s.T())*(sprime.T()/s.T()-1) -
            s.T()/sprime.T())
        if s.type == sprime.type:
            for u in species.get_species_list():
                dcoll += s.nu_equil(u)*u.T()*(2.5 - 1.5*s.T()/u.T() + 1.5/(u.mass/s.mass + u.T()/s.T())*(1 - u.T()/s.T()))

        # contributions from d(logLambda)/dn
        if s.type == sprime.type:
            for u in species.get_species_list():
                dcoll += -s.nu_equil(u)*s.n()*s.T() \
                    * (u.T()/s.T()-1.0) / s.logLambda(u) \
                    * (s.Z**2/(s.T()*(s.n()*s.Z**2/s.T() +
                                      u.n()*u.Z**2/u.T())) +
                       u.mass/s.mass/(s.n()*(u.mass/s.mass+u.T()/s.T())))
        else:
            u = sprime
            dcoll += -s.nu_equil(u)*s.p()*(u.T()/s.T()-1) / s.logLambda(u) * (
                u.Z**2/(u.T()*(s.n()*s.Z**2/s.T() + u.n()*u.Z**2/u.T())) +
                u.T()/s.T()/(u.n()*(u.T()/s.T() + u.mass/s.mass)))
            dcoll += -s.nu_equil(u)*s.n()*s.T() \
                * (u.T()/s.T()-1.0) / s.logLambda(u) \
                * (s.Z**2/(u.T()*(s.n()*s.Z**2/s.T() +
                                  u.n()*u.Z**2/u.T())) +
                   u.T()/s.T()/(u.n()*(u.T()/s.T()+u.mass/s.mass)))

        return dcoll

    def dcolleq_dp(self, sprime, species):
        s = self
        dcoll = s.nu_equil(sprime)*s.n()/sprime.n()*(s.n()/sprime.n() - 1.5*(sprime.T()/s.T()-1)/(sprime.mass/s.mass + sprime.T()/s.T()))

        if s.type == sprime.type:
            for u in species.get_species_list():
                dcoll += s.nu_equil(u)*(-1 + 1.5*(u.T()/s.T()-1)*(u.T()/s.T()/(u.mass/s.mass + u.T()/s.T())-1))

        # contributions from d(logLambda)/dn
        if s.type == sprime.type:
            for u in species.get_species_list():
                dcoll += s.nu_equil(u) \
                    * (u.T()/s.T()-1.0) / s.logLambda(u) \
                    * (u.T()/s.T()*0.5 /
                       (u.T()/s.T()+u.n()/s.n()*(u.Z/s.Z)**2) +
                       (u.mass/s.mass)/(u.T()/s.T()+u.mass/s.mass))
        else:
            u = sprime
            dcoll += s.nu_equil(u) \
                * (u.T()/s.T()-1.0) / s.logLambda(u) \
                * (0.5/(u.T()/s.T()*((s.Z/u.Z)**2*u.T()/s.T() + u.n()/s.n())) +
                   s.n()/(u.n()*(u.T()/s.T()+u.mass/s.mass)))

        return dcoll

    def dPalpha_dn(self, sprime_key, species, physics, dn_frac=0.01):
        # compute dPalpha_s/dn_s' using finite difference
        s = self
        pert = copy.deepcopy(species)
        dn = dn_frac*pert[sprime_key].n_prof
        pert[sprime_key].n_prof += dn
        dPalpha = (physics.compute_alpha_heating(s, pert) - physics.compute_alpha_heating(s, species))/dn
        return dPalpha

    def dPalpha_dp(self, sprime_key, species, physics, dp_frac=0.01):
        # compute dPalpha_s/dp_s' using finite difference
        s = self
        pert = copy.deepcopy(species)
        dp = dp_frac*pert[sprime_key].p_prof
        pert[sprime_key].p_prof += dp
        dPalpha = (physics.compute_alpha_heating(s, pert) - physics.compute_alpha_heating(s, species))/dp
        return dPalpha

    def dPrad_dn(self, sprime_key, species, physics, dn_frac=0.01):
        # compute dPrad_s/dn_s' using finite difference
        s = self
        pert = copy.deepcopy(species)
        dn = dn_frac*pert[sprime_key].n_prof
        pert[sprime_key].n_prof += dn
        if physics.radiation:
            dPrad = -(physics.compute_radiation(s, pert) - physics.compute_radiation(s, species))/dn
        elif physics.bremsstrahlung:
            dPrad = -(physics.compute_bremsstrahlung(s, pert) - physics.compute_bremsstrahlung(s, species))/dn
        return dPrad

    def dPrad_dp(self, sprime_key, species, physics, dp_frac=0.01):
        # compute dPrad_s/dp_s' using finite difference
        s = self
        pert = copy.deepcopy(species)
        dp = dp_frac*pert[sprime_key].p_prof
        pert[sprime_key].p_prof += dp
        if physics.radiation:
            dPrad = -(physics.compute_radiation(s, pert) - physics.compute_radiation(s, species))/dp
        elif physics.bremsstrahlung:
            dPrad = -(physics.compute_bremsstrahlung(s, pert) - physics.compute_bremsstrahlung(s, species))/dp
        return dPrad

    def init_aux_sources(self, grid, geo, norms, imported=None):
        # sources
        # particle source
        if not self.evolve_density:
            self.Sn_labeled['aux'] = GridProfile(0, grid)
        elif self.Sn_aux_import:
            assert imported.imported is not None, 'Error: must specify [import] section to use import = true for density_source'
            aux_source_n = imported.get_density_source(keys=self.Sn_aux_import_keys, species_mass=self.mass, species_charge=self.Z, tag=self.type_tag, norms=norms)
            self.Sn_labeled['aux'] = GridProfile(aux_source_n, grid)
        elif self.Sn_aux_shape == 'gaussian':

            Gaussian = np.vectorize(self.Gaussian)

            # particle source
            aux_source_n = GridProfile(Gaussian(grid.rho, A=1, sigma=self.Sn_aux_width, x0=self.Sn_aux_center), grid)  # Trinity units
            if self.Sn_aux_int:
                target = self.Sn_aux_int  # 10^20 s^-1
                volume_integral = geo.volume_integrate(aux_source_n, normalized=False, interpolate=False)*norms.Sn_ref_SI20  # 10^20 s^-1
                aux_source_n = (target / volume_integral) * aux_source_n
            else:
                # convert height from 10^20 m^-3 s^-1 to Trinity units
                height = self.Sn_aux_height / norms.Sn_ref_SI20
                aux_source_n = height * aux_source_n
            self.Sn_labeled['aux'] = aux_source_n
        else:
            raise ValueError('Invalid density_source profile shape in Trinity input')

        # power source
        if not self.evolve_temperature:
            self.Sp_labeled['aux'] = GridProfile(0, grid)
        elif self.Sp_aux_import:
            assert imported.imported is not None, 'Error: must specify [import] section to use import = true for pressure_source'
            aux_source_p = imported.get_pressure_source(keys=self.Sp_aux_import_keys, species_mass=self.mass, species_charge=self.Z, tag=self.type_tag, norms=norms)
            self.Sp_labeled['aux'] = GridProfile(aux_source_p, grid)
        elif self.Sp_aux_shape == 'gaussian':
            Gaussian = np.vectorize(self.Gaussian)

            aux_source_p = GridProfile(Gaussian(grid.rho, A=1, sigma=self.Sp_aux_width, x0=self.Sp_aux_center), grid)  # Trinity units
            if self.Sp_aux_int:
                target = self.Sp_aux_int  # MW
                volume_integral = geo.volume_integrate(aux_source_p, normalized=False, interpolate=False)*norms.P_ref_MWm3  # MW
                aux_source_p = (target / volume_integral) * aux_source_p
            else:
                # convert height from MW/m^3 to Trinity units
                height = self.Sp_aux_height / norms.P_ref_MWm3
                aux_source_p = height * aux_source_p
            self.Sp_labeled['aux'] = aux_source_p
        else:
            raise ValueError('Invalid pressure_source profile shape in Trinity input')

    # for a particle and heat sources
    def Gaussian(self, x, A=2, sigma=.3, x0=0):
        exp = - ((x - x0) / sigma)**2 / 2
        return A * np.e ** exp

    def get_mass(self):
        '''
        Look-up table for species mass in units of proton mass.
        '''
        if self.type == "hydrogen":
            return 1.0
        if self.type == "deuterium":
            return 2.0
        elif self.type == "tritium":
            return 3.0
        elif self.type == "helium":
            return 4.0
        elif self.type == "boron":
            return 10.811
        elif self.type == "carbon":
            return 12.011
        elif self.type == "oxygen":
            return 15.999
        elif self.type == "neon":
            return 20.18
        elif self.type == "krypton":
            return 83.798
        elif self.type == "tungsten":
            return 183.84
        elif self.type == "electron":
            return 0.000544617021
        else:
            assert False, f"species '{self.type}' has unknown mass. use mass parameter in input file (with units of proton mass)."

    def get_charge(self):
        '''
        Look-up table for species charge in units of e.
        '''
        if self.type == "hydrogen":
            return 1.0
        if self.type == "deuterium":
            return 1.0
        elif self.type == "tritium":
            return 1.0
        elif self.type == "helium":
            return 2.0
        elif self.type == "boron":
            return 5.0
        elif self.type == "carbon":
            return 6.0
        elif self.type == "oxygen":
            return 8.0
        elif self.type == "neon":
            return 10.0
        elif self.type == "krypton":
            return 36.0
        elif self.type == "tungsten":
            return 72.0
        elif self.type == "electron":
            return -1.0
        else:
            assert False, f"species '{self.type}' has unknown charge. use Z parameter in input file (with units of e)."

    def get_type_tag(self):
        '''
        Look-up table for short tag name for diagnostic output
        '''
        if self.type == "hydrogen":
            return "H"
        if self.type == "deuterium":
            return "D"
        elif self.type == "tritium":
            return "T"
        elif self.type == "helium":
            return "He"
        elif self.type == "boron":
            return "B"
        elif self.type == "carbon":
            return "C"
        elif self.type == "oxygen":
            return "O"
        elif self.type == "neon":
            return "Ne"
        elif self.type == "krypton":
            return "Kr"
        elif self.type == "tungsten":
            return "W"
        elif self.type == "electron":
            return "e"
        else:
            return self.type  # use full name
