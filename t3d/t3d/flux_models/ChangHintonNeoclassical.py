from t3d.FluxModels import FluxModel
import numpy as np


class ChangHintonNeo_FluxModel(FluxModel):

    def __init__(self, inputs, grid, time):

        # call base class constructor
        super().__init__(inputs, grid, time)

        model_parameters = inputs.get('model', {})
        self.label = inputs.get('label', 'Chang-Hinton')  # label for diagnostics
        self.vt_sqrt_2 = None

    def compute_fluxes(self, species, geo, norms):
        # in the following, A_sjk indicates A[s, j, k] where
        # s is species index
        # j is rho index
        # k is perturb index

        assert geo.geo_option == 'miller', "Error: Chang-Hinton analytic neoclassical model currently requires geo_option = 'miller'"

        rho_j = self.rho  # r/a
        rmajor = geo.AspectRatio  # R/a
        rmajor_prime = geo.rmajor_prime
        eps = rho_j/rmajor  # r/R
        qsf = geo.qsf
        Bpol = eps/(qsf*(1 + (eps/qsf)**2)**0.5)

        K0 = 0.66
        a2 = 1.03
        b2 = 0.31
        c2 = 0.74

        pflux0_sj = np.zeros((species.N_species, self.N_fluxtubes))
        qflux0_sj = np.zeros((species.N_species, self.N_fluxtubes))
        pflux_sj = np.zeros((species.N_species, self.N_fluxtubes))
        qflux_sj = np.zeros((species.N_species, self.N_fluxtubes))

        # base profiles case
        # get gradient values on flux grid
        # these are 2d arrays, e.g. kn_sj = kap_n[s, j]
        kn0_sj, kT0_sj, kp0_sj, _ = species.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        ns, Ts, nu_ss = species.get_profiles_on_flux_grid(normalize=True, a_minor=geo.a_minor)

        B02_over_B2 = (1 + 1.5*(eps**2 + eps*rmajor_prime) + .375*rmajor_prime*eps**3)/(1 + 0.5*eps*rmajor_prime)
        B02_over_B2_inv = (1-eps**2)**0.5*(1 + 0.5*eps*rmajor_prime)/(1 + rmajor_prime*((1-eps**2)**0.5-1)/eps)
        F = 0.5*(B02_over_B2 - B02_over_B2_inv)/eps**0.5
        K2star = (0.66 + 1.88*eps**0.5 - 1.54*eps)*B02_over_B2

        for i, s in enumerate(species.get_species_list()):
            if s == species[species.first_ion_type]:
                nustar = 4/(3*np.sqrt(np.pi))*nu_ss[i, :]*qsf*rmajor/eps**1.5
                K2 = K0*((K2star/0.66)/(1 + a2*nustar**0.5 + b2*nustar) + c2**2*nustar*eps**1.5/(b2*(1+c2*nustar*eps**1.5))*F)
                qflux0_sj[i, :] = K2*(8/np.pi)**0.5/3.*eps**0.5*kT0_sj[i, :]*nu_ss[i, :]/(0.5*species.norms.vtfac*Bpol**2)
        # print(f'\n  qflux_neo = {qflux0_sj}\n')
        species.add_flux(pflux0_sj, qflux0_sj, label=self.label)

        # perturbed density cases
        # for each evolved density species, need to perturb density
        for stype in species.n_evolve_keys:
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=stype, pert_T=None)
            for i, s in enumerate(species.get_species_list()):
                if s == species[species.first_ion_type]:
                    nustar = 4/(3*np.sqrt(np.pi))*nu_ss[i, :]*qsf*rmajor/eps**1.5
                    K2 = K0*((K2star/0.66)/(1 + a2*nustar**0.5 + b2*nustar) + c2**2*nustar*eps**1.5/(b2*(1+c2*nustar*eps**1.5))*F)
                    qflux_sj[i, :] = K2*(8/np.pi)**0.5/3.*eps**0.5*kT_sj[i, :]*nu_ss[i, :]/(0.5*species.norms.vtfac*Bpol**2)
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkap
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkap
            species.add_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj)

        # perturbed temperature cases
        # for each evolved temperature species, need to perturb temperature
        for stype in species.T_evolve_keys:
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=stype)
            for i, s in enumerate(species.get_species_list()):
                if s == species[species.first_ion_type]:
                    nustar = 4/(3*np.sqrt(np.pi))*nu_ss[i, :]*qsf*rmajor/eps**1.5
                    K2 = K0*((K2star/0.66)/(1 + a2*nustar**0.5 + b2*nustar) + c2**2*nustar*eps**1.5/(b2*(1+c2*nustar*eps**1.5))*F)
                    qflux_sj[i, :] = K2*(8/np.pi)**0.5/3.*eps**0.5*kT_sj[i, :]*nu_ss[i, :]/(0.5*species.norms.vtfac*Bpol**2)
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkap
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkap
            # print(f'\n  qflux_neo = {qflux_sj}\n')
            species.add_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj)
