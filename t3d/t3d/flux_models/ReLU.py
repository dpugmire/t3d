import numpy as np
from t3d.FluxModels import FluxModel
import random


def ReLU(x,a=1,m=1,p=1, noise=0):
    '''
       piecewise-linear function
       can model Gamma( critical temperature gradient scale length ), for example
       x is a number, a and m are constants

       inputs : a number
       outputs: a number
    '''
    if (abs(x) < a):
        return 1e-16
    else:
        sgn = np.sign(x)
        return m*(x - a*sgn)**p*(1+noise*random.uniform(-1,1))


def Linear(x,a=1,m=1,p=1, noise=0):
    '''
       piecewise-linear function
       can model Gamma( critical temperature gradient scale length ), for example
       x is a number, a and m are constants

       inputs : a number
       outputs: a number
    '''
    return np.sign(x-a)*np.abs(m*(x-a))**p*(1+noise*random.uniform(-1,1))


ReLU = np.vectorize(ReLU,otypes=[np.float64])


class ReLU_FluxModel(FluxModel):

    def __init__(self, pars, grid, time):

        # call base class constructor
        super().__init__(pars, grid, time)

        # neoclassical diffusion coefficient
        self.D_neo = pars.get('D_neo', 0.0)
        self.label = pars.get('label', 'ReLU')  # label for diagnostics
        self.p = pars.get('power', 1)
        self.noise = pars.get('noise', 0)
        self.seed = pars.get('random_seed', 10)
        random.seed(self.seed)

    def logfile_info(self) -> str:
        ''' Return string of log file location '''
        return 'None'

    def compute_fluxes(self, species, geo, norms):
        # in the following, A_sjk indicates A[s, j, k] where
        # s is species index
        # j is rho index
        # k is perturb index

        rho_j = self.rho
        pflux0_sj = np.zeros((species.N_species, self.N_fluxtubes))
        qflux0_sj = np.zeros((species.N_species, self.N_fluxtubes))
        pflux_sj = np.zeros((species.N_species, self.N_fluxtubes))
        qflux_sj = np.zeros((species.N_species, self.N_fluxtubes))

        # base profiles case
        # get gradient values on flux grid
        # these are 2d arrays, e.g. kn_sj = kap_n[s, j]
        kn0_sj, kT0_sj, kp0_sj, _ = species.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        for i, s in enumerate(species.get_species_list()):
            gradn = -s.n().toFluxProfile()*kn0_sj[i, :]
            gradp = -s.p().toFluxProfile()*kp0_sj[i, :]
            pflux0_sj[i, :] = -gradn[:]*self.D_neo + Linear(kn0_sj[i, :], a=s.n_relu_critical_gradient0+s.n_relu_critical_gradient1*rho_j[:], m=s.n_relu_flux_slope, p=self.p, noise=self.noise)
            qflux0_sj[i, :] = -gradp[:]*self.D_neo + ReLU(kT0_sj[i, :], a=s.p_relu_critical_gradient0+s.p_relu_critical_gradient1*rho_j[:], m=s.p_relu_flux_slope, p=self.p, noise=self.noise)
        species.add_flux(pflux0_sj, qflux0_sj, label=self.label)

        # perturbed density cases
        # for each evolved density species, need to perturb density
        for stype in species.n_evolve_keys:
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=stype, pert_T=None)
            for i, s in enumerate(species.get_species_list()):
                gradn = -s.n().toFluxProfile()*kn_sj[i, :]
                gradp = -s.p().toFluxProfile()*kp_sj[i, :]
                pflux_sj[i, :] = -gradn[:]*self.D_neo + Linear(kn_sj[i, :], a=s.n_relu_critical_gradient0+s.n_relu_critical_gradient1*rho_j[:], m=s.n_relu_flux_slope, p=self.p, noise=self.noise)
                qflux_sj[i, :] = -gradp[:]*self.D_neo + ReLU(kT_sj[i, :], a=s.p_relu_critical_gradient0+s.p_relu_critical_gradient1*rho_j[:], m=s.p_relu_flux_slope, p=self.p, noise=self.noise)
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkap
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkap
            species.add_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj)

        # perturbed temperature cases
        # for each evolved temperature species, need to perturb temperature
        for stype in species.T_evolve_keys:
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=stype)
            for i, s in enumerate(species.get_species_list()):
                gradn = -s.n().toFluxProfile()*kn_sj[i, :]
                gradp = -s.p().toFluxProfile()*kp_sj[i, :]
                pflux_sj[i, :] = -gradn[:]*self.D_neo + Linear(kn_sj[i, :], a=s.n_relu_critical_gradient0+s.n_relu_critical_gradient1*rho_j[:], m=s.n_relu_flux_slope, p=self.p, noise=self.noise)
                qflux_sj[i, :] = -gradp[:]*self.D_neo + ReLU(kT_sj[i, :], a=s.p_relu_critical_gradient0+s.p_relu_critical_gradient1*rho_j[:], m=s.p_relu_flux_slope, p=self.p, noise=self.noise)
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkap
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkap
            species.add_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj)
