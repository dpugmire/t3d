import t3d.Profiles as pf
from t3d.FluxModels import FluxModel


class Diffusive_FluxModel(FluxModel):
    """
    This test model follows the diffusive model from
    Eq (7.163) in Section 7.8.1 of Michael Barnes' thesis
    """
    # should this test automatically turn off sources?

    def __init__(self, inputs, grid, time, species, geo):

        # call base class constructor
        super().__init__(inputs, grid, time, species, geo)

        self.D = 1

    def compute_Q(self,engine, step=0.1):

        pi = engine.pressure_i.midpoints
        pe = engine.pressure_e.midpoints

        Lpi = - engine.pressure_i.grad_log.profile  # a / L_pi
        Lpe = - engine.pressure_e.grad_log.profile  # a / L_pe

        D = self.D
        Qi = 1.5 * D * Lpi / pi**(-1.5)  # assume p_ref = pi
        Qe = 1.5 * D * Lpe * pe / pi**(-2.5)

        zero = 0*pi
        Gamma = zero  # do not evolve particles

        # Perturb
        Qi_pi = 1.5 * D * (Lpi+step) / pi**(-1.5)  # assume p_ref = pi
        Qe_pe = 1.5 * D * (Lpe+step) * pe / pi**(-2.5)

        dQi_pi = (Qi_pi - Qi) / step
        dQe_pe = (Qe_pe - Qe) / step

        # save
        engine.Gamma = pf.Flux_profile(zero)
        engine.Qi = pf.Flux_profile(Qi)
        engine.Qe = pf.Flux_profile(Qe)
        engine.G_n = pf.Flux_profile(zero)
        engine.G_pi = pf.Flux_profile(zero)
        engine.G_pe = pf.Flux_profile(zero)
        engine.Qi_n = pf.Flux_profile(zero)
        engine.Qi_pi = pf.Flux_profile(dQi_pi)
        engine.Qi_pe = pf.Flux_profile(zero)
        engine.Qe_n = pf.Flux_profile(zero)
        engine.Qe_pi = pf.Flux_profile(zero)
        engine.Qe_pe = pf.Flux_profile(dQe_pe)
