import numpy as np
from t3d.FluxModels import FluxModel
from t3d.Profiles import FluxProfile


class PowerBalance_FluxModel(FluxModel):

    def __init__(self, pars, grid, time):

        # call base class constructor
        super().__init__(pars, grid, time)
        self.label = None

    def compute_fluxes(self, species, geo, norms):
        # in the following, A_sjk indicates A[s, j, k] where
        # s is species index
        # j is rho index
        # k is perturb index
        pflux_sj = np.zeros((species.N_species, self.N_fluxtubes))
        qflux_sj = np.zeros((species.N_species, self.N_fluxtubes))

        for i, s in enumerate(species.get_species_list()):
            Gam = FluxProfile(geo.volume_integrate(s.Sn, profile=True)[:-1], self.grid)
            Q = FluxProfile(geo.volume_integrate(s.Sp, profile=True)[:-1], self.grid)

            # convert to gyroBohm
            Gam_GB = Gam/norms.a_meters**3/((geo.area/geo.Btor**2).plus()*(species.ref_species.p()**1.5/species.ref_species.n()**0.5).plus())
            Q_GB = Q/norms.a_meters**3/(geo.area/geo.Btor**2).plus()/(species.ref_species.p().plus()**2.5/species.ref_species.n().plus()**1.5)
            pflux_sj[i, :] = Gam_GB
            qflux_sj[i, :] = Q_GB

        species.add_flux(pflux_sj, qflux_sj)
