import numpy as np
import sys
import os

from t3d.Grid import Grid
from t3d.Time import Time
from t3d.Species import SpeciesDict
from t3d.Solver import TransportSolver
from t3d.Geometry import Geometry
from t3d.TrinityIO import Input, Logger
from t3d.Physics import Physics
from t3d.FluxModels import FluxModelDict
from t3d.SourceModels import SourceModelDict
from t3d.Import import Import
from t3d.Logbook import info, bold
from termcolor import colored


_version = "1.0.0"


def check_for_stop_file(stopfile="stop") -> bool:
    """Check for a stop file."""
    if os.path.isfile(stopfile):
        bold(' Stop file detected, exiting.', color='blue')
        os.remove(stopfile)
        return True
    return False


class TrinityEngine():

    def __init__(self, trinity_input):

        # Parse the input file data, and store in dictionary self.input_dict
        self.inputs = Input(trinity_input)
        self.input_dict = self.inputs.input_dict
        self.trinity_infile = trinity_input  # save input file name
        self.version = _version

        # initialize grid
        self.grid = Grid(self.input_dict)

        # initialize time
        self.time = Time(self.input_dict)

        # initialize flux models
        self.flux_models = FluxModelDict(self.input_dict, self.grid, self.time)

        # initialize source models
        self.source_models = SourceModelDict(self.input_dict, self.grid, self.time)

        # initialize imported profiles, sources, etc
        self.imported = Import(self.input_dict, self.grid)

        # initialize geometry
        self.geometry = Geometry(self.input_dict, self.grid, self.flux_models, self.imported)

        # initialize all species
        self.species = SpeciesDict(self.input_dict, self.grid)

        # initialize Trinity normalizations (not to be confused with FluxModel normalizations)
        self.norms = TrinNormalizations(Btor=self.geometry.Btor, a_meters=self.geometry.a_minor, vt_sqrt_2=self.flux_models.vt_sqrt_2, m_ref=self.species.ref_species.mass)
        self.time.normalize_inputs(self.norms.t_ref)

        # initialize physics
        self.physics = Physics(self.input_dict, self.grid, self.geometry, self.norms, self.imported)

        # set initial profiles
        self.species.init_profiles_and_sources(self.geometry, self.norms, self.imported)
        self.physics.compute_sources(self.species, self.geometry, self.norms)
        self.species.print_info()

        # initialize solver
        self.solver = TransportSolver(self.grid, self.species, self.geometry, self.physics)

        # initialize logger
        self.logger = Logger(self.trinity_infile, self.input_dict, self.grid, self.time, self.species, self.geometry, self.physics, self.norms)

    # End of __init__ function

    def evolve_profiles(self):

        while self.time.step_idx < self.time.N_steps and self.time.time < self.time.t_max:

            info("")

            if self.time.iter_idx == 0:
                if self.time.time + self.time.dtau > self.time.t_max:
                    self.time.dtau = self.time.t_max - self.time.time
                    info(f"\n{'':>13}*** Clipping timestep to end at max time: dtau = {colored(f'{self.time.dtau:.3e}', 'blue')}. ***")

            if self.geometry.evolve_equilibrium:
                self.geometry.update_equilibrium(self.species, self.time)

            # compute sources from physics and source models
            self.physics.compute_sources(self.species, self.geometry, self.norms)
            for source in self.source_models.get_sources_list():
                info(f" computing {source.label} sources", color='magenta')
                source.compute_sources(self.species, self.geometry, self.norms)

            # zero out fluxes, then compute from flux models
            self.species.clear_fluxes()
            for model in self.flux_models.get_models_list():
                info(f" computing {model.label} fluxes", color='green')
                model.compute_fluxes(self.species, self.geometry, self.norms)

            # collect flux and source results
            for model in self.flux_models.get_models_list():
                model.collect_results(self.species, self.geometry, self.norms)
            for source in self.source_models.get_sources_list():
                source.collect_results(self.species, self.geometry, self.norms)

            # solve transport equations
            self.solver.solve(self.species, self.time)
            rms = self.time.rms

            if self.time.iter_idx == 0:
                info(f" iterate = {self.time.iter_idx:>2} " +
                     f"time index = {self.time.step_idx:>2} " +
                     f"normalized time = {self.time.time:.3e}, " +
                     f"normalized dt = {self.time.dtau:.3f}, " +
                     f"rms = {rms:.3e}, " +
                     f"time = {self.time.time*self.norms.t_ref:.3e} (s)")
            else:
                info(f" iterate = {self.time.iter_idx:>2}{'':>69}" +
                     f"rms = {rms:.3e}", 'blue')

            # info(f"time = {self.time.time*self.norms.t_ref:.3e} (s),  time_index = {self.time.step_idx}, iteration index = {self.time.iter_idx}, dtau = {self.time.dtau}, rms = {rms:.3e}")
            # info(f"time = {self.time.time:.3e}*t_ref, time index = {self.time.step_idx}, iteration index = {self.time.iter_idx}, dtau = {self.time.dtau}, rms = {rms:.3e}")

            self.logger.save(self.species, self.solver, self.time)

            if rms < self.time.rms_converged:
                # evolution converged, trinity stops
                break

            if rms < self.time.newton_threshold or (self.time.iter_idx == self.time.max_newton_iter-1 and rms < self.time.newton_tolerance) or self.time.alpha == 0:
                # iteration converged, trinity starts new time step
                self.time.prev_step_success = True
                self.time.step_idx += 1
                self.time.time += self.time.dtau
                if self.time.iter_idx > 0:
                    # if prev time step halved, do NOT update dtau_success because the halving skipped GX
                    # if prev time was not halved, there is no need to update dtau_success
                    self.time.dtau_success = self.time.dtau
                self.time.iter_idx = 0
                self.solver.advance(self.species)

                if self.physics.update_edge:
                    self.species.update_edge(self.physics)
                if self.physics.update_source:
                    self.species.update_source(self.physics, self.geometry)

                if rms < self.time.dtau_increase_threshold and self.time.step_idx < self.time.N_steps and self.time.alpha > 0:
                    self.time.dtau_old = self.time.dtau
                    self.time.dtau = min(2.0*self.time.dtau, self.time.dtau_max)
                    info(f"\n{'':>13}*** Increasing timestep at time index {self.time.step_idx} ({colored(f'{rms:.3e}', 'blue')} < {self.time.dtau_increase_threshold:.3e}). ***")

            elif np.isnan(rms) or self.time.iter_idx == self.time.max_newton_iter-1:
                info(f"\n{'':>8}*** Iteration error is too large ({colored(f'{rms:.3e}', 'yellow')} > {self.time.newton_tolerance:.3e}) after {self.time.max_newton_iter} iterations. ***")
                self.solver.reset_timestep(self.species, self.time)

                for model in self.flux_models.get_models_list():
                    # reuse data from p=0 when retaking timestep
                    model.overwrite = False
            else:
                # iteration failed, trinity tries another iteration
                self.time.prev_step_success = False
                self.time.iter_idx += 1
                self.solver.advance(self.species)

            self.logger.export()
            self.logger.write_netCDF4(self.trinity_infile)
            self.logger.write_adios2(self.trinity_infile)
            sys.stdout.flush()

            if check_for_stop_file("stop"):
                break

        if self.time.N_steps > 0:
            info(f"\n{'':>14}time index = {self.time.step_idx:>2} " +
                 f"normalized time = {self.time.time:.3e}, " +
                 f"normalized dt = {self.time.dtau:.3f}, rms = {rms:.3e}, " +
                 f"time = {self.time.time*self.norms.t_ref:.3e} (s)")

            info("\n Time evolution finished!")
            # info(f"\n Time evolution finished! time = {self.time.time:.3e}*t_ref, time index = {self.time.step_idx}")
            for model in self.flux_models.get_models_list():
                model.final_info()

        # write final profiles, but note fluxes will not be consistent (because we haven't called compute_fluxes)
        # need to think more about this...
        self.logger.save(self.species, self.solver, self.time, final=True)
        self.logger.export()
        self.logger.write_netCDF4(self.trinity_infile)
        self.logger.write_adios2(self.trinity_infile)


# a class for handling normalizations in Trinity
class TrinNormalizations():
    def __init__(self,
                 Btor=1,      # T
                 m_ref=2.0,   # units of proton mass
                 a_meters=1,  # minor radius, in m
                 vt_sqrt_2=False):

        info("  Initializing normalizations")
        info(f"    Ba      = {Btor:.2f} T")
        info(f"    a_minor = {a_meters:.2f} m")
        info(f"    m_ref   = {m_ref:.2f} m_p")

        # could get these constants from scipy
        self.e = 1.602e-19  # Coulomb
        self.c = 2.99e8     # m/s

        if vt_sqrt_2:
            gam = 2
        else:
            gam = 1

        # reference gyroradius in [m], assuming
        # B_ref = 1 and T_ref = 1 keV
        self.rho_ref = 3.23e-3 * np.sqrt(gam * m_ref)  # m

        # reference thermal speed in [m/s], assuming
        # T_ref = 1 keV
        vT_ref = 3.094e5 * np.sqrt(gam / m_ref)  # m/s

        t_ref = a_meters / vT_ref
        p_ref = 1e20 * 1e3 * self.e  # J/m^3
        rho_star = self.rho_ref / a_meters

        # multiply trinity time (tau) by this to get SI (seconds)
        t_ref = t_ref * rho_star**-2

        # multiply SI (W/m^3) by pressure_source_scale to get Trinity units
        pressure_source_scale = t_ref / p_ref
        P_ref_MWm3 = 1 / pressure_source_scale / 1e6  # MW/m^3

        # multiply SI (10^20 m^-3 s^-1) by particle_source_scale to get Trinity units
        particle_source_scale = t_ref
        Sn_ref_SI20 = 1 / particle_source_scale  # 10^20 m^-3 s^-1

        # info(f"    a_minor = {a_meters:.2f} m")
        info(f"    t_ref = {t_ref:.3f} s")    # deprecated report, to be fixed in the next round.
        # info(f"    P_ref = {P_ref_MWm3:.3f} MW/m^3")
        # info(f"    S_ref = {Sn_ref_SI20:.3f} 10^20/(m^3 s)")
        info(f"    rho_star(T=1keV) = {rho_star / Btor:.3e}\n")

        # save
        self.Btor = Btor
        self.a_meters = a_meters

        self.vT_ref = vT_ref
        self.vtfac = gam
        self.t_ref = t_ref
        self.p_ref = p_ref
        self.rho_star = rho_star
        self.P_ref_MWm3 = P_ref_MWm3
        self.Sn_ref_SI20 = Sn_ref_SI20
