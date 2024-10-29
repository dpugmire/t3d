import numpy as np
import subprocess
from datetime import datetime
import time as _time
from t3d import Profiles as pf
import os
import re
import copy
import sys
from t3d.FluxModels import FluxModel
from netCDF4 import Dataset
import uuid
import math
from t3d.Logbook import logbook

WAIT_TIME = 1  # this should come from the Trinity Engine


def map_to_periodic_range(arr, low_bound, high_bound):
    range_size = high_bound - low_bound
    mapped_arr = ((arr - low_bound) % range_size) + low_bound
    return mapped_arr


class GX_FluxModel(FluxModel):

    def __init__(self, pars, grid, time):

        # call base class constructor
        super().__init__(pars, grid, time)

        # environment variable containing path to gx executable
        GX_PATH = pars.get("gx_path", os.environ.get("GX_PATH") or "")
        self.gx_app = os.path.join(GX_PATH,"gx")

        gx_template = pars.get('gx_template', 'tests/regression/gx_template.in')
        out_dir = pars.get('gx_outputs', 'gx/')
        self.out_dir = out_dir
        self.logfile = pars.get('gx_logfile', 'gx.log')
        self.overwrite = pars.get('overwrite', False)
        self.overwrite_p_neq_0 = False
        self.gpus_per_gx = pars.get('gpus_per_gx', 1)
        # force GX to use the electrostatic limit (regardless of beta in T3D) in GX
        self.electromagnetic = pars.get('electromagnetic', True)
        # list of zeta angles about which flux tubes are centered (for stellarator calculations only)
        self.zeta_centers = pars.get('zeta_center', [0.0])
        if not hasattr(self.zeta_centers, '__len__'):  # allow a scalar input instead of a list
            self.zeta_centers = [self.zeta_centers]
        # interpolate fluxes on theta-zeta grid (stellarator calculations only)
        self.theta_zeta_fluxes = pars.get('theta_zeta_fluxes', False)
        # specify a species should be treated adiabatically by GX
        self.adiabatic_species = pars.get('adiabatic_species', None)
        # scale the GX collisionality by a factor
        self.collisionality_scaling_factor = pars.get('collisionality_scaling_factor', 1.0)
        # scale qfluxes by an artificial factor (useful for scoping)
        self.qflux_scaling_factor = pars.get('qflux_scaling_factor', 1.0)
        # shift tprim used in GX away from nominal tprim by a constant (useful for scoping)
        self.tprim_shift = pars.get('tprim_shift', 0.0)
        # specify some species temperatures that do not need to be perturbed (e.g. don't perturb ion temperatures for ETG calculations)
        self.no_perturb_temperature = pars.get('no_perturb_temperature', [])
        if not hasattr(self.no_perturb_temperature, '__len__'):  # allow a scalar input instead of a list
            self.no_perturb_temperature = [self.no_perturb_temperature]
        # parameter to set perturbation size for gradients
        self.dkap_T = pars.get('dkap_T', 0.5)
        self.dkap_n = pars.get('dkap_n', 0.5)
        self.label = pars.get('label', 'GX')  # label for diagnostics
        # check for stalled GX runs. should be set False when there are not enough GPUs
        # to cover all the flux tubes in the T3D calculation
        self.check_stalls = pars.get('check_stalls', True)
        self.monitor_time = pars.get('monitor_time', 10)  # check on GX output every [monitor_time] seconds
        self.stall_abort_count = pars.get('stall_abort_count', 6)  # abort after this many stall monitors
        self.build_ReLU = pars.get('build_ReLU', False)  # use GX runs to build a ReLU model at each radius
        self.nsteps_ReLU = pars.get('nsteps_ReLU', 1000)  # take [nsteps_ReLU] ReLU steps before re-running GX

        # use a non-default run command (e.g. "srun ...")
        # full command line will be cmd = f"{self.run_command} {gx_app} {f_in}"
        self.run_command = pars.get('run_command', None)

        self.abort_on_nans = pars.get('abort_on_nans', True)
        self.no_restarts = pars.get('no_restarts', False)

        self.dry_run = pars.get('dry_run', False)

        self.vt_sqrt_2 = False  # GX normalizations use vt = sqrt(T/m), without the sqrt(2)
        self.target_chie_over_chii = pars.get('chie_over_chii', 0)  # if non-zero, compute Qe from Qi by using chi ratio

        # Create output directory if not found
        found_path = os.path.exists(out_dir)
        if not found_path:
            os.mkdir(out_dir)

        # Create logger
        self.log = logbook()
        self.log.set_handlers(term_stream=False,
                              file_stream=True,
                              file_handler=os.path.join(self.out_dir, self.logfile))
        self.info = self.log.info  # Short-cut command

        # Check file path
        self.info("  Looking for GX files")
        self.info(f"    Expecting GX template: {gx_template}")
        self.info(f"    Expecting GX executable: {self.gx_app}")
        self.info(f"    GX-Trinity output path: {out_dir}")

        if not self.dry_run:
            found_gx = os.path.isfile(self.gx_app) and os.access(self.gx_app, os.X_OK)
            if not found_gx:
                self.info("ERROR: gx executable not found! Make sure the GX_PATH environment variable is set.")
                raise RuntimeError('GX not found')

        self.info("")

        # load an input template
        self.read_input(gx_template)
        try:
            self.nstep_gx = self.inputs['Time']['nstep']
        except:
            self.nstep_gx = None

        self.t_id_load = 0
        self.p_id_load = 0
        self.dtau_load = 0
        self.processes = []
        self.call_counter = 0
        self.logs = []
        self.gx_index = 0

        self.use_ReLU = False
        if self.build_ReLU:
            self.dpflux_dkn_sj = {}
            self.dqflux_dkn_sj = {}
            self.dheat_dkn_sj = {}
            self.dpflux_dkT_sj = {}
            self.dqflux_dkT_sj = {}
            self.dheat_dkT_sj = {}

        if not self.dry_run:
            try:
                system = os.environ['GK_SYSTEM']
            except:
                self.info("ERROR: must set GK_SYSTEM environment variable to use GX.")
                raise RuntimeError('GK_SYSTEM not set')

            # set up node files for PBS systems like Polaris
            if system == "polaris":  # or other PBS systems eventually
                PBS_NODEFILE = os.environ.get("PBS_NODEFILE")
                NODES_PER_MPI = max(1,int(self.gpus_per_gx/4))
                cmd = f"split --lines={NODES_PER_MPI} --numeric-suffixes=1 --suffix-length=2 {PBS_NODEFILE} {os.path.join(self.out_dir,'local_hostfile')}."
                p = subprocess.Popen(cmd.split())
                p.wait()

    def __del__(self):
        ''' Use destructor to close the log file '''
        self.log.finalize()

    def logfile_info(self) -> str:
        ''' Return string of log file location '''
        return self.log.file_handler.name

    def compute_fluxes(self, species, geo, norms):
        if self.time.step_idx % self.nsteps_ReLU == 0 and self.time.iter_idx == 0:
            self.use_ReLU = False
        if self.use_ReLU:
            self.compute_fluxes_ReLU(species, geo, norms)
        else:
            self.compute_fluxes_gx(species, geo, norms)

    # run GX calculations and pass fluxes and flux jacobian to SpeciesDict species
    def compute_fluxes_gx(self, species, geo, norms):
        # in the following, A_sjk indicates A[s, j, k] where
        # s is species index
        # j is rho index
        # k is perturb index

        self.gx_index = 0

        self.outs_jk = []
        self.dkap_jk = []

        rho_j = self.rho

        # this is for backwards compat, if adiabatic species was specified in [species] tables
        if species.has_adiabatic_species:
            self.adiabatic_species = species.adiabatic_species.type

        # remove adiabatic species from list of species to be used for gx
        if self.adiabatic_species is not None:
            species_gx = species.remove(self.adiabatic_species, in_place=False)
        elif len(self.no_perturb_temperature) > 0:
            species_gx = copy.deepcopy(species)
            for s in self.no_perturb_temperature:
                species_gx.T_evolve_keys.remove(s)
        else:
            species_gx = species

        # base profiles case
        pert_id = 0
        # get gradient values on flux grid
        # these are 2d arrays, e.g. kn_sj = kap_n[s, j]
        kn0_sj, kT0_sj, kp0_sj, dkap = species_gx.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        if self.build_ReLU:
            self.kn0_sj = kn0_sj.copy()
            self.kT0_sj = kT0_sj.copy()
            self.kp0_sj = kp0_sj.copy()
        self.dkap_jk.append(dkap)  # this is a dummy entry, unused
        outs_j = self.run_gx_fluxtubes(rho_j, kn0_sj, kT0_sj, species_gx, species, geo, pert_id)
        self.outs_jk.append(outs_j)

        # perturbed density cases
        # for each evolved density species, need to perturb density
        for stype in species_gx.n_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species_gx.get_grads_on_flux_grid(pert_n=stype, pert_T=None, abs_step=self.dkap_n)
            self.dkap_jk.append(dkap)
            outs_for_fluxtubes = self.run_gx_fluxtubes(rho_j, kn_sj, kT_sj, species_gx, species, geo, pert_id)
            self.outs_jk.append(outs_for_fluxtubes)

        # perturbed temperature cases
        # for each evolved temperature species, need to perturb temperature
        for stype in species_gx.T_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species_gx.get_grads_on_flux_grid(pert_n=None, pert_T=stype, abs_step=self.dkap_T)
            self.dkap_jk.append(dkap)
            outs_for_fluxtubes = self.run_gx_fluxtubes(rho_j, kn_sj, kT_sj, species_gx, species, geo, pert_id)
            self.outs_jk.append(outs_for_fluxtubes)

        self.t_id_load = self.time.step_idx
        self.p_id_load = self.time.iter_idx
        self.dtau_load = self.time.dtau

    def compute_fluxes_ReLU(self, species, geo, norms):
        # remove adiabatic species from list of species to be used for gx
        if self.adiabatic_species is not None:
            species_gx = species.remove(self.adiabatic_species, in_place=False)
        elif len(self.no_perturb_temperature) > 0:
            species_gx = copy.deepcopy(species)
            for s in self.no_perturb_temperature:
                species_gx.T_evolve_keys.remove(s)
        else:
            species_gx = species

        kn_sj, kT_sj, kp_sj, _ = species_gx.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        pflux_sj = self.pflux0_sj.copy()
        qflux_sj = self.qflux0_sj.copy()
        dpflux_dkn_sj = copy.deepcopy(self.dpflux_dkn_sj)
        dpflux_dkT_sj = copy.deepcopy(self.dpflux_dkT_sj)
        dqflux_dkn_sj = copy.deepcopy(self.dqflux_dkn_sj)
        dqflux_dkT_sj = copy.deepcopy(self.dqflux_dkT_sj)
        for i, s in enumerate(species.get_species_list()):
            for ip, stype in enumerate(species_gx.n_evolve_keys):
                pflux_sj[i, :] += self.dpflux_dkn_sj[stype][i, :]*(kn_sj[ip, :] - self.kn0_sj[ip, :])
                qflux_sj[i, :] += np.maximum(self.dqflux_dkn_sj[stype][i, :]*(kn_sj[ip, :] - self.kn0_sj[ip, :]), 0)

            for ip, stype in enumerate(species_gx.T_evolve_keys):
                pflux_sj[i, :] += self.dpflux_dkT_sj[stype][i, :]*(kT_sj[ip, :] - self.kT0_sj[ip, :])
                qflux_sj[i, :] += np.maximum(self.dqflux_dkT_sj[stype][i, :]*(kT_sj[ip, :] - self.kT0_sj[ip, :]), 0)

            for j in np.arange(self.N_fluxtubes):
                if qflux_sj[i, j] < 0:
                    qflux_sj[i, j] = 1e-16
                    # for stype in species.n_evolve_keys:
                    #     dqflux_dkn_sj[stype][i, j] = 1e-16
                    # for stype in species.T_evolve_keys:
                    #     dqflux_dkT_sj[stype][i, j] = 1e-16

                if s.is_adiabatic:
                    pflux_sj[i, j] = 0.0
                    qflux_sj[i, j] = 0.0

        species.add_flux(pflux_sj, qflux_sj)
        self.info(f'\n  qflux_GX_ReLU = {qflux_sj}\n')
        self.info(f'\n  pflux_GX_ReLU = {pflux_sj}\n')
        for stype in species_gx.n_evolve_keys:
            species.add_dflux_dkn(stype, dpflux_dkn_sj[stype], dqflux_dkn_sj[stype])
        for stype in species_gx.T_evolve_keys:
            species.add_dflux_dkT(stype, dpflux_dkT_sj[stype], dqflux_dkT_sj[stype])

    def collect_results(self, species, geo, norms):

        if self.use_ReLU:
            return

        # make sure GX is running normally
        self.monitor()

        # collect parallel runs
        self.wait()

        # read
        _time.sleep(WAIT_TIME)

        if self.adiabatic_species is not None:
            species_gx = species.remove(self.adiabatic_species, in_place=False)
        elif len(self.no_perturb_temperature) > 0:
            species_gx = copy.deepcopy(species)
            for s in self.no_perturb_temperature:
                species_gx.T_evolve_keys.remove(s)
        else:
            species_gx = species

        # read GX output data and pass results to species
        # base
        pert_id = 0
        pflux0_sj, qflux0_sj, heat0_sj = self.read_gx_fluxes(self.outs_jk[pert_id], species.get_species_list())
        species.add_flux(pflux0_sj, qflux0_sj, heat0_sj, label=self.label)
        if self.build_ReLU:
            self.pflux0_sj = pflux0_sj
            self.qflux0_sj = qflux0_sj
            self.heat0_sj = heat0_sj

        # save qflux value for use when deciding whether to use a restart for GX
        self.qflux_prev = qflux0_sj.copy()
        for i, s in enumerate(species.get_species_list()):
            if s.type == self.adiabatic_species:
                self.qflux_prev[i, :] = 1  # for adiabatic species, use a dummy value that won't turn off restarts

        pert_id = pert_id + 1

        # perturbed density cases
        for stype in species_gx.n_evolve_keys:
            dkn_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj, heat_sj = self.read_gx_fluxes(self.outs_jk[pert_id], species.get_species_list())
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkn_j
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkn_j
            dheat_dkn_sj = (heat_sj - heat0_sj) / dkn_j
            species.add_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj, dheat_dkn_sj)
            if self.build_ReLU:
                self.dpflux_dkn_sj[stype] = dpflux_dkn_sj
                self.dqflux_dkn_sj[stype] = dqflux_dkn_sj
                self.dheat_dkn_sj[stype] = dheat_dkn_sj
            pert_id = pert_id + 1

        # perturbed temperature cases
        for stype in species_gx.T_evolve_keys:
            dkT_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj, heat_sj = self.read_gx_fluxes(self.outs_jk[pert_id], species.get_species_list())
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkT_j
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkT_j
            dheat_dkT_sj = (heat_sj - heat0_sj) / dkT_j
            species.add_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj, dheat_dkT_sj)
            if self.build_ReLU:
                self.dpflux_dkT_sj[stype] = dpflux_dkT_sj
                self.dqflux_dkT_sj[stype] = dqflux_dkT_sj
                self.dheat_dkT_sj[stype] = dheat_dkT_sj
            pert_id = pert_id + 1

        # ReLU data is now built, so turn it on
        if self.build_ReLU:
            self.use_ReLU = True

    def monitor(self):
        '''
        Make sure GX sims are actually running and not just hanging
        '''
        self.info("Monitoring GX runs......")

        if len(self.processes) == 0:
            return

        sizes = np.zeros(len(self.processes))
        stall_count = np.zeros(len(self.processes))
        restart_count = np.zeros(len(self.processes))
        any_running = True
        running = [True for p in self.processes]

        while any_running:
            if not any(running):
                any_running = False
                break

            sys.stdout.flush()
            _time.sleep(self.monitor_time)
            for i, p in enumerate(self.processes):
                size = os.path.getsize(self.logs[i])
                # check for "End time" to indicate GX run has ended
                finished = False
                with open(self.logs[i]) as f:
                    for line in f:
                        if re.search('End time', line):
                            finished = True
                if finished or p.poll() is not None:
                    # GX finished successfully
                    if finished:
                        running[i] = False
                        restart_count[i] = 0
                    # if GX stopped running before end, check for error codes
                    elif p.poll() != 0:
                        assert restart_count[i] < 5, f"*** GX log {self.logs[i]} has indicated an error several times. There must be something wrong. Exiting T3D..."
                        # error code indicates an issue with the GX run
                        err_file = self.logs[i] + "-error-" + uuid.uuid4().hex  # get a unique file ID in case this run fails multiple times
                        self.info(f'ERROR: GX log {self.logs[i]} indicates an error (error code={p.poll()}). Saving bugged log to {err_file}. Restarting this GX sim...')
                        restart_count[i] += 1
                        cmd = p.args
                        os.rename(self.logs[i], err_file)  # save bugged log
                        # resubmit subprocess, and replace it in processes list
                        self.info(f"> {cmd}")
                        with open(self.logs[i], 'w') as fp:
                            newp = subprocess.Popen(cmd, stdout=fp, stderr=fp)
                        self.processes[i] = newp
                        # reset counters
                        sizes[i] = 0
                        stall_count[i] = 0
                elif size > sizes[i]:
                    running[i] = True
                    sizes[i] = size
                    stall_count[i] = 0
                elif stall_count[i] > self.stall_abort_count:
                    # if there is no new GX output after a minute,
                    # GX is probably stalled
                    err_file = self.logs[i] + "-error-" + uuid.uuid4().hex  # get a unique file ID in case this run fails multiple times
                    self.info(f'ERROR: GX log {self.logs[i]} is stalled. Saving bugged log to {err_file}. Restarting this GX sim...')
                    restart_count[i] += 1
                    cmd = p.args
                    # kill the stalled processes
                    p.kill()
                    p.wait()
                    os.rename(self.logs[i], err_file)  # save bugged log
                    _time.sleep(30)
                    # resubmit subprocess, and replace it in processes list
                    self.info(f"> {cmd}")
                    with open(self.logs[i], 'w') as fp:
                        newp = subprocess.Popen(cmd, stdout=fp, stderr=fp)
                    self.processes[i] = newp
                    # reset counters
                    sizes[i] = 0
                    stall_count[i] = 0
                else:
                    if stall_count[i] > 2:
                        self.info(f'    Warning... GX output {self.logs[i]} may be stalling... stall_count = {int(stall_count[i])}')
                    if self.check_stalls:
                        stall_count[i] += 1

    def wait(self):

        # wait for a list of subprocesses to finish
        #    and reset the list

        exitcodes = [p.wait() for p in self.processes]
        self.info(f'GX exitcodes = {exitcodes}')
        self.processes = []  # reset
        self.logs = []  # reset

        # could add some sort of timer here

    def read_input(self, fin):

        with open(fin) as f:
            data = f.readlines()

        obj = {}
        header = ''
        for line in data:

            # strip comments
            if line.find('#') > -1:
                end = line.find('#')
                line = line[:end]

            # parse headers
            if line.find('[') == 0:
                header = line.split('[')[1].split(']')[0]
                obj[header] = {}
                continue

            # skip blanks
            if line.find('=') < 0:
                continue

            # store data
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()

            if header == '':
                obj[key] = value
            else:
                obj[header][key] = value

        # check for all possible header types in GX input file,
        # and initialize empty if not found
        try:
            obj['Dimensions']
        except:
            obj['Dimensions'] = {}
        try:
            obj['Domain']
        except:
            obj['Domain'] = {}
        try:
            obj['Time']
        except:
            obj['Time'] = {}
        try:
            obj['Initialization']
        except:
            obj['Initialization'] = {}
        try:
            obj['Restart']
        except:
            obj['Restart'] = {}
        try:
            obj['Dissipation']
        except:
            obj['Dissipation'] = {}
        try:
            obj['Diagnostics']
        except:
            obj['Diagnostics'] = {}
        try:
            obj['Geometry']
        except:
            obj['Geometry'] = {}
        try:
            obj['Physics']
        except:
            obj['Physics'] = {}
        try:
            obj['species']
        except:
            obj['species'] = {}
        try:
            obj['Boltzmann']
        except:
            obj['Boltzmann'] = {}

        self.inputs = obj
        self.filename = fin

    def write_input(self, fout='temp.in'):

        # do not overwrite
        # if (os.path.exists(fout) and not self.overwrite):
        #    self.info(f'  input exists, skipping write {fout}')
        #    return

        if np.__version__[0] == '2':
            np.set_printoptions(legacy='1.25')

        with open(fout,'w') as f:

            for item in self.inputs.items():

                if (type(item[1]) is not dict):
                    print('  %s = %s ' % item, file=f)
                    continue

                header, nest = item
                print('\n[%s]' % header, file=f)

                longest_key = max(nest.keys(), key=len)
                N_space = len(longest_key)
                for pair in nest.items():
                    s = '  {:%i}  =  {}' % N_space
                    print(s.format(*pair), file=f)

        self.info(f'  wrote input: {fout}')

    # run a GX flux tube calculation at each radius, given gradient values kns and kts
    def run_gx_fluxtubes(self, rho, kns, kts, species, species_all, geo, pert_id, dummy=False):

        t_id = self.time.step_idx  # time integer
        p_id = self.time.iter_idx  # Newton iteration number
        dtau_tag = self.time.dtau      # T3D timestep

        if self.time.prev_step_success:
            # handle case where prev step succeeded, but GX was skipped (because GX was halved)
            if self.p_id_load == 0:
                # skip running a new flux tube. This can reach back to save state multiple halvings ago.
                self.dtau_load = self.time.dtau_success

        else:
            # handle case of failed timestep
            if dtau_tag < self.time.dtau_success:
                # an entire chain of Newton iterations has failed
                if p_id == 0:
                    # skip running a new flux tube, read existing data
                    dtau_tag = self.time.dtau_success
                    self.p_id_load = 0
                elif p_id == 1:
                    # run a new flux tube, but restart from an existing dtau
                    self.dtau_load = self.time.dtau_success

        gx_inputs = self.inputs

        if not dummy:
            # handle adiabatic species
            if self.adiabatic_species is not None:
                T_adiab = species_all[self.adiabatic_species].T().toFluxProfile()
                T_ref = species_all.ref_species.T().toFluxProfile()
                tau_fac = T_ref/T_adiab
                if self.adiabatic_species == "electron":
                    adiab_type = "electron"
                else:
                    adiab_type = "ion"
            N_species = species.N_species

            # get profile values on flux grid
            # these are 2d arrays, e.g. ns = ns[species_idx, rho_idx]
            ns, Ts, nu_ss = species.get_profiles_on_flux_grid(normalize=True, a_minor=geo.a_minor)
            beta_ref = species.ref_species.beta(geo.Btor).toFluxProfile()

        # loop over flux tubes, one at each rho
        outs = []
        for r_id in np.arange(self.N_fluxtubes):
            for a_id, zeta in enumerate(self.zeta_centers):

                path = self.out_dir
                if len(self.zeta_centers) > 1:
                    tag = f"t{t_id:02}-d{dtau_tag}-p{p_id}-r{r_id}-a{a_id}-j{pert_id}"
                    f_load = f"t{self.t_id_load:02}-d{self.dtau_load}-p{self.p_id_load}-r{r_id}-a{a_id}-j{pert_id}.restart.nc"
                else:
                    tag = f"t{t_id:02}-d{dtau_tag}-p{p_id}-r{r_id}-j{pert_id}"
                    f_load = f"t{self.t_id_load:02}-d{self.dtau_load}-p{self.p_id_load}-r{r_id}-j{pert_id}.restart.nc"

                fout = tag + '.in'
                f_save = tag + '.restart.nc'

                if geo.geo_option == 'vmec':
                    gx_inputs['Geometry']['geo_option'] = "'vmec'"
                    gx_inputs['Geometry']['desired_normalized_toroidal_flux'] = rho[r_id]**2
                    gx_inputs['Geometry']['zeta_center'] = zeta
                    gx_inputs['Geometry']['vmec_file'] = f"'{geo.geo_file}'"
                elif geo.geo_option == 'desc':
                    gx_inputs['Geometry']['geo_option'] = "'desc'"
                    gx_inputs['Geometry']['geo_file'] = f"'{geo.geo_file}'"
                    gx_inputs['Geometry']['rhotor'] = rho[r_id]
                    gx_inputs['Geometry']['zeta_center'] = zeta
                elif geo.geo_option == 'geqdsk':
                    gx_inputs['Geometry']['geo_option'] = "'gs2_geo'"
                    gx_inputs['Geometry']['rhoc'] = rho[r_id]
                    gx_inputs['Geometry']['iflux'] = 1
                    gx_inputs['Geometry']['bishop'] = 1
                    gx_inputs['Geometry']['equal_arc'] = "true"
                    gx_inputs['Geometry']['efit_eq'] = "true"
                    gx_inputs['Geometry']['eqfile'] = f"'{geo.geo_file}'"
                    if self.grid.flux_label == 'torflux':
                        gx_inputs['Geometry']['irho'] = 1
                    elif self.grid.flux_label == 'rminor':
                        gx_inputs['Geometry']['irho'] = 2
                    gx_inputs['Geometry']['eqfile'] = f"'{geo.geo_file}'"
                elif geo.geo_option == 'miller':
                    gx_inputs['Geometry']['geo_option'] = "'miller'"
                    gx_inputs['Geometry']['rhoc'] = rho[r_id]
                    gx_inputs['Geometry']['Rmaj'] = geo.AspectRatio
                    gx_inputs['Geometry']['R_geo'] = geo.R_geo[r_id]
                    gx_inputs['Geometry']['qinp'] = geo.qsf[r_id]
                    gx_inputs['Geometry']['shat'] = geo.shat[r_id]
                    gx_inputs['Geometry']['shift'] = geo.rmajor_prime[r_id]
                    gx_inputs['Geometry']['akappa'] = geo.kappa[r_id]
                    gx_inputs['Geometry']['akappri'] = geo.kappa_prime[r_id]
                    gx_inputs['Geometry']['tri'] = geo.delta[r_id]
                    gx_inputs['Geometry']['tripri'] = geo.delta_prime[r_id]
                    gx_inputs['Geometry']['betaprim'] = -beta_ref[r_id]*np.sum(ns[:, r_id]*Ts[:, r_id]*(kns[:, r_id]+kts[:, r_id]))

                if dummy:
                    gx_inputs['Time']['nstep'] = 1
                    gx_inputs['Dimensions']['nspecies'] = 1
                    gx_inputs['Restart']['restart'] = 'false'
                    gx_inputs['species']['mass'] = [1.0]
                    gx_inputs['species']['z'] = [1.0]
                    gx_inputs['species']['dens'] = [1.0]
                    gx_inputs['species']['temp'] = [1.0]
                    gx_inputs['species']['fprim'] = [1.0]
                    gx_inputs['species']['tprim'] = [1.0]
                    gx_inputs['species']['vnewk'] = [1.0]
                    gx_inputs['species']['type'] = ['ion']
                    gx_inputs['Boltzmann']['add_Boltzmann_species'] = 'false'
                else:
                    if self.nstep_gx:
                        gx_inputs['Time']['nstep'] = self.nstep_gx
                    else:
                        gx_inputs['Time']['nstep'] = 10000000

                    gx_inputs['Dimensions']['nspecies'] = N_species
                    gx_inputs['Dimensions']['nperiod'] = 1

                    # set reference beta
                    if self.electromagnetic:
                        gx_inputs['Physics']['beta'] = beta_ref[r_id]
                    else:
                        # force beta = 5e-4 to get electrostatic limit in GX to avoid omega_H mode when beta=0 (or fapar=0)
                        # keep actual beta value as comment for reference
                        gx_inputs['Physics']['beta'] = f"5e-4  # {beta_ref[r_id]}"

                    # set species parameters (these are lists)
                    gx_inputs['species']['mass'] = list(species.get_masses(normalize=True))
                    gx_inputs['species']['z'] = list(species.get_charges(normalize=True))
                    gx_inputs['species']['dens'] = list(ns[:, r_id])
                    gx_inputs['species']['temp'] = list(Ts[:, r_id])
                    gx_inputs['species']['fprim'] = list(kns[:, r_id])
                    gx_inputs['species']['tprim'] = list(kts[:, r_id] + self.tprim_shift)
                    gx_inputs['species']['vnewk'] = list(nu_ss[:, r_id]*self.collisionality_scaling_factor)
                    gx_inputs['species']['type'] = species.get_types_ion_electron()

                    if self.adiabatic_species is not None:
                        gx_inputs['Boltzmann']['add_Boltzmann_species'] = 'true'
                        gx_inputs['Boltzmann']['tau_fac'] = tau_fac[r_id]
                        gx_inputs['Boltzmann']['Boltzmann_type'] = f"'{adiab_type}'"
                    else:
                        gx_inputs['Boltzmann']['add_Boltzmann_species'] = 'false'

                    # Load restart if this is the first trinity timestep,
                    # or if the heat flux from the previous GX run was very small
                    if (self.no_restarts or (t_id == 0 and p_id == 0) or np.all(np.abs(self.qflux_prev[:,r_id]) < 1e-10) or np.any(np.isnan(self.qflux_prev[:,r_id]))):
                        gx_inputs['Restart']['restart'] = 'false'
                    elif not os.path.exists(os.path.join(path,f_load)):
                        # gx restart does not exist, start from scratch
                        gx_inputs['Restart']['restart'] = 'false'
                    else:
                        gx_inputs['Restart']['restart'] = 'true'
                        gx_inputs['Restart']['restart_from_file'] = '"{:}"'.format(os.path.join(path, f_load))
                        # restart from the same file (prev time step), to ensure parallelizability

                    # save restart file (always)
                    gx_inputs['Restart']['save_for_restart'] = 'true'

                # execute
                self.write_input(os.path.join(path, fout))
                out = self.submit_gx_job(tag, path)  # this returns a file name
                outs.append(out)
            # end zeta_center loop
        # end radius loop
        _time.sleep(2)

        return outs

    def submit_gx_job(self,tag,path):

        f_in = os.path.join(path, tag + '.in')
        f_tag = os.path.join(path, tag)
        gx_app = self.gx_app

        skip = False
        if os.path.exists(f_tag+".nc") or os.path.exists(f_tag+".out.nc"):
            skip = True
        if self.overwrite or (self.overwrite_p_neq_0 and self.time.iter_idx > 0):
            skip = False

        if not skip:
            # attempt to call
            system = os.environ['GK_SYSTEM']

            if self.run_command is not None:
                cmd = f"{self.run_command} {gx_app} {f_in}"
            elif system == 'stellar':
                cmd = f"srun -N 1 -t 2:00:00 --ntasks=1 --gpus-per-task=1 --exclusive {gx_app} {f_in}"  # stellar
            elif system == 'traverse':
                cmd = f"srun -u -N 1 -c 1 -t 6:00:00 --ntasks={self.gpus_per_gx} --gpus-per-node={min(4, self.gpus_per_gx)} --exact --overcommit {gx_app} {f_in}"
            elif system == 'satori':
                cmd = f"srun -N 1 -t 2:00:00 --ntasks=1 --gres=gpu:1 {gx_app} {f_in}"
            elif system == 'perlmutter':
                cmd = f"srun -u -N {int(math.ceil(self.gpus_per_gx/4))} -c 1 -t 24:00:00 --ntasks={self.gpus_per_gx} --gpus-per-node={min(4, self.gpus_per_gx)} --exact --overcommit {gx_app} {f_in}"
            elif system == 'polaris':
                i = self.gx_index
                d = 3-i*self.gpus_per_gx % 4
                ds = [f"{d - j}" for j in range(min(4,self.gpus_per_gx))]
                os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(ds)
                # this will distribute MPI processes for GX spaced by 8 cores among the 32 physical cores per Polaris node (--cpu-bind core)
                # note that --cpu-bind depth would distribute the processes among the 64 logical cores per node, which is not what we want
                cmd = f"mpiexec -n {self.gpus_per_gx} --ppn {min(4, self.gpus_per_gx)} --depth=8 --cpu-bind core --hostfile {os.path.join(path,'local_hostfile')}.{int((i*min(4,self.gpus_per_gx))/4)+1:02} {gx_app} {f_in}"
            elif system == 'summit':
                cmd = f"jsrun -n {self.gpus_per_gx} -a 1 -c 1 -g 1 {gx_app} {f_in}"
            else:
                cmd = f"{gx_app} {f_in}"

            self.info(f"> {cmd}")
            self.print_time()
            f_log = os.path.join(path, 'log.' + tag)
            with open(f_log, 'w') as fp:

                p = subprocess.Popen(cmd.split(), stdout=fp, stderr=fp)
                self.processes.append(p)
                self.logs.append(f_log)
                sys.stdout.flush()

        else:
            self.info(f'  gx output {f_tag} already exists')

        self.call_counter += 1

        self.gx_index += 1

        return f_tag  # this is a file stem

    def read_gx_fluxes(self, out_j, species_list):

        pflux = np.zeros((len(species_list), self.N_fluxtubes))
        qflux = np.zeros((len(species_list), self.N_fluxtubes))
        heat = np.zeros((len(species_list), self.N_fluxtubes))
        chi = np.zeros((len(species_list), self.N_fluxtubes))

        out_id = 0
        for r_id in np.arange(self.N_fluxtubes):
            # fluxes will be summed over all field lines
            if self.theta_zeta_fluxes:
                pflux_sz = None
                qflux_sz = None
                heat_sz = None
                theta_z = None
                zeta_z = None
                grho_z = None
            for a_id in np.arange(len(self.zeta_centers)):
                out = out_j[out_id]
                self.info(f'  read_gx_fluxes: reading output for {out}')
                new_gx_diag = False
                try:
                    f = Dataset(out+".nc", mode='r')
                except:
                    try:
                        f = Dataset(out+".out.nc", mode='r')
                        new_gx_diag = True
                    except:
                        assert False, f'  read_gx_fluxes: could not read output for {out}'

                # read qflux[t,s] = time trace of heat flux for each species
                is_gx = 0
                for i, s in enumerate(species_list):
                    if s.type == self.adiabatic_species:
                        pflux[i, r_id] = 0.0
                        qflux[i, r_id] = 0.0
                    else:
                        if self.theta_zeta_fluxes:
                            assert new_gx_diag, 'theta_zeta_fluxes require new gx diagnostics'

                            if i == 0:  # only need to get geometry stuff for first species
                                scale = f.groups['Geometry'].variables['theta_scale'][:]
                                theta = f.groups['Grids'].variables['theta'][:]*scale
                                nz = len(theta)

                                iota = 1/f.groups['Geometry'].variables['q'][:]
                                nfp = f.groups['Geometry'].variables['nfp'][:]
                                zeta_center = f.groups['Geometry'].variables['zeta_center'][:]
                                alpha = -iota*zeta_center
                                zeta = (theta - alpha)/iota

                                theta_p = map_to_periodic_range(theta, -np.pi, np.pi)
                                zeta_p = map_to_periodic_range(zeta, -np.pi/nfp, np.pi/nfp)
                                if theta_z is None:
                                    theta_z = np.zeros(nz*len(self.zeta_centers))
                                    zeta_z = np.zeros(nz*len(self.zeta_centers))
                                    grho_z = np.zeros(nz*len(self.zeta_centers))
                                theta_z[a_id*nz:(a_id+1)*nz] = theta_p
                                zeta_z[a_id*nz:(a_id+1)*nz] = zeta_p

                                jacobian = f.groups['Geometry'].variables['jacobian'][:]
                                grho = f.groups['Geometry'].variables['grho'][:]
                                fluxDenom = np.sum(jacobian*grho)
                                flux_fac = jacobian/fluxDenom
                                grho_z[a_id*nz:(a_id+1)*nz] = jacobian*grho

                            pflux_zt = f.groups['Diagnostics'].variables['ParticleFlux_zst'][:,is_gx,:]
                            pflux_z = np.mean(pflux_zt[int(len(pflux_zt[:,0])/2):,:], axis=0)*fluxDenom
                            if pflux_sz is None:
                                pflux_sz = np.zeros((len(species_list), nz*len(self.zeta_centers)))
                            pflux_sz[i,a_id*nz:(a_id+1)*nz] = pflux_z
                        else:
                            if new_gx_diag:
                                pflux_t = f.groups['Diagnostics'].variables['ParticleFlux_st'][:,is_gx]
                            else:
                                pflux_t = f.groups['Fluxes'].variables['pflux'][:,is_gx]
                            pflux[i, r_id] += self.median_estimator(pflux_t)

                        if self.theta_zeta_fluxes:
                            qflux_zt = f.groups['Diagnostics'].variables['HeatFlux_zst'][:,is_gx,:]
                            qflux_z = np.mean(qflux_zt[int(len(qflux_zt[:,0])/2):,:], axis=0)*fluxDenom
                            if qflux_sz is None:
                                qflux_sz = np.zeros((len(species_list), nz*len(self.zeta_centers)))
                            qflux_sz[i,a_id*nz:(a_id+1)*nz] = qflux_z
                        else:
                            if new_gx_diag:
                                qflux_t = f.groups['Diagnostics'].variables['HeatFlux_st'][:,is_gx]
                            else:
                                qflux_t = f.groups['Fluxes'].variables['qflux'][:,is_gx]
                            qflux[i, r_id] += self.median_estimator(qflux_t)

                        if self.theta_zeta_fluxes:
                            heat_zt = f.groups['Diagnostics'].variables['TurbulentHeating_zst'][:,is_gx,:]
                            heat_z = np.mean(heat_zt[int(len(heat_zt[:,0])/2):,:], axis=0)*fluxDenom
                            if heat_sz is None:
                                heat_sz = np.zeros((len(species_list), nz*len(self.zeta_centers)))
                            heat_sz[i,a_id*nz:(a_id+1)*nz] = heat_z
                        else:
                            if new_gx_diag:
                                heat_t = f.groups['Diagnostics'].variables['TurbulentHeating_st'][:,is_gx]
                                heat[i, r_id] += self.median_estimator(heat_t)
                        is_gx += 1
                out_id += 1
            # end zeta_center loop
            if self.theta_zeta_fluxes:
                from scipy.interpolate import griddata
                from scipy.integrate import trapz
                grid_zeta, grid_theta = np.mgrid[-np.pi/nfp:np.pi/nfp:100j, -np.pi:np.pi:100j]
                interp_grho = griddata((zeta_z, theta_z), grho_z, (grid_zeta, grid_theta), method='cubic', fill_value=1e-10)
                integrated_grho = trapz(trapz(interp_grho, grid_zeta[:,0]), grid_theta[0,:])
                for i, s in enumerate(species_list):
                    interp_pflux = griddata((zeta_z, theta_z), pflux_sz[i,:], (grid_zeta, grid_theta), method='cubic', fill_value=1e-10)
                    interp_qflux = griddata((zeta_z, theta_z), qflux_sz[i,:], (grid_zeta, grid_theta), method='cubic', fill_value=1e-10)
                    interp_heat = griddata((zeta_z, theta_z), heat_sz[i,:], (grid_zeta, grid_theta), method='cubic', fill_value=1e-10)

                    pflux[i, r_id] = trapz(trapz(interp_pflux, grid_zeta[:,0]), grid_theta[0,:])/integrated_grho
                    qflux[i, r_id] = trapz(trapz(interp_qflux, grid_zeta[:,0]), grid_theta[0,:])/integrated_grho
                    heat[i, r_id] = trapz(trapz(interp_heat, grid_zeta[:,0]), grid_theta[0,:])/integrated_grho
        # end radius loop

        if not self.theta_zeta_fluxes:
            # average fluxes over all field lines (already summed via += above)
            qflux = qflux/len(self.zeta_centers)
            pflux = pflux/len(self.zeta_centers)
            heat = heat/len(self.zeta_centers)

        qflux = qflux*self.qflux_scaling_factor

        # compute chi = Q / n grad T
        for i, s in enumerate(species_list):

            n_mid = s.n().toFluxProfile()
            p_mid = s.p().toFluxProfile()
            gradT_mid = -s.T().gradient_as_FluxProfile()
            #  Q_GB is nTv(rho/a)^2. To denormalize from Trinity units into physical units
            #  we only need the scaling, which is (n * T^5/2)
            nT5_2 = p_mid**2.5/n_mid**1.5  # n*T^(5/2)
            chi[i] = qflux[i] * nT5_2 / n_mid / gradT_mid

        # set Qe as a function of Qi, if using chi model
        if self.target_chie_over_chii != 0:
            # assume ion is 0, electron is 1, and electron is adia
            for i, s in enumerate(species_list):
                if s.type == self.adiabatic_species:
                    i_other = 1-i

                    # instead of scaling Q, get the chi=Q/n grad T, and scale that instead
                    chi[i] = self.target_chie_over_chii * chi[i_other]
                    n_mid = s.n().toFluxProfile()
                    p_mid = s.p().toFluxProfile()
                    gradT_mid = -s.T().gradient_as_FluxProfile()
                    nT5_2 = p_mid**2.5/n_mid**1.5  # n * T^(5/2)
                    qflux[i] = chi[i] * n_mid * gradT_mid / nT5_2

        self.info(f'\n  {self.label}: qflux = {qflux}\n')
        self.info(f'\n  {self.label}: pflux = {pflux}\n')

        if self.abort_on_nans and np.any(np.isnan(qflux)):
            assert False, f"Error: {self.label} fluxes have nans. Aborting..."

        return pflux, qflux, heat

    def median_estimator(self, flux):

        N = len(flux)
        med = np.ma.median([np.ma.median(flux[::-1][:k]) for k in np.arange(1,N)])

        return med

    def get_gx_eqdsk_geometry(self, geo):
        # do a dummy init-only GX calculation for each flux tube so that the
        # GX geometry information (B_ref, a_ref, grho, area) can be read
        out_j = self.run_gx_fluxtubes(self.rho, None, None, None, None, geo, 'i', dummy=True)
        # collect parallel runs
        self.wait()
        # read
        _time.sleep(WAIT_TIME)

        Btor_eq = []
        a_minor_eq = []
        grho_eq = []
        area_eq = []
        R_major_eq = []

        for r_id in np.arange(self.N_fluxtubes):
            out = out_j[r_id]
            new_gx_diag = False
            try:
                f = Dataset(out+".nc", mode='r')
            except:
                try:
                    f = Dataset(out+".out.nc", mode='r')
                    new_gx_diag = True
                except:
                    assert False, f'  read_gx_fluxes: could not read output for {out}'

            if new_gx_diag:
                Btor_eq.append(f.groups['Inputs']['B_ref'][:])     # This value will be the same for every rho. No need for a profile
                a_minor_eq.append(f.groups['Inputs']['a_ref'][:])   # This value will be the same for every rho. No need for a profile
                grho_eq.append(f.groups['Inputs']['grhoavg'][:])
                area_eq.append(f.groups['Inputs']['surfarea'][:])
                R_major_eq.append(f.groups['Geometry']['rmaj'][:])  # What do we want? R(axis)? Rgeo? R_avg(rho)? Check usage. Not used?
            else:
                Btor_eq.append(f.groups['Geometry']['B_ref'][:])     # This value will be the same for every rho. No need for a profile
                a_minor_eq.append(f.groups['Geometry']['a_ref'][:])   # This value will be the same for every rho. No need for a profile
                grho_eq.append(f.groups['Geometry']['grhoavg'][:])
                area_eq.append(f.groups['Geometry']['surfarea'][:])
                R_major_eq.append(f.groups['Geometry']['Rmaj'][:])  # What do we want? R(axis)? Rgeo? R_avg(rho)? Check usage. Not used?

        geo.Btor_eq = float(Btor_eq[0])
        geo.a_minor_eq = float(a_minor_eq[0])
        geo.R_major_eq = float(R_major_eq[0])

        geo.grho = pf.FluxProfile(np.asarray(grho_eq), self.grid)
        geo.area = pf.FluxProfile(np.asarray(area_eq), self.grid)

    def final_info(self):

        self.info(f"Total number of GX calls = {self.call_counter}")

    def print_time(self):

        dt = datetime.now()
        self.info(f'  time: {dt}')
        # ts = datetime.timestamp(dt)
        # self.info(f'  time: {ts}')
