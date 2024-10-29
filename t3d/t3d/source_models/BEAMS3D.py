import numpy as np
import subprocess
# from datetime import datetime
# import time as _time
import os
import sys
# from collections import OrderedDict
from t3d.SourceModels import SourceModel
from t3d.Logbook import logbook


class BEAMS3D_SourceModel(SourceModel):

    def __init__(self, pars, grid, time):

        # call base class constructor
        super().__init__(pars, grid, time)

        # environment variable containing path to xbeams3d executable
        BEAMS3D_PATH = self.BEAMS3D_PATH = pars.get('beams3d_path', os.environ.get("BEAMS3D_PATH") or "")
        self.beams3d_app = os.path.join(BEAMS3D_PATH, "xbeams3d")

        out_dir = self.out_dir = pars.get('beams3d_outputs', 'beams3d/')
        self.logfile = pars.get('beams3d_logfile', 'beams3d.log')
        beams3d_template = pars.get('beams3d_template', 'tests/regression/beams3d_template.in')
        self.overwrite = pars.get('overwrite', False)
        self.label = pars.get('label', 'BEAMS3D')  # label for diagnostics
        self.particle_source = pars.get('particle_source', False)
        self.power_source = pars.get('power_source', True)
        self.run_command = pars.get('run_command', None)
        self.run_options = pars.get('run_options', '')
        self.implicit = pars.get('implicit', False)

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
        self.info('  Looking for BEAMS3D files')
        self.info(f'    Expecting BEAMS3D template: {beams3d_template}')
        self.info(f'    Expecting BEAMS3D executable: {self.beams3d_app}')
        self.info(f'    BEAMS3D-Trinity output path: {out_dir}')

        found_beams3d = os.path.exists(self.beams3d_app)
        if not found_beams3d:
            self.info("  Error: xbeams3d executable not found! Make sure the BEAMS3D_PATH environment variable is set.")
            raise RuntimeError('BEAMS3D not found')

        self.info("")

        # load template
        self.read_input(beams3d_template)

        try:
            system = os.environ['GK_SYSTEM']
        except:
            self.info("  Error: must set GK_SYSTEM environment variable to use GX.")
            raise RuntimeError('GK_SYSTEM not set')

        self.processes = []
        self.call_counter = 0
        self.logs = []

    def __del__(self):
        ''' Use destructor to close the log file '''
        self.log.finalize()

    def read_input(self, beams3d_template):
        import f90nml
        self.input_data = f90nml.read(beams3d_template)

    def compute_sources(self, species, geo, norms):
        '''
        run BEAMS3D calculations and pass results to species.

        in the following, A_sjk indicates A[s, j, k] where
        s is species index
        j is rho index
        k is perturb index
        '''

        self.outs_n_pert = {}
        self.outs_T_pert = {}
        self.dn_jk = {}
        self.dT_jk = {}

        rho_j = self.rho

        # base profiles case
        pert_id = 0
        # get profile values on grid
        # these are 2d arrays, e.g. n_sj = n[s, j]
        n0_sj, T0_sj, df = species.get_profiles_on_grid(normalize=False, pert_n=None, pert_T=None)
        self.out_base = self.run_beams3d(rho_j, n0_sj, T0_sj, species, geo, pert_id)
        pert_id = pert_id + 1

        if self.implicit:
            # perturbed density cases
            # for each evolved density species, need to perturb density
            for stype in species.n_evolve_keys:
                self.dn_jk[stype] = []
                self.outs_n_pert[stype] = []
                for k in np.arange(self.N_radial-1):
                    pert_id = pert_id + 1
                    n_sj, T_sj, dn = species.get_profiles_on_grid(normalize=False, pert_n=stype, pert_T=None, pert_idx=k)
                    self.dn_jk[stype].append(dn)
                    out = self.run_beams3d(rho_j, n_sj, T_sj, species, geo, pert_id)
                    self.outs_n_pert[stype].append(out)

            # perturbed temperature cases
            # for each evolved temperature species, need to perturb temperature
            for stype in species.T_evolve_keys:
                self.dT_jk[stype] = []
                self.outs_T_pert[stype] = []
                for k in np.arange(self.N_radial-1):
                    pert_id = pert_id + 1
                    n_sj, T_sj, dT = species.get_profiles_on_grid(normalize=False, pert_n=None, pert_T=stype)
                    self.dT_jk[stype].append(dT)
                    out = self.run_beams3d(rho_j, n_sj, T_sj, species, geo, pert_id)
                    self.outs_T_pert[stype].append(out)

    def run_beams3d(self, rho, ns, Ts, species, geo, pert_id):
        '''
        run a BEAMS3D calculation
        '''
        import f90nml

        t_id = self.time.step_idx  # time integer
        p_id = self.time.iter_idx  # Newton iteration number

        outs = []

        # write BEAMS3D input file (fortran namelist)
        inputs = self.input_data
        inputs['beams3d_input']['te_aux_s'] = rho.tolist()
        inputs['beams3d_input']['te_aux_f'] = Ts[species.sidx_electron, :].tolist()
        inputs['beams3d_input']['ne_aux_s'] = rho.tolist()
        inputs['beams3d_input']['ne_aux_f'] = ns[species.sidx_electron, :].tolist()

        m_proton_cgs = 1.67e-24  # mass of proton in grams
        Zs = species.get_charges(normalize=False, drop='electron')
        ms = species.get_masses(normalize=False, drop='electron')*m_proton_cgs
        inputs['beams3d_input']['ni_aux_z'] = Zs.tolist()
        inputs['beams3d_input']['ni_aux_m'] = ms.tolist()

        inputs['beams3d_input']['ni_aux_s'] = rho.tolist()
        nis = np.delete(ns, species.sidx_electron, axis=0)
        inputs['beams3d_input']['ni_aux_f'] = nis.tolist()

        dir = f"t{t_id:02}-p{p_id}-{pert_id}"
        tag = self.vmec_tag
        input_file = os.path.join(self.out_dir, f"input.{tag}")
        f90nml.write(inputs, input_file, force=True)

        out_file = self.submit_beams3d_job(tag, self.out_dir)
        return out_file

    def submit_beams3d_job(self, tag, path):
        system = os.environ['GK_SYSTEM']

        out_file = os.path.join(path, f"{tag}.h5")
        if (not os.path.exists(out_file) or self.overwrite):
            if self.run_command is not None:
                cmd = f"{self.run_command} {self.beams3d_app} -vmec {tag} -depo {self.run_options}"
            elif system == 'perlmutter':
                cmd = f"srun -u -t 0:30:0 -N 1 --exact --overcommit --ntasks=32 --cpus-per-task=1 --gpus-per-node=0 {self.beams3d_app} -vmec {tag} -depo {self.run_options}"
            else:
                cmd = f"{self.beams3d_app} -vmec {tag} -depo {self.run_options}"

            self.info(f"> {cmd}")

            wd = os.getcwd()
            os.chdir(path)
            p = subprocess.Popen(cmd, shell=True)
            os.chdir(wd)
            self.processes.append(p)
            sys.stdout.flush()
        else:
            self.info(f'  beams3d output {out_file} already exists')

        return out_file

    def collect_results(self, species, geo, norms):
        # collect parallel runs
        self.wait()

        # read BEAMS3D output data and pass results to species
        # base
        Sn0_sj, Sp0_sj = self.read_beams3d_deposition(self.out_base, species, geo, norms)
        species.add_source(Sn0_sj, Sp0_sj, label=self.label)

        pert_id = 0
        if self.implicit:
            # perturbed density cases
            for stype in species.n_evolve_keys:
                for k in np.arange(self.N_radial-1):
                    dn_j = self.dn_jk[stype][k]
                    Sn_sj, Sp_sj = self.read_beams3d_deposition(self.outs_n_pert[stype][k], species, geo, norms)
                    dSn_dn_sj = (Sn_sj - Sn0_sj) / dn_j
                    dSp_dn_sj = (Sp_sj - Sp0_sj) / dn_j
                    species.add_dS_dn(stype, dSn_dn_sj, dSp_dn_sj)
                    pert_id = pert_id + 1

            # perturbed temperature cases
            for stype in species.T_evolve_keys:
                for k in np.arange(self.N_radial-1):
                    dT_j = self.dT_jk[stype][k]
                    Sn_sj, Sp_sj = self.read_beams3d_deposition(self.outs_T_pert[stype][k], species, geo, norms)
                    dSn_dT_sj = (Sn_sj - Sn0_sj) / dT_j
                    dSp_dT_sj = (Sp_sj - Sp0_sj) / dT_j
                    species.add_dS_dT(stype, dSn_dT_sj, dSp_dT_sj)
                    pert_id = pert_id + 1

    def wait(self):

        # wait for a list of subprocesses to finish
        #    and reset the list

        exitcodes = [p.wait() for p in self.processes]
        self.info(f'BEAMS3D exitcodes = {exitcodes}')
        self.processes = []  # reset

    def read_beams3d_deposition(self, out_file, species, geo, norms):
        import h5py

        Sn = np.zeros((species.N_species, self.N_radial))
        Sp = np.zeros((species.N_species, self.N_radial))

        # read hdf5 data file
        data = h5py.File(out_file, 'r')
        nr = data['ns_prof1'][0]
        raxis = np.linspace(0, 1, nr)
        ndot_prof = data['ndot_prof'][:]      # 1/(m^3 s)
        epower_prof = data['epower_prof'][:]  # W/m^3
        ipower_prof = data['ipower_prof'][:]  # W/m^3

        # interpolate onto T3D rho grid and normalize to T3D units
        f = interp1d(raxis, ndot_prof, bounds_error=False, fill_value='extrapolate')
        ndot = f(self.rho)*1e-20/norms.Sn_ref_SI20

        f = interp1d(raxis, epower_prof, bounds_error=False, fill_value='extrapolate')
        epower = f(self.rho)*1e-6/norms.P_ref_MWm3

        f = interp1d(raxis, ipower_prof, bounds_error=False, fill_value='extrapolate')
        ipower = f(self.rho)*1e-6/norms.P_ref_MWm3

        # assume deposition only goes to electrons and bulk ions
        for i, s in enumerate(species.get_species_list()):
            if s.type == species.bulk_ion.type:
                Sn[i, r_id] = ndot
                Sp[i, r_id] = ipower
            elif s.type == "electron":
                Sn[i, r_id] = ndot
                Sp[i, r_id] = epower

        return Sn, Sp
