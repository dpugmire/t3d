import numpy as np
import subprocess
from datetime import datetime
import time as _time
import os
import sys
from t3d.FluxModels import FluxModel
from t3d.Logbook import logbook

WAIT_TIME = 1  # this should come from the Trinity Engine


class SFINCS_FluxModel(FluxModel):

    def __init__(self, pars, grid, time):

        # call base class constructor
        super().__init__(pars, grid, time)

        # environment variable containing path to sfincs.x executable
        SFINCS_PATH = self.SFINCS_PATH = os.environ.get("SFINCS_PATH") or pars.get('sfincs_path', "")

        out_dir = self.out_dir = pars.get('sfincs_outputs', 'sfincs/')
        self.logfile = pars.get('sfincs_logfile', 'sfincs.log')
        sfincs_template = pars.get('sfincs_template', 'tests/sfincs/sfincs_template.in')
        self.overwrite = pars.get('overwrite', False)
        self.label = pars.get('label', 'SFINCS')  # label for diagnostics

        # SFINCS uses sqrt(2T/m) as v normalization but we modify its behavior
        # to allow simulations with GX
        self.vt_sqrt_2 = False

        # use a non-default run command (e.g. "srun ...")
        # full command line will be cmd = f"{self.run_command} {gx_app} {f_in}"
        self.run_command = pars.get('run_command', None)
        self.cpus_per_sfincs = pars.get('cpus_per_sfincs', 1)
        self.mpis_per_sfincs = pars.get('mpis_per_sfincs', 1)
        # total number of allocated cpus
        self.cpus_total = pars.get('cpus_total', 96)
        self.used_cpus = 0
        self.sfincs_index = 0

        found_path = os.path.exists(out_dir)
        if not found_path:
            os.mkdir(out_dir)

        # Create logger
        self.log = logbook()
        self.log.set_handlers(term_stream=False,
                              file_stream=True,
                              file_handler=os.path.join(self.out_dir, self.logfile))
        self.info = self.log.info  # Short-cut command
        self.errr = self.log.errr  # Short-cut command

        # Check file path
        self.info("  Looking for SFINCS files")
        self.info(f"    Expecting SFINCS template: {sfincs_template}")
        self.info(f"    Expecting SFINCS executable: {SFINCS_PATH+'/sfincs'}")
        self.info(f"    SFINCS-Trinity output path: {out_dir}")

        found_sfincs = os.path.exists(SFINCS_PATH+"/sfincs")
        if not found_sfincs:
            self.errr("sfincs executable not found! Make sure the SFINCS_PATH environment variable is set.")
            raise RuntimeError('SFINCS not found')

        self.info("")

        self.read_sfincs_input(sfincs_template)

        try:
            system = os.environ['GK_SYSTEM']
        except:
            self.errr("Must set GK_SYSTEM environment variable to use SFINCS.")
            raise RuntimeError('GK_SYSTEM not set')

        # set up node files for PBS systems like Polaris
        if system == "polaris":  # or other PBS systems eventually
            PBS_NODEFILE = os.environ.get("PBS_NODEFILE")
            NODES_PER_MPI = max(1,int(self.mpis_per_sfincs*self.cpus_per_sfincs/32))
            cmd = f"split --lines={NODES_PER_MPI} --numeric-suffixes=1 --suffix-length=2 {PBS_NODEFILE} {os.path.join(self.out_dir,'local_hostfile')}."
            p = subprocess.Popen(cmd.split())
            p.wait()

        self.first = True
        self.processes = []
        self.firstrunning = 0

    def __del__(self):
        ''' Use destructor to close the log file '''
        self.log.finalize()

    def logfile_info(self) -> str:
        ''' Return string of log file location '''
        return self.log.file_handler.name

    # run SFINCS calculations and pass fluxes and flux jacobian to SpeciesDict species
    def compute_fluxes(self, species, geo, norms):
        # in the following, A_sjk indicates A[s, j, k] where
        # s is species index
        # j is rho index
        # k is perturb index

        self.outs_jk = []
        self.dkap_jk = []
        self.sfincs_index = 0
        rho_j = self.rho

        if geo.geo_option == 'vmec':
            self.namelist["geometryparameters"]["geometryscheme"] = "5"
            self.namelist["geometryparameters"]["equilibriumfile"] = '"../../../{}"'.format(geo.geo_file.strip())
            self.namelist["geometryparameters"]["inputradialcoordinate"] = "3"
            self.namelist["geometryparameters"]["inputradialcoordinateforgradients"] = "3"
            self.namelist["geometryparameters"]["vmecradialoption"] = "0"

        self.namelist["speciesparameters"]["zs"] = ' '.join(map(str, species.get_charges(normalize=True)))
        self.namelist["speciesparameters"]["mhats"] = ' '.join(map(str, species.get_masses(normalize=True)))
        self.namelist['general']['ambipolarsolve'] = '.true.'
        self.namelist['general']['ambipolarsolveoption'] = '2'

        # set normalization parameters so that SFINCS reference parameters match T3D ones:
        # n_bar = 10^20 m^-3
        # T_bar = 1 keV
        # B_bar = 1 T
        # m_bar = m_ref
        # R_bar = a_meters
        self.namelist['physicsparameters']['Delta'] = norms.rho_star*np.sqrt(2/norms.vtfac)
        self.namelist['physicsparameters']['alpha'] = 1.0
        self.namelist['physicsparameters']['nu_n'] = 8.33e-3*norms.a_meters

        # base profiles case
        pert_id = 0
        # get gradient values on flux grid
        # these are 2d arrays, e.g. kn_sj = kap_n[s, j]
        kn0_sj, kT0_sj, kp0_sj, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        self.dkap_jk.append(dkap)  # this is a dummy entry, unused
        outs_j = self.run_sfincs(rho_j, kn0_sj, kT0_sj, species, geo, pert_id)
        self.outs_jk.append(outs_j)

        # perturbed density cases
        # for each evolved density species, need to perturb density
        for stype in species.n_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=stype, pert_T=None)
            self.dkap_jk.append(dkap)
            outs_j = self.run_sfincs(rho_j, kn_sj, kT_sj, species, geo, pert_id)
            self.outs_jk.append(outs_j)

        # perturbed temperature cases
        # for each evolved temperature species, need to perturb temperature
        for stype in species.T_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=stype)
            self.dkap_jk.append(dkap)
            outs_j = self.run_sfincs(rho_j, kn_sj, kT_sj, species, geo, pert_id)
            self.outs_jk.append(outs_j)

    def collect_results(self, species, geo, norms):
        # collect parallel runs
        self.wait()

        # read
        _time.sleep(WAIT_TIME)

        # read SFINCS output data and pass results to species
        # base
        pert_id = 0
        pflux0_sj, qflux0_sj = self.read_sfincs_fluxes(self.outs_jk[pert_id], species, geo, norms)
        species.add_flux(pflux0_sj, qflux0_sj, None, label=self.label)

        pert_id = pert_id + 1

        # perturbed density cases
        for stype in species.n_evolve_keys:
            dkn_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj = self.read_sfincs_fluxes(self.outs_jk[pert_id], species, geo, norms)
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkn_j
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkn_j
            species.add_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj, None)
            pert_id = pert_id + 1

        # perturbed temperature cases
        for stype in species.T_evolve_keys:
            dkT_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj = self.read_sfincs_fluxes(self.outs_jk[pert_id], species, geo, norms)
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkT_j
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkT_j
            species.add_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj, None)
            pert_id = pert_id + 1

    def wait(self):

        # wait for a list of subprocesses to finish
        #    and reset the list

        exitcodes = [p.wait() for p in self.processes]
        self.info(f'SFINCS exitcodes = {exitcodes}')
        self.processes = []  # reset
        self.firstrunning = 0
        self.used_cpus = 0
        # could add some sort of timer here

    # run a SFINCS calculation, given gradient values kns and kts
    def run_sfincs(self, rho, kns, kts, species, geo, pert_id):
        t_id = self.time.step_idx  # time integer
        p_id = self.time.iter_idx  # Newton iteration number

        # get profile values on flux grid
        # these are 2d arrays, e.g. ns = ns[species_idx, rho_idx]
        ns, Ts, nu_ss = species.get_profiles_on_flux_grid(normalize=False, a_minor=geo.a_minor)

        run_dir = self.out_dir + f"/t{t_id:02}-p{p_id}-{pert_id}"
        found_path = os.path.exists(run_dir)
        if not found_path:
            os.mkdir(run_dir)

        flux_files = []

        flux_dirs = [run_dir + f"/rho{n:02}" for n in np.arange(self.N_fluxtubes)]
        for r_id, flux_dir in enumerate(flux_dirs):
            found_path = os.path.exists(flux_dir)
            if not found_path:
                os.mkdir(flux_dir)
            input_file = flux_dir + "/input.namelist"

            sfincs_namelist = self.namelist
            sfincs_namelist["speciesparameters"]["nhats"] = ' '.join(map(str, ns[:, r_id]))
            sfincs_namelist["speciesparameters"]["thats"] = ' '.join(map(str, Ts[:, r_id]))
            sfincs_namelist["speciesparameters"]["dnhatdrns"] = ' '.join(map(str, -ns[:, r_id]*kns[:, r_id]))
            sfincs_namelist["speciesparameters"]["dthatdrns"] = ' '.join(map(str, -Ts[:, r_id]*kts[:, r_id]))
            sfincs_namelist["geometryparameters"]["rn_wish"] = rho[r_id]

            geo_file_path = geo.geo_file.strip()
            geo_filename = os.path.basename(geo.geo_file)
            if (os.path.exists(os.path.join(flux_dir,geo_filename)) is False):
                cmd = f"cp {geo_file_path} {flux_dir}"
                subprocess.Popen(cmd, shell=True)
            self.namelist["geometryparameters"]["equilibriumfile"] = '"{}"'.format(geo_filename)

            self.write_sfincs_input(input_file,sfincs_namelist)
            flux_files.append(self.submit_sfincs_job(flux_dir))
        return flux_files

    def submit_sfincs_job(self,flux_dir):

        flux_file = flux_dir + "/sfincsOutput.h5"
        sfincs_app = f"{self.SFINCS_PATH}/sfincs"

        if (os.path.exists(flux_file) is False or self.overwrite):
            system = os.environ['GK_SYSTEM']
            if self.run_command is not None:
                cmd = f"{self.run_command} {sfincs_app}"
            elif system == 'perlmutter':
                cmd = f"srun -u -t 0:30:0 -N 1 --exact --overcommit --ntasks={self.cpus_per_sfincs} --cpus-per-task=1 --gpus-per-node=0 --mem-per-cpu={int(256/self.N_fluxtubes)}G {sfincs_app}"
            elif system == 'stellar':
                cmd = f"export OMP_NUM_THREADS={self.cpus_per_sfincs}; srun -t 4:00:0 --ntasks={self.mpis_per_sfincs} -c {self.cpus_per_sfincs} {sfincs_app} > output.out"
            elif system == 'polaris':
                # SFINCS will only be run on hardware threads 32-63 on each node, so that GX or another GPU-based flux model can use threads 0-31 on the same node
                i = self.sfincs_index
                sfincs_per_node = max(1, 32//self.mpis_per_sfincs)
                cpu_list = ':'.join(map(str, (np.arange(min(32, self.mpis_per_sfincs))+self.mpis_per_sfincs*(i % sfincs_per_node)+32) % 64))
                cmd = f"mpiexec -n {self.mpis_per_sfincs} --ppn {min(32, (self.mpis_per_sfincs))} --cpu-bind list:{cpu_list} --env OMP_NUM_THREADS={self.cpus_per_sfincs} --hostfile ../../local_hostfile.{i//sfincs_per_node+1:02} {sfincs_app} -ksp_view -mat_mumps_cntl_1 1e-3 > output.out"
            elif system == 'summit':
                cmd = f"jsrun -n 1 -a {self.mpis_per_sfincs} -c {self.mpis_per_sfincs} -g 0 -EOMP_NUM_THREADS={self.cpus_per_sfincs} {sfincs_app} -ksp_view -mat_mumps_cntl_1 1e-3 > output.out"
            else:
                cmd = f"mpirun -np 2 {sfincs_app} -ksp_max_it 500 >output.out"

            self.info(f"> {cmd}")
            self.print_time()

            wd = os.getcwd()
            os.chdir(flux_dir)
            p = subprocess.Popen(cmd, shell=True)
            os.chdir(wd)
            self.processes.append(p)
            sys.stdout.flush()
            # self.used_cpus += self.cpus_per_sfincs*self.mpis_per_sfincs
            # if self.used_cpus + self.cpus_per_sfincs*self.mpis_per_sfincs > self.cpus_total:
            #    self.processes[self.firstrunning].wait()
            #    for process in self.processes[self.firstrunning:]:
            #        if process.poll() is not None:
            #            self.used_cpus -= self.cpus_per_sfincs*self.mpis_per_sfincs
            #            self.firstrunning += 1
            # if system == 'lomideb':
            #    p.wait()
        else:
            self.info(f'  sfincs output {flux_file} already exists')

        self.sfincs_index += 1

        return flux_file

    def read_sfincs_fluxes(self, outs_j, species, geo, norms):
        pflux = np.zeros((species.N_species, self.N_fluxtubes))
        qflux = np.zeros((species.N_species, self.N_fluxtubes))

        for r_id in np.arange(self.N_fluxtubes):
            out = outs_j[r_id]

            import h5py
            h5file = h5py.File(out,'r')
            for i, s in enumerate(species.get_species_list()):
                if self.namelist['physicsparameters']['includephi1'] == '.true.':
                    pflux[i, r_id] = h5file['particleFlux_vd_rN'][i][-1]
                    qflux[i, r_id] = h5file['heatFlux_vd_rN'][i][-1]
                else:
                    pflux[i, r_id] = h5file['particleFlux_vm_rN'][i][-1]
                    qflux[i, r_id] = h5file['heatFlux_vm_rN'][i][-1]

            h5file.close()

        # SFINCS fluxes are in normalized sfincs units. need to normalize to gyroBohm
        # note SFINCS uses vt = sqrt(2T/m), but to make this model compatible with other
        # models with vt_sqrt_2 = False, multiply by sqrt(2) factors
        for i, s in enumerate(species.get_species_list()):
            qflux[i, :] = qflux[i, :]*np.sqrt(8)*geo.Btor**2*species.ref_species.mass**0.5/(species.ref_species.n()*species.ref_species.T()**2.5).toFluxProfile()/geo.grho*norms.rho_star**-2
            pflux[i, :] = pflux[i, :]*np.sqrt(2)*geo.Btor**2*species.ref_species.mass**0.5/(species.ref_species.n()*species.ref_species.T()**1.5).toFluxProfile()/geo.grho*norms.rho_star**-2

        self.info(f'\n  {self.label}: qflux = {qflux}\n')
        self.info(f'\n  {self.label}: pflux = {pflux}\n')
        return pflux, qflux

    def write_sfincs_input(self, fout, namelist):
        # do not overwrite
        if (os.path.exists(fout) and not self.overwrite):
            self.info(f'  input exists, skipping write {fout}')
            return
        with open(fout,'w') as f:

            for item in namelist.items():

                if (type(item[1]) is not dict):
                    print('  %s = %s ' % item, file=f)
                    continue

                header, nest = item
                print('\n&%s' % header, file=f)
                if len(nest.keys()) <= 0:
                    print('/\n', file=f)
                    continue
                longest_key = max(nest.keys(), key=len)
                N_space = len(longest_key)
                for pair in nest.items():
                    s = '  {:%i}  =  {}' % N_space
                    print(s.format(*pair), file=f)
                print('/\n', file=f)

        self.info(f'  wrote input: {fout}')

    def read_sfincs_input(self,fin):
        with open(fin) as f:
            data = f.readlines()

        obj = {}
        header = ''
        for line in data:

            # strip comments
            if line.find('!') > -1:
                end = line.find('!')
                line = line[:end]

            # parse headers
            if line.find('&') == 0:
                header = line.split('&')[1].strip().lower()
                obj[header] = {}
                continue

            if len(line.strip()) > 0 and line.strip()[0] == '/':
                header = ''
            # skip blanks
            if line.find('=') < 0:
                continue

            # store data
            key, value = line.split('=')
            key = key.strip().lower()
            value = value.strip()

            if header == '':
                obj[key] = value
            else:
                obj[header][key] = value

        # check for all possible header types in GX input file,
        # and initialize empty if not found
        try:
            obj['general']
        except:
            obj['general'] = {}
        try:
            obj['geometryparameters']
        except:
            obj['geometryparameters'] = {}
        try:
            obj['speciesparameters']
        except:
            obj['speciesparameters'] = {}
        try:
            obj['physicsparameters']
        except:
            obj['physicsparameters'] = {}
        try:
            obj['resolutionparameters']
        except:
            obj['resolutionparameters'] = {}
        try:
            obj['othernumericalparameters']
        except:
            obj['othernumericalparameters'] = {}
        try:
            obj['preconditioneroptions']
        except:
            obj['preconditioneroptions'] = {}
        try:
            obj['export_f']
        except:
            obj['export_f'] = {}
        try:
            obj['adjointoptions']
        except:
            obj['adjointoptions'] = {}

        self.namelist = obj
        self.filename = fin

    def print_time(self):

        dt = datetime.now()
        self.info(f'  time: {dt}')
        # ts = datetime.timestamp(dt)
        # print('  time', ts)
