import numpy as np
import subprocess
from datetime import datetime
import time as _time
import os
import sys
from t3d.FluxModels import FluxModel
from t3d.Logbook import logbook

WAIT_TIME = 1  # this should come from the Trinity Engine


class KNOSOS_FluxModel(FluxModel):

    def __init__(self, pars, grid, time):

        # call base class constructor
        super().__init__(pars, grid, time)

        # environment variable containing path to knosos.x executable
        KNOSOS_PATH = self.KNOSOS_PATH = os.environ.get("KNOSOS_PATH") or pars.get('knosos_path', "")

        out_dir = self.out_dir = pars.get('knosos_outputs', 'knosos/')
        self.logfile = pars.get('knosos_logfile', 'knosos.log')
        self.adiabatic_electrons = pars.get('adiabatic_electrons', False)
        self.overwrite = pars.get('overwrite', False)
        self.overwrite_p_neq_0 = False
        self.dirdb = pars.get('DIRDB', None)
        self.dirs = pars.get('DIRS', None)
        self.sdkes = pars.get('SDKES', None)
        self.label = pars.get('label', 'KNOSOS')  # label for diagnostics

        self.vt_sqrt_2 = None  # KNOSOS doesn't care about vt definition

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
        self.info("  Looking for KNOSOS files")
        self.info(f"    Expecting KNOSOS executable: {KNOSOS_PATH}/knosos.x")
        self.info(f"    KNOSOS-Trinity output path: {out_dir}")

        found_knosos = os.path.exists(KNOSOS_PATH+"/knosos.x")
        if not found_knosos:
            self.info("ERROR: knosos.x executable not found! Make sure the KNOSOS_PATH environment variable is set.")
            raise RuntimeError('KNOSOS not found')

        # write the input.model file
        with open(out_dir+"/input.model", 'w') as f:
            print("&model\nPREDICTIVE = .TRUE.\n/", file=f)
        # write the input.surfaces file
        with open(out_dir+"/input.surfaces", 'w') as f:
            # KNOSOS uses s = rho**2 as the radial coordinate
            print(f"&surfaces\nNS = {self.N_fluxtubes}\nS = {','.join(map(str,self.rho**2))}", file=f)
            if self.dirdb is not None:
                print(f"DIRDB = '{self.dirdb}'", file=f)
            if self.dirs is not None:
                print(f"DIRS = {self.dirs}", file=f)
            if self.sdkes is not None:
                print(f"SDKES = {self.sdkes}", file=f)
            print("/", file=f)
        with open(out_dir+"/input.parameters", 'w') as f:
            print("&parameters\nERMIN=-40\nERMAX=40\n/", file=f)

        self.first = True
        self.processes = []

    def __del__(self):
        ''' Use destructor to close the log file '''
        self.log.finalize()

    def logfile_info(self) -> str:
        ''' Return string of log file location '''
        return self.log.file_handler.name

    def make_booz(self, geo_file):
        # KNOSOS requires the vmec file to be pre-processed by BOOZ_XFORM
        try:
            import booz_xform as bx
        except:
            self.info("ERROR: KNOSOS model requires booz_xform to be installed. Use 'pip install booz_xform'")
            exit(1)
        self.info("    Running BOOZ_XFORM...")
        booz = bx.Booz_xform()
        booz.verbose = 0
        booz.read_wout(geo_file)
        booz.run()
        # write netCDF file that KNOSOS will read
        booz.write_boozmn(f"{self.out_dir}/boozmn.nc")
        self.info(f"    Wrote {self.out_dir}/boozmn.nc")

    # run KNOSOS calculations and pass fluxes and flux jacobian to SpeciesDict species
    def compute_fluxes(self, species, geo, norms):
        # in the following, A_sjk indicates A[s, j, k] where
        # s is species index
        # j is rho index
        # k is perturb index

        # write input.species, but only on first call
        if self.first:
            self.first = False

            with open(self.out_dir + "/input.species", "w") as f:
                print("&species", file=f)
                print(f"NBB = {species.N_species}", file=f)
                charge_list = f"{species['electron'].Z},{species.bulk_ion.Z}"
                mass_list = f"{species['electron'].mass},{species.bulk_ion.mass}"
                if self.adiabatic_electrons:
                    reg_list = "-2,0"
                else:
                    reg_list = "0,0"
                frac_list = "1,1"
                if species.N_species > 2:
                    other_ion_charges = np.zeros(species.N_species-2)
                    other_ion_masses = np.zeros(species.N_species-2)
                    for i, s in enumerate(species.other_ions_dict.values()):
                        other_ion_charges[i] = s.Z
                        other_ion_masses[i] = s.mass
                        reg_list = reg_list + ",0"
                        frac_list = frac_list + ",0.001"  # assume all non-bulk ions are trace
                    charge_list = charge_list + "," + ','.join(map(str,other_ion_charges))
                    mass_list = mass_list + "," + ','.join(map(str,other_ion_masses))
                print(f"ZB = {charge_list}", file=f)
                print(f"AB = {mass_list}", file=f)
                print(f"REGB = {reg_list}", file=f)
                print(f"FRACB = {frac_list}", file=f)
                print("/", file=f)

        self.outs_jk = []
        self.dkap_jk = []

        rho_j = self.rho

        # base profiles case
        pert_id = 0
        # get gradient values on flux grid
        # these are 2d arrays, e.g. kn_sj = kap_n[s, j]
        kn0_sj, kT0_sj, kp0_sj, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        self.dkap_jk.append(dkap)  # this is a dummy entry, unused
        outs_j = self.run_knosos(rho_j, kn0_sj, kT0_sj, species, geo, pert_id)
        self.outs_jk.append(outs_j)

        # perturbed density cases
        # for each evolved density species, need to perturb density
        for stype in species.n_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=stype, pert_T=None)
            self.dkap_jk.append(dkap)
            outs_j = self.run_knosos(rho_j, kn_sj, kT_sj, species, geo, pert_id)
            self.outs_jk.append(outs_j)

        # perturbed temperature cases
        # for each evolved temperature species, need to perturb temperature
        for stype in species.T_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=stype)
            self.dkap_jk.append(dkap)
            outs_j = self.run_knosos(rho_j, kn_sj, kT_sj, species, geo, pert_id)
            self.outs_jk.append(outs_j)

    def collect_results(self, species, geo, norms):
        # collect parallel runs
        self.wait()

        # read
        _time.sleep(WAIT_TIME)

        # read KNOSOS output data and pass results to species
        # base
        pert_id = 0
        pflux0_sj, qflux0_sj = self.read_knosos_fluxes(self.outs_jk[pert_id], species, geo, norms)
        species.add_flux(pflux0_sj, qflux0_sj, None, label=self.label)

        pert_id = pert_id + 1

        # perturbed density cases
        for stype in species.n_evolve_keys:
            dkn_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj = self.read_knosos_fluxes(self.outs_jk[pert_id], species, geo, norms)
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkn_j
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkn_j
            species.add_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj, None)
            pert_id = pert_id + 1

        # perturbed temperature cases
        for stype in species.T_evolve_keys:
            dkT_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj = self.read_knosos_fluxes(self.outs_jk[pert_id], species, geo, norms)
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkT_j
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkT_j
            species.add_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj, None)
            pert_id = pert_id + 1

    def wait(self):

        # wait for a list of subprocesses to finish
        #    and reset the list

        exitcodes = [p.wait() for p in self.processes]
        self.info(f'KNOSOS exitcodes = {exitcodes}')
        self.processes = []  # reset

        # could add some sort of timer here

    # run a KNOSOS calculation, given gradient values kns and kts
    def run_knosos(self, rho, kns, kts, species, geo, pert_id):

        t_id = self.time.step_idx  # time integer
        p_id = self.time.iter_idx  # Newton iteration number
        dtau = self.time.dtau      # T3D timestep

        # handle case of failed timestep, read existing p=0 iteration to avoid re-running
        if dtau < self.time.dtau_old and p_id == 0:
            dtau = self.time.dtau_success

        # get profile values on flux grid
        ne = species['electron'].n().toFluxProfile()
        Te = species['electron'].T().toFluxProfile()
        Ti = species.bulk_ion.T().toFluxProfile()

        for i, s in enumerate(species.get_species_list()):
            if s.type == species.bulk_ion.type:
                kTi = kts[i,:]
            elif s.type == "electron":
                kne = kns[i,:]
                kTe = kts[i,:]

        run_dir = self.out_dir + f"/t{t_id:02}-p{p_id}-dt{dtau}-{pert_id}"
        found_path = os.path.exists(run_dir)
        if not found_path:
            os.mkdir(run_dir)
        profile_file = run_dir + "/profiles_trin.txt"

        # even though KNOSOS uses s = rho**2 as the radial coordinate, the profiles_trin.txt file is set up to use
        # rho and d(log n)/drho etc. the conversion from rho to s happens inside KNOSOS when profiles_trin.txt is read.
        with open(profile_file, "w") as f:
            print("rho, ne (10^20 m^-3), d(log ne)/drho, Ti (keV), d(log Ti)/drho, Te (keV), d(log Te)/drho", file=f)
            for r_id in np.arange(self.N_fluxtubes):
                print(f"{rho[r_id]}\t{ne[r_id]}\t{-kne[r_id]}\t{Ti[r_id]}\t{-kTi[r_id]}\t{Te[r_id]}\t{-kTe[r_id]}", file=f)

        flux_files = self.submit_knosos_job(run_dir)
        return flux_files

    def submit_knosos_job(self,run_dir):

        flux_files = [run_dir + f"/knosos_for_predictive.txt.{n:02}" for n in np.arange(self.N_fluxtubes)]

        if (any([os.path.exists(f) for f in flux_files]) is False or self.overwrite or (self.overwrite_p_neq_0 and self.time.iter_idx > 0)):
            system = os.environ['GK_SYSTEM']
            if system == 'perlmutter':
                cmd = f"srun -u -t 0:30:0 -N 1 --exact --overcommit --ntasks={self.N_fluxtubes} --cpus-per-task=1 --gpus-per-node=0 --mem-per-cpu={int(256/self.N_fluxtubes)}G {self.KNOSOS_PATH}/knosos.x"
            elif system == 'stellar':
                cmd = f"srun -t 0:30:0 -N 1 --exclusive --ntasks={self.N_fluxtubes} {self.KNOSOS_PATH}/knosos.x"
            elif system == 'summit':
                cmd = f"jsrun -n 1 -a {self.N_fluxtubes} -c {self.N_fluxtubes} -g 0 {self.KNOSOS_PATH}/knosos.x"

            self.info(f"> {cmd}")
            self.print_time()

            wd = os.getcwd()
            os.chdir(run_dir)
            p = subprocess.Popen(cmd, shell=True)
            os.chdir(wd)
            self.processes.append(p)
            sys.stdout.flush()
        else:
            self.info(f'  knosos output {flux_files[0][:-3]} already exists')

        return flux_files  # this is a list of file names

    def read_knosos_fluxes(self, outs_j, species, geo, norms):
        pflux = np.zeros((species.N_species, self.N_fluxtubes))
        qflux = np.zeros((species.N_species, self.N_fluxtubes))

        for r_id in np.arange(self.N_fluxtubes):
            out = outs_j[r_id]

            with open(out, 'r') as f:
                f.readline()  # skip header
                data = list(map(float,f.readline().split()))
                try:
                    Q_i = data[2]
                    Q_e = data[3]
                    if species.N_species > 2:
                        Q_Z = data[5:species.N_species+2]
                    Gam_i = data[species.N_species+2]
                    Gam_e = data[species.N_species+3]
                    if species.N_species > 2:
                        Gam_Z = data[species.N_species+4:]

                    for i, s in enumerate(species.get_species_list()):
                        if s.type == species.bulk_ion.type:
                            qflux[i, r_id] = Q_i  # W/m^2
                            pflux[i, r_id] = Gam_i  # 1/(m^2 s)
                        elif s.type == "electron":
                            qflux[i, r_id] = Q_e  # W/m^2
                            pflux[i, r_id] = Gam_e  # 1/(m^2 s)
                except:
                    self.info(f"ERROR: KNOSOS did not produce fluxes at r_id = {r_id}. Setting fluxes to zero.")

        # KNOSOS fluxes are in SI units. need to normalize to gyroBohm
        for i, s in enumerate(species.get_species_list()):
            qflux[i, :] = qflux[i, :]*geo.Btor**2/geo.a_minor/(species.ref_species.n()*species.ref_species.T()**2.5).toFluxProfile()/(1e6*norms.P_ref_MWm3)
            pflux[i, :] = pflux[i, :]*geo.Btor**2/geo.a_minor/(species.ref_species.n()*species.ref_species.T()**1.5).toFluxProfile()/(1e20*norms.Sn_ref_SI20)

        self.info(f'\n  {self.label}: qflux = {qflux}\n')
        self.info(f'\n  {self.label}: pflux = {pflux}\n')
        return pflux, qflux

    def print_time(self):

        dt = datetime.now()
        self.info(f'  time: {dt}')
        # ts = datetime.timestamp(dt)
        # self.info(f'  time: {ts}')
