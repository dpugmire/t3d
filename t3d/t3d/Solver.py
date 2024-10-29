import numpy as np
from t3d.Logbook import info, errr


class TransportSolver():

    def __init__(self, grid, species, geo, physics):

        self.grid = grid
        self.geo = geo
        self.physics = physics

        # number of evolved profiles
        self.N_profiles = species.N_profiles
        self.N_radial = grid.N_radial

        # initialize an N_prof x N_prof identity matrix
        self.I_mat = np.identity(self.N_profiles*(grid.N_radial))

        y_init = species.get_vec_from_profs()

        self.y_m = y_init
        self.y_old = y_init
        self.y_curr = y_init

        self.y_hist = []
        self.y_hist.append(y_init)
        self.y_error = np.zeros_like(y_init)
        self.chi_error = 0
        self.reset = False

    def psimat(self, species):

        species.compute_psi_n_Fn(self.geo)
        species.compute_psi_p_Fn(self.geo)
        species.compute_psi_n_Fp(self.geo, self.physics)
        species.compute_psi_p_Fp(self.geo, self.physics)

        mat = []
        for s in species.n_evolve_keys:
            row = []
            for sprime in species.n_evolve_keys:
                row.append(species[s].psi_n_Fn[sprime])
            for sprime in species.T_evolve_keys:
                row.append(species[s].psi_p_Fn[sprime])
            mat.append(row)

        for s in species.T_evolve_keys:
            row = []
            for sprime in species.n_evolve_keys:
                row.append(species[s].psi_n_Fp[sprime])
            for sprime in species.T_evolve_keys:
                row.append(species[s].psi_p_Fp[sprime])
            mat.append(row)

        return np.block(mat)

    def F_plus_vec(self, species):

        Fvec = []
        for s in species.n_evolve_keys:
            Fn = np.zeros(self.N_radial)
            Fn[:-1] = species[s].Fn_plus(species.ref_species, self.geo)
            Fvec.append(Fn)

        for s in species.T_evolve_keys:
            Fp = np.zeros(self.N_radial)
            Fp[:-1] = species[s].Fp_plus(species.ref_species, self.geo)
            Fvec.append(Fp)

        return np.array(Fvec).flatten()

    def F_minus_vec(self, species):

        Fvec = []
        for s in species.n_evolve_keys:
            Fn = np.zeros(self.N_radial)
            Fn[:-1] = species[s].Fn_minus(species.ref_species, self.geo)
            Fvec.append(Fn)

        for s in species.T_evolve_keys:
            Fp = np.zeros(self.N_radial)
            Fp[:-1] = species[s].Fp_minus(species.ref_species, self.geo)
            Fvec.append(Fp)

        return np.array(Fvec).flatten()

    def Svec(self, species):
        Svec = []
        for s in species.n_evolve_keys:
            Sn = np.zeros(self.N_radial)
            Sn[:-1] = species[s].Sn[:-1]
            Svec.append(Sn)

        for s in species.T_evolve_keys:
            Sp = np.zeros(self.N_radial)
            Sp[:-1] = 2./3.*species[s].Sp[:-1]
            Svec.append(Sp)

        return np.array(Svec).flatten()

    def solve(self, species, time):

        alpha = time.alpha
        iter_idx = time.iter_idx
        step_idx = time.step_idx

        d1 = time.d1(step_idx)
        d0 = time.d0(step_idx)
        dm1 = time.dm1(step_idx)
        F_plus = self.F_plus_vec(species)
        F_minus = self.F_minus_vec(species)
        S = self.Svec(species)
        G = np.tile((self.geo.G_fac/self.grid.drho).profile, species.evolve_count)

        if iter_idx == 0 and not self.reset:
            self.y_old = self.y_m
            self.y_m = self.y_curr

            self.F_plus_m = F_plus
            self.F_minus_m = F_minus
            self.S_m = S

        psimat = self.psimat(species)
        self.jacob = self.I_mat*d1 + alpha*psimat
        self.rhs = -self.y_m*d0 - self.y_old*dm1 + alpha*psimat@self.y_curr - alpha*(G*(F_plus - F_minus) - S) - (1-alpha)*(G*(self.F_plus_m - self.F_minus_m) - self.S_m)

        try:
            self.y_new = np.linalg.inv(self.jacob)@self.rhs
        except:
            errr("Error in linear solve")
            return np.nan

        time.rms = self.check_error(self.y_new, self.y_curr, species)

    def check_error(self, y_new, y_old, species):
        offset = 0
        rel_diff = (y_new - y_old)/y_old
        err2 = 0
        for i in np.arange(species.evolve_count):
            err2 += np.sum(rel_diff[offset:offset + self.N_radial-1]**2)/(self.N_radial-1)
            offset = offset + self.N_radial
        return np.sqrt(err2 / species.evolve_count)

    def advance(self, species):
        self.y_curr = self.y_new
        # copy vector data into species profiles
        species.get_profs_from_vec(self.y_curr)
        self.reset = False

    def reset_timestep(self, species, time):
        self.y_curr = self.y_m
        species.get_profs_from_vec(self.y_curr)
        time.dtau_old = time.dtau
        time.dtau = time.dtau/time.dtau_adjust
        time.iter_idx = 0
        self.reset = True
        info(f"*** Resetting timestep at time index {time.step_idx}. new dtau = {time.dtau} ***")
