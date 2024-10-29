from t3d.Logbook import info


class Time():

    def __init__(self, inputs):

        # read time parameters from input file

        time_parameters = inputs.get('time', {})
        self.max_newton_iter = time_parameters.get('max_newton_iter', 4)       # maximum number of newton iterations
        self.newton_threshold = time_parameters.get('newton_threshold', 0.02)  # stop newton iteration when err < newton_threshold
        self.newton_tolerance = time_parameters.get('newton_tolerance', 0.1)   # allow advance to next timestep if err < newton_tolerance after max_newton_iter
        self.alpha = time_parameters.get('alpha', 1)                           # implicitness parameter (1.0 -> fully implicit, 0.0 -> fully explicit)
        self.dtau = time_parameters.get('dtau', 0.5)                           # initial timestep
        self.dtau_max = time_parameters.get('dtau_max', 10.0)                  # maximum allowed timestep
        self.dtau_adjust = time_parameters.get('dtau_adjust', 2.0)             # adjustment factor when decreasing timestep
        self.dtau_increase_threshold = time_parameters.get('dtau_increase_threshold', self.newton_threshold/4.)  # increase timestep if err < dtau_increase_threshold
        self.N_steps = time_parameters.get('N_steps', 1000)                    # number of timesteps
        self.t_max = time_parameters.get('t_max', 1000)                        # end time
        self.use_SI = time_parameters.get('use_SI', False)                     # interpret t_max and dtau time inputs as SI
        self.rms_converged = time_parameters.get('rms_converged', 1e-16)       # rms converged halting condition

        # init some variables
        self.time = 0
        self.step_idx = 0
        self.iter_idx = 0
        self.gx_idx = 0
        self.prev_p_id = 0
        self.dtau_old = self.dtau
        self.dtau_success = self.dtau
        self.prev_step_success = False
        self.newton_mode = False
        self.rms = 0

    def normalize_inputs(self, t_ref):
        """Function to compute normalized time inputs if specified in SI"""

        if self.use_SI:
            # interpret Trinity time inputs as SI units
            info("  Time inputs set in SI units, normalizing")
            info(f"    t_max: {self.t_max} s, {self.t_max/t_ref:.3f} in normalized units")
            info(f"    dtau: {self.dtau} s, {self.dtau/t_ref:.3f} in normalized units\n")

            self.t_max = self.t_max / t_ref
            self.dtau = self.dtau / t_ref
            self.dtau_max = self.dtau_max / t_ref
            self.dtau_adjust = self.dtau_adjust / t_ref
            self.dtau_increase_threshold = self.dtau_increase_threshold / t_ref

            self.dtau_old = self.dtau
            self.dtau_success = self.dtau

        else:
            info("  Normalized time inputs specified")
            info(f"    t_max: {self.t_max:.3f} which corresponds to {self.t_max*t_ref:.3e} s")
            info(f"    dtau: {self.dtau:.3f} which corresponds to {self.dtau*t_ref:.3e} s\n")

    def d1(self, step_idx):
        if self.alpha == 1 and step_idx > 0:
            return (2.*self.dtau + self.dtau_old)/(self.dtau+self.dtau_old)/self.dtau
        else:
            return 1/self.dtau

    def d0(self, step_idx):
        if self.alpha == 1 and step_idx > 0:
            return -(self.dtau + self.dtau_old)/self.dtau_old/self.dtau
        else:
            return -1/self.dtau

    def dm1(self, step_idx):
        if self.alpha == 1 and step_idx > 0:
            return self.dtau / (self.dtau_old * (self.dtau + self.dtau_old))
        else:
            return 0
