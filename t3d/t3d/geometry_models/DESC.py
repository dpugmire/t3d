import numpy as np
import os
from t3d.Logbook import info


class DescHandler():

    def __init__(self, desc_file, nr=100, use_vmec_format=None):

        from desc import set_device

        try:
            set_device('gpu')
            import desc.io
        except:
            info("  Desc failed to find GPU, falling back to CPU", color='yellow')
            import desc.io

        from desc.grid import LinearGrid
        from desc.vmec import VMECIO

        self.rho = np.linspace(0, 1, nr)
        self.desc_grid = LinearGrid(rho=self.rho)

        # load desc output
        # check if we are using VMEC format or DESC format
        if desc_file[-3:] == '.nc':
            if use_vmec_format is not None:
                self.use_vmec_format = use_vmec_format
            else:
                self.use_vmec_format = True
            eq = VMECIO.load(desc_file, L=10, M=10, N=10)
            self.out_vmec = desc_file
        else:
            if use_vmec_format is not None:
                self.use_vmec_format = use_vmec_format
            else:
                self.use_vmec_format = False
            eq = desc.io.load(desc_file)
            if hasattr(eq, '__len__'):
                eq = eq[-1]
            self.out_vmec = desc_file[-3:] + "_wout.nc"

        self.get_geometry(eq)

        # store for later
        self.desc_eq = eq

    def get_geometry(self, eq):
        from desc.grid import LinearGrid

        # gets area and grad rho from DESC for Trinity
        if hasattr(eq, '__len__'):
            eq = eq[-1]

        grid = self.desc_grid

        # area = dVdrho * <|grad rho|>
        self.area = eq.compute('S(r)', grid=grid)['S(r)']
        self.dVdrho = eq.compute('V_r(r)', grid=grid)['V_r(r)']
        self.grad_rho = self.area / self.dVdrho

        self.B_ref = eq.compute('|B|', grid=grid)['|B|'][0]
        self.B0 = eq.compute('<|B|>', grid=LinearGrid(rho=0.0))['<|B|>'][0]
        self.iota23 = eq.compute('iota', grid=LinearGrid(rho=0.66667))['iota'][0]
        self.aminor = eq.compute('a')['a']
        self.Rmajor = eq.compute('R0')['R0']

    def update_desc_eq(self, geometry, species, time):

        try:
            axis = geometry.grid.rho
        except:
            # backwards compatibility for older T3D outputs (05/2024)
            axis = geometry.grid.rho_axis

        # skip if profiles did not change above threshold
        p_trin = species.get_pressure_SI()
        p_desc = self.get_pressure_SI(axis)

        # get the point-wise maximum
        p_delta = np.max(p_trin / p_desc)
        if p_delta < geometry.desc_pressure_threshold:
            info(f"  p_delta: {p_delta}, skipping eq update")
            return

        # Entering EQ update
        info("ENTERING EQ UPDATE")
        info(f"   p_delta {p_delta}")

        # setup file name
        t_idx = time.step_idx
        p_idx = time.iter_idx

        path = geometry.geo_outputs  # defaults to geo/, this is a fine place for gx, lets put desc elsewhere
        # add a tag specifying desc and trinity file name
        # path = f'desc_eqs_{trin}/'
        os.makedirs(path, exist_ok=True)
        desc_output = f"{path}/desc-t{t_idx:02d}-p{p_idx:d}.h5"

        if self.use_vmec_format:
            desc_output = os.path.join(path, f"desc-t{t_idx:02d}-p{p_idx:d}.nc")
        else:
            desc_output = os.path.join(path, f"desc-t{t_idx:02d}-p{p_idx:d}.h5")
        desc_log = os.path.join(path, f"desc-t{t_idx:02d}-p{p_idx:d}.log")

        if not geometry.overwrite:

            # skip if file detected
            if os.path.exists(desc_output):
                info("Found desc file:", desc_output)
                info("  skipping eq re-eval")

                geometry.geo_file = desc_output
                return

        eq = self.desc_eq
        if hasattr(eq, '__len__'):
            eq = eq[-1]

        # setup desc pressure
        axis = geometry.grid.rho
        if axis[-1] < 1.0:
            # assume zero pressure at LCFS
            p_trin = np.append(p_trin, 0.0)
            axis = np.append(axis, 1.0)
        from desc.profiles import PowerSeriesProfile
        pres_new = PowerSeriesProfile.from_values(axis, p_trin)

        # use perturbative solve for small changes
        if p_delta < geometry.desc_pressure_perturb:
            perturb_desc = True
            info("Attempting perturbative DESC equilibrium solve...")
        else:
            info("Pressure delta too large, attempting full DESC solve...")
            perturb_desc = False

        from contextlib import redirect_stdout
        info(f"   DESC output is being redirected to {desc_log}.")
        with open(desc_log, 'w', buffering=1) as f:
            with redirect_stdout(f):
                # run equilibrium

                if perturb_desc:
                    try:
                        # perturb is faster than solve
                        eq.perturb(deltas={"p_l": pres_new.params - eq.pressure.params}, verbose=3)
                    except:
                        info(" perturbative DESC failed, attempting full solve")
                        eq.pressure = pres_new
                        eq.solve(verbose=3)
                else:
                    eq.pressure = pres_new
                    eq.solve(verbose=3)

                # update trinity area and grad rho
                self.get_geometry(eq)

                # save output
                if self.use_vmec_format:
                    from desc.vmec import VMECIO
                    VMECIO.save(eq, desc_output, surfs=512)
                else:
                    eq.save(desc_output)
                info("  saved desc output", desc_output)

        # update geo_file
        geometry.geo_file = desc_output

        return True

    def get_pressure_SI(self,axis):
        from desc.grid import LinearGrid
        grid = LinearGrid(rho=axis)
        p_SI = self.desc_eq.compute("p", grid=grid)['p']

        return p_SI


DescReader = DescHandler  # backwards compatibility
