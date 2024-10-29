# This lives as an instance of the Trinity Engine
# to handle flux tubes and GX's geometry module

from t3d import Profiles as pf
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from t3d.geometry_models.VMEC import VmecReader
from t3d.geometry_models.DESC import DescHandler
from t3d.Logbook import info, errr


class Geometry():

    def __init__(self, inputs, grid, flux_models=None, imported=None):

        # read geo parameters from input file
        geo_parameters = inputs.get('geometry', {})
        rescale_parameters = inputs.get('rescale', {})

        self.L_mult = rescale_parameters.get('L_mult', 1)
        self.L_def = rescale_parameters.get('L_def', ' ')
        self.a = rescale_parameters.get('a_minor', 0)
        self.R = rescale_parameters.get('R_major', 0)

        self.B_mult = rescale_parameters.get('B_mult', 1)
        self.B_def = rescale_parameters.get('B_def', ' ')
        self.Ba = rescale_parameters.get('Ba', 0)
        self.volavgB = rescale_parameters.get('volavgB', 0)
        self.vmecB0 = rescale_parameters.get('vmecB0', 0)

        self.AspectRatio = geo_parameters.get('AspectRatio', 0)
        self.iota23 = geo_parameters.get('iota23', 0.75)

        self.compute_surface_areas = geo_parameters.get('compute_surface_areas', True)
        self.geo_option = geo_parameters.get('geo_option', 'basic')
        self.geo_file = geo_parameters.get('geo_file', '')
        self.geo_import = geo_parameters.get('import', False)
        self.geo_outputs = geo_parameters.get('geo_outputs', 'geo/')
        self.overwrite = geo_parameters.get('overwrite', False)

        self.evolve_equilibrium = geo_parameters.get('evolve_equilibrium', False)
        self.desc_pressure_threshold = geo_parameters.get('desc_pressure_delta_threshold', 1.25)  # determines whether to update desc
        self.desc_pressure_perturb = geo_parameters.get('desc_pressure_delta', 2)  # determines whether to perturb or full solve
        self.desc_use_vmec_format = geo_parameters.get('desc_use_vmec_format', None)

        self.grid = grid
        self.imported = imported
        self.flux_models = flux_models
        self.N_fluxtubes = len(grid.midpoints)

        self.B0 = None  # only defined for some geo options

        self.init_geometry()

        info('')
        info('  Global Geometry Information', 'magenta')
        info(f'    R_major: {self.AspectRatio * self.a_minor:.2f} m', 'magenta')
        info(f'    a_minor: {self.a_minor:.2f} m', 'magenta')
        if self.B0:
            info(f"    B0: {self.B0:.2f} T", 'magenta')
        info(f'    Ba: {self.Btor:.2f} T', 'magenta')
        info('')

    def init_geometry(self):

        if self.geo_option == 'basic':
            info("\n  Using basic geometry")

            if (self.L_mult == 0):
                self.L_mult = 1
                errr("L_mult cannot be set to zero. Forcing L_mult = 1")

            if (self.L_mult != 1):
                info(f"  Scaling all linear sizes by a factor of L_mult = {self.L_mult:.2f}")

            if (self.AspectRatio == 0):
                self.AspectRatio = 4
                errr("A non-zero value of AspectRatio is required. Forcing AspectRatio = 4.")

            else:
                info(f"Aspect Ratio = {self.AspectRatio:.2f}")

            if (self.L_def == 'minor' or self.a != 0):
                if (self.a == 0):
                    self.a = 1
                    errr(f"A non-zero value of a_minor is required. Forcing a_minor = {self.a_minor:.2f} m.")
                self.a_minor = self.a * self.L_mult
                self.R_major = self.a_minor * self.AspectRatio

            else:
                if (self.R == 0):
                    self.R = 4
                    errr(f"A non-zero value of R_major is required. Forcing R_major = {self.a_minor*self.AspectRatio:.2f} m.")
                self.R_major = self.R * self.L_mult
                self.a_minor = self.R_major / self.AspectRatio

            self.volume = 2*np.pi**2*self.R_major*self.a_minor**2
            area = (self.grid.midpoints * 2 * self.volume) / self.a_minor**2
            grho = np.ones(self.N_fluxtubes) * self.a_minor
            area_grid = self.grid.rho * 2 * self.volume / self.a_minor**2
            grho_grid = np.ones(self.N_fluxtubes+1) * self.a_minor

            # store results in self object
            self.grho = pf.FluxProfile(grho, self.grid)
            self.area = pf.FluxProfile(area, self.grid)
            self.grho_grid = pf.GridProfile(grho_grid, self.grid)
            self.area_grid = pf.GridProfile(area_grid, self.grid)

            if (self.B_mult == 0):
                self.B_mult = 1
                errr("B_mult cannot be set to zero. Forcing B_mult = 1")

            if (self.B_mult != 1):
                info(f"  Scaling all magnetic fields by a factor of B_mult = {self.B_mult:.2f}")

            if (self.B_def == 'LCFS'):
                if (self.Ba == 0):
                    self.Ba = self.B_mult
                    errr(f"A non-zero value of Ba is required. Forcing Ba = {self.Ba:.2f} T.")

            elif (self.B_def == 'volavgB'):
                if (self.volavgB == 0):
                    self.Ba = self.B_mult
                    errr(f"A non-zero value of volavgB is required. Forcing volavgB = {self.Ba:.2f} T.")
                    info("  Note: volavgB is assumed to be equal to Ba in the absence of a numerical equilibrium")
                else:
                    self.Ba = self.volavgB

            elif (self.B_def == 'vmecB0'):
                if (self.vmecB0 == 0):
                    self.Ba = self.B_mult
                    errr(f"A non-zero value of vmecB0 is required. Forcing vmecB0 = {self.Ba:.2f} T.")
                    info("  Note: vmecB0 is assumed to be equal to Ba in the absence of a numerical equilibrium")
                else:
                    self.Ba = self.vmecB0

            self.Btor = self.Ba
            self.B0 = self.Ba
            self.volavgB = self.Ba

        elif self.geo_option == 'vmec':
            info("\n  Using geometric information from VMEC")
            if self.grid.flux_label != 'torflux':
                errr("geo_option = 'vmec' requires flux_label = 'torflux'")
            assert self.grid.flux_label == 'torflux'

            if self.flux_models and 'KNOSOS' in self.flux_models.models_dict.keys():
                # KNOSOS requires the vmec file to be pre-processed by BOOZ_XFORM
                self.flux_models.models_dict['KNOSOS'].make_booz(self.geo_file)

            self.read_VMEC()

            if (self.L_mult == 0):
                self.L_mult = 1
                errr("L_mult cannot be set to zero. Forcing L_mult = 1.")
            elif (self.L_mult == 1):
                info("  Using L_mult = 1.0")

            if (self.L_mult != 1):
                info(f"  Scaling the VMEC linear sizes by a factor of L_mult = {self.L_mult:.2f}")

            if (self.AspectRatio != 0):
                info("  The aspect ratio cannot be modified in the VMEC context. It is determined by the VMEC data. \n")

            self.AspectRatio = self.R_major_vmec / self.a_minor_vmec
            info(f"  Aspect Ratio = {self.AspectRatio:.2f}")

            # compute additional scaling factors
            if (self.L_def == 'minor'):
                assert self.a != 0, "Must specify a_minor value to use L_def = 'minor'"
                self.L_scale = self.L_mult * self.a / self.a_minor_vmec
                info("  Minor radius set to value of a_minor in the input file * L_mult.")
            elif (self.L_def == 'major'):
                assert self.R != 0, "Must specify R_major value to use L_def = 'major'"
                self.L_scale = self.L_mult * self.R / self.R_major_vmec
                info("  Major radius set to value of R_major in the input file * L_mult.")
            else:
                self.L_scale = self.L_mult
                info("  Minor radius set to value of a_minor in VMEC file * L_mult.")
            self.a_minor = self.a_minor_vmec * self.L_scale
            self.R_major = self.R_major_vmec * self.L_scale
            self.volume = self.volume_vmec * self.L_scale**3

            info(f"\n    In the VMEC output file, volavgB/Ba = {self.volavgB_vmec/self.Ba_vmec:.2f}.")
            info(f"    In the VMEC output file, vmecB0/Ba = {self.B0_vmec/self.Ba_vmec:.2f}.\n")

            if (self.B_mult == 0):
                self.B_mult = 1
                errr("B_mult cannot be set to zero. Forcing B_mult = 1.")
            elif (self.B_mult == 1):
                info("  Using B_mult = 1.0")

            if (self.B_mult != 1):
                info(f"  Scaling all magnetic fields by a factor of B_mult = {self.B_mult:.2f}")

            # compute additional scaling factors based on different B_def definitions
            if (self.B_def == 'LCFS'):
                if (self.Ba != 0):
                    self.B_scale = self.B_mult * self.Ba / self.Ba_vmec
                    info("  Btor == (Ba from input file) * (B_mult).")
            elif (self.B_def == 'volavgB'):
                assert self.volavgB != 0, "Must specify volavgB value to use B_def = 'volavgB'"
                info("  Btor == (volavgB from input file) / (volavgB from VMEC) * (Ba from VMEC) * (B_mult).")
                self.B_scale = self.B_mult * self.volavgB / self.volavgB_vmec
            elif (self.B_def == 'vmecB0'):
                assert self.vmecB0 != 0, "Must specify vmecB0 value to use B_def = 'vmecB0'"
                self.B_scale = self.B_mult * self.vmecB0 / self.B0_vmec
                info("  Btor == (vmecB0 from input file) * (Ba from VMEC) * (B_mult) / (B0 from VMEC).")
                info("    I.e, the value of Ba used internally by T3D produces Btor according to your input values of vmecB0 and B_mult.")
                info("    Normally, one would probably use B_mult = 1 for this option.")
            else:
                self.B_scale = self.B_mult
                info("  No explicit choice specified for the definition of normalizing magnetic field. Using default:")
                info("  Btor = (Ba from VMEC file) * (B_mult).")

            # scale fields
            self.Btor = self.Ba_vmec * self.B_scale
            self.B0 = self.B0_vmec * self.B_scale
            self.volavgB = self.volavgB_vmec * self.B_scale

        elif self.geo_option == 'geqdsk':
            assert 'GX' in self.flux_models.models_dict.keys(), "Error: geo_option = 'geqdsk' currently requires model = 'GX'"
            self.flux_models['GX'].get_gx_eqdsk_geometry(self)

            if (self.AspectRatio != 0):
                info("  The aspect ratio cannot be modified in the geqdsk context. It is determined by the geqdsk data. ")

            self.AspectRatio = self.R_major_eq  # Check this! Normally, GX would report R in units of a. Maybe it has been multiplied out?

            if (self.L_mult == 0):
                self.L_mult = 1
                info("  Error: L_mult cannot be set to zero in the geqdsk context. Forcing L_mult = 1 ")

            if (self.L_mult == 1 and self.a_minor_eq != 0):
                self.a_minor = self.a_minor_eq
                info("  Setting a_minor to the value found in the geqdsk file. ")

            if (self.L_mult != 1 and self.a_minor_eq != 0):
                self.a_minor = self.a_minor_eq * self.L_mult
                info(f"  Setting a_minor to {self.a_minor_eq}, the value found in the geqdsk file, times L_mult ")

            if (self.L_mult == 1 and self.a != 0):
                self.a_minor = self.a
                info("  Setting a_minor to the value of a_minor found in the input file ")

            if (self.L_mult != 1 and self.a != 0):
                self.a_minor = self.a * self.L_mult
                info(f"  Setting a_minor = a_minor * L_mult = {self.a} * {self.L_mult}, as specified in the input file ")

            if (self.B_mult == 0):
                self.B_mult = 1
                info("  Error: B_mult cannot be set to zero in the geqdsk context. Forcing B_mult = 1")

            if (self.B_mult == 1 and self.Ba == 0):
                self.Btor = self.Btor_eq
                info("  Setting Btor to the value found in the geqdsk file. ")

            if (self.B_mult != 1 and self.Ba == 0):
                self.Btor = self.Btor_eq * self.B_mult
                info(f"  Setting Btor to {self.Btor_eq}, the value found in the geqdsk file, times B_mult ")

            if (self.B_mult == 1 and self.Ba != 0):
                self.Btor = self.Ba
                info("  Setting Btor to the value of Ba found in the input file")

            if (self.B_mult != 1 and self.Ba != 0):
                self.Btor = self.Ba * self.B_mult
                info(f"  Setting Btor = Ba * B_mult = {self.Ba} * {self.B_mult}, as specified in the input file ")

        elif self.geo_option == 'miller':
            assert self.grid.flux_label == 'rminor', "Error: geo_option = 'miller' requires flux_label = 'rminor'"
            if self.geo_import:
                assert self.imported.imported is not None, 'Error: must specify [import] section to use import = true in [geometry]'
                self.imported.get_geometry(self)

        elif self.geo_option == 'desc':

            self.read_DESC()

            if (self.L_mult == 0):
                self.L_mult = 1
                info("  Error: L_mult cannot be set to zero. Forcing L_mult = 1.")

            if (self.AspectRatio != 0):
                info("  The aspect ratio cannot be modified in the DESC context. It is determined by the DESC data. \n")

            self.AspectRatio = self.R_major_desc / self.a_minor_desc

            if (self.L_mult == 1 and self.a == 0):
                self.a_minor = self.a_minor_desc
                info("  Setting a_minor to the value found in the DESC file.")

            if (self.L_mult != 1 and self.a == 0):
                self.a_minor = self.a_minor_desc * self.L_mult
                info(f"  Setting a_minor to {self.a_minor_desc}, the value found in the DESC file, times L_mult")

            if (self.L_mult == 1 and self.a != 0):
                self.a_minor = self.a
                info("  Setting a_minor to the value of a_minor found in the input file")

            if (self.L_mult != 1 and self.a != 0):
                self.a_minor = self.a * self.L_mult
                info(f"  Setting a_minor = a_minor * L_mult = {self.a} * {self.L_mult}, as specified in the input file ")

            if (self.B_mult == 0):
                self.B_mult = 1
                errr("B_mult cannot be set to zero. Forcing B_mult = 1.")
            elif (self.B_mult == 1):
                info("  Using B_mult = 1.0")

            if (self.B_mult != 1):
                info(f"  Scaling all magnetic fields by a factor of B_mult = {self.B_mult:.2f}")

            # compute additional scaling factor if Ba is specified in input file
            if self.Ba != 0:
                info(f"  Setting Btor to the value of Ba found in the input file, times B_mult = {self.B_mult} ")
                self.B_scale = self.B_mult * self.Ba / self.Ba_desc
            else:
                info(f"  Setting Btor to the value of Ba found in the DESC file, times B_mult = {self.B_mult} ")
                self.B_scale = self.B_mult

            # scale magnetic field values
            self.Btor = self.Ba_desc * self.B_scale
            self.B0 = self.B0_desc * self.B_scale

            if self.flux_models and 'KNOSOS' in self.flux_models.models_dict.keys():
                # KNOSOS requires the vmec file to be pre-processed by BOOZ_XFORM
                self.flux_models.models_dict['KNOSOS'].make_booz(self.DescHandler.out_vmec)

        else:
            assert False, f"Error: geo_option = '{self.geo_option}' not recognized."

        if not hasattr(self, 'grho_grid'):
            self.grho_grid = self.grho.toGridProfile(axis_val=self.grho[0])
        if not hasattr(self, 'area_grid'):
            self.area_grid = self.area.toGridProfile(axis_val=0)
        self.G_fac = self.grho_grid/self.area_grid

    def update_equilibrium(self, species, time):

        # skip if profiles are not converged
        if time.iter_idx != 0:
            return

        if self.geo_option == 'desc':
            info("Updating equilibrium using DESC...")
            updated = self.DescHandler.update_desc_eq(self, species, time)
        else:
            assert False, f"update_equilibrium not yet available for geo_option = '{self.geo_option}'"

        # re-initialize trinity flux tubes with updated equilibrium
        if updated:
            self.init_geometry()

    def line_integrate(self, f, profile=False, normalized=False):
        """
        Computes line integral of f
        """

        if isinstance(f, pf.FluxProfile):
            assert False, 'Geometry.lineavg(f) requires f to be a GridProfile'

        if profile:
            integral = f.__class__(np.cumsum(f.profile*self.grid.drho), self.grid)
        else:
            integral = np.sum(f.profile*self.grid.drho)

        if normalized:
            fac = 1  # returns line integral normalized to a
        else:
            fac = self.a_minor

        return integral*fac

    def volume_integrate(self, f, profile=False, normalized=False, interpolate=False):
        """
        Computes volume integral of f
        """

        if isinstance(f, pf.GridProfile):
            grho = self.grho_grid.profile
            area = self.area_grid.profile
        elif isinstance(f, pf.FluxProfile):
            grho = self.grho.profile
            area = self.area.profile
        else:
            assert False, 'Geometry.volavg(f) requires f to be a GridProfile or FluxProfile'

        if profile:
            if interpolate:
                integrand = interp1d(f.axis, f.profile*area/grho, fill_value='extrapolate')
                integral = 0*f.profile
                for i in np.arange(len(f.profile)):
                    integral[i] = integrate.quad(integrand, 0, f.axis[i], limit=100, epsabs=1e-5)[0]
                integral = f.__class__(integral, self.grid)
            else:
                integral = f.__class__(np.cumsum(f.profile*area/grho)*self.grid.drho, self.grid)
        else:
            if interpolate:
                integrand = interp1d(f.axis, f.profile*area/grho, fill_value='extrapolate')
                integral = integrate.quad(integrand, 0, f.axis[-1], limit=100, epsabs=1e-5)[0]
            else:
                integral = np.sum(f.profile*area/grho)*self.grid.drho

        if normalized:
            fac = 1  # returns volume integral normalized to a^3
        else:
            fac = self.a_minor**3

        return integral*fac

    def volume_average(self, f):
        """
        Computes volume average of f
        """

        return self.volume_integrate(f) / self.volume_integrate(f.__class__(1, self.grid))

    def line_average(self, f, profile=False, normalized=False):
        """
        Computes line integral of f
        """

        if isinstance(f, pf.FluxProfile):
            assert False, 'Geometry.lineavg(f) requires f to be a GridProfile'

        if profile:
            integral = f.__class__(np.cumsum(f.profile*self.grid.drho), self.grid)
        else:
            integral = np.mean(f.profile*self.grid.drho)

        if normalized:
            fac = 1  # returns line integral normalized to a
        else:
            fac = self.a_minor

        return integral*fac

    def read_VMEC(self):
        """
        Loads VMEC wout file into Trinity
        """

        self.vmec_wout = self.geo_file
        wout = self.vmec_wout

        if wout == '':
            assert False, 'Error: must specify a VMEC file using geo_file'
        else:
            info(f"\n  Loading VMEC geometry from {wout}")

        # load global geometry from VMEC
        vmec = VmecReader(wout)
        self.R_major_vmec = float(vmec.Rmajor)
        self.a_minor_vmec = float(vmec.aminor)
        self.Ba_vmec = vmec.Ba_vmec
        self.iota_vmec = vmec.iotaf
        self.volavgB_vmec = vmec.volavgB
        self.B0_vmec = vmec.B0
        self.volume_vmec = vmec.volume

        # interpolate grho and area onto Trinity grids
        interp = interp1d(vmec.rho, vmec.grad_rho, fill_value='extrapolate')
        self.grho = pf.FluxProfile(interp(self.grid.midpoints), self.grid)
        self.grho_grid = pf.GridProfile(interp(self.grid.rho), self.grid)

        # area = dVdrho * <|grad rho|>
        interp = interp1d(vmec.rho, vmec.area, fill_value='extrapolate')
        self.area = pf.FluxProfile(interp(self.grid.midpoints), self.grid)
        self.area_grid = pf.GridProfile(interp(self.grid.rho), self.grid)

        interp = interp1d(vmec.rho, vmec.dVdrho, fill_value='extrapolate')
        self.dVdrho = pf.FluxProfile(interp(self.grid.midpoints), self.grid)
        self.dVdrho_grid = pf.GridProfile(interp(self.grid.rho), self.grid)

        # iota23 = iota(rho=0.67), used in ISS04
        interp = interp1d(vmec.rho, vmec.iotaf)
        self.iota23 = interp(0.66667)

        # normalize area and grho
        # Sanity check: For circular tokamak, normalized area = 4 pi**2 a R / a**2 = 4 pi**2 R/a
        self.area = self.area / self.a_minor_vmec**2
        self.area_grid = self.area_grid / self.a_minor_vmec**2
        self.grho = self.grho * self.a_minor_vmec
        self.grho_grid = self.grho_grid * self.a_minor_vmec

        # save for later
        self.VmecReader = vmec

    def read_DESC(self):
        """
        Loads DESC h5 file into Trinity
        """

        self.desc_file = self.geo_file

        if self.desc_file == '':
            assert False, 'Error: must specify a DESC file using geo_file'
        else:
            info(f"\nLoading DESC geometry from {self.desc_file}...")

        # load global geometry from DESC
        desc = DescHandler(self.desc_file, use_vmec_format=self.desc_use_vmec_format)
        self.R_major_desc = float(desc.Rmajor)
        self.a_minor_desc = float(desc.aminor)

        self.B_ref = pf.FluxProfile(float(desc.B_ref), self.grid)
        self.a_ref = pf.FluxProfile(desc.aminor, self.grid)
        self.Ba_desc = float(desc.B_ref)
        self.B0_desc = float(desc.B0)
        self.iota23 = float(desc.iota23)

        # interpolate grho and area onto Trinity grids
        interp = interp1d(desc.rho[1:], desc.grad_rho[1:], fill_value='extrapolate')
        self.grho = pf.FluxProfile(interp(self.grid.midpoints), self.grid)
        self.grho_grid = pf.GridProfile(interp(self.grid.rho), self.grid)

        # area = dVdrho * <|grad rho|>
        interp = interp1d(desc.rho, desc.area, fill_value='extrapolate')
        self.area = pf.FluxProfile(interp(self.grid.midpoints), self.grid)
        self.area_grid = pf.GridProfile(interp(self.grid.rho), self.grid)

        # normalize area and grho
        self.area = self.area / self.a_minor_desc**2
        self.grho = self.grho * self.a_minor_desc
        self.area_grid = self.area_grid / self.a_minor_desc**2
        self.grho_grid = self.grho_grid * self.a_minor_desc

        # save for later
        self.DescHandler = desc
