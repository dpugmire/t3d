.. _input_file:

Inputs
======

The T3D input file
------------------

The T3D input file is parsed with `toml <https://github.com/ToruNiina/toml11>`_. Input file names should be suffixed with ``.in``, such as ``example.in``. Such an input file can be run via

.. code-block:: bash

  t3d example.in

Example input files can be found in the :ref:`quickstart` pages, and in the ``tests/regression`` `directory <https://bitbucket.org/gyrokinetics/t3d/src/main/tests/regression/>`_ in the T3D repository.

A typical input file will be of the form

.. code-block:: toml

  [grid]
  N_radial = 9
  ...

  [time]
  ...

  ...

All parameters belong to particular groups, e.g. ``[grid]`` or ``[time]``, etc. 

In the following we describe each possible input parameter in each group. In practice, many of these parameters can be left unspecified so that default values are used.

grid
++++

The ``[grid]`` group controls the radial grid used for the calculation, as well as the choice of radial coordinate.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[grid]``
     - ``N_radial``
     - The number of radial grid points where profiles (e.g. density, temperature) will be evolved.
     - 10
   * - ``[grid]``
     - ``flux_label``
     - Choice of radial coordinate (rho). Can be either 'torflux' (rho = sqrt(toroidal_flux/toroidal_flux_LCFS)) or 'rminor' (rho = r/a, the normalized minor radius).
     - 'torflux'
   * - ``[grid]``
     - ``rho_edge``
     - Largest value of rho on the grid.
     - 0.8
   * - ``[grid]``
     - ``rho_inner``
     - Smallest value of rho on the grid (currently not available as an input).
     - :math:`\texttt{rho_edge}/(2\,\texttt{N_radial}-1)`

time
++++

The ``[time]`` group contains parameters pertaining to the timestepping and Newton iteration.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[time]``
     - ``dtau``
     - The initial timestep size.
     - 0.5
   * - ``[time]``
     - ``t_max``
     - End time. Use this or ``N_steps`` to control the length of the simulation.
     - 1000
   * - ``[time]``
     - ``N_steps``
     - Number of timesteps. Use this or ``t_max`` to control the length of the simulation.
     - 1000
   * - ``[time]``
     - ``alpha``
     - Implicitness parameter (1.0 -> fully implicit, 0.0 -> fully explicit)
     - 1.0
   * - ``[time]``
     - ``max_newton_iter``
     - Maximum number of Newton iterations 
     - 4
   * - ``[time]``
     - ``newton_threshold``
     - Stop Newton iteration when rms_err < ``newton_threshold`` and advance to next timestep
     - 0.02
   * - ``[time]``
     - ``newton_tolerance``
     - Allow advance to next timestep if rms_err < ``newton_tolerance`` after ``max_newton_iter``, even if rms_err > ``newton_threshold``
     - 0.1
   * - ``[time]``
     - ``dtau_adjust``
     - Adjustment factor when decreasing timestep, so that dtau_new = dtau_old/``dtau_adjust``
     - 2.0
   * - ``[time]``
     - ``dtau_increase_threshold``
     - Increase timestep by factor of 2 if rms_err < ``dtau_increase_threshold``
     - ``newton_threshold``/4
   * - ``[time]``
     - ``dtau_max``
     - The maximum allowable timestep size.
     - 10.0

model
+++++

The ``[[model]]`` group controls the models used to compute fluxes (e.g. turbulence models, neoclassical models, etc). Note that the double square bracket denotes an array in toml, so that multiple ``[[model]]`` blocks can be used. Fluxes from different models will be summed (e.g. Q_tot = Q_turb + Q_neo).

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[model]``
     - ``model``
     - Name of the flux model. See :ref:`flux_models` for options.
     - "GX"

Each model has additional parameters that can/should be specified. See the section corresponding to each model in :ref:`flux_models`.

species
+++++++

The ``[[species]]`` group controls the plasma species in the calculation. Note that the double square bracket denotes an array in toml, so that multiple ``[[species]]`` blocks can be used, one for each species.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[species]``
     - ``type``
     - Name of the plasma species. Can be arbitrary, but "electron", "hydrogen", "deuterium", "tritium", "boron", "carbon" have default values for charge, mass, etc.
     - "deuterium"
   * - ``[species]``
     - ``mass``
     - Mass of species in units of proton mass. If ``type`` is one of the standard species listed above there is a smart default, otherwise this parameter must be specified.
     - Smart default or None
   * - ``[species]``
     - ``Z``
     - Charge of species in units of electron charge. If ``type`` is one of the standard species listed above there is a smart default, otherwise this parameter must be specified.
     - Smart default or None
   * - ``[species]``
     - ``adiabatic``
     - Treat species as adiabatic in turbulence and/or neoclassical calculations.
     - False
   * - ``[species]``
     - ``adiabatic``
     - Treat species as adiabatic in turbulence and/or neoclassical calculations.
     - False
   * - ``[species]``
     - ``use_as_reference``
     - Use species as reference species for normalizations in some flux models. Should be specified ``True`` for only one species.
     - First ion species listed
   * - ``[species]``
     - ``bulk``
     - Use species as bulk species. Should be specified ``True`` for only one ion species. Should not be used for electrons. 
     - First ion species listed
   * - ``[species]``
     - ``tag``
     - Short name for species, to be used in diagnostic output, e.g. "e" for ``type="electron"``. If ``type`` is one of the standard species listed above there is a smart default.
     - Smart default or ``type``
   * - ``[species]``
     - ``density``
     - Table controlling density initialization and evolution. See ``species.density`` parameters below.
     - {}
   * - ``[species.density]``
     - ``evolve``
     - Evolve the density in time.
     - True
   * - ``[species.density]``
     - ``init_to``
     - Initialize the density equal to some fraction of the density of another species, but allow profiles evolve independently after initialization. This can be either a string corresponding to a species, e.g. ``init_to = "electron"``, or a string containing a multiplicative fraction times the species, e.g. ``init_to = "0.5 * electron"``.
     - 
   * - ``[species.density]``
     - ``equal_to``
     - Initialize and maintain (as profiles evolve) the density equal to some fraction of the density of another species. This can be either a string corresponding to a species, e.g. ``equal_to = "electron"``, or a string containing a multiplicative fraction times the species, e.g. ``equal_to = "0.5 * electron"``.
     - 
   * - ``[species.density]``
     - ``shape``
     - Shape of initial density profile. Options are ``'parabolic'``, ``'exponential'``, ``'custom'``.
     - 'parabolic'
   * - ``[species.density]``
     - ``core``
     - For ``shape = 'parabolic'``, specifies density at :math:`\rho = 0`, in :math:`10^{20}` :math:`m^{-3}`.
     - 4
   * - ``[species.density]``
     - ``volavg``
     - For ``shape = 'parabolic'``, specifies volume-average density, in :math:`10^{20}` :math:`m^{-3}`.
     - 
   * - ``[species.density]``
     - ``lineavg``
     - For ``shape = 'parabolic'``, specifies line-average density, in :math:`10^{20}` :math:`m^{-3}`.
     - 
   * - ``[species.density]``
     - ``edge``
     - For ``shape = 'parabolic'``, specifies density at :math:`\rho = \rho_{edge}`, in :math:`10^{20}` :math:`m^{-3}`.
     - 0.5
   * - ``[species.density]``
     - ``sep``
     - For ``shape = 'parabolic'``, specifies density at :math:`\rho = 1`, in :math:`10^{20}` :math:`m^{-3}`.
     - 
   * - ``[species.density]``
     - ``alpha``
     - For ``shape = 'parabolic'``, density profile is proportional to ``(1 - (rho/rho_edge)**alpha1)**alpha``.
     - 1
   * - ``[species.density]``
     - ``alpha1``
     - For ``shape = 'parabolic'``, density profile is proportional to ``(1 - (rho/rho_edge)**alpha1)**alpha``.
     - 2
   * - ``[species.density]``
     - ``coefs``
     - For ``shape = 'exponential'``, density profile is given by ``coefs[0]*np.exp(-coefs[1]*x**2 - coefs[2]*x**4 - coefs[3]*x**6``, with ``x`` = :math:`\rho`.
     - [0, 0, 0, 0]
   * - ``[species.density]``
     - ``import``
     - Import initial density profile from external file.
     - False
   * - ``[species.density]``
     - ``key``
     - Key (string) corresponding to density for import.
     - 
   * - ``[species.density]``
     - ``func``
     - For ``shape = 'custom'``, string literal containing python function with signature ``def init(rho):``, which will be used to initialize the density profile.
     - 
   * - ``[species]``
     - ``temperature``
     - Table controlling temperature initialization and evolution. See ``species.temperature`` parameters below.
     - {}
   * - ``[species.temperature]``
     - ``evolve``
     - Evolve the temperature in time.
     - True
   * - ``[species.temperature]``
     - ``init_to``
     - Initialize the temperature equal to some fraction of the temperature of another species, but allow profiles evolve independently after initialization. This can be either a string corresponding to a species, e.g. ``init_to = "electron"``, or a string containing a multiplicative fraction times the species, e.g. ``init_to = "0.5 * electron"``.
     - 
   * - ``[species.temperature]``
     - ``equal_to``
     - Initialize and maintain (as profiles evolve) the temperature equal to some fraction of the temperature of another species. This can be either a string corresponding to a species, e.g. ``equal_to = "electron"``, or a string containing a multiplicative fraction times the species, e.g. ``equal_to = "0.5 * electron"``.
     - 
   * - ``[species.temperature]``
     - ``shape``
     - Shape of initial temperature profile. Options are ``'parabolic'``, ``'exponential'``, ``'custom'``.
     - 'parabolic'
   * - ``[species.temperature]``
     - ``core``
     - For ``shape = 'parabolic'``, specifies temperature at :math:`\rho = 0`, in keV
     - 4
   * - ``[species.temperature]``
     - ``volavg``
     - For ``shape = 'parabolic'``, specifies volume-average temperature, in keV
     - 
   * - ``[species.temperature]``
     - ``lineavg``
     - For ``shape = 'parabolic'``, specifies line-average temperature, in keV
     - 
   * - ``[species.temperature]``
     - ``edge``
     - For ``shape = 'parabolic'``, specifies temperature at :math:`\rho = \rho_{edge}`, in keV
     - 0.5
   * - ``[species.temperature]``
     - ``sep``
     - For ``shape = 'parabolic'``, specifies temperature at :math:`\rho = 1`, in keV
     - 
   * - ``[species.temperature]``
     - ``alpha``
     - For ``shape = 'parabolic'``, temperature profile is proportional to ``(1 - (rho/rho_edge)**alpha1)**alpha``.
     - 1
   * - ``[species.temperature]``
     - ``alpha1``
     - For ``shape = 'parabolic'``, temperature profile is proportional to ``(1 - (rho/rho_edge)**alpha1)**alpha``.
     - 2
   * - ``[species.temperature]``
     - ``coefs``
     - For ``shape = 'exponential'``, temperature profile is given by ``coefs[0]*np.exp(-coefs[1]*x**2 - coefs[2]*x**4 - coefs[3]*x**6``, with ``x`` = :math:`\rho`.
     - [0, 0, 0, 0]
   * - ``[species.temperature]``
     - ``import``
     - Import initial temperature profile from external file.
     - False
   * - ``[species.temperature]``
     - ``key``
     - Key (string) corresponding to temperature for import.
     - 
   * - ``[species.temperature]``
     - ``func``
     - For ``shape = 'custom'``, string literal containing python function with signature ``def init(rho):``, which will be used to initialize the temperature profile.
     - 
   * - ``[species]``
     - ``aux_particle_source``
     - Table controlling the auxiliary particle source. See ``species.aux_particle_source`` parameters below.
     - {}
   * - ``[species.aux_particle_source]``
     - ``shape``
     - Shape of aux particle source. Options are ``'gaussian'``.
     - 'gaussian'
   * - ``[species.aux_particle_source]``
     - ``height``
     - For ``shape='gaussian'``, peak value of Gaussian aux particle source profile, in :math:`10^{20}` :math:`m^{-3} s^{-1}`
     - 0
   * - ``[species.aux_particle_source]``
     - ``integrated``
     - For ``shape='gaussian'``, desired volume-integrated aux particle source, in :math:`10^{20}` :math:`s^{-1}`
     - 
   * - ``[species.aux_particle_source]``
     - ``width``
     - For ``shape='gaussian'``, width of Gaussian aux particle source profile
     - 0.1
   * - ``[species.aux_particle_source]``
     - ``center``
     - For ``shape='gaussian'``, Gaussian aux particle source profile is centered at :math:`\rho =` ``center``.
     - 0
   * - ``[species]``
     - ``aux_power_source``
     - Table controlling the auxiliary power source. See ``species.aux_power_source`` parameters below.
     - {}
   * - ``[species.aux_power_source]``
     - ``shape``
     - Shape of aux power source. Options are ``'gaussian'``.
     - 'gaussian'
   * - ``[species.aux_power_source]``
     - ``height``
     - For ``shape='gaussian'``, peak value of Gaussian aux power source profile, in MW/ :math:`m^{3}`
     - 0
   * - ``[species.aux_power_source]``
     - ``integrated``
     - For ``shape='gaussian'``, desired volume-integrated aux power source, in MW
     - 
   * - ``[species.aux_power_source]``
     - ``width``
     - For ``shape='gaussian'``, width of Gaussian aux power source profile
     - 0.1
   * - ``[species.aux_power_source]``
     - ``center``
     - For ``shape='gaussian'``, Gaussian aux power source profile is centered at :math:`\rho =` ``center``.
     - 0
   
Note that since quasineutrality must be satisfied, the density of the last ion species listed in the input file will be set by quasineutrality (throughout the T3D simulation). This will effectively override parameters in the ``density`` table of that species if they were specified.


geometry
++++++++

The ``[geometry]`` group controls the (tokamak or stellarator) geometry used in the calculation.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[geometry]``
     - ``geo_option``
     - Type of geometry to use. Options are ``'vmec'``, ``'miller'``, ``'geqdsk'``
     -
   * - ``[geometry]``
     - ``geo_file``
     - Geometry file to generate geometry from. For ``geo_option = 'vmec'``, for example, this would be the VMEC ``*wout*.nc`` file.
     -
   * - ``[geometry]``
     - ``import``
     - Import geometry from file specified in [import]. Typically this only used for ``geo_option = 'miller'`` when importing data from TRANSP u-files.
     -
 

Note that there are some requirements for the various options:

- ``geo_option = 'vmec'`` requires ``flux_label = 'torflux'`` (in ``[grid]``). 
- ``geo_option = 'geqdsk'`` currently requires that GX be used for the flux model.
- ``geo_option = 'miller'`` requires ``flux_label = 'rminor'`` and ``import = true``.

rescale
+++++++

The ``[rescale]`` group allows one to change the magnetic field or size of a device that is represented by a VMEC output file.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[rescale]``
     - ``L_def``
     - Choose a radius to be specified. Options are ``'major'``, ``'minor'``. Note: The aspect ratio is fixed to the value found in the VMEC output file.
     -
   * - ``[rescale]``
     - ``R_major``
     - If L_def = ``'major``' then R_major sets the major radius in meters. 
     - L_mult
   * - ``[rescale]``
     - ``a_minor``
     - If L_def = ``'minor``' then a_minor sets the minor radius in meters. 
     - L_mult
   * - ``[rescale]``
     - ``L_mult``
     - Multiply all linear lengths by L_mult (float). 
     - 1.0
   * - ``[rescale]``
     - ``B_def``
     - Choose a definition of the magnetic field to be targeted. Options are ``'LCFS'``, ``'volavgB'``. ``'vmecB0'``. 
     -
   * - ``[rescale]``
     - ``Ba``
     - If B_def = ``'LCFS``' then Ba (float) sets magnetic field (T) through the last closed flux surface, defined to be the toroidal flux through the LCFS divided by the toroidally averaged area of the LCFS. 
     - 0
   * - ``[rescale]``
     - ``vmecB0``
     - If B_def = ``'vmecB0``' then vmecB0 (float) sets the on-axis magnetic field (T), defined to be <R B>/<R>, where <...> is the toroidal average.
     - B_mult
   * - ``[rescale]``
     - ``volavgB``
     - If B_def = ``'volavgB``' then volavgB (float) sets the volume averaged magnetic field strength (T).
     - B_mult
   * - ``[rescale]``
     - ``B_mult``
     - Multiply the magnetic field by B_mult (float). 
     - 1.0

       
physics
+++++++

The ``[physics]`` group controls various physics knobs for the calculation.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[physics]``
     - ``collisions``
     - Include collisional equilibration between species.
     - True
   * - ``[physics]``
     - ``alpha_heating``
     - Include alpha heating.
     - True
   * - ``[physics]``
     - ``f_tritium``
     - For alpha heating: if only deuterium ion species is present, assumes a fraction ``f_tritium`` of the deuterium ions are tritium. 
     - 0.5
   * - ``[physics]``
     - ``radiation``
     - Include radiation of impurity species if they are present. Includes line, bremsstrahlung, recombination.
     - False
   * - ``[physics]``
     - ``bremsstrahlung``
     - Include only bremsstrahlung radiation, based on :math:``Z_{eff}`` (doesn't necessarily require an impurity species to be included in the calculation)
     - False
   * - ``[physics]``
     - ``Zeff``
     - Specify Zeff as a constant or as a table (table specification follows e.g. density table specification in ``[species]``)
     - False
   * - ``[physics]``
     - ``turbulent_exchange``
     - Include turbulent exchange terms in pressure equation. 
     - False

import
++++++

The ``[import]`` group controls data that is imported from external files of various formats:

 - ``type = 'trinity'`` allows importing from an existing T3D .npy output file
 - ``type = 'columns'`` allows importing from a text file with data in columns
 - ``type = 'transp'`` allows importing from a TRANSP-style u-file

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[import]``
     - ``type``
     - Type of import. Options are: ``'columns'``, ``'trinity'``, ``'transp'``
     - 
   * - ``[import]``
     - ``file``
     - File to import
     - ''
   * - ``[import]``
     - ``frame``
     - For ``type = 'trinity'``, frame index to import data from. 
     - -1
   * - ``[import]``
     - ``columns``
     - For ``type = 'columns'``, list of keys (strings) corresponding to columns in file. E.g. ``columns = ['rho', 'n_e', 'T_e']``
     - []
   * - ``[import]``
     - ``divide_by``
     - For ``type = 'columns'``, list of factors to divide columns by to get expected units (:math:`\rho` should be normalized to range from 0 to 1, :math:`10^{20} m^{-3}` for density, keV for temperature, MW/ :math:`m^3` for power, etc.)
     - []
   * - ``[import]``
     - ``transp_time``
     - For ``type = 'transp'``, timeslice to import data from. 
     - 0.0

log
+++

The ``[log]`` group controls the output format. T3D will output a python pickle file which can be used for plotting but is also used as a restart file. T3D output data can also be written in NetCDF4 or ADIOS2 format.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[log]``
     - ``output_netcdf``
     - Write T3D output in NetCDF4 format
     - False
   * - ``[log]``
     - ``output_adios2``
     - Write T3D output in ADIOS2 format
     - True
