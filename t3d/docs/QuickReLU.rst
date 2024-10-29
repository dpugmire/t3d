.. _quickrelu:
  
Running your first T3D simulation (W7X ReLU regression test)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In this tutorial we run the W7X ReLU regression test. This test uses W7X geometry (via a VMEC equilibrium file) and a simple parametric turbulence model.

To run the test, make sure you are in the main ``t3d`` directory, and then simply use

.. code-block:: bash

  t3d tests/regression/test-w7x-relu.in -l tests/regression/test-w7x-relu.out

The code will then proceed to output data to the screen:

.. code-block:: bash

  $ t3d tests/regression/test-w7x-relu.in 
  Running T3D calculation.
  
    Loading input file: tests/regression/test-w7x-relu.in 
  
  
    Grid Information
      Radial coordinate is rho = sqrt(toroidal_flux/toroidal_flux_LCFS)
      N_radial: 9
      rho grid:             [0.1   0.175 0.25  0.325 0.4   0.475 0.55  0.625 0.7  ]
      flux (midpoint) grid:   [0.138 0.212 0.287 0.362 0.438 0.512 0.587 0.662]
  
    Loading VMEC geometry from tests/data/wout_w7x.nc
    Global Geometry Information
      R_major: 5.49 m
      a_minor: 0.49 m
      B_0: 2.6043144343662488 T
  
  This calculation contains ['hydrogen', 'electron'] species.
  The 'hydrogen' species is treated as the bulk ions.
  Evolving densities: []
  The 'hydrogen' density will be set by quasineutrality.
  Evolving temperatures: ['hydrogen']
  The 'electron' species will be treated adiabatically.
  Using 'hydrogen' as the reference species for turbulence calculations.
  Total number of (parallelizable) flux tube calculations per step = 16.
  
    Initializing normalizations
      n_ref = 1e+20 m^-3
      T_ref = 1000.0 eV
      B_ref = 1 T
      m_ref = 1.0 m_p
  
      a_ref = 0.49 m
      t_ref = 0.037 s
      P_ref = 0.428 MW/m^3
      S_ref = 26.708 10^20/(m^3 s)
      rho_star_ref = 6.533e-03
  
  time = 0.000e+00*t_ref, time index = 0, iteration index = 0, dtau = 0.1, rms = 5.359e-02
  time = 0.000e+00*t_ref, time index = 0, iteration index = 1, dtau = 0.1, rms = 3.757e-04
  *** Increasing timestep at time index 1 (3.757e-04 < 2.500e-03). new dtau = 0.2 ***
  time = 1.000e-01*t_ref, time index = 1, iteration index = 0, dtau = 0.2, rms = 8.345e-02
  time = 1.000e-01*t_ref, time index = 1, iteration index = 1, dtau = 0.2, rms = 9.990e-04
  *** Increasing timestep at time index 2 (9.990e-04 < 2.500e-03). new dtau = 0.4 ***
  time = 3.000e-01*t_ref, time index = 2, iteration index = 0, dtau = 0.4, rms = 9.319e-02
  time = 3.000e-01*t_ref, time index = 2, iteration index = 1, dtau = 0.4, rms = 2.261e-03
  /Users/nmandell/Codes/t3d/t3d/Geometry.py:102: IntegrationWarning: The occurrence of roundoff error is detected, which prevents 
    the requested tolerance from being achieved.  The error may be 
    underestimated.
    integral[i] = integrate.quad(integrand, 0, f.axis[i], limit=100, epsabs=1e-5)[0]
  *** Increasing timestep at time index 3 (2.261e-03 < 2.500e-03). new dtau = 0.8 ***
  time = 7.000e-01*t_ref, time index = 3, iteration index = 0, dtau = 0.8, rms = 5.692e-02
  time = 7.000e-01*t_ref, time index = 3, iteration index = 1, dtau = 0.8, rms = 1.397e-03
  *** Increasing timestep at time index 4 (1.397e-03 < 2.500e-03). new dtau = 1.6 ***
  time = 1.500e+00*t_ref, time index = 4, iteration index = 0, dtau = 1.6, rms = 6.343e-03
  time = 3.100e+00*t_ref, time index = 5, iteration index = 0, dtau = 1.6, rms = 8.834e-03
  time = 4.700e+00*t_ref, time index = 6, iteration index = 0, dtau = 1.6, rms = 6.224e-03
  time = 6.300e+00*t_ref, time index = 7, iteration index = 0, dtau = 1.6, rms = 1.207e-03
  *** Increasing timestep at time index 8 (1.207e-03 < 2.500e-03). new dtau = 3.2 ***
  time = 7.900e+00*t_ref, time index = 8, iteration index = 0, dtau = 3.2, rms = 1.165e-03
  *** Increasing timestep at time index 9 (1.165e-03 < 2.500e-03). new dtau = 6.4 ***
  Time evolution finished! time = 1.110e+01*t_ref, time index = 9
  
  T3D Complete, exiting normally.

.. note::

  The ``-l`` option specifies the name of the logfile the T3D terminal output is piped to.
