.. _flux_models:

Flux Models
===========

In non-trivial use cases, T3D requires interfacing with one or more external codes that compute transport fluxes. The available interfaces and instructions for using them are provided below. The available interfaces can also be found in ``t3d/flux_models``. If you would like to add a new flux model interface, see :ref:`Adding New Flux Models <add_model>`.

The model(s) used in the calculation are controlled by the ``[[model]]`` group in the T3D input file, via

.. code-block:: toml

  [[model]]
    model = "..."

The possible values of this parameter are given in the headings below. For each model, there are additional ``[[model]]`` parameters that can/should be specified, as detailed below.

.. _gx_model:

model = "GX"
------------

`GX <https://gx.rtfd.io/en/latest/>`_ is a gyrokinetic turbulence code that runs on NVIDIA GPUs. After `installing <https://gx.readthedocs.io/en/latest/Install.html>`_ GX on your target system, set the ``GX_PATH`` environment variable via::

  export GX_PATH=[/path/to/gx_repo]

so that the ``gx`` executable can be found at ``$GX_PATH/gx``. If T3D cannot find the ``gx`` executable it will throw an error::

  Error: gx executable not found! Make sure the GX_PATH environment variable is set.

Additionally, T3D requires a template GX input file, which will be used to set parameters such as the numerical resolution for the GX calculation. The (relative) path to the template file should be passed via the ``gx_template`` parameter in ``[[model]]``. This need not be a full GX input file (although it can be), as T3D will automatically modify and/or add many parameters. An example GX template is given in ``tests/regression/gx_template.in``. The most critical tunable parameters are in ``[Dimensions]`` (numerical resolution), ``[Domain]`` (box dimensions), and ``[Time]`` (length of GX calculation).

In addition to setting ``model = "GX"``, the following parameters in the ``[model]`` group can/should be specified in the T3D input file.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - ``[model]``
     - ``gx_template``
     - Path to template input file for GX.
     - "tests/regression/gx_template.in"
   * - ``[model]``
     - ``gx_outputs``
     - GX files (input files and output files) will be written to this directory.
     - "gx/"
   * - ``[model]``
     - ``overwrite``
     - If False, read fluxes from existing outputs instead of re-running GX if the GX output files corresponding to the current timestep/iteration/radius are found. Useful for "replaying" a simulation or continuing a simulation.
     - False
   * - ``[model]``
     - ``gpus_per_gx``
     - The number of GPUs to use for each GX calculation. The total number of GPUs used for the T3D calculation can be up to (``N_radial``-1) * ``gpus_per_gx`` * (number of evolved profiles+1).
     - 1
   * - ``[model]``
     - ``electromagnetic``
     - If False, beta will be set to :math:`10^{-4}` in the GX input file to effectively force the GX calculation into the electrostatic limit.
     - True
 

model = "KNOSOS"
----------------

.. `KNOSOS <https://github.com/joseluisvelasco/KNOSOS>`_ is a neoclassical transport code for stellarators. After `installing <https://github.com/joseluisvelasco/KNOSOS/blob/master/MANUAL/KNOSOSManual.pdf>`_ KNOSOS on your target system, set the ``KNOSOS_PATH`` environment variable via::
.. 
..   export KNOSOS_PATH=[/path/to/KNOSOS/SOURCES/]
.. 
.. so that the ``knosos.x`` executable can be found at ``$KNOSOS_PATH/knosos.x``. If T3D cannot find the ``knosos.x`` executable it will throw an error::
.. 
..   Error: knosos.x executable not found! Make sure the KNOSOS_PATH environment variable is set.

model = "ReLU"
--------------

.. _add_model:

Adding New Flux Models
----------------------
