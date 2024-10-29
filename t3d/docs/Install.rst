.. _install:

Installing T3D
++++++++++++++

This page contains information for obtaining the T3D source code and building the code.

Obtaining the source code
-------------------------

Clone the `BitBucket <https://bitbucket.org/gyrokinetics/t3d>`_ repository using::

    git clone https://bitbucket.org/gyrokinetics/t3d

Navigate into the ``t3d`` directory to begin.

Building the code
-----------------
Installation is done with ``Setuptools`` and the ``pyproject.toml`` file. You can do this in a ``conda`` environment (recommended) OR a python ``venv`` virtual environment.

.. _conda:

Installing in a conda environment (recommended)
###############################################

First create your ``conda`` environment::

  conda env create -f environment.yml

Activate your environment::

  conda activate t3d

Install ``t3d`` using ``pip``::

  pip install -e .

.. _venv:

Installing in a venv environment (requires python >= 3.10)
##########################################################

First create your ``venv`` virtual environment::

  python3 -m venv /path/to/your/venvs/t3d

Activate your environment::

  source /path/to/your/venvs/t3d/bin/activate

Install ``t3d`` using ``pip``::

  pip install -e .

.. note::

  The ``-e`` option makes the installation editable. You can make changes to the code without recreating the installation. You only need to reinstall, using the same command as above, if you modify the ``pyproject.yml`` file.

Installing optional dependencies
################################

Some models available for coupling into T3D require additional Python package dependencies. To install the dependencies required for using e.g. the GX and KNOSOS models, use::

  pip install -e .[gx,knosos]

For example, this will install the ``booz_xform`` `package <https://hiddensymmetries.github.io/booz_xform/>`_ required for interfacing with KNOSOS. However, note that ``booz_xform`` itself may require installing additional `dependencies <https://hiddensymmetries.github.io/booz_xform/getting_started.html#requirements>`_.

.. note::

  The ``booz_xform`` package requires a local NetCDF installation. You may need to load an environment module, depending on where you are using T3D.

Using external flux models
##########################
 
In non-trivial use cases, T3D requires interfacing with one or more external codes that compute transport fluxes. The available interfaces are provided in ``t3d/flux_models``, and currently include GX (a gyrokinetic turbulence code) and KNOSOS (a stellarator neoclassical code). For more details about setting up T3D to use an available flux model, see :ref:`Flux Models <flux_models>`.

