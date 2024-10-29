.. _quickgx:
  
Running your first T3D+GX simulation (W7X GX regression test)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In this tutorial we run the W7X GX regression test. This test uses W7X geometry (via a VMEC equilibrium file) and turbulence calculations using the GX code. Make sure you have installed GX by following the instructions in :ref:`Flux Models <gx_model>`.

To run the test, you will need access to at least 1 (and up to 16) NVIDIA GPU(s). For example, on a system with resource management (e.g. SLURM or PBS), you should either request an interactive job with up to 16 GPUs and then run the below command once the job starts, or create a job script that requests up to 16 GPUs and run the below command in the script.

.. code-block:: bash

  t3d tests/regression/test-w7x-gx.in

On 16 A100 GPUs, this test should take about 30 min. 

.. note::

  You do not need to use ``srun`` or another MPI command when running ``t3d``. 

Example job scripts
-------------------

Below you can find example job scripts to run the W7X GX regression test on 16 GPUs on various machines. These jobs should be submitted from the `top-level` (``t3d``) directory. These examples assume that T3D has been installed in a conda environment called ``t3d`` (see :ref:`Installing T3D <conda>`).

Perlmutter (NERSC)
==================

.. code-block:: bash

  #!/bin/bash
  #SBATCH -N 4
  #SBATCH -n 16
  #SBATCH --gpus-per-node=4
  #SBATCH --time=0:30:00
  #SBATCH -A [INSERT ACCOUNT HERE]
  #SBATCH --qos=debug
  #SBATCH -C gpu
  #SBATCH -J t3d
  
  # load modules here, or source ~/.bashrc

  conda activate t3d
  t3d tests/regression/test-w7x-gx.in -l tests/regression/test-w7x-gx.out

Traverse (Princeton)
====================

.. code-block:: bash

  #!/bin/bash
  #SBATCH -N 4
  #SBATCH -n 16
  #SBATCH --gpus-per-node=4
  #SBATCH --time=0:30:00
  #SBATCH -J t3d
  #SBATCH --exclusive

  # load modules here, or source ~/.bashrc
  
  conda activate t3d
  t3d tests/regression/test-w7x-gx.in -l tests/regression/test-w7x-gx.out

Summit (ORNL)
=============

.. code-block:: bash

  #!/bin/bash
  #BSUB -P [INSERT ACCOUNT HERE]
  #BSUB -W 1:00
  #BSUB -nnodes 3
  #BSUB -J t3d
  #BSUB -o t3d.%J
  #BSUB -e t3d.%J
  #BSUB -B
  #BSUB -N
  
  # load modules here, or source ~/.bashrc
  
  conda deactivate
  conda activate t3d
  t3d tests/regression/test-w7x-gx.in -l tests/regression/test-w7x-gx.out

.. note::

  The ``-l`` option specifies the name of the logfile the T3D terminal output is piped to.
