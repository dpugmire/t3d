.. _developer:

Developer Notes
+++++++++++++++

This page contains information for developers.

Branching Model
---------------

T3D is currently being developed using a `trunk-based <https://www.atlassian.com/continuous-delivery/continuous-integration/trunk-based-development>`_ development strategy. This works best with small incremental changes. Long-lived branches should only be used for larger changes, such as new model development. It is the responsibility of developers to keep their branches current with the main branch.


Python Style Guide
------------------

We strive to adhere to the `PEP 8 style guide <https://peps.python.org/pep-0008/>`_ as much as possible. The CI will automatically update the formatting when a development branch is merged into the main branch. Developers are encouraged to run

.. code-block:: bash

  flake8
  
or

.. code-block:: bash

  autopep8
  
while working on the code, by simply running either command from the ``t3d`` home directory. Note that ``flake8`` will only report on formatting issues whereas ``autopep8`` will also fix them. Errors and formatting issues that are currently being ignored are listed in the ``.flake8`` and ``pyproject.toml`` files.


Code Testing 
------------

Unit Testing
############

To run all the unit tests, use the following from the main directory

.. code-block:: bash

  t3d-unittests

which is a short-cut command for ``python3 -m unittest``. Individual unit tests can be performed using, for example

.. code-block:: bash

  python3 -m unittest tests/unit/test_import.py

See the python files in ``tests/unit`` for the available unit tests. Developers are encouraged to add new unit tests.

Regression Testing
##################

Regression testing can be performed by running the test cases in the ``tests/regression/`` directory. Currently there are only reference results for the serial tests with the ReLU model.

.. code-block:: bash

  t3d tests/regression/test-w7x-relu.in
  t3d tests/regression/test-jet-relu.in

The `bpcmp <https://github.com/PrincetonUniversity/bpcmp>`_ utility is used to compare the ADIOS2 bp output file from the new and old results. This utility is installed as part of the ``t3d`` installation.

.. code-block:: text

  bpcmp --ignore-atts title history inputs infile -r 1e-9 -v 1 tests/regression/test-w7x-relu.bp tests/regression/ref-w7x-relu.bp

The ``-v`` option sets the verbosity level with options of 0 for quiet mode, 1 to list all attributes/variables with differences, and 2 to list the status of the comparison for all attributes/variables.

Continuous Integration (CI)
###########################

CI is performed using bitbucket pipelines. Unit and regression tests are run on branches pushed to the remote repository. he regression tests only perform serial tests with the ReLU model. Upon merging into the main branch a linting stage is performed using ``autopep8`` which updates the formatting to (mostly) adhere to the `PEP 8 style guide <https://peps.python.org/pep-0008/>`_. 


Updating Documentation
----------------------

T3D documentation is maintained with Read-the-Docs. To review your documentation changes locally, first install the required packages

.. code-block:: bash

  pip install -e .[docs]

Then build the documentation

.. code-block:: bash

  sphinx-autobuild docs/ docs/_build/html

Some warnings and errors will be printed to the terminal screen. The documentation can be viewed by going to `http://127.0.0.1:8000 <http://127.0.0.1:8000>`_.

.. note::

  The local documentation will update as changes are saved.


Terminal and file log output
----------------------------

A ``Logbook`` class was developed to stream output to both the terminal and a specified log file. To use, python files should import the Logbook with
.. code-block:: python

  from Logbook import info, bold, emph, warn, errr, debug

Do not import unused functions. These functions should be used instead ``print``. A color can be passed to this function so that terminal output is colorized. All colors are stripped from the output string before writing to the log file otherwise the log file will contain ascii characters when viewed with most editors. For example,

.. code-block:: python

  info(msg=f"x = {x}", color=green)

The color input is optional and will use the default color if not specified. Other output functions include

.. code-block:: python

  bold(msg=f"x = {x}", color=green)  # The output will be the specified color and bold
  emph(msg=f"x = {x}")  # The output will be blue and bold
  warn(msg=f"x = {x}")  # The output will be yellow, bold, and include a WARNING prefix
  errr(msg=f"x = {x}")  # The output will be red, bold, and include a ERROR prefix
  debug(msg=f"x = {x}")  # The output will only be written to the log file

Flux models have their own instance of the Logbook class which only write output to files.

Available colors include:

* red
* green
* yellow
* blue
* magenta
* cyan
* and light\_ versions of the above
